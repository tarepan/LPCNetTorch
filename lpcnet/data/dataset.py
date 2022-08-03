"""Datasets"""


from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
from hashlib import md5

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset
from tqdm import tqdm
from omegaconf import MISSING
from speechdatasety.interface.speechcorpusy import AbstractCorpus, ItemId               # pyright: ignore [reportMissingTypeStubs]
from speechdatasety.helper.archive import try_to_acquire_archive_contents, save_archive # pyright: ignore [reportMissingTypeStubs]
from speechdatasety.helper.adress import dataset_adress                                 # pyright: ignore [reportMissingTypeStubs]
from speechdatasety.helper.access import generate_saver_loader                          # pyright: ignore [reportMissingTypeStubs]

from ..domain import FPitchCoeffSt1nStcBatch
from .domain import FeatSeries, PitchSeries, LPCoeffSeries, St1SeriesNoisy, StSeriesClean, FPitchCoeffSt1nStcDatum
from .transform import ConfTransform, load_raw, preprocess, augment, collate


CorpusItems = Tuple[AbstractCorpus, List[Tuple[ItemId, Path]]]


@dataclass
class ConfFPitchCoeffSt1nStcDataset:
    """Configuration of FPitchCoeffSt1nStc dataset.
    Args:
        adress_data_root - Root adress of data
        att1 - Attribute #1
        ndim_feat - The size of feature series's feature dimension
        sample_per_frame - Waveform samples per acoustic frame [samples/frame]
        frame_per_chunk - Acoustic frames per training chunk [frame/chunk]
        path_sample_series - Path of s_t_1_s_t
        path_feat_lpc_series - Path of feat_lpc
        lpc_order - The LPC order
        padding_frame - The number of padding [frame]
        lookahead_frame - The number of lookahead [frame]
    """
    adress_data_root: Optional[str] = MISSING
    attr1: int = MISSING
    transform: ConfTransform = ConfTransform()
    ndim_feat: int = MISSING
    sample_per_frame: int = MISSING
    frame_per_chunk: int = MISSING # 15
    path_sample_series: str = MISSING
    path_feat_lpc_series: str = MISSING
    lpc_order: int = MISSING # 16
    padding_frame: int = MISSING # 4
    lookahead_frame: int = MISSING # 2

class FPitchCoeffSt1nStcDataset(Dataset[FPitchCoeffSt1nStcDatum]):
    """The Feat/Pitch/LPCoeff/S_t_1_noisy/S_t_clean dataset from the corpus.
    """
    # def __init__(self, conf: ConfFPitchCoeffSt1nStcDataset, items: CorpusItems):
    def __init__(self, conf: ConfFPitchCoeffSt1nStcDataset):
        """
        Args:
            conf: The Configuration
            items: Corpus instance and filtered item information (ItemId/Path pair)
        """

        # Store parameters
        self._conf = conf
        # self._corpus = items[0]
        # self._items = items[1]

        # # Calculate data path
        # conf_specifier = f"{conf.attr1}{conf.transform}"
        # item_specifier = f"{list(map(lambda item: item[0], self._items))}"
        # exp_specifier = md5((conf_specifier+item_specifier).encode()).hexdigest()
        # self._adress_archive, self._path_contents = dataset_adress(
        #     conf.adress_data_root, self._corpus.__class__.__name__, "HogeFuga", exp_specifier
        # )
        # self._save_hogefuga, self._load_hogefuga = generate_saver_loader(HogeFuga_, ["hoge", "fuga"], self._path_contents)

        # # Deploy dataset contents
        # ## Try to 'From pre-generated dataset archive'
        # contents_acquired = try_to_acquire_archive_contents(self._adress_archive, self._path_contents)
        # ## From scratch
        # if not contents_acquired:
        #     print("Dataset archive file is not found.")
        #     self._generate_dataset_contents()

        sample_per_chunk = conf.sample_per_frame * conf.frame_per_chunk

        # np.memmap for partial access to single big file
        ## Series of (s_{t-1}, s_t) :: (T=t_s, IO=2) linear_s16pcm
        s_t_1_s_t_series = np.memmap(conf.path_sample_series, dtype='int16', mode='r').reshape((-1, 2))

        # The number of chunks in <data.s16>
        num_chunk: int = len(s_t_1_s_t_series) // sample_per_chunk - 1
        ## Why -1? For look-ahead clipping?

        pad_left_frame = conf.padding_frame - conf.lookahead_frame

        # Samples
        # Padding - s[0:pad_left] is only for padding, so no needs of AR&GT waveform
        s_t_1_s_t_series = s_t_1_s_t_series[pad_left_frame * conf.sample_per_frame:]
        # Discard chippings
        s_t_1_s_t_series = s_t_1_s_t_series[:num_chunk * sample_per_chunk]
        # Chunk-nize :: (T=t_s, IO=2) -> (Chunk, T=spl_cnk, IO=2)
        s_t_1_s_t_series = s_t_1_s_t_series.reshape((num_chunk, sample_per_chunk, 2))

        # :: (Chunk, ...St1SeriesNoisy)
        self._s_t_1_noisy_series = s_t_1_s_t_series[:, :, 0]
        # :: (Chunk, ...StSeriesClean)
        self._s_t_clean_series   = s_t_1_s_t_series[:, :, 1]

        # Features
        feat_lpc_series_linearized = np.memmap(conf.path_feat_lpc_series, dtype='float32', mode='r')
        dim_feat_lpc = conf.ndim_feat + conf.lpc_order
        byte_value = feat_lpc_series_linearized.strides[-1]
        byte_frame = dim_feat_lpc * byte_value
        byte_chunk = conf.frame_per_chunk * byte_frame
        # Overlap-strided frames :: (Chunk, T=frm_cnk+pad, F=feat+order)
        feat_lpc_series: NDArray[np.float32] = np.lib.stride_tricks.as_strided( # type: ignore
            feat_lpc_series_linearized,
            shape=(num_chunk, conf.frame_per_chunk + conf.padding_frame, dim_feat_lpc),
            strides=(byte_chunk, byte_frame, byte_value)
        )

        # :: (Chunk, ...FeatSeries)
        self._feat_series    = feat_lpc_series[:, :,                  :-1*conf.lpc_order]
        # :: (Chunk, ...LPCoeffSeries)
        self._lpcoeff_series = feat_lpc_series[:, :, -1*conf.lpc_order:                 ]

        # feat_series[:,:,-1] - Pitch Correlation series / feat_series[:,:,-2] - Pitch Period series
        pitch_period_series = self._feat_series[:, :, -2]
        # :: (Chunk, ...PitchSeries)
        self._pitch_period_series = (.1 + 50 * pitch_period_series + 100).astype("int16")

    # def _generate_dataset_contents(self) -> None:
    #     """Generate dataset with corpus auto-download and static preprocessing.
    #     """

    #     print("Generating new dataset...")

    #     # Lazy contents download
    #     self._corpus.get_contents()

    #     # Preprocessing - Load/Transform/Save
    #     for item_id, item_path in tqdm(self._items, desc="Preprocessing", unit="item"):
    #         piyo = load_raw(self._conf.transform.load, item_path)
    #         hoge_fuga = preprocess(self._conf.transform.preprocess, piyo)
    #         self._save_hogefuga(item_id, hoge_fuga)

    #     print("Archiving new dataset...")
    #     save_archive(self._path_contents, self._adress_archive)
    #     print("Archived new dataset.")

    #     print("Generated new dataset.")

    def __getitem__(self, n: int) -> FPitchCoeffSt1nStcDatum:
        """(API) Load the n-th datum from the dataset with tranformation.
        """
        feat_series:        FeatSeries     =         self._feat_series[n]
        pitch_series:       PitchSeries    = self._pitch_period_series[n]
        lpcoeff_series:     LPCoeffSeries  =      self._lpcoeff_series[n]
        s_t_1_noisy_series: St1SeriesNoisy =  self._s_t_1_noisy_series[n]
        s_t_clean_series:   StSeriesClean  =    self._s_t_clean_series[n]

        return augment(self._conf.transform.augment, (feat_series, pitch_series, lpcoeff_series, s_t_1_noisy_series, s_t_clean_series))

    def __len__(self) -> int:
        return len(self._s_t_1_noisy_series)

    def collate_fn(self, items: List[FPitchCoeffSt1nStcDatum]) -> FPitchCoeffSt1nStcBatch:
        """(API) datum-to-batch function."""
        return collate(items)
