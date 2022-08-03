"""Data transformation"""

from dataclasses import dataclass
from typing import List
from pathlib import Path

from omegaconf import MISSING
from torch import from_numpy, stack # pyright: ignore [reportUnknownVariableType] ; because of PyTorch ; pylint: disable=no-name-in-module

from ..domain import FPitchCoeffSt1nStcBatch
from .domain import FeatSeries, St1SeriesNoisy, FPitchCoeffSt1nStc, \
    FeatSeriesDatum, PitchSeriesDatum, LPCoeffSeriesDatum, St1SeriesNoisyDatum, StSeriesCleanDatum, FPitchCoeffSt1nStcDatum


# [Data transformation]
#
#      load        preprocessing            augmentation              collation
#     -----> raw -----------------> item -----------------> datum -----------------> batch
#                 before training            in Dataset             in DataLoader

###################################################################################################################################
# [Load]
"""
(delele here when template is used)

[Design Notes - Load as transformation]
    Loading determines data shape, and load utilities frequently modify the data.
    In this meaning, load has similar 'transform' funtionality.
    For this reason, `load_raw` is placed here.
"""

@dataclass
class ConfLoad:
    """
    Configuration of piyo loading.
    Args:
        sampling_rate - Sampling rate
    """
    sampling_rate: int = MISSING

def load_raw(conf: ConfLoad, path: Path) -> St1SeriesNoisyDatum:
    """Load raw data 'piyo' from the adress."""

    # Audio Example (librosa is not handled by this template)
    import librosa # pyright: ignore [reportMissingImports, reportUnknownVariableType] ; pylint: disable=import-outside-toplevel,import-error
    piyo: St1SeriesNoisyDatum = librosa.load(path, sr=conf.sampling_rate, mono=True)[0] # pyright: ignore [reportUnknownMemberType]

    return piyo

###################################################################################################################################
# [Preprocessing]

@dataclass
class ConfPiyo2Hoge:
    """
    Configuration of piyo-to-hoge preprocessing.
    Args:
        amp - Amplification factor
    """
    amp: float = MISSING

def piyo_to_hoge(conf: ConfPiyo2Hoge, piyo: St1SeriesNoisyDatum) -> St1SeriesNoisy:
    """Convert piyo to hoge.
    """
    # Amplification :: (T,) -> (T,)
    hoge: St1SeriesNoisy = piyo * conf.amp

    return hoge


@dataclass
class ConfPiyo2Fuga:
    """
    Configuration of piyo-to-fuga preprocessing.
    Args:
        div - Division factor
    """
    div: float = MISSING

def piyo_to_fuga(conf: ConfPiyo2Fuga, piyo: St1SeriesNoisyDatum) -> FeatSeries:
    """Convert piyo to fuga.
    """
    # Division :: (T,) -> (T,)
    fuga: FeatSeries = piyo / conf.div

    return fuga

@dataclass
class ConfPreprocess:
    """
    Configuration of item-to-datum augmentation.
    Args:
        len_clip - Length of clipping
    """
    piyo2hoge: ConfPiyo2Hoge = ConfPiyo2Hoge()
    piyo2fuga: ConfPiyo2Fuga = ConfPiyo2Fuga()

def preprocess(conf: ConfPreprocess, raw: St1SeriesNoisyDatum) -> FPitchCoeffSt1nStc:
    """Preprocessing (raw_to_item) - Process raw data into item.

    Piyo -> Hoge & Fuga
    """
    return piyo_to_hoge(conf.piyo2hoge, raw), piyo_to_fuga(conf.piyo2fuga, raw)

###################################################################################################################################
# [Augmentation]

@dataclass
class ConfAugment:
    """
    Configuration of item-to-datum augmentation.
    Args:
        placeholder - Pass through
    """
    padding: int = MISSING
    lookahead: int = MISSING

def augment(conf: ConfAugment, f_pitch_coeff_st1n_stc: FPitchCoeffSt1nStc) -> FPitchCoeffSt1nStcDatum:
    """Augmentation (item_to_datum) - Dynamically modify item into datum.

    Pass-through
    """
    feat_series, pitch_series, lpcoeff_series, s_t_1_series_noisy, s_t_series_clean = f_pitch_coeff_st1n_stc

    # Pass-through
    feat_series_datum:        FeatSeriesDatum     = feat_series
    pitch_series_datum:       PitchSeriesDatum    = pitch_series
    s_t_1_series_noisy_datum: St1SeriesNoisyDatum = s_t_1_series_noisy
    s_t_series_clean_datum:   StSeriesCleanDatum  = s_t_series_clean

    # Unneeded padding removal :: (T=frm_cnk+pad, Order) -> (T=frm_cnk, Order)
    pad_left, pad_right = (conf.padding - conf.lookahead), conf.lookahead
    lpcoeff_series_datum: LPCoeffSeriesDatum = lpcoeff_series[:, pad_left:-pad_right]

    return feat_series_datum, pitch_series_datum, lpcoeff_series_datum, s_t_1_series_noisy_datum, s_t_series_clean_datum

###################################################################################################################################
# [collation]

def collate(datums: List[FPitchCoeffSt1nStcDatum]) -> FPitchCoeffSt1nStcBatch:
    """Collation (datum_to_batch) - Bundle multiple datum into a batch."""

    # Pass-through
    feat_series        = stack([from_numpy(datum[0]) for datum in datums])
    pitch_series       = stack([from_numpy(datum[1]) for datum in datums])
    lpcoeff_series     = stack([from_numpy(datum[2]) for datum in datums])
    s_t_1_noisy_series = stack([from_numpy(datum[3]) for datum in datums])
    s_t_clean_series   = stack([from_numpy(datum[4]) for datum in datums])

    return feat_series, pitch_series, lpcoeff_series, s_t_1_noisy_series, s_t_clean_series

###################################################################################################################################

@dataclass
class ConfTransform:
    """Configuration of data transform."""
    load: ConfLoad = ConfLoad()
    preprocess: ConfPreprocess = ConfPreprocess()
    augment: ConfAugment = ConfAugment()
