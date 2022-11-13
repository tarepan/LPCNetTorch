"""Data transformation"""

from dataclasses import dataclass
from typing import List
from pathlib import Path
import math

from omegaconf import MISSING
from torch import from_numpy, stack # pyright: ignore [reportUnknownVariableType] ; because of PyTorch ; pylint: disable=no-name-in-module
import numpy as np
from numpy.typing import NDArray
import librosa

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

def ls16_to_mu8q(series: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Convert linear int16 range (q|c) series into mulaw uint8 range quantized series."""
    return 128 + librosa.mu_compress(series/32768., mu=255, quantize=True) # pyright: ignore [reportUnknownMemberType, reportUnknownVariableType]; because of librosa


def mu8q_to_ls16c(series: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Convert mulaw uint8 range quantized series into linear int16 range continuous series."""
    return 32768. * librosa.mu_expand(series-128, mu=255.0, quantize=True) # pyright: ignore [reportUnknownMemberType, reportUnknownVariableType]; because of librosa


def emulate_noisy_sample_frame(
    s_t_clean_frame_ls16q: NDArray[np.int16],
    coeffs: NDArray[np.float32],
    noise_ms8q: NDArray[np.int8],
    s_0P1_noisy_ls16c: NDArray[np.float32],
    ) -> NDArray[np.float32]:
    """Emulate LPCNet-inferred s_t_noisy sample series of a frame.

    Args:
        s_t_clean_frame_ls16q - s_t_clean ground-truth series of the frame
        coeffs                - LP coefficients [a1, ..., aP] of the frame
        noise_ms8q            - Inference error (noise) of each samples in mulaw domain
        s_0P1_noisy_ls16c     - Past samples s_{-P} ~ s_{-1}
    Returns:
        s_t_noisy_frame_ls16c

    Residual e_t_estim is inferred as quantized value by NN, so it contain QuantizationError (QuantErr) and InferenceError (InferErr).
    As a result, inferred sample s_t_estim contains QuantErr and InferErr.
    If we use ground-truth sample s_t_clean for AR training, training input distribution do not match s_t_estim AR input distribution.
    For this reason, noisy sample series s_t_noisy seems to be needed for good inference quality.

    Conceptually, s_t_noisy is equal to 's_t_clean + QuantErr + InferErr'.
    It is proven by this equations.

    e_t_ideal   = s_t_clean   - LP(s_t1_tP_noisy)    # Ideal residual       - perfectly reconstruct sample under erroneous (noisy) previous samples
    e_q_t_ideal = e_t_ideal   + QuantErr             # Ideal quant residual - Network infer quantized value, so QuantErr is unavoidable
    e_t_noisy   = e_q_t_ideal + InferErr             # Noisy residual       - Realistically, Network outputs erronous (noisy) residual
    s_t_noisy   = e_t_noisy   + LP(s_t1_tP_noisy)    # Noisy sample         - Realistically, output sample is noisy

    s_t_noisy =                           e_t_noisy                     + LP(s_t1_tP_noisy)
            = (                   e_q_t_ideal               + InferErr) + LP(s_t1_tP_noisy)
            = ((         e_t_ideal              + QuantErr) + InferErr) + LP(s_t1_tP_noisy)
            = (((s_t_clean - LP(s_t1_tP_noisy)) + QuantErr) + InferErr) + LP(s_t1_tP_noisy)
            = s_t_clean + QuantErr + InferErr ■

    QuantErr and InferErr are calculated in residual domain, and e_q_t_ideal needs s_t1_tP_noisy because of LP.
    It means that, for s_t_noisy, we needs s_t1_noisy ~ s_tP_noisy.
    For this setup, we must use AR generation of s_t_noisy (not parallelizable).
    """

    order = len(coeffs)
    # [a_P, ..., a_1]
    coeffs = np.flip(coeffs) # pyright: ignore [reportUnknownMemberType]; because of numpy
    # Append samples for initial LP
    s_t_noisy_frame_ls16c = s_0P1_noisy_ls16c

    for s_t_clean_ls16q, noise_t_ms8q in zip(s_t_clean_frame_ls16q, noise_ms8q):
        # For typing
        s_t_clean_ls16q: NDArray[np.int16] = s_t_clean_ls16q
        noise_t_ms8q: NDArray[np.int8] = noise_t_ms8q

        p_t_noisy_ls16c: NDArray[np.float32] = np.dot(coeffs, s_t_noisy_frame_ls16c[-order:])          # pyright: ignore [reportUnknownMemberType]; because of numpy
        e_q_t_ideal_mu8q = ls16_to_mu8q(s_t_clean_ls16q - p_t_noisy_ls16c)
        e_t_noisy_mu8q: NDArray[np.uint8] = np.clip(e_q_t_ideal_mu8q + noise_t_ms8q, 0, 255)           # pyright: ignore [reportUnknownMemberType]; because of numpy
        s_t_noisy_ls16c = p_t_noisy_ls16c + mu8q_to_ls16c(e_t_noisy_mu8q)
        s_t_noisy_frame_ls16c: NDArray[np.float32] = np.append(s_t_noisy_frame_ls16c, s_t_noisy_ls16c) # pyright: ignore [reportUnknownMemberType]; because of numpy

    # Remove samples for initial LP
    s_t_noisy_frame_ls16c = s_t_noisy_frame_ls16c[order:]

    return s_t_noisy_frame_ls16c


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
    # Currently, not implemented
    #   - Fixed High-pass filter
    #   - Random equalizer
    #   - preemphasis
    #   - gain

    # We should implement
    # shift

    # Feature extraction (wave -> MelSpc & LPCoeff)

    # s_{t-1}_noisy by LPC


    order = 12
    tiny = 1e-4
    sample_per_frame = 160
    rng = np.random.default_rng()
    s_t_0P1_noisy_ls16c: NDArray[np.float32] = np.array([0. for _ in range(order)], dtype=np.float32)
    for _ in range(10):
        noise_std = -1.5*math.log(tiny + np.random.random()) - 0.5*math.log(tiny + np.random.random())
        noise_ms8q = np.around(rng.laplace(0.0, noise_std *.707, sample_per_frame)).astype(np.int8) # pyright: ignore [reportUnknownMemberType]; because of numpy
        s_t_noisy_ls16c = emulate_noisy_sample_frame(s_t_clean_frame_ls16q, coeffs, noise_ms8q, s_t_0P1_noisy_ls16c)
        s_t_0P1_noisy_ls16c = s_t_noisy_ls16c[-order:]
        # save quantized

    return piyo_to_hoge(conf.piyo2hoge, raw), piyo_to_fuga(conf.piyo2fuga, raw)


def generate_s_t1_noisy():
    """
    Generate noisy s_{t-1} sample series from clean s_t sample series for Network input.

    This function assume that signal will be estimated by Linear Prediction `s_t = e_t + Σa_p*s_t_p`.
    e_t is quantized value estimated by NN, so it has QuantizationError and InferenceError (below equation).
    s_t = NN + Σa_p*s_t_p = e_t + InferErr + QuantErr + Σa_p*s_t_p
    QuantizationError is not unavoidable, so we mimic inferenceError with noise for robust training.
    First, calculate ideal e_t, which has QuantizationError.
    結局NNの頭でmulaw8bitに圧縮するけど意味あるのかこれ？
    e_tにノイズ乗っけるだけじゃダメ? -> わからないけど、s_t_noisy を s_t_p_noisy から作るので結局全長ARにはなる
    """
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
    pitch_series_datum:       PitchSeriesDatum    = pitch_series.astype(np.int32)       # type: ignore
    s_t_1_series_noisy_datum: St1SeriesNoisyDatum = s_t_1_series_noisy.astype(np.int32) # type: ignore
    s_t_series_clean_datum:   StSeriesCleanDatum  = s_t_series_clean.astype(np.int32)   # type: ignore

    # Unneeded padding removal :: (T=frm_cnk+pad, Order) -> (T=frm_cnk, Order)
    pad_left, pad_right = (conf.padding - conf.lookahead), conf.lookahead
    lpcoeff_series_datum: LPCoeffSeriesDatum = lpcoeff_series[pad_left:-pad_right]

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
