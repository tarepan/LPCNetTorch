"""Data domain"""


from typing import Tuple

import numpy as np
from numpy.typing import NDArray


# `XX_` is for typing

# [Statically-preprocessed item]
#     FeatSeries     :: (T=frm_cnk+pad, F)     - Feature series
#     PitchSeries    :: (T=frm_cnk+pad,)       - Pitch Period series
#     LPCoeffSeries  :: (T=frm_cnk+pad, Order) - LinearPrediction Coefficient series
#     St1SeriesNoisy :: (T=spl_cnk,)           - Lagged/Delayed sample series (linear_s16pcm waveform) with noise
#     StSeriesClean  :: (T=spl_cnk,)           - Sample series (linear_s16pcm waveform) w/o noise

FeatSeries     = NDArray[np.float32]
FeatSeries_: FeatSeries         = np.array([[1., 2.]], dtype=np.float32) # pyright: ignore [reportUnknownMemberType]
PitchSeries    = NDArray[np.int16]
PitchSeries_: PitchSeries       = np.array([1, 2],     dtype=np.int16)   # pyright: ignore [reportUnknownMemberType]
LPCoeffSeries  = NDArray[np.float32]
LPCoeffSeries_: LPCoeffSeries   = np.array([[1., 2.]], dtype=np.float32) # pyright: ignore [reportUnknownMemberType]
St1SeriesNoisy = NDArray[np.int16]
St1SeriesNoisy_: St1SeriesNoisy = np.array([1, 2,],    dtype=np.int16)   # pyright: ignore [reportUnknownMemberType]
StSeriesClean  = NDArray[np.int16]
StSeriesClean_: StSeriesClean   = np.array([1, 2],     dtype=np.int16)   # pyright: ignore [reportUnknownMemberType]

## the item
FPitchCoeffSt1nStc = Tuple[FeatSeries, PitchSeries, LPCoeffSeries, St1SeriesNoisy, StSeriesClean]
FPitchCoeffSt1nStc_: FPitchCoeffSt1nStc = (FeatSeries_, PitchSeries_, LPCoeffSeries_, St1SeriesNoisy_, StSeriesClean_)


# [Dynamically-transformed Dataset datum]
#     FeatSeriesDatum     :: (T=frm_cnk+pad, F)     - Feature series
#     PitchSeriesDatum    :: (T=frm_cnk+pad,)       - Pitch Period series
#     LPCoeffSeriesDatum  :: (T=frm_cnk,     Order) - LinearPrediction Coefficient series
#     St1SeriesNoisyDatum :: (T=spl_cnk,)           - Lagged/Delayed sample series (linear_s16pcm waveform) with noise
#     StSeriesCleanDatum  :: (T=spl_cnk,)           - Sample series (linear_s16pcm waveform) w/o noise
#
#     Pitch and Feature become short by padding-less ('valid') Convs, so needs padding in input

FeatSeriesDatum     = NDArray[np.float32]
PitchSeriesDatum    = NDArray[np.int16]
LPCoeffSeriesDatum  = NDArray[np.float32]
St1SeriesNoisyDatum = NDArray[np.int16]
StSeriesCleanDatum  = NDArray[np.int16]
## the datum
FPitchCoeffSt1nStcDatum = Tuple[FeatSeriesDatum, PitchSeriesDatum, LPCoeffSeriesDatum, St1SeriesNoisyDatum, StSeriesCleanDatum]
