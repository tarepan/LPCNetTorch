"""Domain"""


from typing import Tuple

from torch import Tensor # pyright: ignore [reportUnknownVariableType] ; because of PyTorch ; pylint: disable=no-name-in-module


# [Batch]
#     FeatSeriesBatched     :: (B, T=frm_cnk+pad, F)     - Padded feature series
#     PitchSeriesBatched    :: (B, T=frm_cnk+pad, F)     - Padded Pitch Period series
#     LPCoeffSeriesBatched  :: (B, T=frm_cnk,     Order) - LinearPrediction Coefficient series
#     St1SeriesNoisyBatched :: (B, T=spl_cnk)            - Lagged/Delayed sample series (linear_s16pcm waveform) with noise
#     StSeriesCleanBatched  :: (B, T=spl_cnk)            - Sample series (linear_s16pcm waveform) w/o noise

FeatSeriesBatched     = Tensor # FloatTensor
PitchSeriesBatched    = Tensor # IntTensor
LPCoeffSeriesBatched  = Tensor # FloatTensor
St1SeriesNoisyBatched = Tensor # IntTensor (linear_s16pcm needs only 16bit/ShortTensor, but nn.Embedding needs 32bit or 64bit)
StSeriesCleanBatched  = Tensor # IntTensor (linear_s16pcm needs only 16bit/ShortTensor, but nn.Embedding needs 32bit or 64bit)

## the batch
FPitchCoeffSt1nStcBatch = Tuple[FeatSeriesBatched, PitchSeriesBatched, LPCoeffSeriesBatched, St1SeriesNoisyBatched, StSeriesCleanBatched]
