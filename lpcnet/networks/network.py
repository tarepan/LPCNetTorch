"""The Network"""


from dataclasses import dataclass
from typing import Optional, Tuple

from torch import Tensor, tensor, cat, repeat_interleave, roll
import torch.nn as nn
from omegaconf import MISSING, SI

from ..domain import FPitchCoeffSt1nStcBatch, FeatSeriesBatched, LPCoeffSeriesBatched, PitchSeriesBatched, St1SeriesNoisyBatched
from .framenet import FrameNet, ConfFrameNet
from .samplenet import SampleNet, ConfSampleNet
from .components.linear_prediction import linear_prediction, linear_prediction_series


@dataclass
class ConfNetwork:
    """Configuration of the Network.

    Args:
        sample_per_frame - The number of samples in single frame [sample/frame]
        lp_order - LinearPrediction order
    """
    sample_per_frame: int = MISSING
    lp_order: int = MISSING
    frame_net: ConfFrameNet = ConfFrameNet(
        ndim_h_o_feat=SI("${..}"),)
    sample_net: ConfSampleNet = ConfSampleNet()

class Network(nn.Module):
    """The LPCNet Network.
    """
    def __init__(self, conf: ConfNetwork):
        super().__init__()
        self._sample_per_frame = conf.sample_per_frame
        self._order = conf.lp_order

        self.frame_net = FrameNet(conf.frame_net)
        self.sample_net = SampleNet(conf.sample_net)

    # Typing of PyTorch forward API is poor.
    def forward(self, # pyright: ignore [reportIncompatibleMethodOverride]
        feat_series: FeatSeriesBatched,
        pitch_series: PitchSeriesBatched,
        lpcoeff_series: LPCoeffSeriesBatched,
        s_t_1_noisy_series: St1SeriesNoisyBatched,
        ) -> Tuple[Tensor, Tensor]:
        """(PT API) Forward a batch.

        Returns:
            e_t_pd_series    :: (B, T=spl_cnk, JDist) - Series of residual_t's (Joint) Probability Distribution
            p_t_noisy_series :: (B, T=spl_cnk)        - Series of linear prediction @t
        """

        # FrameNet :: (B, T=frm_cnk+pad, F) -> (B, T=frm_cnk, F) -> (B, T=spl_cnk, F)
        cond_t_f_series: Tensor = self.frame_net(feat_series, pitch_series)
        cond_t_s_series = repeat_interleave(cond_t_f_series, self._sample_per_frame, dim=1)

        # LPC calculation (E2E)
        # lpc_series = lpc()
        # Frame repeat :: (B, T_f, Feat) -> 

        # Linear Prediction
        # (B, T=frm_cnk, Order) -> (B, T=spl_cnk, Order)
        lpcoeff_series = repeat_interleave(lpcoeff_series, self._sample_per_frame, dim=1)
        # ((B, T=spl_cnk), (B, T=spl_cnk, Order)) -> (B, T=spl_cnk)
        p_t_noisy_series = linear_prediction_series(s_t_1_noisy_series, lpcoeff_series, self._order)

        # Residual :: -> (B, T=spl_cnk)
        p_t_1_noisy_series = roll(p_t_noisy_series, shifts=1, dims=1)
        e_t_1_noisy_series = s_t_1_noisy_series - p_t_1_noisy_series

        # SampleNet :: ((B, T), (B, T), (B, T, F)) -> (B, T, Dist)
        e_t_pd_series: Tensor = self.sample_net(s_t_1_noisy_series, p_t_noisy_series, e_t_1_noisy_series, cond_t_s_series)

        return e_t_pd_series, p_t_noisy_series

    def generate(self, batch: FPitchCoeffSt1nStcBatch) -> Tensor:
        """Run inference with a batch.

        Args:
            batch - Series of s_t_1 / pitch / Feature / LPCoefficient
        Returns:
            o_pred :: (Batch, T, Feat=dim_o) - Prediction
        """

        _, feat_series, pitch_series, lpcoeff_series = batch

        # Update cell parameters
        self.sample_net.update_gru_cells()

        # Feat2Cond :: (B, T=t_f, F) -> (B, T=<t_f, F) -> (B, T=<t_s, F)
        cond_t_f_series: Tensor = self.frame_net(feat_series, pitch_series)
        cond_t_s_series = repeat_interleave(cond_t_f_series, self._sample_per_frame, dim=1)

        # Coeff upsampling
        lpcoeff_series = repeat_interleave(lpcoeff_series, self._sample_per_frame, dim=1)

        # :: (Batch, T, Feat=dim_i) -> (Batch, T, Feat=dim_o)
        ndim_b: int = lpcoeff_series.size()[0]
        s_t_n: Tensor = tensor([128. for _ in range(ndim_b)]) # (B, Order)
        e_t_1: Tensor = tensor([128. for _ in range(ndim_b)]) # (B,)
        cond_t: Tensor =
        # GRU hidden states
        h_a: Optional[Tensor] = None
        h_b: Optional[Tensor] = None
        # :: (B, T)
        s_t_series: Tensor = 

        # Sample Generation
        for _ in range(cond_scale):
            # :: ((B, Order), (Order)) -> (B,)
            p_t = linear_prediction(s_t_n, coeffs)
            # :: -> ((B,), (B, F), (B, F))
            e_t, h_a, h_b = self.sample_net.generate(s_t_n[:, 0], p_t, e_t_1, cond_t, h_a, h_b, ndim_b)
            # ((B,), (B,)) -> (B, 1)
            s_t_pred = (p_t + e_t).unsqueeze(-1)
            # Record :: ((B, T=t), (B, 1)) -> (B, T=t+1)
            s_t_series = cat((s_t_series, s_t_pred), dim=-1)
            # AR - Samples :: ((B, 1), (B, Order=order-1)) -> (B, Order=order)
            s_t_n = cat([s_t_pred, s_t_n[:, :-1]], dim=-1)
            # AR - Residual
            e_t_1 = e_t

        return s_t_series
