"""The Network"""


from dataclasses import dataclass
from typing import Optional, Tuple

from torch import nn, Tensor, tensor, int32, cat, repeat_interleave, roll # pylint: disable=no-name-in-module
from omegaconf import MISSING, SI

from ..domain import FeatSeriesBatched, LPCoeffSeriesBatched, PitchSeriesBatched, St1SeriesNoisyBatched
from .framenet import FrameNet, ConfFrameNet
from .samplenet import SampleNet, ConfSampleNet
from .components.linear_prediction import linear_prediction, linear_prediction_series
from .components.mulaw import linear_s16pcm, mlaw2lin


@dataclass
class ConfNetwork:
    """Configuration of the Network.

    Args:
        sample_per_frame - The number of samples in single frame [sample/frame]
        lp_order - LinearPrediction order
    """
    sample_per_frame: int = MISSING
    lp_order: int = MISSING
    ndim_cond_feat: int = MISSING
    frame_net: ConfFrameNet = ConfFrameNet(
        ndim_h_o_feat=SI("${..ndim_cond_feat}"),)
    sample_net: ConfSampleNet = ConfSampleNet(
        ndim_cond_feat=SI("${..ndim_cond_feat}"),)

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
        stateful: bool = True,
        ) -> Tuple[Tensor, Tensor]:
        """(PT API) Forward a batch.

        Args:
            stateful - Whether to run SampleRateNet with stateful mode or not
        Returns:
            e_t_mlaw_logp_series :: (B, T=spl_cnk, JDist) - Series of mulaw_u8pcm residual_t's (Joint) Log-Probability Distribution
            p_t_noisy_series     :: (B, T=spl_cnk)        - Series of linear prediction @t
        """

        # FrameNet :: (B, T=frm_cnk+pad, F) -> (B, T=frm_cnk, F) -> (B, T=spl_cnk, F)
        cond_t_f_series: Tensor = self.frame_net(feat_series, pitch_series)
        cond_t_s_series = repeat_interleave(cond_t_f_series, self._sample_per_frame, dim=1)

        # LPC calculation (E2E)
        # pass

        # Linear Prediction
        # (B, T=frm_cnk, Order) -> (B, T=spl_cnk, Order)
        lpcoeff_series = repeat_interleave(lpcoeff_series, self._sample_per_frame, dim=1)
        # ((B, T=spl_cnk), (B, T=spl_cnk, Order)) -> (B, T=spl_cnk), linear_s16
        p_t_noisy_series = linear_prediction_series(s_t_1_noisy_series, lpcoeff_series, self._order)

        # Residual :: -> (B, T=spl_cnk), linear_s16
        p_t_1_noisy_series = roll(p_t_noisy_series, shifts=1, dims=1)
        e_t_1_noisy_series = s_t_1_noisy_series - p_t_1_noisy_series

        # SampleNet :: ((B, T), (B, T), (B, T, F)) -> (B, T, Dist), dist. of mulaw_u8pcm
        e_t_mlaw_logp_series: Tensor = self.sample_net(s_t_1_noisy_series, p_t_noisy_series, e_t_1_noisy_series, cond_t_s_series, stateful)

        return e_t_mlaw_logp_series, p_t_noisy_series

    def generate(self,
        feat_series: FeatSeriesBatched,
        pitch_series: PitchSeriesBatched,
        lpcoeff_series: LPCoeffSeriesBatched,
        ) -> Tensor:
        """Run inference with a batch.

        Returns:
            s_t_series_estim :: (Batch, T) - Generated waveform, linear_s16pcm
        """

        # Update cell parameters
        self.sample_net.update_gru_cells()

        # Acquire device
        device = self.device()

        # Feat2Cond :: (B, T, F) -> (B, T=t_f, F) -> (B, T=t_s, F)
        cond_t_f_series: Tensor = self.frame_net(feat_series, pitch_series)
        cond_t_s_series = repeat_interleave(cond_t_f_series, self._sample_per_frame, dim=1)

        # Coeff upsampling :: (B, T=t_f, Order) -> (B, T=t_s, Order)
        lpcoeff_series = repeat_interleave(lpcoeff_series, self._sample_per_frame, dim=1)

        # Sample Generation
        ndim_b, len_t, _ = cond_t_s_series.size()
        s_t_n = tensor([[0 for _ in range(self._order)] for _ in range(ndim_b)], dtype=int32, device=device) # (B, T=order) - s_{t-1} ~ s_{t-order}, zeros of linear_s16pcm
        e_t_1 = tensor([0 for _ in range(ndim_b)],                               dtype=int32, device=device) # (B,) - e_{t-1}, zeros of linear_s16pcm
        h_a: Optional[Tensor] = None # (B, Hidden) - GRU_A hidden state
        h_b: Optional[Tensor] = None # (B, Hidden) - GRU_B hidden state
        s_t_series_estim: Tensor = tensor([[] for _ in range(ndim_b)],           dtype=int32, device=device) # (B, T) - Generated sample series
        for i in range(len_t):
            coeff_t, cond_t = lpcoeff_series[:, i], cond_t_s_series[:, i]

            # Linear Prediction :: ((B, T=order), (Order=order)) -> (B,), linear_s16
            p_t = linear_prediction(s_t_n, coeff_t)

            # Residual sampling :: -> ((B,), (B, F), (B, F))
            e_t_mlaw, h_a, h_b = self.sample_net.generate(s_t_n[:, 0], p_t, e_t_1, cond_t, h_a, h_b, ndim_b) # mlaw_u8pcm
            e_t = mlaw2lin(e_t_mlaw) # mlaw_u8pcm -> linear_s16

            # Sample :: ((B,), (B,)) -> (B, 1), linear_s16pcm
            s_t_estim = linear_s16pcm(p_t + e_t).unsqueeze(-1)

            # Record :: ((B, T=t), (B, 1)) -> (B, T=t+1)
            s_t_series_estim = cat((s_t_series_estim, s_t_estim), dim=-1)
            # AR - Samples :: ((B, T=1), (B, T=order-1)) -> (B, T=order)
            s_t_n = cat([s_t_estim, s_t_n[:, :-1]], dim=-1)
            # AR - Residual, μlaw-scale
            e_t_1 = e_t

        return s_t_series_estim

    def device(self) -> str:
        """Acquire current device."""
        return self.sample_net.device()
