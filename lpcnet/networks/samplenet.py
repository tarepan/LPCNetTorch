"""The SampleNet"""

from dataclasses import dataclass
from typing import Optional, Tuple

from torch import Tensor, cat, randn # pylint: disable=no-name-in-module
import torch.nn as nn
from torch.distributions import Categorical
from omegaconf import MISSING, SI

from .components.mulaw import lin2mlaw
from .components.diffemb import ConfDifferentialEmbedding, DifferentialEmbedding
from .components.dualfc import DualFC, ConfDualFC
from .components.tree_sampling import tree_to_pdf


def l2m(linear: Tensor) -> Tensor:
    """
    Returns:
        mulaw :: (..., 1) - Unsqueezed μ-law signal
    """
    return lin2mlaw(linear).unsqueeze(-1)


def get_gru_cell(gru: nn.GRU) -> nn.GRUCell:
    """Transfer (learned) GRU state to a new GRUCell.
    """

    # Instantiation
    gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)

    # Weight Transfer
    gru_cell.weight_hh.data = gru.weight_hh_l0.data # type:ignore
    gru_cell.weight_ih.data = gru.weight_ih_l0.data # type:ignore
    gru_cell.bias_hh.data = gru.bias_hh_l0.data     # type:ignore
    gru_cell.bias_ih.data = gru.bias_ih_l0.data     # type:ignore

    return gru_cell


@dataclass
class ConfSampleNet:
    """Configuration of the SampleNet.
    Args:
        ndim_emb - The size of embedding vector
        ndim_cond_feat - The size of conditioning series's feature dimension
        size_gru_a - The unit size of GRU_A's hidden/output
        size_gru_b - The unit size of GRU_A's hidden/output
    """
    sample_level: int = MISSING
    ndim_emb: int = MISSING
    emb: ConfDifferentialEmbedding = ConfDifferentialEmbedding(
        codebook_size=SI("${..sample_level}"),
        ndim_emb=SI("${..ndim_emb}"),)
    ndim_cond_feat: int = MISSING
    size_gru_a: int = MISSING # 384
    size_gru_b: int = MISSING # 16
    dual_fc : ConfDualFC = ConfDualFC(
        ndim_i_feat=SI("${..size_gru_b}"),
        ndim_o_feat=SI("${..sample_level}"),)

class SampleNet(nn.Module):
    """The FrameRateNetwork.
    """
    def __init__(self, conf: ConfSampleNet):
        super().__init__()
        self._size_gru_a = conf.size_gru_a

        # Shared signal embedding (for s/sample, p/LP, e/residual)
        self.emb = DifferentialEmbedding(conf.emb)

        # todo: recurrent_constraint = WeightClip(0.992), recurrent_regularizer=quant
        # todo: Sparsification & Quantization
        self.gru_a = nn.GRU(3*conf.ndim_emb + conf.ndim_cond_feat, conf.size_gru_a, batch_first=True)
        ## State@LastForward for stateful training
        self._prev_h_a: Optional[Tensor] = None
        self.gru_cell_a = get_gru_cell(self.gru_a)

        # todo: kernel_constraint & recurrent_constraint = WeightClip(0.992), kernel_regularizer & recurrent_regularizer = quant
        # todo: Sparsification & Quantization
        self.gru_b = nn.GRU(conf.size_gru_a + conf.ndim_cond_feat, conf.size_gru_b, batch_first=True)
        ## State@LastForward for stateful training
        self._prev_h_b: Optional[Tensor] = None
        self.gru_cell_b = get_gru_cell(self.gru_b)

        # DualFC
        self.dual_fc = DualFC(conf.dual_fc)

    # Typing of PyTorch forward API is poor.
    def forward(self, # pyright: ignore [reportIncompatibleMethodOverride]
        s_t_1_noisy_series: Tensor,
        p_t_noisy_series: Tensor,
        e_t_1_noisy_series: Tensor,
        cond_t_s_series: Tensor,
        ) -> Tensor:
        """(PT API) Forward a batch.

        Generate the probability distribution of residual@t (e_t) from ARs (s_t_1 & e_t_1) and LinearPrediction (p_t)

        Args:
            s_t_1_noisy_series :: (B, T=t_s)    - Lagged/Delayed sample series (waveform) with noise
            p_t_noisy_series   :: (B, T=t_s)    - Linear Prediction series with noise
            e_t_1_noisy_series :: (B, T=t_s)    - Lagged/Delayed residual series with noise
            cond_t_s_series    :: (B, T=t_s, F) - Conditioning vector series with sample-series time scale
        Returns:
            e_t_pd_series :: (B, T=t_s, JDist) - Series of residual_t's (Joint) Probability Distribution
        """

        ndim_b, ndim_t = tuple(s_t_1_noisy_series.size())

        # Packing :: ((B, T), (B, T), (B, T)) -> (B, T, 3)
        i_t_series = cat([l2m(s_t_1_noisy_series), l2m(p_t_noisy_series), l2m(e_t_1_noisy_series)], dim=-1)

        # Additive Gaussian Noise
        i_t_series = i_t_series + randn((ndim_b, ndim_t, 3)) * .3

        # Embedding :: (B, T, 3) -> (B, T, 3, Emb=emb) -> (B, T, F=3*emb)
        emb_t_series: Tensor = self.emb(i_t_series).reshape(ndim_b, ndim_t, -1)

        # Conditioning :: ((B, T, F=3*emb), (B, T, F=cond)) -> (B, T, F=3*emb+cond)
        i_rnn_a = cat([emb_t_series, cond_t_s_series], dim=-1)

        # GRU_A :: (B, T, F) -> ((B, T, F=gru_a), (B, F))
        o_rnn_a, prev_h_a = self.gru_a(i_rnn_a, self._prev_h_a)
        self._prev_h_a = prev_h_a.detach()

        # Additive Gaussian Noise
        o_rnn_a: Tensor = o_rnn_a + randn((ndim_b, ndim_t, self._size_gru_a)) * .005

        # Conditioning :: ((B, T, F=gru_a), (B, T, F=cond)) -> (B, T=t_s, F=gru_a+cond)
        i_rnn_b = cat([o_rnn_a, cond_t_s_series], dim=-1)

        # GRU_B :: (B, T, F) -> ((B, T, F), (B, F))
        o_rnn_b, prev_h_b = self.gru_b(i_rnn_b, self._prev_h_b)
        self._prev_h_b = prev_h_b.detach()

        # DualFC :: (B, T, F) -> (B, T, CProb=2**Q)
        bit_cond_probs = self.dual_fc(o_rnn_b)

        # P(e_t) series :: (B, T, CProb=2**Q) -> (B, T, JDist=2**Q)
        e_t_pd_series = tree_to_pdf(bit_cond_probs)

        return e_t_pd_series

    def update_gru_cells(self) -> None:
        """Transfer weights of GRU_A and GRU_B to cells."""
        self.gru_cell_a = get_gru_cell(self.gru_a)
        self.gru_cell_b = get_gru_cell(self.gru_b)

    def generate(self,
        s_t_1: Tensor,
        p_t: Tensor,
        e_t_1: Tensor,
        cond_t: Tensor,
        prev_h_a: Optional[Tensor],
        prev_h_b: Optional[Tensor],
        ndim_b: int,
        ) -> Tuple[Tensor, Tensor, Tensor]: # pyright: ignore [reportIncompatibleMethodOverride]
        """Generate a sample.

        Generate the probability distribution of residual@t (e_t) from ARs (s_t_1 & e_t_1) and LinearPrediction (p_t)

        Args:
            p_t      :: (B,)   - Linear Prediction @t
            s_t_1    :: (B,)   - Sample @t-1
            e_t_1    :: (B,)   - Residual @t-1
            cond_t   :: (B, F) - Conditioning vector @t
            prev_h_a :: (B, F) - GRU_A's hidden state @t-1, None means h_a=0
            prev_h_b :: (B, F) - GRU_B's hidden state @t-1, None means h_b=0
            ndim_b             - Batch size
        Returns:
            e_t :: (B,)   - generated residual @t
            h_a :: (B, F) - GRU_A's hidden state @t
            h_b :: (B, F) - GRU_B's hidden state @t
        """

        # Embedding (w/o noise) :: ((B,), (B,), (B,)) -> (B, F=3*emb)
        emb_t: Tensor = self.emb(cat([l2m(s_t_1), l2m(p_t), l2m(e_t_1)], dim=-1)).reshape(ndim_b, -1)

        # Transform (w/o noise) :: (B, F) -> (B, JDist)
        h_a = self.gru_cell_a(cat([emb_t, cond_t], dim=-1), prev_h_a)
        h_b = self.gru_cell_b(cat([h_a,   cond_t], dim=-1), prev_h_b)
        e_t_pd = tree_to_pdf(self.dual_fc(h_b))

        # Sampling :: (B, JDist) -> (B,)
        dist_t = Categorical(e_t_pd)
        e_t = dist_t.sample() # pyright: ignore [reportUnknownMemberType]

        return e_t, h_a, h_b
