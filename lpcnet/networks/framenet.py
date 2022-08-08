"""The FrameNet"""

from typing import List
from dataclasses import dataclass

from torch import Tensor, cat # pylint: disable=no-name-in-module
import torch.nn as nn
from omegaconf import MISSING

from .components.receptive_field import calc_rf
from .components.nn_wrapper import TransposeLast


@dataclass
class ConfFrameNet:
    """Configuration of the FrameNet.
    Args:
        ndim_i_feat     - The size of input series's feature dimension
        codebook_size   - The size of Embedding codebook
        ndim_emb        - The size of pitch embedding series's feature/emb dimension
        ndim_h_o_feat   - The size of hidden/output series's feature dimension
        kernel_size     - Convolution's kernel size
        num_conv_layer  - The number of convolutional layers
        num_segfc_layer - The number of segmental FC layers
    """
    ndim_i_feat: int = MISSING
    codebook_size: int = MISSING
    ndim_emb: int = MISSING
    ndim_h_o_feat: int = MISSING
    kernel_size: int = MISSING
    num_conv_layer: int = MISSING
    num_segfc_layer: int = MISSING
    padding: int = MISSING


class FrameNet(nn.Module):
    """The FrameRateNetwork.
    """
    def __init__(self, conf: ConfFrameNet):
        super().__init__()

        # Validation - Compatibility between padding config and Conv 'valid' padding size
        valid_padding = calc_rf(conf.num_conv_layer, conf.kernel_size) - 1
        assert conf.padding == valid_padding, f"Invalid padding size, {conf.padding} != {valid_padding}"

        # Pitch Period embedding: (B, T) -> (B, T, Emb)
        self.emb = nn.Embedding(conf.codebook_size, conf.ndim_emb)
        nn.init.uniform_(self.emb.weight, a=-0.05, b=+0.05)

        # ConvSegFC: Conv1d_c/k/s1-tanh-Conv1d_c/k/s1-tanh-SegFC_c-tanh-SegFC_c-tanh
        # :: (B, T=frm_cnk+pad, F) -> (B, T=frm_cnk, F)
        layers: List[nn.Module] = []

        ## Transpose :: (B, T, F) -> (B, F, T)
        layers += [TransposeLast()]

        ## Conv :: (Batch, F=i_f+emb, T=frm_cnk+pad) -> (B, F, T=frm_cnk)
        for i in range(conf.num_conv_layer):
            dim_i = (conf.ndim_i_feat + conf.ndim_emb) if i == 0 else conf.ndim_h_o_feat
            conv = nn.Conv1d(dim_i, conf.ndim_h_o_feat, conf.kernel_size, stride=1)
            # TF init
            nn.init.xavier_uniform_(conv.weight, gain=1.0)
            nn.init.zeros_(conv.bias) # type:ignore
            layers += [conv, nn.Tanh()]

        ## Transpose :: (B, F, T) -> (B, T, F)
        layers += [TransposeLast()]

        ## SegFC :: (B, T, F) -> (B, T, F)
        for _ in range(conf.num_segfc_layer):
            linear = nn.Linear(conf.ndim_h_o_feat, conf.ndim_h_o_feat)
            nn.init.xavier_uniform_(linear.weight, gain=1.0)
            nn.init.zeros_(linear.bias) # type:ignore
            layers += [linear, nn.Tanh()]

        ## ConvSegFC subnetwork
        self.conv_segfc = nn.Sequential(*layers)

    # Typing of PyTorch forward API is poor.
    def forward(self, feat_series: Tensor, pitch_series: Tensor) -> Tensor: # pyright: ignore [reportIncompatibleMethodOverride]
        """(PT API) Forward a batch.

        Arguments:
            feat_series  :: (B, T=frm_cnk+pad, F) - Feature series
            pitch_series :: (B, T=frm_cnk+pad)    - Pitch Period series
        Returns:
                         :: (B, T=frm_cnk    , F) - Conditioning vector series
        """

        # Pitch embedding :: (B, T) -> (B, T, Emb)
        pitch_emb_series = self.emb(pitch_series)

        # ConvSegFC :: ((B, T=frm_cnk+pad, F=feat), (B, T=frm_cnk+pad, Emb=emb)) -> (B, T=frm_cnk+pad, F=feat+emb) -> (Batch, T=frm_cnk, F)
        return self.conv_segfc(cat((feat_series, pitch_emb_series), dim=-1))
