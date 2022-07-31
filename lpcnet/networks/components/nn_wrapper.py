"""NN Wrapper of functions"""

from torch import nn, Tensor, transpose # pylint: disable=no-name-in-module


class TransposeLast(nn.Module):
    """Transpose Tensor(..., F, E) into Tensor(..., E, F)."""
    # Typing of PyTorch forward API is poor.
    def forward(self, ipt: Tensor) -> Tensor: # pyright: ignore [reportIncompatibleMethodOverride]
        """(PT API) Forward a batch.

        Args:
            ipt :: (..., F, E) - Input
        Returns:
            opt :: (..., E, F) - Output
        """
        return transpose(ipt, -1, -2)
