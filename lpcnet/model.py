"""The model"""


from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn, exp, permute
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from omegaconf import MISSING

from .domain import FPitchCoeffSt1nStcBatch
from .data.domain import St1SeriesNoisyDatum
from .data.transform import ConfTransform, augment, collate, load_raw, preprocess
from .networks.network import Network, ConfNetwork
from .networks.components.mulaw import lin2mlawpcm, mlaw2lin, s16pcm_to_unit


@dataclass
class ConfOptim:
    """Configuration of optimizer.
    Args:
        learning_rate: Optimizer learning rate
        sched_decay_rate: LR shaduler decay rate
        sched_decay_step: LR shaduler decay step
    """
    learning_rate: float = MISSING
    # sched_decay_rate: float = MISSING
    # sched_decay_step: int = MISSING

@dataclass
class ConfModel:
    """Configuration of the Model.
    """
    sampling_rate: int = MISSING
    net: ConfNetwork = ConfNetwork()
    optim: ConfOptim = ConfOptim()
    transform: ConfTransform = ConfTransform()

class Model(pl.LightningModule):
    """The model.
    """

    def __init__(self, conf: ConfModel):
        super().__init__()
        self.save_hyperparameters()
        self._conf = conf
        self.net = Network(conf.net)
        self.loss = nn.NLLLoss()

    def forward(self, batch: FPitchCoeffSt1nStcBatch): # pyright: ignore [reportIncompatibleMethodOverride] ; pylint: disable=arguments-differ
        """(PL API) Run inference toward a batch.
        Returns:
            (Batch, T) - Generated sample series, linear_s16pcm
        """
        feat_series, pitch_series, lpcoeff_series, _, _ = batch
        return self.net.generate(feat_series, pitch_series, lpcoeff_series)

    # Typing of PL step API is poor. It is typed as `(self, *args, **kwargs)`.
    def training_step(self, batch: FPitchCoeffSt1nStcBatch): # pyright: ignore [reportIncompatibleMethodOverride] ; pylint: disable=arguments-differ
        """(PL API) Train the model with a batch.
        """

        feat_series, pitch_series, lpcoeff_series, s_t_1_noisy_series, s_t_clean_series = batch

        # Forward :: ... -> ((B, T=spl_cnk, JDist), (B, T=spl_cnk))
        e_t_mlaw_logp_series_estim, p_t_noisy_series = self.net(feat_series, pitch_series, lpcoeff_series, s_t_1_noisy_series)

        # Ideal residual under noisy AR conditions
        e_t_mlaw_series_ideal = lin2mlawpcm(s_t_clean_series - p_t_noisy_series)

        # NLL loss toward Log-Probability :: (B, JDist, T=spl_cnk) vs (B, T=spl_cnk)
        loss = self.loss(permute(e_t_mlaw_logp_series_estim, (0, 2, 1)), e_t_mlaw_series_ideal)

        self.log('loss', loss) #type: ignore ; because of PyTorch-Lightning
        return {"loss": loss}

    def validation_step(self, batch: FPitchCoeffSt1nStcBatch, batch_idx: int): # pyright: ignore [reportIncompatibleMethodOverride] ; pylint: disable=arguments-differ,unused-argument
        """(PL API) Validate the model with a batch.
        """

        feat_series, pitch_series, lpcoeff_series, s_t_1_noisy_series, s_t_clean_series = batch

        # Forward :: ... -> ((B, T=t_s, JDist), (B, T=t_s,))
        e_t_mlaw_logp_series_estim, p_t_noisy_series = self.net(feat_series, pitch_series, lpcoeff_series, s_t_1_noisy_series, stateful=False)
        s_t_series_fwd_estim = p_t_noisy_series + mlaw2lin(Categorical(exp(e_t_mlaw_logp_series_estim)).sample())
        s_t_series_fwd_estim_unit = s16pcm_to_unit(s_t_series_fwd_estim)

        # Loss
        e_t_mlaw_series_ideal = lin2mlawpcm(s_t_clean_series - p_t_noisy_series)
        loss_fwd = self.loss(permute(e_t_mlaw_logp_series_estim, (0, 2, 1)), e_t_mlaw_series_ideal)

        # Inference :: ... -> (Batch, T=t_s), linear_s16
        s_t_series_estim = self.net.generate(feat_series, pitch_series, lpcoeff_series)
        s_t_series_estim_unit = s16pcm_to_unit(s_t_series_estim)


        # Logging
        # [PyTorch](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_audio)
        #                                                      ::Tensor(1, L)
        self.logger.experiment.add_audio(f"tf_{batch_idx}", s_t_series_fwd_estim_unit, global_step=self.global_step, sample_rate=self._conf.sampling_rate) # type: ignore
        self.logger.experiment.add_audio(f"ar_{batch_idx}", s_t_series_estim_unit,     global_step=self.global_step, sample_rate=self._conf.sampling_rate) # type: ignore

        return {"val_loss": loss_fwd}

    # def test_step(self, batch, batch_idx: int): # pyright: ignore [reportIncompatibleMethodOverride] ; pylint: disable=arguments-differ
    #     """(PL API) Test a batch. If not provided, test_step == validation_step."""
    #     return anything_for_`test_epoch_end`

    def configure_optimizers(self): # type: ignore ; because of PyTorch-Lightning (no return typing, so inferred as Void)
        """(PL API) Set up a optimizer.
        """
        conf = self._conf.optim
        decay: float = 2.5e-5

        optim = Adam(self.net.parameters(), lr=conf.learning_rate, betas=(0.9, 0.99), eps=1e-07)
        sched = {
            "scheduler": LambdaLR(optim, lr_lambda=lambda step: 1./(1. + decay * step)),
            "interval": "step",
        }

        return {
            "optimizer": optim,
            "lr_scheduler": sched,
        }

    # def predict_step(self, batch: HogeFugaBatch, batch_idx: int): # pyright: ignore [reportIncompatibleMethodOverride] ; pylint: disable=arguments-differ
    #     """(PL API) Run prediction with a batch. If not provided, predict_step == forward."""
    #     return pred

    def sample(self) -> St1SeriesNoisyDatum:
        """Acquire sample input toward preprocess."""

        # Audio Example (librosa is not handled by this template)
        import librosa # pyright: ignore [reportMissingImports, reportUnknownVariableType] ; pylint: disable=import-outside-toplevel,import-error
        path: Path = librosa.example("libri2") # pyright: ignore [reportUnknownMemberType, reportUnknownVariableType]

        return load_raw(self._conf.transform.load, path)

    def load(self, path: Path) -> St1SeriesNoisyDatum:
        """Load raw inputs.
        Args:
            path - Path to the input.
        """
        return load_raw(self._conf.transform.load, path)

    def preprocess(self, piyo: St1SeriesNoisyDatum, to_device: Optional[str] = None) -> FPitchCoeffSt1nStcBatch:
        """Preprocess raw inputs into model inputs for inference."""

        conf = self._conf.transform
        hoge_fuga = preprocess(conf.preprocess, piyo)
        hoge_fuga_datum = augment(conf.augment, hoge_fuga)
        batch = collate([hoge_fuga_datum])

        # To device
        device = torch.device(to_device) if to_device else torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        batch = (batch[0].to(device), batch[1].to(device), batch[2])

        return batch
