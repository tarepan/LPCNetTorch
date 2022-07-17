"""Run training"""


import pytorch_lightning as pl
# import torchaudio
from lightlightning import train # pyright: ignore [reportMissingTypeStubs]

from .model import Model
from .data.datamodule import Data
from .config import load_conf


def main_train():
    """Training with cli arguments.
    """

    # Load default/extend/CLI configs.
    conf = load_conf()

    # Setup
    pl.seed_everything(conf.seed)
    model = Model(conf.model)
    model.train()
    datamodule = Data(conf.data)

    # Train
    train(model, conf.train, datamodule)


if __name__ == "__main__":  # pragma: no cover
    main_train()
