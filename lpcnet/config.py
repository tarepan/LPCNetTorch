"""Whole Configuration"""


from typing import Optional
from dataclasses import dataclass

from omegaconf import MISSING
from configen import generate_conf_loader # pyright: ignore [reportMissingTypeStubs]
from lightlightning import ConfTrain      # pyright: ignore [reportMissingTypeStubs]

from .data.transform import ConfTransform
from .data.datamodule import ConfData
from .model import ConfModel


CONF_DEFAULT_STR = """
seed: 1234
path_extend_conf: null
padding: 4
lookahead: 2
transform:
    load:
        sampling_rate: 16000
    preprocess:
        piyo2hoge:
            amp: 1.2
        piyo2fuga:
            div: 3.0
    augment:
        padding: "${padding}"
        lookahead: "${lookahead}"
model:
    sampling_rate: 16000
    net:
        sample_per_frame: 160
        lp_order: 16
        ndim_cond_feat: 128
        frame_net:
            ndim_i_feat: 20
            codebook_size: 256
            ndim_emb: 64
            kernel_size: 3
            num_conv_layer: 2
            num_segfc_layer: 2
            padding: 4
        sample_net:
            sample_level: 256
            ndim_emb: 128
            size_gru_a: 384
            size_gru_b: 16
            noise_gru: 0.005
    optim:
        learning_rate: 0.001
    transform: "${transform}"
data:
    adress_data_root: ""
    corpus:
        train:
            name: "TEST"
            download: False
        val:
            name: "TEST"
            download: False
        test:
            name: "TEST"
            download: False
        n_val: 1
        n_test: 1
    dataset:
        transform: "${transform}"
        ndim_feat: 20
        sample_per_frame: 160
        frame_per_chunk: 15
        path_sample_series: ./train_waves.s16  
        path_feat_lpc_series: ./train_features.f32
        lpc_order: 16
        padding_frame: "${padding}"
        lookahead_frame: "${lookahead}"
    loader:
        batch_size_train: 128
        batch_size_val: 1
        batch_size_test: 1
        num_workers: null
        pin_memory: null
        drop_last: True
train:
    gradient_clipping: null
    # In original LPCNet paper, 120 epochs is 230k steps (c.f. 20epochs/767Ksteps@lpcnet_efficiency)
    max_epochs: 1000
    val_interval_epoch: 3
    profiler: null
    ckpt_log:
        dir_root: "."
        name_exp: "default"
        name_version: "version_0"
"""

@dataclass
class ConfGlobal:
    """Configuration of everything.
    Args:
        seed - PyTorch-Lightning's seed for every random system
        path_extend_conf - Path of configuration yaml which extends default config
        padding - Total Padding [frame]
        lookahead - Lookahead [frame]
    """
    seed: int = MISSING
    path_extend_conf: Optional[str] = MISSING
    padding: int = MISSING
    lookahead: int = MISSING
    transform: ConfTransform = ConfTransform()
    model: ConfModel = ConfModel()
    data: ConfData = ConfData()
    train: ConfTrain = ConfTrain()


# Exported
load_conf = generate_conf_loader(CONF_DEFAULT_STR, ConfGlobal)
"""Load configuration type-safely.
"""
