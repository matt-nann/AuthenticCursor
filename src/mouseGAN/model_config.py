import os
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Union

class LR_SCHEDULERS(Enum):
    LOSS_GAP_AWARE = 'loss_gap_aware_lr_scheduler'
    """ params:
        ideal_loss: the ideal loss of D. See Table 1 in https://arxiv.org/pdf/2302.00089.pdf
        discLossDecay: decay factor used calculate the exponential moving average of discriminator loss
        loss_min: the value of x at which the scheduler achieves its minimum allowed
            value h_min. Specifically, when loss < ideal_loss, the scheduler
            gradually decreases the LR (as long as x < loss_min). For x >= loss_min, the
            scheduler's output is capped to the minimum allowed value h_min. In the
            paper we set this to 0.1*ideal_loss.
        loss_max: the value of x at which the scheduler achieves its maximum allowed
            value f_max. Specifically, when loss >= ideal_loss, the scheduler
            gradually increases the LR (as long as x < loss_max). For x >= loss_max, the
            scheduler's output is capped to the maximum allowed value f_max. In the
            paper we set this to 0.1*ideal_loss.
        lr_shrinkMin: a scalar in (0, 1] denoting the minimum allowed value of the
            scheduling function. In the paper we used lr_shrinkMin=0.1.
        lr_growthMax: a scalar (>= 1) denoting the maximum allowed value of the 
            scheduling function. In the paper we used lr_growthMax=2.0.
    """
    STEP = 'step_lr_scheduler'
    """ params:
        step_size
        gamma
    """
    REDUCE_ON_PLATEAU_EMA = 'reduce_on_plateau_with_emaSmoothing_lr_scheduler'

class LOSS_FUNC(Enum):
    WGAN_GP = 'wgan_gp_loss_function'
    LSGAN = 'lsgan_loss_function'

@dataclass
class C_D_lrScheduler:
    cooldown: int
    type: LR_SCHEDULERS = LR_SCHEDULERS.LOSS_GAP_AWARE
    ideal_loss: float = 0.5 # LSGAN
    # ideal_loss: float = 0 # WGAN-GP
    loss_min: float = 0.1 * ideal_loss
    loss_max: float = 0.1 * ideal_loss
    lr_shrinkMin: float = 0.1
    discLossDecay: float = 0.8
    lr_growthMax: float = 2.0
@dataclass
class C_G_lrScheduler:
    patience: int
    cooldown: int
    type: LR_SCHEDULERS = LR_SCHEDULERS.REDUCE_ON_PLATEAU_EMA
    factor: float = 0.5
    min_lr: float = 1e-9
    verbose: bool = False
    ema_alpha: float = 0.4
    threshold_mode: str = 'rel'
    threshold: float = 1 / 100
@dataclass
class C_MiniBatchDisc:
    num_kernels: int = 5
    kernel_dim: int = 3

# loss function param dataclasses
@dataclass
class C_WGAN_GP:
    lambda_gp: float = 10
@dataclass
class C_LSGAN:
    pass
@dataclass
class C_LOSS_FUNC:
    type : LOSS_FUNC = LOSS_FUNC.LSGAN
    params: Union[C_WGAN_GP, C_LSGAN] = C_LSGAN()

    def __post_init__(self):
        if self.type == LOSS_FUNC.WGAN_GP:
            assert isinstance(self.params, C_WGAN_GP)
        elif self.type == LOSS_FUNC.LSGAN:
            assert isinstance(self.params, C_LSGAN)

@dataclass
class C_Discriminator:
    hidden_units: int = 128
    num_layers: int = 4
    lr: float = 0.0001
    # A Bidirectional LSTM (BiLSTM) captures information from both past and future states by processing the sequence in both forward and backward directions. This provides additional context, improving performance in tasks where future context is informative.
    bidirectional: bool = False 
    miniBatchDisc: Optional[C_MiniBatchDisc] = C_MiniBatchDisc()
    useEndDeviationLoss: bool = False
    
@dataclass
class C_Generator:
    hidden_units: int = 128
    lr: float = 0.0001
    useOutsideTargetLoss: bool = False
    drop_prob: float = 0.5

@dataclass
class Config:
    num_epochs: int
    BATCH_SIZE: int
    num_feats: int
    latent_dim: int
    num_target_feats: int
    MAX_SEQ_LEN: int
    G_lr_scheduler: Optional[C_G_lrScheduler] = None
    D_lr_scheduler: Optional[C_D_lrScheduler] = None
    locationMSELoss: bool = False
    lossFunc: C_LOSS_FUNC = C_LOSS_FUNC()
    discriminator: C_Discriminator = C_Discriminator()
    generator: C_Generator = C_Generator()

    def __post_init__(self):
        if self.G_lr_scheduler is None:
            self.G_lr_scheduler = C_G_lrScheduler(patience=self.BATCH_SIZE, cooldown=int(self.BATCH_SIZE/8))
        if self.D_lr_scheduler is None:
            self.D_lr_scheduler = C_D_lrScheduler(cooldown=int(self.BATCH_SIZE/8))