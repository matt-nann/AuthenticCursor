import torch
from torch import inf
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

import numpy as np

class GapScheduler(LRScheduler):
    """
        Args:
        ideal_loss: the ideal loss of D. See Table 1 in https://arxiv.org/pdf/2302.00089.pdf
        https://github.com/STELIORD/google-research/blob/715a347918b2946f4dcb5b16ef5bb8ae588d6b4a/adversarial_nets_lr_scheduler/demo.ipynb#L45
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
        cooldown: number of steps to wait before scaling the learning rate in the SAME direction again
    """
    def __init__(self, optimizer, 
                ideal_loss,
                discLossDecay=0.95, loss_min=None, loss_max=None, 
                lr_shrinkMin=0.1, lr_growthMax=2.0, cooldown=0
                ,verbose=False):
        self.ideal_loss = ideal_loss
        self.discLossDecay = discLossDecay
        self.loss_min = loss_min if loss_min is not None else 0.1*ideal_loss
        self.loss_max = loss_max if loss_max is not None else 0.1*ideal_loss
        self.lr_shrinkMin = lr_shrinkMin
        self.lr_growthMax = lr_growthMax
        super(GapScheduler, self).__init__(optimizer)
        self.verbose = verbose
        self.scale_history = []
        self.LR_history = []
        self.step_history = []
        self.lr_max = 0.001
        self.lr_min = 1*10**(-9)
        self.cooldown = cooldown

    def get_lr(self):
        lr_scaleFactor = self.lr_scheduler()
        if self.scale_history:
            shrinkTwice = self.scale_history[-1] < 1.0 and lr_scaleFactor < 1.0
            # growTwice = self.scale_history[-1] > 1.0 and lr_scaleFactor > 1.0
            growTwice = False
            if (shrinkTwice or growTwice) and self.cooldown_timer > 0:
                self.cooldown_timer -= 1
                return self.base_lrs
            else:
                self.cooldown_timer = self.cooldown
        self.scale_history.append(lr_scaleFactor)
        self.LR_history.append(self.base_lrs[0])
        self.step_history.append(self.optimizer._step_count)
        return [np.clip(lr_scaleFactor * base_lr, self.lr_min, self.lr_max) for base_lr in self.base_lrs]
    
    def lr_scheduler(self):
        """Gap-aware Learning Rate Scheduler for Adversarial Networks.

        SIMPLE EXPLANATION:
        The function helps to adjust the learning rate based on how far the current loss is from the ideal loss. 
        If the current loss is higher than the ideal loss, it tries to increase the learning rate to decrease the loss faster. 
        If the current loss is less than the ideal loss, it tries to decrease the learning rate to avoid overshooting the minimum. 
        This can help to stabilize the training of adversarial networks and prevent drastic loss fluctuations.

        The scheduler changes the learning rate of the discriminator (D) during 
        training in an attempt to keep D's current loss close to that of D's ideal
        loss, i.e., D's loss when the distribution of the generated data matches
        that of the real data. The scheduler is called at every training step.
        See the paper for more details: https://arxiv.org/pdf/2302.00089.pdf

        Args:
        loss: the loss of the discriminator D on the training data. In the paper,
            we estimate this quantity by using an exponential moving average of
            the loss of D over all batches seen so far (see the section "Example
            Usage for Training a GAN" in this colab for an example of the
            exponential moving average).
        ideal_loss: the ideal loss of D. See Table 1 in our paper for the
            ideal loss of common GAN loss functions.

        Returns:
        A scalar in [h_min, f_max], which can be used as a multiplier for the
            learning rate of D. 
        """
        with torch.no_grad():
            x = torch.abs(self.smoothDiscLoss - self.ideal_loss)
            # exponential interpolation of the optimality gap to learning rate multiplier, 
            #   worked better than linear according to paper
            f_x = torch.clamp(torch.pow(self.lr_growthMax, x/self.loss_max), 1.0, self.lr_growthMax)
            # clamped between 1 and lr_growthMax to ensure the learning rate is always increased but not too much 
            h_x = torch.clamp(torch.pow(self.lr_shrinkMin, x/self.loss_min), self.lr_shrinkMin, 1.0)
            lr_scaleFactor = f_x if self.smoothDiscLoss > self.ideal_loss else h_x
            return lr_scaleFactor.item()

    def _initial_step(self):
        """Initialize step counts and performs a step"""
        self.optimizer._step_count = 0
        self._step_count = 0
        self.smoothDiscLoss = None
        self.cooldown_timer = 0

    def step(self, discriminator_batch_loss):
        if self.smoothDiscLoss is None:
            self.smoothDiscLoss = discriminator_batch_loss
        else:
            self.smoothDiscLoss = self.discLossDecay * self.smoothDiscLoss + (1-self.discLossDecay) * discriminator_batch_loss 
        # print("\t\tLR_sch loss: ", temp, " -> ", self.smoothDiscLoss.item())
        super(GapScheduler, self).step()
        self.base_lrs = list(map(lambda group: group['lr'], self.optimizer.param_groups))


class ReduceLROnPlateauWithEMA:
    """
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
        ema_alpha (float): Exponential moving average smoothing factor for the metric. Default: 0.25.

    """
    def __init__(self, optimizer, *args, ema_alpha=.25, **kwargs):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, *args, **kwargs)
        self.alpha = ema_alpha  # define the EMA smoothing factor here
        self.prev_metric = None
        self.LR_history = []

    def step(self, metric):
        if self.prev_metric is None:
            self.prev_metric = metric
        # Compute the EMA of the metric
        smoothed_metric = self.alpha * metric + (1 - self.alpha) * self.prev_metric
        self.prev_metric = smoothed_metric
        if self.scheduler.verbose:
            print(f"\t\t EMA metric: {smoothed_metric:.3f} metric {metric:.3f}")
        self.LR_history.append(self.scheduler.optimizer.param_groups[0]['lr'])
        self.scheduler.step(smoothed_metric)

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)