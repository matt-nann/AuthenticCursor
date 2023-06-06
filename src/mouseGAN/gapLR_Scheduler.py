import torch
from torch.optim.lr_scheduler import _LRScheduler

class GapScheduler(_LRScheduler):
    """
        Args:
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
    def __init__(self, optimizer, 
                ideal_loss,
                discLossDecay=0.95,
                loss_min=None, loss_max=None, lr_shrinkMin=0.1, lr_growthMax=2.0):
        self.ideal_loss = ideal_loss
        self.discLossDecay = discLossDecay
        self.loss_min = loss_min if loss_min is not None else 0.1*ideal_loss
        self.loss_max = loss_max if loss_max is not None else 0.1*ideal_loss
        self.lr_shrinkMin = lr_shrinkMin
        self.lr_growthMax = lr_growthMax
        super(GapScheduler, self).__init__(optimizer)

    def get_lr(self):
        return [self.lr_scheduler(base_lr) for base_lr in self.base_lrs]
    
    def lr_scheduler(self, base_lr):
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
        x = torch.abs(self.smoothDiscLoss - self.ideal_loss)
        # exponential interpolation of the optimality gap to learning rate multiplier, 
        #   worked better than linear according to paper
        f_x = torch.clamp(torch.pow(self.lr_growthMax, x/self.loss_max), 1.0, self.lr_growthMax)
        # clamped between 1 and f_max to ensure the learning rate is always increased but not too much 
        h_x = torch.clamp(torch.pow(self.lr_shrinkMin, x/self.loss_min), self.lr_shrinkMin, 1.0)
        lr_scaleFactor = f_x if self.smoothDiscLoss > self.ideal_loss else h_x
        return base_lr * lr_scaleFactor

    def _initial_step(self):
        """Initialize step counts and performs a step"""
        self.optimizer._step_count = 0
        self._step_count = 0
        self.smoothDiscLoss = self.ideal_loss

    def step(self, discriminator_batch_loss):
        self.smoothDiscLoss = self.discLossDecay * self.smoothDiscLoss + (1-self.discLossDecay) * discriminator_batch_loss 
        super().step()