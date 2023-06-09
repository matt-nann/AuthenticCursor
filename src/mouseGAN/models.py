import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable
from enum import Enum
from dataclasses import dataclass
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from .dataProcessing import MouseGAN_Data
from .abstractModels import GeneratorBase, DiscriminatorBase, GAN
from .minibatchDiscrimination import MinibatchDiscrimination
from .LR_schedulers import *

class LR_SCHEDULERS(Enum):
    LOSS_GAP_AWARE = 1 
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
    STEP = 2
    """ params:
        step_size
        gamma
    """

class LOSS_FUNC(Enum):
    WGAN_GP = 1
    LSGAN = 2

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) == nn.LSTMCell:
        torch.nn.init.xavier_uniform_(m.weight_hh)
        torch.nn.init.xavier_uniform_(m.weight_ih)
        m.bias_hh.data.fill_(0.01)
        m.bias_ih.data.fill_(0.01)
    elif type(m) == nn.LSTM:
        torch.nn.init.xavier_uniform_(m.weight_hh_l0)
        torch.nn.init.xavier_uniform_(m.weight_ih_l0)
        m.bias_hh_l0.data.fill_(0.01)
        m.bias_ih_l0.data.fill_(0.01)
    elif type(m) == nn.LeakyReLU:
        pass
    elif type(m) == nn.Dropout:
        pass
    else:
        if type(m) != MinibatchDiscrimination and type(m) != Generator or type(m) != Discriminator:
            print('No initialization for', type(m))

class Generator(GeneratorBase):
    ''' C-RNN-GAN generator
    '''
    def __init__(self, device, num_feats, latent_dim, target_dim, MAX_SEQ_LEN, 
                 hidden_units=256, drop_prob=0.6, learning_rate=0.001):
        super(Generator, self).__init__(latent_dim=latent_dim, lr=learning_rate)
        # params
        self.device = device
        self.MAX_SEQ_LEN = MAX_SEQ_LEN
        self.hidden_dim = hidden_units
        self.num_feats = num_feats
        # double the number features
        self.fc_layer1 = nn.Linear(in_features=(num_feats * 2 + target_dim), out_features=hidden_units)
        self.lstm_cell1 = nn.LSTMCell(input_size=hidden_units, hidden_size=hidden_units)
        self.leaky_relu = nn.LeakyReLU(.2)
        self.dropout = nn.Dropout(p=drop_prob)
        self.lstm_cell2 = nn.LSTMCell(input_size=hidden_units, hidden_size=hidden_units)
        self.fc_layer2 = nn.Linear(in_features=hidden_units, out_features=num_feats)
    
        for m in self.modules():
            init_weights(m)
        
        # generating a sequence length
        # self.linear = nn.Linear(latent_dim + target_dim, 1)
        # self.sigmoid = nn.Sigmoid()
    
    def generate_noise(self, batch_size):
        # sampling from spherical distribution
        z = torch.randn([batch_size, self.MAX_SEQ_LEN, self.num_feats]).to(self.device)
        z = z / z.norm(dim=-1, keepdim=True).to(self.device)
        return z

    def forward(self, z, buttonTarget, states):
        # z: (batch_size, seq_len, num_feats)
        # z here is the uniformly random vector
        batch_size, seq_len, num_feats = z.shape

        # # Generate the sequence length
        # seq_len = self.sigmoid(self.linear(z))  # this is a tensor of sequence lengths
        # seq_len = (seq_len * MAX_SEQ_LEN).long()  # this is a tensor of sequence lengths
        # seq_len = seq_len + (seq_len == 0).long()

        # split to seq_len * (batch_size * num_feats)
        z = torch.split(z, 1, dim=1)
        z = [z_step.squeeze(dim=1) for z_step in z]

        # create dummy-previous-output for first timestep
        prev_gen = torch.empty([batch_size, num_feats]).uniform_().to(self.device)
        # TODO should I continuously pass in a new noise vector for every timestep

        # manually process each timestep
        state1, state2 = states # (h1, c1), (h2, c2)
        gen_feats = []
        for i in range(seq_len):
            z_step = z[i]
            # concatenate current input features and previous timestep output features, and buttonTarget every sequence step
            # print("z_step.shape: ", z_step.shape, "prev_gen.shape: ", prev_gen.shape, "buttonTarget.shape: ", buttonTarget.shape)
            concat_in = torch.cat((z_step, prev_gen, buttonTarget), dim=-1)
            out = self.leaky_relu(self.fc_layer1(concat_in))
            h1, c1 = self.lstm_cell1(out, state1)
            h1 = self.dropout(h1) # feature dropout only (no recurrent dropout)
            h2, c2 = self.lstm_cell2(h1, state2)
            prev_gen = self.fc_layer2(h2)
            gen_feats.append(prev_gen)

            state1 = (h1, c1)
            state2 = (h2, c2)

        # seq_len * (batch_size * num_feats) -> (batch_size * seq_len * num_feats)
        gen_feats = torch.stack(gen_feats, dim=1)

        states = (state1, state2)
        return gen_feats, states

    def init_hidden(self, batch_size):
        ''' Initialize hidden state '''
        # create NEW tensor with SAME TYPE as weight
        weight = next(self.parameters()).data
        hidden = ((weight.new(batch_size, self.hidden_dim).zero_().to(self.device),
                    weight.new(batch_size, self.hidden_dim).zero_().to(self.device)),
                    (weight.new(batch_size, self.hidden_dim).zero_().to(self.device),
                    weight.new(batch_size, self.hidden_dim).zero_().to(self.device)))
        return hidden


class Discriminator(DiscriminatorBase):
    ''' C-RNN-GAN discrminator
    '''
    def __init__(self, device, num_feats, target_dim, 
                hidden_units=256, drop_prob=0.6, learning_rate=0.001,
                miniBatchDisc=True, num_kernels=None, kernel_dim=None,
                use_D_endDeviationLoss=False
                ):
        super(Discriminator, self).__init__(lr=learning_rate)
        self.use_D_endDeviationLoss = use_D_endDeviationLoss
        self.miniBatchDisc = miniBatchDisc
        self.device = device
        self.hidden_dim = hidden_units
        self.num_target_feats = target_dim
        self.num_layers = 2
        lstm_input_dim = num_feats + target_dim

        if miniBatchDisc:
            self.miniBatch = MinibatchDiscrimination(lstm_input_dim, num_kernels, kernel_dim)
            lstm_input_dim += num_kernels
        # NOTE not using dropout because the number of input features is so small
        self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=hidden_units,
                            num_layers=self.num_layers, batch_first=True, dropout=drop_prob,
                            bidirectional=True)
        self.score_layer = nn.Linear(in_features=(2*hidden_units), out_features=1)
        # Spectral Normalization operates by normalizing the weights of the neural network layer using the spectral norm, 
        # which is the maximum singular value of the weights matrix. This normalization technique ensures Lipschitz continuity 
        # and controls the Lipschitz constant of the function represented by the neural network, which is important for the stability of GANs. 
        # This is especially critical in the discriminator network of GANs, where controlling the Lipschitz constant can prevent mode collapse 
        # and help to produce higher quality generated samples.
        self.score_layer = torch.nn.utils.spectral_norm(self.score_layer)
        if use_D_endDeviationLoss:
            self.endLoc_layer = nn.Linear(in_features=(2*hidden_units), out_features=2)
        for m in self.modules():
            init_weights(m)

    def forward(self, trajectory, buttonTarget, state):
        ''' 
        trajectory: (batch_size, seq_len, num_feats)
        buttonTarget: (batch_size, num_target_feats)
        '''
        input_feats = torch.cat((trajectory, buttonTarget.unsqueeze(1).repeat(1, trajectory.shape[1], 1)), dim=-1)
        if self.miniBatchDisc:
            x = self.miniBatch(input_feats)
        else:
            x = input_feats
        lstm_out, state = self.lstm(x, state)
        score = self.score_layer(lstm_out)
        endLocation = None
        if self.use_D_endDeviationLoss:
            endLocation = self.endLoc_layer(lstm_out)
            endLocation = torch.mean(endLocation, dim=1) # (batch_size, 2)
        """
        the discriminator analyzes sequential data, providing a score for each time step. 
        By not using the last time step's output, a more comprehensive representation of 
        the entire sequence is obtained thus avoiding information loss.
        """
        num_dims = len(score.shape)
        reduction_dims = tuple(range(1, num_dims))
        score = torch.mean(score, dim=reduction_dims) # (batch_size)
        return score, endLocation

    def init_hidden(self, batch_size):
        ''' Initialize hidden state '''
        weight = next(self.parameters()).data
        layer_mult = 2 # for being bidirectional
        hidden = (weight.new(self.num_layers * layer_mult, batch_size,
                                self.hidden_dim).zero_().to(self.device),
                    weight.new(self.num_layers * layer_mult, batch_size,
                                self.hidden_dim).zero_().to(self.device))
        return hidden

class MouseGAN(GAN):
    """
    TODO better description needed
    In Wasserstein GANs (WGAN), the critic aims to maximize the difference between its evaluations of real and generated samples, leading to a positive loss, 
    while the generator minimizes the critic's evaluations of its generated samples, leading to a negative loss.
    """
    def __init__(self, dataset: MouseGAN_Data, device, num_feats, target_dims, MAX_SEQ_LEN,
                miniBatchDisc=True, num_kernels=5, kernel_dim=3,
                g_lr=0.0001, d_lr=0.0001,
                latent_dim = 100, lambda_gp = 10, discriminator_steps=5, 
                lr_scheduler=None, schedulerParams: dataclass=None, schedulerParamsG: dataclass=None,
                lossFunc=LOSS_FUNC.LSGAN, 
                use_D_endDeviationLoss=False, use_G_OutsideTargetLoss=False, locationMSELoss = False,
                verbose=False,
                ) -> GAN:
        if miniBatchDisc and (num_kernels is None or kernel_dim is None):
            raise ValueError("num_kernels and kernel_dim must be specified if using minibatch discrimination")
        self.device = device
        self.verbose = verbose
        self.discriminator_steps = discriminator_steps
        self.lr_scheduler = lr_scheduler
        self.initial_g_lr = g_lr
        self.initial_d_lr = d_lr

        if not isinstance(dataset, MouseGAN_Data):
            raise ValueError("dataset must be an instance of MouseGAN_Data")
        self.dataset = dataset
        self.std_traj = torch.Tensor(dataset.std_traj).to(device)
        self.std_button = torch.Tensor(dataset.std_button).to(device)
        self.mean_traj = torch.Tensor(dataset.mean_traj).to(device)
        self.mean_button = torch.Tensor(dataset.mean_button).to(device)
        if (use_D_endDeviationLoss or use_G_OutsideTargetLoss):
            self.locationMSELoss = locationMSELoss
            self.criterion_locaDev = nn.MSELoss() if locationMSELoss else nn.L1Loss()
        self.use_D_locationDeviationLoss = use_D_endDeviationLoss
        self.use_G_OutsideTargetLoss = use_G_OutsideTargetLoss

        generator = Generator(device, num_feats, latent_dim, target_dims, MAX_SEQ_LEN, learning_rate=g_lr).to(device)
        discriminator = Discriminator(device, num_feats, target_dims, learning_rate=d_lr,
                            miniBatchDisc=miniBatchDisc, num_kernels=num_kernels, kernel_dim=kernel_dim,
                            use_D_endDeviationLoss=use_D_endDeviationLoss, 
                            ).to(device)
        self.lossFunc = lossFunc

        super().__init__(generator, discriminator)

        if lr_scheduler:
            if lr_scheduler.value == LR_SCHEDULERS.STEP.value:
                self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer_G, **schedulerParams)
                self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_D, **schedulerParams)
            elif lr_scheduler.value == LR_SCHEDULERS.LOSS_GAP_AWARE.value:
                self.scheduler_D = GapScheduler(self.optimizer_D, **schedulerParams)
                if schedulerParamsG:
                    self.scheduler_G = ReduceLROnPlateauWithEMA(self.optimizer_G, 'min', **schedulerParamsG)
            self.schedulerParams = schedulerParams
            self.schedulerParamsG = schedulerParamsG

        self.lambda_gp = lambda_gp
        self.device = device

        # LSGAN
        self.criterion = nn.MSELoss()

    def plotLearningRateScales(self):
        if self.lr_scheduler:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(self.scheduler_G.scale_history))), y=self.scheduler_G.scale_history, name="Generator"))
            fig.add_trace(go.Scatter(x=list(range(len(self.scheduler_D.scale_history))), y=self.scheduler_D.scale_history, name="Discriminator"))
            fig.update_layout(title="Learning rate scales", xaxis_title="Epoch", yaxis_title="Learning rate")
            fig.show()

    def calcFinalTrajLocations(self, g_feats, buttonTargets):
        # calculate the final locations of the trajectories
        rawTrajectories = g_feats * self.std_traj + self.mean_traj
        rawButtonTargets = buttonTargets * self.std_button + self.mean_button
        targetWidths = rawButtonTargets[:, 0] 
        targetHeights = rawButtonTargets[:, 1]
        startingLocations = rawButtonTargets[:, 2:4]
        realFinalLocations = rawButtonTargets[:, 4:6]
        g_finalLocations = startingLocations + rawTrajectories.sum(dim=1)
        return g_finalLocations, realFinalLocations, targetWidths, targetHeights

    def endDeviationLoss(self, g_finalLocations, realFinalLocations):
        """
        the discriminator is penalized for incorrectly classifying the final location of both fake and real mouse trajectories
        E.I. the generator creates a sequence of mouse movements, the discriminator has to analyze the series of delta movements and predict the final location
        NOT comparing the final generated location to the real final location or vice versa
        """
        d_loss_dev = self.criterion_locaDev(g_finalLocations, realFinalLocations)
        d_loss_dev = torch.log(d_loss_dev)
        return d_loss_dev

    def outsideTargetLoss(self, g_finalLocations, targetWidths, targetHeights):
        """
        the generator is penalized for generating trajectories that are outside the target area
        """
        # Coordinates of the button's edges
        x1, y1 = -targetWidths / 2, -targetHeights / 2
        # Calculate distances from the point to each edge of the button
        dx1 = x1 - g_finalLocations[:, 0]
        dx2 = g_finalLocations[:, 0] - (x1 + targetWidths)
        dy1 = y1 - g_finalLocations[:, 1]
        dy2 = g_finalLocations[:, 1] - (y1 + targetHeights)
        # If a distance is negative, the point is inside the button with respect to that edge
        insideBounds = (dx1 <= 0) & (dx2 <= 0) & (dy1 <= 0) & (dy2 <= 0)
        # Get the maximum distance for x and y (0 if the point is inside the button)
        dx = torch.max(dx1, dx2)
        dy = torch.max(dy1, dy2)
        # Calculate the distances to the nearest corner or edge
        dx_dy_gt_0 = (dx > 0) & (dy > 0)  # both dx and dy are > 0, point is outside the button
        dists = torch.where(dx_dy_gt_0, torch.sqrt(dx**2 + dy**2), torch.max(dx, dy))  # calculate distance to the corner or edge
        # Apply the mask, so that distance is 0 for points inside the button
        masked_dists = torch.where(insideBounds, torch.zeros_like(dists), dists)
        if self.locationMSELoss:
            g_losses = masked_dists
        else:
            g_losses = masked_dists ** 0.5
        return g_losses.mean()

    def compute_gradient_penalty(self, real_samples, fake_samples, buttonTargets, d_state, phi=1):
        """        
        helps ensure that the GAN learns smoothly and generates realistic samples by measuring and penalizing abrupt changes in the discriminator's predictions.

        doesn't work on MPS device -> RuntimeError: derivative for aten::linear_backward is not implemented, https://github.com/pytorch/pytorch/issues/92206 the issue is closed and solved on github but I wonder if it's not released yet
        """
        assert real_samples.shape == fake_samples.shape
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand((real_samples.size(0), 1, 1)).to(self.device).requires_grad_(False)
        # Get random interpolation between real and fake samples
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        # calculate probability of interpolated examples
        with torch.backends.cudnn.flags(enabled=False):
            score_interpolated, _ = self.discriminator(interpolated, buttonTargets, d_state)
        ones = torch.ones(score_interpolated.size()).to(self.device).requires_grad_(True)
        gradients = torch.autograd.grad(
            outputs=score_interpolated,
            inputs=interpolated,
            grad_outputs=ones,
            create_graph=True)[0]
        gradients = gradients.reshape(gradients.size(0), -1)
        gradient_penalty = (
            torch.mean((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2)
        )   
        return gradient_penalty
    
    def discriminatorLoss(self, d_real_out, d_real_predictedEnd, d_fake_out, d_fake_predictedEnd,
                        mouse_trajectories, fake_traj, buttonAndLocations, d_state):
        buttonTargets = buttonAndLocations[:, 0:4]
        if self.lossFunc.value == LOSS_FUNC.WGAN_GP.value:
            if self.use_D_locationDeviationLoss:
                raise NotImplementedError("WGAN_GP loss function doesn't support locationDeviationLoss")
            gradient_penalty = self.compute_gradient_penalty(mouse_trajectories, fake_traj, buttonTargets, d_state, phi=1)
            # Compute the WGAN loss for the discriminator
            d_loss = torch.mean(d_fake_out) - torch.mean(d_real_out) + self.lambda_gp * gradient_penalty
            # the discriminator tries to minimize the loss by giving out lower scores to fake samples and high scores to real samples, 
            # the discriminator is pentalized for abrupt changes in it's predictions
        elif self.lossFunc.value == LOSS_FUNC.LSGAN.value:
            d_real_out = d_real_out.view(-1)
            d_fake_out = d_fake_out.view(-1)
            loss_disc_real = self.criterion(d_real_out, torch.ones_like(d_real_out))
            loss_disc_fake = self.criterion(d_fake_out, torch.zeros_like(d_fake_out))
            d_loss = (loss_disc_real + loss_disc_fake) / 2
        else:
            raise ValueError("Invalid loss function")
        # additional loss components
        if self.use_D_locationDeviationLoss:
            g_finalLocations, realFinalLocations, _, _ = self.calcFinalTrajLocations(fake_traj, buttonAndLocations)
            d_loss_real_dev = self.endDeviationLoss(d_real_predictedEnd, realFinalLocations)
            d_loss_fake_dev = self.endDeviationLoss(d_fake_predictedEnd, g_finalLocations)
            d_loss_dev = (d_loss_real_dev + d_loss_fake_dev) / 2
            # d_loss_real_dev.backward(retain_graph=True)
            # d_loss_fake_dev.backward(retain_graph=True)
            print("d_loss: ", d_loss.item(), "d_loss_real_dev: ", d_loss_real_dev.item(), "d_loss_fake_dev: ", d_loss_fake_dev.item())
            d_loss += d_loss_dev
        return d_loss
    
    def generatorLoss(self, z, buttonTargets, g_states, d_state):
        if self.lossFunc.value == LOSS_FUNC.WGAN_GP.value:
            # The generator's optimizer (self.optimizer_G) tries to minimize this loss, which is equivalent to maximizing the average discriminator's score for the generated data. As this loss is minimized, the generator gets better at producing data that looks real to the discriminator.ine)
            g_loss = - torch.mean(d_logits_gen)
        elif self.lossFunc.value == LOSS_FUNC.LSGAN.value:
            fake_traj, _ = self.generator(z, buttonTargets, g_states) # need to redo generator pass because the previous gradient graph is discarded once the discriminator is zeroed
            d_logits_gen, _ = self.discriminator(fake_traj, buttonTargets, d_state)
            d_logits_gen = d_logits_gen.view(-1)
            g_loss = self.criterion(d_logits_gen, torch.ones_like(d_logits_gen))
        else:
            raise ValueError("Invalid loss function")
        # additional loss components
        if self.use_G_OutsideTargetLoss:
            g_finalLocations, _, targetWidths, targetHeights = self.calcFinalTrajLocations(fake_traj, buttonTargets)
            g_loss += self.outsideTargetLoss(g_finalLocations, targetWidths, targetHeights)
        return g_loss

    def train_epoch(self, dataloader):
        g_loss_total, d_loss_total = 0.0, 0.0
        for i, dataTuple in enumerate(dataloader, 0): 
            mouse_trajectories, buttonAndLocations, trajectoryLengths = dataTuple
            mouse_trajectories = mouse_trajectories.to(self.device)
            buttonAndLocations = buttonAndLocations.to(self.device).squeeze(1)
            buttonTargets = buttonAndLocations[:, :4]
            real_batch_size = mouse_trajectories.shape[0]

            g_state = self.generator.init_hidden(real_batch_size)
            d_state = self.discriminator.init_hidden(real_batch_size)

            z = self.generator.generate_noise(real_batch_size)
            fake_traj, _ = self.generator(z, buttonTargets, g_state)

            ### train discriminator ###
            # for ii in range(self.discriminator_steps):
            d_real_out, d_real_predictedEnd = self.discriminator(mouse_trajectories, buttonTargets, d_state)
            d_fake_out, d_fake_predictedEnd = self.discriminator(fake_traj, buttonTargets, d_state)

            d_loss = self.discriminatorLoss(d_real_out, d_real_predictedEnd, d_fake_out, d_fake_predictedEnd,
                                            mouse_trajectories, fake_traj, buttonAndLocations, d_state)
            self.optimizer_D.zero_grad() # clear previous gradients
            d_loss.backward() # retain_graph=True compute gradients of all variables wrt loss
            self.optimizer_D.step() # perform updates using calculated gradients

            g_loss = self.generatorLoss(z, buttonTargets, g_state, d_state)
            self.optimizer_G.zero_grad()
            g_loss.backward()
            self.optimizer_G.step()

            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()

            if self.verbose:
                print("\tBatch %d/%d, d_loss = %.3f, g_loss = %.3f" % (i + 1, len(dataloader), d_loss.item(),  g_loss.item()), end="\n")
            # print(f"\t\tfake score: %.3f real score: %.3f gradient_penalty: %.3f lambda_gp: %.3f" % (torch.mean(d_fake_out).item(), torch.mean(d_real_out).item(), gradient_penalty, self.lambda_gp))
            # print(f"\t\tfake score: %.3f real score: %.3f" % (torch.mean(d_fake_out).item(), torch.mean(d_real_out).item()))

            if self.lr_scheduler and self.lr_scheduler.value == LR_SCHEDULERS.LOSS_GAP_AWARE.value:
                self.scheduler_D.step(d_loss)
                if self.schedulerParamsG:
                    self.scheduler_G.step(g_loss.item())
                # print(f"\t\tschedulers step D_lr: {self.optimizer_D.param_groups[0]['lr']}, G_lr: {self.optimizer_G.param_groups[0]['lr']}")

        if self.lr_scheduler and self.lr_scheduler.value != LR_SCHEDULERS.LOSS_GAP_AWARE.value:
            self.scheduler_D.step()
            self.scheduler_G.step()

        return d_loss_total / len(dataloader), g_loss_total / len(dataloader)

    def generate(self, rawButtonTargets):
        """
        returns unnormalized trajectories
        """      
        normButtonTargets = self.dataset.normalizeButtonTargets(rawButtonTargets)[:,:4]
        normButtonTargets = normButtonTargets.type(torch.FloatTensor).to(self.device)
        samples = rawButtonTargets.shape[0]
        self.generator.eval()
        with torch.no_grad():
            z = self.generator.generate_noise(samples)
            g_state = self.generator.init_hidden(samples)
            fake_traj, _ = self.generator(z, normButtonTargets, g_state)
            generated_traj = self.dataset.denormalizeTraj(fake_traj.cpu().numpy())
        return generated_traj
    
    def visualTrainingVerfication(self, epoch=None, batch=None, batches=None):
        fig = go.Figure()
        samples = 5
        rawButtonTargets = self.dataset.createButtonTargets(samples,
                                low_radius = 200, high_radius = 300,
                                max_width = 200, min_width = 50,
                                max_height = 100, min_height = 25,)
                                # axial_resolution = AXIAL_RESOLUTION)
        max_y = np.max(rawButtonTargets[:,3])
        min_y = np.min(rawButtonTargets[:,3])
        max_x = np.max(rawButtonTargets[:,2])
        min_x = np.min(rawButtonTargets[:,2])
        min_width = np.min(rawButtonTargets[:,1])
        min_height = np.min(rawButtonTargets[:,0])
        _rawButtonTargets = torch.tensor(rawButtonTargets, dtype=torch.float32).to(self.device)
        generated_trajs = self.generate(_rawButtonTargets)

        for i in range(samples):
            generated_traj = generated_trajs[i]
            rawButtonTarget = rawButtonTargets[i]
            df_sequence = pd.DataFrame(generated_traj, columns=self.dataset.trajColumns)
            df_target = pd.DataFrame([rawButtonTarget], columns=self.dataset.targetColumns)

            self.dataset.SHOW_ONE = True

            df_sequence['distance'] = np.sqrt(df_sequence['dx']**2 + df_sequence['dy']**2)
            df_sequence['velocity'] = df_sequence['distance'] / self.dataset.FIXED_TIMESTEP
            df_abs = self.dataset.convertToAbsolute(df_sequence, df_target)

            max_x = np.max([max_x, df_abs['x'].max()])
            min_x = np.min([min_x, df_abs['x'].min()])
            max_y = np.max([max_y, df_abs['y'].max()])
            min_y = np.min([min_y, df_abs['y'].min()])

            self.dataset.SHOW_ONE = True
            fig.add_trace(go.Scatter(x=df_abs['x'], y=df_abs['y'],
                    mode='lines+markers',
                    showlegend=False,
                    marker=dict(
                                size=5, 
                                # symbol= "arrow-bar-up", angleref="previous",
                                # size=15,
                                # color='grey',),
                                color=df_abs['velocity'], colorscale='Viridis', 
                                # colorbar=dict(title="Velocity")),
                                # turn off colorbar
                    )))

        x0, y0 = -min_width/2, -min_height/2
        x_i, y_i = min_width/2, min_height/2
        square = go.layout.Shape(
            type='rect',
            x0=x0,
            y0=y0,
            x1=x_i,
            y1=y_i,
            line=dict(color='black', width=2),
            fillcolor='rgba(0, 0, 255, 0.3)',
        )
        title = f"Generated Trajectories"
        if epoch is not None:
            title += f" Epoch {epoch}"
        if batch and batches:
            title += f" Batch {batch}/{batches}"
        fig.update_layout(
            shapes=[square],
            width=400,
            height=400,
            xaxis=dict(range=[min_x*1.1, max_x*1.1],),
            yaxis=dict(range=[min_y*1.1, max_y*1.1],),
            title=title,
            margin=dict(l=0, r=0, b=0, t=30),
        )
        fig.show()

    def discriminatorLearningRates(self):
        df = pd.DataFrame(self.scheduler_D.scale_history, columns=['scale'])
        df['epoch'] = df.index
        df['lr'] = df['scale'].cumprod() * self.initial_d_lr

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df['epoch'], y=df['lr'],))
        fig.update_layout(
            title="Learning Rate for discriminator over steps",
            width=800,
            height=800,)
        fig.show()