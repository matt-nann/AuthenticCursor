import os
import time
import tempfile
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from dataclasses import asdict
import wandb
import plotly.graph_objects as go
import plotly.io as pio
import os
import psutil
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from .model_config import LOSS_FUNC, LR_SCHEDULERS, Config
from .dataProcessing import MouseGAN_Data
from .abstractModels import GeneratorBase, DiscriminatorBase, GAN
from .minibatchDiscrimination import MinibatchDiscrimination
from .LR_schedulers import *

def print_memory_usage(tag):
    process = psutil.Process(os.getpid())
    print(f'{tag} : Memory usage: {process.memory_info().rss / 1024 ** 2} MB')

# Custom context manager to handle temporary file removal
class TempFileContext:
    def __enter__(self):
        self.tmp_file = tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False)
        self.tmp_filename = self.tmp_file.name
        return self.tmp_filename

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tmp_file.close()
        os.remove(self.tmp_filename)

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
    elif type(m) == nn.LayerNorm:
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    else:
        if isinstance(m, nn.ModuleList) or isinstance(m, MinibatchDiscrimination) or isinstance(m, Generator) or isinstance(m, Discriminator):
            pass
        else:
            raise ValueError("Unexpected module type {}".format(type(m)))

class Generator(GeneratorBase):
    """
    
    """
    def __init__(self, device, config : Config, mean_length, std_length):
        c_g = config.generator
        self.c_g = c_g
        self.config = config
        super(Generator, self).__init__(latent_dim=config.latent_dim, lr=c_g.lr)
        self.num_lstm_layers = c_g.num_lstm_layers  # number of LSTM layers
        self.device = device
        self.MAX_SEQ_LEN = config.MAX_SEQ_LEN
        self.hidden_units = c_g.hidden_units
        self.mean_length = mean_length
        self.std_length = std_length
        numInput = self.config.latent_dim + config.num_feats + config.num_target_feats
        self.fc_input_g = nn.Linear(in_features=numInput, out_features=c_g.hidden_units)
        self.fc_sequenceLength = nn.Linear(in_features=c_g.hidden_units, out_features=1) # predicting the stop token
        # Use a list to manage all LSTM cells
        self.lstm_cells_g = nn.ModuleList([nn.LSTMCell(input_size=c_g.hidden_units, hidden_size=c_g.hidden_units) for _ in range(self.num_lstm_layers)])
        # TODO why does layer normalization work
        if c_g.layer_normalization:
            self.layer_norms_g = nn.ModuleList([nn.LayerNorm(c_g.hidden_units) for _ in range(self.num_lstm_layers)])
        self.leaky_relu = nn.LeakyReLU(.2)
        self.dropout = nn.Dropout(p=c_g.drop_prob)
        self.fc_output_g = nn.Linear(in_features=c_g.hidden_units, out_features=config.num_feats)
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            init_weights(m)
    
    def generate_noise(self, batch_size, seed=None):
        # sampling from spherical distribution
        with torch.random.fork_rng(devices=[0] if self.device.type == 'cuda' else []):
            if seed is not None:
                torch.manual_seed(seed)
                if self.device.type == 'cuda':
                    torch.cuda.manual_seed(seed)
            z = torch.randn([batch_size, self.config.latent_dim], device=self.device)
            z = z / z.norm(dim=-1, keepdim=True)
        return z
        """
        BATCH_SIZE = 10000000
        z = torch.randn([BATCH_SIZE, 100])
        z = z / z.norm(dim=-1, keepdim=True)
        covariance_matrix = torch.matmul(z.T, z) / BATCH_SIZE
        is_spherical = torch.allclose(covariance_matrix, torch.eye(covariance_matrix.shape[0]) * covariance_matrix.diag().mean(), atol=1e-5)
        # always comes out to be false but everything points to that this is right
        """
    def forward(self, z, buttonTarget, states):
        batch_size, latent_dim = z.shape
        prev_gen = torch.zeros([batch_size, self.config.num_feats], device=self.device).uniform_()
        state = states  # List of LSTM hidden states
        # Skip / Residual Connections : effective alleviating vanishing gradient problem for deep networks, 
        #   allows gradients to propagate directly through the residual connections bypassing non-linear activation functions that cause gradients to explode or vanish
        if self.c_g.residual_connections:
            hidden_states = []
        concat_in = torch.cat((z, prev_gen, buttonTarget), dim=-1)
        normSequenceLengths = self.fc_sequenceLength(self.leaky_relu(self.fc_input_g(concat_in)))
        sequenceLengths = self.std_length * normSequenceLengths + self.mean_length
        MAX_SEQ_LEN = np.clip(round(sequenceLengths.max().item()), min=1, max=self.MAX_SEQ_LEN)
        # print( "MAX_SEQ_LEN: ", MAX_SEQ_LEN, "raw sequenceLengths: ", sequenceLengths.squeeze())
        gen_feats = torch.zeros([batch_size, MAX_SEQ_LEN, self.config.num_feats], device=self.device)
        # print("Generator")
        # print("\tMAX_SEQ_LEN", MAX_SEQ_LEN)
        for i in range(MAX_SEQ_LEN):
            # concatenate current input features and previous timestep output features, and buttonTarget every sequence step
            concat_in = torch.cat((z, prev_gen, buttonTarget), dim=-1)
            out = self.leaky_relu(self.fc_input_g(concat_in))
            for j in range(self.num_lstm_layers): # Pass through each LSTM cell
                hidden, cell = self.lstm_cells_g[j](out if j == 0 else state[j-1][0], state[j])  # Pass the output from the previous layer or the input for the first layer
                if self.c_g.layer_normalization:
                    hidden = self.layer_norms_g[j](hidden) 
                hidden = self.dropout(hidden)
                state[j] = (hidden, cell)
                if self.c_g.residual_connections: # add the residual connection
                    if j > 0:
                        state[j] = (state[j][0] + hidden_states[-1], state[j][1])
                    hidden_states.append(state[j][0])
            # torch.greater(sequenceLengths, i).float() is non differentiable and will block the gradient to fc_sequenceLength
            gen_feats[:, i, :] = self.fc_output_g(state[-1][0]) * torch.sigmoid(sequenceLengths - i).float() # The output from the final LSTM layer
        sequenceLengths = sequenceLengths.squeeze(dim=-1)
        sequenceLengths= torch.round(sequenceLengths.float()) # .long() switches require gradients to off
        sequenceLengths = torch.clamp(sequenceLengths, min=1, max=MAX_SEQ_LEN)
        mask = torch.arange(gen_feats.shape[1], device=self.device).expand(batch_size, gen_feats.shape[1]) < sequenceLengths.unsqueeze(dim=-1)
        return gen_feats, sequenceLengths, mask, state

    def init_hidden(self, batch_size):
        ''' Initialize hidden state for each LSTM layer '''
        weight = next(self.parameters()).data
        hidden = []
        for _ in range(self.num_lstm_layers):
            h = weight.new(batch_size, self.hidden_units, device=self.device).zero_()
            c = weight.new(batch_size, self.hidden_units, device=self.device).zero_()
            hidden.append((h, c))
        return hidden

class Discriminator(DiscriminatorBase):
    ''' C-RNN-GAN discrminator
    '''
    def __init__(self, device, config: Config):
        c_d = config.discriminator
        self.config = config
        super(Discriminator, self).__init__(lr=c_d.lr)
        self.useEndDeviationLoss = c_d.useEndDeviationLoss
        self.useMiniBatchDisc = c_d.miniBatchDisc is not None 
        self.miniBatchDisc = c_d.miniBatchDisc
        self.device = device
        self.hidden_units = c_d.hidden_units
        self.num_layers = c_d.num_lstm_layers
        lstm_input_dim = config.num_feats + config.num_target_feats

        if self.useMiniBatchDisc:
            self.miniBatch_d = MinibatchDiscrimination(self.miniBatchDisc, lstm_input_dim)
            lstm_input_dim += self.miniBatchDisc.num_kernels
        # NOTE not using dropout because the number of input features is so small
        self.lstm_d = nn.LSTM(input_size=lstm_input_dim, hidden_size=self.hidden_units,
                            num_layers=c_d.num_lstm_layers, batch_first=True, bidirectional=c_d.bidirectional)
        self.num_lstm_output_feats = 2 * self.hidden_units if c_d.bidirectional else self.hidden_units
        self.score_layer_d = nn.Linear(in_features=(self.num_lstm_output_feats), out_features=1)
        # Spectral Normalization operates by normalizing the weights of the neural network layer using the spectral norm, 
        # which is the maximum singular value of the weights matrix. This normalization technique ensures Lipschitz continuity 
        # and controls the Lipschitz constant of the function represented by the neural network, which is important for the stability of GANs. 
        # This is especially critical in the discriminator network of GANs, where controlling the Lipschitz constant can prevent mode collapse 
        # and help to produce higher quality generated samples.
        if c_d.spectral_norm:
            self.score_layer_d = torch.nn.utils.spectral_norm(self.score_layer_d)
        if self.useEndDeviationLoss:
            self.endLoc_layer_d = nn.Linear(in_features=(self.num_lstm_output_feats), out_features=2)
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            init_weights(m)

    def forward(self, input_feats, lengths, state, real=False):
        lengths = lengths.long()
        if self.miniBatchDisc:
            x = self.miniBatch_d(input_feats)
        else:
            x = input_feats
        lstm_out, state = self.lstm_d(x, state)
        score = self.score_layer_d(lstm_out)
        endLocation = None
        if self.useEndDeviationLoss:
            if self.lstm_d.bidirectional:
                """ The LSTM output contains the hidden states at each sequence step. 
                The forward pass's last hidden state is at the end of the sequence (index -1), 
                while the backward pass's last hidden state is at the beginning (index 0). However, each of these are still full-sized hidden states. """
                # TODO this needs to be altered for sequences of different lengths?
                forward_hidden = lstm_out[range(lstm_out.size(0)), lengths - 1, :self.hidden_units]
                backward_hidden = lstm_out[:, 0, self.hidden_units:]
                lstm_out = torch.cat((forward_hidden, backward_hidden), dim=-1)
            else:
                lstm_out = lstm_out[range(lstm_out.size(0)), lengths - 1, :]
            endLocation = self.endLoc_layer_d(lstm_out)
        """ the discriminator analyzes sequential data, providing a score for each time step. 
        By not using the last time step's output, a more comprehensive representation of 
        the entire sequence is obtained thus avoiding information loss. """
        num_dims = len(score.shape)
        reduction_dims = tuple(range(1, num_dims))
        score = torch.mean(score, dim=reduction_dims)  # (batch_size)
        return score, endLocation
    
    def build_input(self, traj, button, mask):
        d_input = torch.cat((traj, button.unsqueeze(1).repeat(1, traj.shape[1], 1)), dim=-1) * mask.unsqueeze(-1)
        return d_input
    
    def build_mask(self, maxSeqLength, lengths):
        # Create a 2D mask using broadcasting based on the actual lengths
        mask = torch.arange(maxSeqLength, device=self.device).expand(len(lengths), -1) < lengths.unsqueeze(1)
        return mask

    def init_hidden(self, batch_size):
        ''' Initialize hidden state '''
        weight = next(self.parameters()).data
        layer_mult = 2 if self.lstm_d.bidirectional else 1
        hidden = (weight.new(self.num_layers * layer_mult, batch_size,
                                self.hidden_units, device=self.device).zero_(),
                    weight.new(self.num_layers * layer_mult, batch_size,
                                self.hidden_units, device=self.device).zero_())
        return hidden

class MouseGAN(GAN):
    """
    TODO better description needed
    In Wasserstein GANs (WGAN), the critic aims to maximize the difference between its evaluations of real and generated samples, leading to a positive loss, 
    while the generator minimizes the critic's evaluations of its generated samples, leading to a negative loss.
    """
    def __init__(self, dataset: MouseGAN_Data, trainLoader: DataLoader, testLoader: DataLoader,
                device, c : Config,IN_COLAB=False, verbose=False,printBatch=False, dataType=) -> GAN:
        self.c = c
        self.device = device
        self.verbose = verbose
        self.printBatch = printBatch
        self.IN_COLAB = IN_COLAB
        # if not isinstance(dataset, MouseGAN_Data):
        #     raise ValueError("dataset must be an instance of MouseGAN_Data")
        self.dataset = dataset
        self.fakeDataProperties = dataset.fakeDatasetProperties
        self.trainLoader = trainLoader
        self.trainBatches = len(trainLoader)
        self.testLoader = testLoader
        self.testBatches = len(testLoader)
        self.std_traj = torch.Tensor(dataset.std_traj).to(device)
        self.std_condition = torch.Tensor(dataset.std_condition).to(device)
        self.mean_traj = torch.Tensor(dataset.mean_traj).to(device)
        self.mean_condition = torch.Tensor(dataset.mean_condition).to(device)
        if (c.discriminator.useEndDeviationLoss or c.generator.useOutsideTargetLoss):
            self.locationMSELoss = c.locationMSELoss
            self.criterion_locaDev = nn.MSELoss() if c.locationMSELoss else nn.L1Loss()

        self.visualTrainingVerfication = 

        self.criterionMSE = nn.MSELoss()
        self.criterionMAE = nn.L1Loss()

        self.gradientScaler = GradScaler() # gradient scaling, multiple small grad
        self.generator = Generator(device, c, dataset.mean_length, dataset.std_length).to(device)
        self.discriminator = Discriminator(device, c).to(device)
        
        """
        Gradient clipping by norm prevents the gradient norm from exceeding a certain threshold, hence preserving the direction of the gradients while scaling down their magnitude. 
        two ways:
            1) DURING backpropagation: recommended as unhealthy gradients are clipped at each layer before propagating to the next layer. Preventing a snowball effect.
            2) AFTER backpropagation: NOT recommended because "unhealthy" large gradients might have already propagated through the network.
                If the gradients of all layers saturate at the threshold (clip) value this might prevent convergence. Early unhealthy gradients cascade across the network making more gradients getting clipped.
        https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch
        """
        def hook_gener(grad):
            # clip_grad_norm_ is in place so can't create a lambda function
            clip_grad_norm_(grad, c.discriminator.gradient_maxNorm)
            return grad
        def hook_discr(grad):
            # clip_grad_norm_ is in place so can't create a lambda function
            clip_grad_norm_(grad, c.discriminator.gradient_maxNorm)
            return grad
        if c.discriminator.gradient_maxNorm is not None:
            for p in self.discriminator.parameters():
                p.register_hook(hook_discr)
        if c.generator.gradient_maxNorm is not None: 
            for p in self.generator.parameters():
                p.register_hook(hook_gener)

        super().__init__()

        if c.G_lr_scheduler:
            schVal = c.G_lr_scheduler.type.value
            schConfig = asdict(c.G_lr_scheduler)
            del schConfig['type']
            if schVal == LR_SCHEDULERS.STEP.value:
                self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer_G, **schConfig)
            elif schVal == LR_SCHEDULERS.REDUCE_ON_PLATEAU_EMA.value:
                self.scheduler_G = ReduceLROnPlateauWithEMA(self.optimizer_G, 'min', **schConfig)
            else:
                raise ValueError("LR_SCHEDULERS.LOSS_GAP_AWARE not supported for generator")
        if c.D_lr_scheduler:
            schVal = c.D_lr_scheduler.type.value
            schConfig = asdict(c.D_lr_scheduler)
            del schConfig['type']
            if schVal == LR_SCHEDULERS.STEP.value:
                self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_D, **schConfig)
            elif schVal == LR_SCHEDULERS.REDUCE_ON_PLATEAU_EMA.value:
                self.scheduler_D = ReduceLROnPlateauWithEMA(self.optimizer_D, 'min', **schConfig)
            elif schVal == LR_SCHEDULERS.LOSS_GAP_AWARE.value:
                self.scheduler_D = GapScheduler(self.optimizer_D, **schConfig)

    def train(self, modelSaveInterval=None, sample_interval=None, num_plot_paths=10, 
              output_dir=os.getcwd(), catchErrors=False, batch_print_percentage=0.1, visualCheckInterval=1):
        try:
            self.freeze_d = False
            self.discrim_loss = []
            self.gen_loss = []
            self.batch_print_percentage = batch_print_percentage
            try:
                wandb.watch(self.generator, log='all', log_freq=10, log_graph=True, idx=0)
                wandb.watch(self.discriminator, log='all', log_freq=10, log_graph=True, idx=1)
            except: 
                ...
            for epoch in range(self.startingEpoch, self.c.num_epochs + self.startingEpoch):
                self.epoch_metrics = {}
                s_time = time.time()
                d_loss, g_loss = self.train_epoch(epoch)
                with torch.no_grad():
                    if sample_interval and (epoch % sample_interval) == 0:
                        raise NotImplementedError
                        # Saving 3 predictions
                        self.save_prediction(epoch, num_plot_paths,
                                            output_dir=output_dir)
                    if modelSaveInterval and (epoch % modelSaveInterval) == 0 and epoch != 0 and epoch != self.startingEpoch:
                        self.save_models(epoch)
                    self.epoch_metrics.update({'epoch': epoch, 'd_loss': d_loss, 'g_loss': g_loss, 'epochTime': time.time()-s_time})
                    self.discrim_loss.append(d_loss)
                    self.gen_loss.append(g_loss)
                    if epoch % visualCheckInterval == 0:
                        self.visualTrainingVerfication(epoch=epoch)
                    self.validation()
                    if self.verbose:
                        epochMetricsFormatted = {k: f'{v:.5f}' if isinstance(v, float) else v for k, v in self.epoch_metrics.items()}
                        print(epochMetricsFormatted)
                    try:
                        wandb.log(self.epoch_metrics, step=(epoch+1) * self.trainBatches)
                    except:
                        ...
            self.plot_loss(output_dir)
            self.save_models(epoch)
        except Exception as e:
            if catchErrors:
                print(e)
                print("Training failed")
            else:
                raise e
            
    def calcRawButtonTargets(self, normButtonLocs):
        rawButtonTargets = normButtonLocs * self.std_condition + self.mean_condition
        targetWidths = rawButtonTargets[:, 0] 
        targetHeights = rawButtonTargets[:, 1]
        startingLocations = rawButtonTargets[:, 2:4]
        realFinalLocations = rawButtonTargets[:, 4:6]
        return targetWidths, targetHeights, startingLocations, realFinalLocations
    
    def denormalizeTraj(self, traj, mask):
        return (traj * self.std_traj + self.mean_traj) * mask.unsqueeze(2)

    def calcFinalTrajLocations(self, fake_traj, startingLocations):
        """
        all return values are in unnormalized form
        """
        g_finalLocations = startingLocations + fake_traj.sum(dim=1)
        return g_finalLocations
    
    def endDeviationLoss(self, g_finalLocations, realFinalLocations):
        """
        the discriminator is penalized for incorrectly classifying the final location of both fake and real mouse trajectories
        E.I. the generator creates a sequence of mouse movements, the discriminator has to analyze the series of delta movements and predict the final location
        NOT comparing the final generated location to the real final location or vice versa
        """
        d_loss_dev = self.criterion_locaDev(g_finalLocations, realFinalLocations)
        # when the d_loss_dev < 1.0 then torch.log(d_loss_dev) is negative, shifting input left by 1.0 to prevent any negative losses and rapid changes
        d_loss_dev = torch.log(d_loss_dev + 1)
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
            # Lastly, you can add a small epsilon value before taking the square root to avoid taking the square root of zero:?
            g_losses = ((masked_dists + 1e-8) ** 0.5)
        return g_losses.mean()
    
    def pathLengthLoss(self, rawFakeTraj, rawRealTraj):
        fakePathLength = torch.sqrt(torch.square(rawFakeTraj[:,:,0]) + torch.square(rawFakeTraj[:,:,1])).sum(dim=1)
        realPathLength = torch.sqrt(torch.square(rawRealTraj[:,:,0]) + torch.square(rawRealTraj[:,:,1])).sum(dim=1)
        g_path_length_loss = torch.abs(fakePathLength.mean() - realPathLength.mean())
        return g_path_length_loss

    def compute_gradient_penalty(self, real_samples, fake_samples, buttonTargets, d_state, phi=1):
        raise NotImplementedError("no longer working with gan architecture changes, need to update for variable lengths")
        """        
        helps ensure that the GAN learns smoothly and generates realistic samples by measuring and penalizing abrupt changes in the discriminator's predictions.

        doesn't work on MPS device -> RuntimeError: derivative for aten::linear_backward is not implemented, https://github.com/pytorch/pytorch/issues/92206 the issue is closed and solved on github but I wonder if it's not released yet
        """
        assert real_samples.shape == fake_samples.shape
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand((real_samples.size(0), 1, 1), device=self.device).requires_grad_(False)
        # Get random interpolation between real and fake samples
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        # calculate probability of interpolated examples
        with torch.backends.cudnn.flags(enabled=False):
            score_interpolated, _ = self.discriminator(interpolated, buttonTargets, d_state)
        ones = torch.ones(score_interpolated.size(), device=self.device).requires_grad_(True)
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
    def discriminatorLoss(self, d_real_out, d_fake_out, raw_fake_traj, realFinalLocs, startingLocs, validation=False):
        d_real_logits, d_real_predictedEnd = d_real_out
        d_fake_logits, d_fake_predictedEnd = d_fake_out
        post = "_val" if validation else ""
        if self.c.lossFunc.type.value == LOSS_FUNC.WGAN_GP.value:
            # if self.c.discriminator.useEndDeviationLoss:
            raise NotImplementedError("WGAN_GP needs to be updated for variable length sequences")
            # gradient_penalty = self.compute_gradient_penalty(mouse_trajectories, fake_traj, buttonTargets, d_state, phi=1)
            # Compute the WGAN loss for the discriminator
            # d_loss = torch.mean(d_real_logits) - torch.mean(d_fake_logits) + self.c.lossFunc.params.lambda_gp * gradient_penalty
            # the discriminator tries to minimize the loss by giving out lower scores to fake samples and high scores to real samples, 
            # the discriminator is pentalized for abrupt changes in it's predictions
            # self.batchMetrics["gradient_penalty" + post] = gradient_penalty.item()
        elif self.c.lossFunc.type.value == LOSS_FUNC.LSGAN.value:
            loss_disc_real = self.criterionMSE(d_real_logits, torch.ones_like(d_real_logits))
            loss_disc_fake = self.criterionMSE(d_fake_logits, -torch.ones_like(d_fake_logits)) # modified to -1 from normal LSGAN 0 target
            d_loss = (loss_disc_real + loss_disc_fake) / 2
            self.batchMetrics["d_loss_real" + post] = loss_disc_real.item()
            self.batchMetrics["d_loss_fake" + post] = loss_disc_fake.item()
        # Positive scores generally indicate that the discriminator considers the sample as real, while negative scores indicate the sample is classified as fake.
        else:
            raise ValueError("Invalid loss function")
        d_loss_base = d_loss.clone() # clones the computational graph too
        self.batchMetrics["d_loss_base" + post] = d_loss.item()
        with torch.no_grad():
            # additional loss components
            if self.c.discriminator.useEndDeviationLoss:
                g_finalLocations = startingLocs + raw_fake_traj.sum(dim=1)
                d_loss_real_dev = self.endDeviationLoss(d_real_predictedEnd * self.std_traj + self.mean_traj, realFinalLocs)
                d_loss_fake_dev = self.endDeviationLoss(d_fake_predictedEnd * self.std_traj + self.mean_traj, g_finalLocations)
                d_loss_dev = (d_loss_real_dev + d_loss_fake_dev) / 2
                d_loss += d_loss_dev
                self.batchMetrics["d_loss_real_dev" + post] = d_loss_real_dev.item()
                self.batchMetrics["d_loss_fake_dev" + post] = d_loss_fake_dev.item()
            self.batchMetrics['d_real_logits' + post] = d_real_logits.mean().item()
            self.batchMetrics['d_fake_logits' + post] = d_fake_logits.mean().item()
            self.batchMetrics["d_loss" + post] = d_loss.item()
            # print("d_loss", d_loss.item(), "d_loss_real", loss_disc_real.item(), "d_loss_fake", loss_disc_fake.item(), "d_loss_dev", d_loss_dev.item() if self.c.discriminator.useEndDeviationLoss else 0)
        return d_loss, d_loss_base
    
    def generatorLoss(self, z, normButtonLocs, real_lengths, raw_real_traj,
                      targetWidths, targetHeights, startingLocations,
                      validation=False):
        _batch_size = z.size(0)
        g_states = self.generator.init_hidden(_batch_size)
        d_states = self.discriminator.init_hidden(_batch_size)
        normButtons = normButtonLocs[:, 0:4]
        post = "_val" if validation else "" 
        fake_traj, fake_lengths, fake_mask, _ = self.generator(z, normButtons, g_states)
        d_input = self.discriminator.build_input(fake_traj, normButtons, fake_mask)
        d_logits_gen, _ = self.discriminator(d_input, fake_lengths, d_states, real=False)
        if self.c.lossFunc.type.value == LOSS_FUNC.WGAN_GP.value:
            # The generator's optimizer (self.optimizer_G) tries to minimize this loss, which is equivalent to maximizing the average discriminator's score for the generated data. As this loss is minimized, the generator gets better at producing data that looks real to the discriminator.ine)
            g_loss = - torch.mean(d_logits_gen)
        elif self.c.lossFunc.type.value == LOSS_FUNC.LSGAN.value:
            # need to redo generator pass because the previous gradient graph is discarded once the discriminator is zeroed
            d_logits_gen = d_logits_gen.view(-1)
            g_loss = self.criterionMSE(d_logits_gen, torch.ones_like(d_logits_gen))
        else:
            raise ValueError("Invalid loss function")
        self.batchMetrics["g_loss_base" + post] = g_loss.item()
        g_loss_base = g_loss.clone() # clones the computational graph too
        with torch.no_grad():
            if self.c.generator.usePathLengthLoss or self.c.generator.useOutsideTargetLoss:
                raw_fake_traj = self.denormalizeTraj(fake_traj, fake_mask)
            if self.c.generator.usePathLengthLoss:
                g_path_length_loss = self.pathLengthLoss(raw_fake_traj, raw_real_traj)
                self.batchMetrics["g_path_length_loss" + post] = g_path_length_loss.item()
                # g_loss += g_path_length_loss * self.c.generator.pathLengthLossWeight
            if self.c.generator.useSeqLengthLoss:
                # print("real_lengths", real_lengths, "fake_lengths", fake_lengths)
                g_seqLength_loss = self.criterionMAE(real_lengths.reshape(1,-1).float(), fake_lengths.reshape(1,-1).float())
                self.batchMetrics["g_length_loss" + post] = g_seqLength_loss.item()
                # g_loss += g_seqLength_loss * self.c.generator.lengthLossWeight
            # additional loss components
            if self.c.generator.useOutsideTargetLoss:
                g_finalLocations = self.calcFinalTrajLocations(raw_fake_traj, startingLocations)
                g_loss_missed = self.outsideTargetLoss(g_finalLocations, targetWidths, targetHeights)
                self.batchMetrics["g_loss_missed" + post] = g_loss_missed.item()
                g_loss += g_loss_missed * self.c.generator.outsideTargetLossWeight
            self.batchMetrics["g_loss" + post] = g_loss.item()
            # print("g_loss", g_loss.item(), "g_loss_base: ", g_loss_base.item(), "g_seqLength_loss", g_seqLength_loss.item(), "g_loss_missed", g_loss_missed.item() if self.c.generator.useOutsideTargetLoss else 0, "g_path_length_loss", g_path_length_loss.item())
        return g_loss, g_loss_base
    
    def prepare_batch(self, dataTuple):
        mouse_trajectories, normButtonLocs, real_lengths = dataTuple
        real_lengths = real_lengths.to(self.device)
        mouse_trajectories = mouse_trajectories.to(self.device)
        normButtonLocs = normButtonLocs.to(self.device).squeeze(1)
        normButtons = normButtonLocs[:, :4]
        return mouse_trajectories, normButtons, normButtonLocs, real_lengths

    def run_batch(self, i_batch, batchData, is_training, freeze_d=False, freeze_g=False):
        real_traj, normButtons, normButtonLocs, real_lengths = self.prepare_batch(batchData)
        targetWidths, targetHeights, startingLocs, realFinalLocs = self.calcRawButtonTargets(normButtonLocs)
        real_mask = self.discriminator.build_mask(real_traj.shape[1], real_lengths)
        rawRealTraj = self.denormalizeTraj(real_traj, real_mask)
        _batch_size = real_traj.shape[0]
        g_states = self.generator.init_hidden(_batch_size)
        d_states = self.discriminator.init_hidden(_batch_size)

        z = self.generator.generate_noise(_batch_size)
        fake_traj, fake_lengths, fake_mask, _ = self.generator(z, normButtons, g_states)
        raw_fake_traj = self.denormalizeTraj(fake_traj, fake_mask)

        d_real_input = self.discriminator.build_input(real_traj, normButtons, real_mask)
        d_fake_input = self.discriminator.build_input(fake_traj, normButtons, fake_mask)

        d_real_out = self.discriminator(d_real_input, real_lengths, d_states, real=True)
        d_fake_out = self.discriminator(d_fake_input, fake_lengths, d_states, real=False)

        d_loss, d_loss_base = self.discriminatorLoss(d_real_out, d_fake_out, raw_fake_traj, 
                                                                         realFinalLocs, startingLocs, validation=not is_training)
        if is_training and not freeze_d:
            self.optimizer_D.zero_grad()  # clear previous gradients
            d_loss.backward() # retain_graph=True compute gradients of all variables wrt loss
            self.optimizer_D.step() # perform updates using calculated gradients
        g_loss, g_loss_base = self.generatorLoss(z, normButtonLocs, real_lengths, rawRealTraj, 
                                                 targetWidths, targetHeights, startingLocs, validation=not is_training)
        if is_training and not freeze_g:
            self.optimizer_G.zero_grad()
            g_loss.backward()
            self.optimizer_G.step()
            # print("fc_sequenceLength grad: ", self.generator.fc_sequenceLength.weight.grad)  # check the gradients
            # print("fake_lengths.requires_grad: ", fake_lengths.requires_grad)
            # raise
        if is_training:
            return d_loss, d_loss_base, g_loss, g_loss_base
        else:
            return d_loss, g_loss, d_real_out[0], d_fake_out[0] # getting logits from d_output

    def validation(self):
        self.generator.eval()
        self.discriminator.eval()
        val_g_loss_total, val_d_loss_total, correct = 0.0, 0.0, 0
        with torch.no_grad():
            for i, batchData in enumerate(self.testLoader, 0): 
                d_loss, g_loss, d_real_out, d_fake_out = self.run_batch(i, batchData, is_training=False)
                val_g_loss_total += g_loss.item()
                val_d_loss_total += d_loss.item()
                if self.c.lossFunc.type.value == LOSS_FUNC.LSGAN.value:
                    correct += (d_real_out > 0).sum().item() + (d_fake_out < 0).sum().item()
            if self.c.lossFunc.type.value == LOSS_FUNC.LSGAN.value:
                accuracy = correct / (self.testBatches * self.c.BATCH_SIZE * 2)
                self.epoch_metrics['val_accuracy'] = accuracy
            self.epoch_metrics['val_d_loss'] = val_d_loss_total / self.testBatches
            self.epoch_metrics['val_g_loss'] = val_g_loss_total / self.testBatches

    def train_epoch(self, epoch):
        self.generator.train()
        self.discriminator.train()
        g_loss_total, d_loss_total = 0.0, 0.0
        for i, batchData in enumerate(self.trainLoader):
            self.batchMetrics = {}
            d_loss, d_loss_base, g_loss, g_loss_base = self.run_batch(i, batchData, is_training=True)
            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()
            if self.c.D_lr_scheduler and self.c.D_lr_scheduler.type.value == LR_SCHEDULERS.LOSS_GAP_AWARE.value:
                self.scheduler_D.step(d_loss_base)
                self.batchMetrics["D_lr"] = self.optimizer_D.param_groups[0]['lr']
            if self.c.G_lr_scheduler and self.c.G_lr_scheduler.type.value == LR_SCHEDULERS.REDUCE_ON_PLATEAU_EMA.value:
                self.scheduler_G.step(g_loss.item())
                self.batchMetrics["G_lr"] = self.optimizer_G.param_groups[0]['lr']
            # print("D_lr:", self.optimizer_D.param_groups[0]['lr'], "G_lr:", self.optimizer_G.param_groups[0]['lr'])
            if self.printBatch and i % int(self.trainBatches * self.batch_print_percentage) == 0:
                print("\tBatch %d/%d, d_loss = %.3f, g_loss = %.3f" % (i + 1, self.trainBatches, d_loss.item(),  g_loss.item()), end="\n")
            try:
                if i != self.trainBatches - 1:
                    wandb.log(self.batchMetrics, step=epoch * self.trainBatches + i)
            except:
                ...
        if self.c.D_lr_scheduler and self.c.D_lr_scheduler.type.value != LR_SCHEDULERS.LOSS_GAP_AWARE.value:
            self.scheduler_D.step()
            self.epoch_metrics["D_lr"] = self.optimizer_D.param_groups[0]['lr']
        if self.c.G_lr_scheduler and self.c.G_lr_scheduler.type.value != LR_SCHEDULERS.REDUCE_ON_PLATEAU_EMA.value:
            self.scheduler_G.step()
            self.epoch_metrics["G_lr"] = self.optimizer_G.param_groups[0]['lr']
        return d_loss_total /  self.trainBatches, g_loss_total / self.trainBatches

    def generate(self, rawButtonLocs):
        """
        param rawButtonTargets: tensor of shape (batch_size, 4) containing button locations in pixels, must be unnormalized
        returns unnormalized trajectories
        """      
        normButtons = (rawButtonLocs - self.mean_condition) / self.std_condition
        normButtons = normButtons[:,:4].type(torch.FloatTensor).to(self.device)
        samples = rawButtonLocs.shape[0]
        self.generator.eval()
        with torch.no_grad():
            z = self.generator.generate_noise(samples, seed=1)
            g_states = self.generator.init_hidden(samples)
            fake_trajs, fake_stop_tokens, fake_mask, _ = self.generator(z, normButtons, g_states)
            generated_trajs = fake_trajs * self.std_traj + self.mean_traj
        return generated_trajs, fake_stop_tokens

    def plotGeneratedMouseTrajectories(self, samples=50, epoch=None, batch=None, batches=None,
                                  rawButtonTargets = None):
        fig = go.Figure()
        if rawButtonTargets is None:
            rawButtonTargets = self.dataset.createButtonTargets(samples, **self.fakeDataProperties, seed=1)
                                    # axial_resolution = AXIAL_RESOLUTION)
        max_y = np.max(rawButtonTargets[:,3])
        min_y = np.min(rawButtonTargets[:,3])
        max_x = np.max(rawButtonTargets[:,2])
        min_x = np.min(rawButtonTargets[:,2])
        min_width = np.min(rawButtonTargets[:,0])
        min_height = np.min(rawButtonTargets[:,1])
        _rawButtonTargets = torch.tensor(rawButtonTargets, dtype=torch.float32).to(self.device)
        generated_trajs, fake_lengths = self.generate(_rawButtonTargets)
        _rawButtonTargets = _rawButtonTargets.detach().cpu().numpy()
        generated_trajs = generated_trajs.detach().cpu().numpy()
        fake_lengths = fake_lengths.detach().cpu().numpy()
        shapes = []
        for i in range(samples):
            generated_traj = generated_trajs[i][:round(fake_lengths[i])]
            rawButtonTarget = rawButtonTargets[i]
            df_sequence = pd.DataFrame(generated_traj, columns=self.dataset.trajColumns)
            df_target = pd.DataFrame([rawButtonTarget], columns=self.dataset.targetColumns)
            width = rawButtonTarget[0]
            height = rawButtonTarget[1]

            self.dataset.SHOW_ONE = True

            df_sequence['distance'] = np.sqrt(df_sequence['dx']**2 + df_sequence['dy']**2)
            df_sequence['velocity'] = df_sequence['distance'] / self.dataset.FIXED_TIMESTEP
            df_abs = self.dataset.convertToAbsolute(df_sequence.copy(), df_target.copy())
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
            # a big red dot to starting location
            fig.add_trace(go.Scatter(x=[df_abs['x'].iloc[0]], y=[df_abs['y'].iloc[0]], mode='markers', showlegend=False, marker=dict(size=8, symbol='star',
                                                                                                                                      color='red')))
            x0, y0 = -width/2, -height/2
            x_i, y_i = width/2, height/2
            max_x = np.max([max_x, df_abs['x'].max(), x_i])
            min_x = np.min([min_x, df_abs['x'].min(), x0])
            max_y = np.max([max_y, df_abs['y'].max(), y_i])
            min_y = np.min([min_y, df_abs['y'].min(), y0])
            square = go.layout.Shape(
                type='rect',
                x0=x0,
                y0=y0,
                x1=x_i,
                y1=y_i,
                line=dict(color='black', width=2),
                fillcolor='rgba(0, 0, 255, 0.3)',
            )
            shapes.append(square)
        title = f"Generated Trajectories"
        if epoch is not None:
            title += f" Epoch {epoch}"
        if batch and batches:
            title += f" Batch {batch}/{batches}"
        fig.update_layout(
            shapes=shapes[0:2],
            width=400,
            height=400,
            xaxis=dict(range=[min_x*1.1, max_x*1.1],),
            yaxis=dict(range=[min_y*1.1, max_y*1.1],),
            title=title,
            margin=dict(l=0, r=0, b=0, t=30),
        )
        if not self.IN_COLAB:
            fig.show()

        try:
            # Convert the figure to a JPEG image and log using wandb
            with TempFileContext() as tmp_filename:
                image_bytes = pio.to_image(fig, format='jpeg')
                with open(tmp_filename, 'wb') as tmp_file:
                    tmp_file.write(image_bytes)
                wandb.log({"generator_output": wandb.Image(tmp_filename)})
        except Exception as e:
            ...

    def plotGeneratedSineCurves(self, samples=50, epoch=None, batch=None, batches=None):
        fig = go.Figure()
        rawButtonTargets = self.dataset.createButtonTargets(samples, **self.fakeDataProperties, seed=1)
        ...

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
    
    def find_learning_rates_for_GAN(self, pretrainEpochs=1):
        preOptimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.003, betas=(0.5, 0.999))
        for epoch in range(pretrainEpochs):
            for i, batchData in enumerate(self.trainLoader):
                self.batchMetrics = {}
                d_loss, d_loss_base, g_loss, g_loss_base = self.run_batch(i, batchData, is_training=True, freeze_d=True, freeze_g=True)
                preOptimizer.zero_grad()
                d_loss_base.backward()
                preOptimizer.step()
            print(f"Pretrain Epoch {epoch} / {pretrainEpochs} d_loss: {d_loss.item()} d_loss_base: {d_loss_base.item()} g_loss: {g_loss.item()} g_loss_base: {g_loss_base.item()}")
        self._testModel_LRs(testDiscriminator=False, start_lr=1e-12)
        self.visualTrainingVerfication(samples=10)
        self.discriminator.init_weights() # reset weights
        self.generator.init_weights() # reset weights
        self._testModel_LRs(testDiscriminator=True)
        self.visualTrainingVerfication(samples=10)

    def _testModel_LRs(self, testDiscriminator=False, start_lr=1e-9, end_lr=10, num_iter=100, smooth_f=0.05, diverge_th=2):
        import plotly.graph_objects as go
        fig = go.Figure()
        for i_repeat in range(3):
            lrs = []
            losses = []
            best_loss = float('inf')
            model = self.generator if not testDiscriminator else self.discriminator
            model.init_weights() # reset weights
            optimizer = torch.optim.Adam(model.parameters(), lr=start_lr, betas=(0.5, 0.999))
            lr_scheduler = ExponentialLR(optimizer, end_lr, num_iter)
            model.train()
            start_loss = None
            for i_batch, batchData in enumerate(self.trainLoader):
                self.batchMetrics = {}
                d_loss, d_loss_base, g_loss, g_loss_base = self.run_batch(i_batch, batchData, is_training=True, freeze_d=True, freeze_g=True)
                loss = d_loss_base if testDiscriminator else g_loss
                optimizer.zero_grad()  # clear previous gradients
                loss.backward() # retain_graph=True compute gradients of all variables wrt loss
                optimizer.step() # perform updates using calculated gradients
                if i_batch == 0:
                    smooth_loss = loss.item()
                    start_loss = smooth_loss
                else:
                    smooth_loss = smooth_f * loss.item() + (1 - smooth_f) * smooth_loss
                losses.append(smooth_loss)
                # print(f"Batch {i_batch}/{len(self.trainLoader)}: loss={smooth_loss:.4f}, lr={lr_scheduler.get_lr()[0]:.10f}")
                lr_scheduler.step()
                lrs.append(lr_scheduler.get_lr()[0])
                # Check if the loss has diverged; if it has, stop the test
                if i_batch > 0 and smooth_loss > diverge_th * start_loss:
                    print("Stopping early, the loss has diverged")
                    break
                if smooth_loss < best_loss or i_batch == 0:
                    best_loss = smooth_loss
            fig.add_trace(go.Scatter(x=lrs, y=losses, name=f"Repeat {i_repeat}"))
        fig.update_layout(
            xaxis_type="log",
            xaxis_title="Learning Rate",
            yaxis_title="Loss",
            title='Generator' if not testDiscriminator else 'Discriminator' + ' Learning Rate Finder',
            width=800,
            height=800,)
        fig.show()

class ExponentialLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]