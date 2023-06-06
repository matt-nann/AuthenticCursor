import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable

from .abstractModels import GeneratorBase, DiscriminatorBase, GAN
from .minibatchDiscrimination import MinibatchDiscrimination

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
        print('No initialization for', type(m))

class Generator(GeneratorBase):
    ''' C-RNN-GAN generator
    '''
    def __init__(self, device, num_feats, latent_dim, target_dim, MAX_SEQ_LEN, hidden_units=256, drop_prob=0.6):
        super(Generator, self).__init__(latent_dim=latent_dim)
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
                hidden_units=256, drop_prob=0.6, 
                miniBatchDisc=True, num_kernels=None, kernel_dim=None):
        super(Discriminator, self).__init__()
        # params
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
        self.fc_layer = nn.Linear(in_features=(2*hidden_units), out_features=1)
        # Spectral Normalization operates by normalizing the weights of the neural network layer using the spectral norm, 
        # which is the maximum singular value of the weights matrix. This normalization technique ensures Lipschitz continuity 
        # and controls the Lipschitz constant of the function represented by the neural network, which is important for the stability of GANs. 
        # This is especially critical in the discriminator network of GANs, where controlling the Lipschitz constant can prevent mode collapse 
        # and help to produce higher quality generated samples.
        self.fc_layer = torch.nn.utils.spectral_norm(self.fc_layer)
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
        out = self.fc_layer(lstm_out)
        """
        The current discriminator architecture utilizes LSTM for sequential data, applying a fully connected layer and sigmoid activation function at each time step, resulting in a sequence of scores. 
        The mean of these scores is taken to represent the whole sequence's score. This is because, in many sequence-to-sequence problems, 
        only using the last LSTM output can cause information loss. The LSTM captures temporal dependencies, and each output encapsulates the information up to that specific time step. 
        By averaging all time step outputs, we can ensure a more comprehensive representation of the entire sequence, which is especially crucial for longer sequences.
        """
        num_dims = len(out.shape)
        reduction_dims = tuple(range(1, num_dims))
        out = torch.mean(out, dim=reduction_dims)

        return out, lstm_out, state

    def init_hidden(self, batch_size):
        ''' Initialize hidden state '''
        weight = next(self.parameters()).data
        layer_mult = 2 # for being bidirectional
        hidden = (weight.new(self.num_layers * layer_mult, batch_size,
                                self.hidden_dim).zero_().to(self.device),
                    weight.new(self.num_layers * layer_mult, batch_size,
                                self.hidden_dim).zero_().to(self.device))
        return hidden

class BasicGAN(GAN):
    def __init__(self, conditional_freezing=False):
        raise NotImplementedError("BasicGAN is not implemented yet, only WGAN_GP is implemented")
        generator = Generator()
        discriminator = Discriminator()
        super().__init__(generator, discriminator, conditional_freezing)
        self.seq_shape = generator.seq_shape

class WGAN_GP(GAN):
    """
    TODO better description needed
    In Wasserstein GANs (WGAN), the critic aims to maximize the difference between its evaluations of real and generated samples, leading to a positive loss, 
    while the generator minimizes the critic's evaluations of its generated samples, leading to a negative loss.
    """
    def __init__(self, device, num_feats, target_dims, MAX_SEQ_LEN,
                miniBatchDisc=True, num_kernels=5, kernel_dim=3,
                latent_dim = 100, lambda_gp = 10) -> GAN:
        if miniBatchDisc and (num_kernels is None or kernel_dim is None):
            raise ValueError("num_kernels and kernel_dim must be specified if using minibatch discrimination")
        self.device = device
        generator = Generator(device, num_feats, latent_dim, target_dims, MAX_SEQ_LEN).to(device)
        discriminator = Discriminator(device, num_feats, target_dims, 
                            miniBatchDisc=miniBatchDisc, num_kernels=num_kernels, kernel_dim=kernel_dim).to(device)
        super().__init__(generator, discriminator)
        self.lambda_gp = lambda_gp
        self.device = device

    def compute_gradient_penalty(self, real_samples, fake_samples, buttonTarget, d_state, phi=1):
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
            prob_interpolated, _, _ = self.discriminator(interpolated, buttonTarget, d_state)
        ones = torch.ones(prob_interpolated.size()).to(self.device).requires_grad_(True)
        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=ones,
            create_graph=True)[0]
        gradients = gradients.reshape(gradients.size(0), -1)
        gradient_penalty = (
            torch.mean((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2)
        )   
        return gradient_penalty

    def train_epoch(self, dataloader):
        g_loss_total, d_loss_total = 0.0, 0.0
        for i, dataTuple in enumerate(dataloader, 0): 
            mouse_trajectories, buttonTargets, trajectoryLengths = dataTuple
            # if len(mouse_trajectories) != BATCH_SIZE:
            #     continue
            mouse_trajectories = mouse_trajectories.to(self.device)
            buttonTargets = buttonTargets.to(self.device).squeeze(1)

            real_batch_size = mouse_trajectories.shape[0]

            g_states = self.generator.init_hidden(real_batch_size)
            d_state = self.discriminator.init_hidden(real_batch_size)

            z = self.generator.generate_noise(real_batch_size)
            fake_data, _ = self.generator(z, buttonTargets, g_states)

            ### train discriminator ###
            d_real_out, _, _ = self.discriminator(mouse_trajectories, buttonTargets, d_state)
            d_fake_out, _, _ = self.discriminator(fake_data, buttonTargets, d_state)
            gradient_penalty = self.compute_gradient_penalty(mouse_trajectories, fake_data, buttonTargets, d_state, phi=1)
            # Compute the WGAN loss for the discriminator
            d_loss = torch.mean(d_fake_out) - torch.mean(d_real_out) + self.lambda_gp * gradient_penalty
            d_loss.backward()
            self.optimizer_D.step()
            self.optimizer_D.zero_grad()

            ### train generator ###
            # TODO do I need to generate the data a second time? the generator wouldn't have change when training the discriminator
            fake_data, _ = self.generator(z, buttonTargets, g_states)
            d_logits_gen, _, _ = self.discriminator(fake_data, buttonTargets, d_state)
            g_loss = - torch.mean(d_logits_gen)
            g_loss.backward()

            self.optimizer_G.step()
            self.optimizer_G.zero_grad()

            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()

            print("\tBatch %d/%d, d_loss = %.3f, g_loss = %.3f" % (i + 1, len(dataloader), d_loss.item(),  g_loss.item()), end= "\r" if i != len(dataloader) - 1 else "\n")

        return d_loss_total / len(dataloader), g_loss_total / len(dataloader)