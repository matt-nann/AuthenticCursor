import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_SEQ_LEN = 200

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

class Generator(nn.Module):
    ''' C-RNN-GAN generator
    '''
    
    def __init__(self, device, num_feats, latent_dim, target_dim, hidden_units=256, drop_prob=0.6):
        super(Generator, self).__init__()
        self.device = device

        # params
        self.hidden_dim = hidden_units
        self.num_feats = num_feats
        # double the number features
        self.fc_layer1 = nn.Linear(in_features=(num_feats * 2 + target_dim), out_features=hidden_units)
        self.lstm_cell1 = nn.LSTMCell(input_size=hidden_units, hidden_size=hidden_units)
        self.leaky_relu = nn.LeakyReLU(.2)
        self.dropout = nn.Dropout(p=drop_prob)
        self.lstm_cell2 = nn.LSTMCell(input_size=hidden_units, hidden_size=hidden_units)
        self.fc_layer2 = nn.Linear(in_features=hidden_units, out_features=num_feats)

        # generating a sequence length
        # self.linear = nn.Linear(latent_dim + target_dim, 1)
        # self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            init_weights(m)

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

class Discriminator(nn.Module):
    ''' C-RNN-GAN discrminator
    '''
    def __init__(self, device, num_feats, num_target_feats, hidden_units=256, drop_prob=0.6):
        super(Discriminator, self).__init__()
        self.device = device
        # params
        self.hidden_dim = hidden_units
        self.num_target_feats = num_target_feats
        self.num_layers = 2

        # NOTE not using dropout because the number of input features is so small
        # self.dropout = nn.Dropout(p=drop_prob)    
        self.lstm = nn.LSTM(input_size=num_feats + num_target_feats, hidden_size=hidden_units,
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
        ''' Forward prop
        '''
        # note_seq: (batch_size, seq_len, num_feats)
        # buttonTarget: (batch_size, num_target_feats)
        input_feats = torch.cat((trajectory, buttonTarget.unsqueeze(1).repeat(1, trajectory.shape[1], 1)), dim=-1)
        # drop_in = self.dropout(input_feats) # input with dropout
        # (batch_size, seq_len, num_directions*hidden_size)
        lstm_out, state = self.lstm(input_feats, state)
        # (batch_size, seq_len, 1)
        out = self.fc_layer(lstm_out)
        out = torch.sigmoid(out)

        num_dims = len(out.shape)
        reduction_dims = tuple(range(1, num_dims))
        # (batch_size)
        # The current discriminator architecture utilizes LSTM for sequential data, applying a fully connected layer and sigmoid activation function at each time step, resulting in a sequence of scores. 
        # The mean of these scores is taken to represent the whole sequence's score. This is because, in many sequence-to-sequence problems, 
        # only using the last LSTM output can cause information loss. The LSTM captures temporal dependencies, and each output encapsulates the information up to that specific time step. 
        # By averaging all time step outputs, we can ensure a more comprehensive representation of the entire sequence, which is especially crucial for longer sequences.
        out = torch.mean(out, dim=reduction_dims)

        return out, lstm_out, state

    def init_hidden(self, batch_size):
        ''' Initialize hidden state '''
        # create NEW tensor with SAME TYPE as weight
        weight = next(self.parameters()).data
        layer_mult = 2 # for being bidirectional
        hidden = (weight.new(self.num_layers * layer_mult, batch_size,
                                self.hidden_dim).zero_().to(self.device),
                    weight.new(self.num_layers * layer_mult, batch_size,
                                self.hidden_dim).zero_().to(self.device))
        return hidden