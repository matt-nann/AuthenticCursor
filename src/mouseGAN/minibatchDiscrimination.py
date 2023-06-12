import torch
import torch.nn as nn

from .model_config import C_MiniBatchDisc

class MinibatchDiscrimination(nn.Module):
    """
    Prevents generator creating similar results (mode collapse) by allowing discriminator to receive batch statistics
    """
    def __init__(self, config : C_MiniBatchDisc, input_features):
        super(MinibatchDiscrimination, self).__init__()
        if (config.num_kernels is None) or (config.kernel_dim is None):
            raise ValueError("num_kernels and kernel_dim must be specified if using minibatch discrimination")
        self.input_features = input_features
        self.num_kernels = config.num_kernels
        self.kernel_dim = config.kernel_dim
        self.T = nn.Parameter(torch.randn(input_features, self.num_kernels * self.kernel_dim))

    def forward(self, x):
        # input x shape : (batch_size, sequence_length, number_features)
        batch_size, seq_length, number_features = x.size()
        # reshape input tensor to (batch_size*seq_length, number_features)
        x_reshape = x.view(-1, self.input_features)
        # Compute minibatch discrimination on reshaped tensor
        M = torch.matmul(x_reshape, self.T).view(-1, self.num_kernels, self.kernel_dim)
        diffs = M.unsqueeze(0) - M.transpose(0, 1).unsqueeze(2)
        abs_diffs = torch.sum(torch.abs(diffs), dim=2)
        minibatch_features = torch.sum(torch.exp(-abs_diffs), dim=2).T
        # Reshape minibatch features tensor back to (batch_size, seq_length, minibatch_features_dim)
        minibatch_features_reshape = minibatch_features.view(batch_size, seq_length, -1)
        # concatenate along the last dimension
        return torch.cat((x, minibatch_features_reshape), dim=2)