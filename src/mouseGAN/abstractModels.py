import abc
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import Variable
import glob
import time

CKPT_DIR = os.getcwd() + '/data/local/ganModels'
try:
  from google.colab import drive
  drive.mount('/content/drive')   # This will prompt for authorization.
  CKPT_DIR = '/content/drive/My Drive/mouseGAN_models'  # or the directory in your Google Drive where you want to save the models
except:
  ...

LOAD_PRETRAINED = True

def find_latest_model(model_type, path):
    list_of_files = glob.glob(os.path.join(path, model_type + '*.pt')) 
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file
  
def find_epoch_model(model_type, epoch, path):
    list_of_files = glob.glob(os.path.join(path, model_type + f'{epoch}.pt')) 
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

class GeneratorBase(nn.Module, metaclass=abc.ABCMeta):
    """Abstract Generator class for the GAN."""

    def __init__(self, latent_dim=100, lr=0.0002, beta1=0.5, beta2=0.999, eps=1e-8):
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.latent_dim = latent_dim

    def optim_parameters(self):
        return {'lr': self.lr, 'betas': (self.beta1, self.beta2), 'eps': self.eps}

    def generate_noise(self, batch_size):
        
        # # sampling from spherical distribution
        # z = torch.randn([real_batch_sz, MAX_SEQ_LEN, num_feats]).to(device)
        # z = z / z.norm(dim=-1, keepdim=True)

        return torch.randn(batch_size, self.latent_dim)

    @abc.abstractmethod
    def forward(self, z):
        pass

class DiscriminatorBase(nn.Module, metaclass=abc.ABCMeta):
    """Abstract Discriminator class for the GAN."""

    def __init__(self, lr=0.0002, beta1=0.5, beta2=0.999, eps=1e-8):
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
    
    def optim_parameters(self):
        return {'lr': self.lr, 'betas': (self.beta1, self.beta2), 'eps': self.eps}
    
    @abc.abstractmethod
    def forward(self, x):
        pass

class GAN(metaclass=abc.ABCMeta):
    """Abstract GAN class."""

    def __init__(self, conditional_freezing=False):
        """
        Args:
            generator: Generator class
            discriminator: Discriminator class
        """
        self.loss = nn.BCELoss()
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), **self.discriminator.optim_parameters())
        self.optimizer_G = optim.Adam(self.generator.parameters(), **self.generator.optim_parameters())

        ### parameters
        self.conditional_freezing = conditional_freezing
        
        self.startingEpoch = 0

    def eval(self):
        self.generator.eval()
        self.discriminator.eval()

    def loadPretrained(self, startingEpoch=None):
        if startingEpoch is not None:
            latest_g_model = find_epoch_model('g', startingEpoch, CKPT_DIR)
            latest_d_model = find_epoch_model('d', startingEpoch, CKPT_DIR)
            if latest_g_model is None or latest_d_model is None:
                print("Specific Epoch models not found. Loading latest models.")
                startingEpoch = None
        if startingEpoch is None:
            latest_g_model = find_latest_model('g', CKPT_DIR)
            latest_d_model = find_latest_model('d', CKPT_DIR)
        if latest_g_model is not None:
            self.generator.load_state_dict(torch.load(latest_g_model, map_location=self.device))
            print(f"Loaded generator model: {latest_g_model}")
        if latest_d_model is not None:
            self.discriminator.load_state_dict(torch.load(latest_d_model, map_location=self.device))
            print(f"Loaded discriminator model: {latest_d_model}")
        if latest_g_model is not None and latest_d_model is not None:
            if isinstance(startingEpoch, str):
                startingEpoch = 0
            else:
                startingEpoch = min(int(latest_g_model.split('/')[-1].split('.')[0][1:]), int(latest_d_model.split('/')[-1].split('.')[0][1:]))
            print(f"Starting from epoch {startingEpoch}")
        else:
            print("No pretrained models found. Starting from scratch.")
            startingEpoch = 0
        self.startingEpoch = startingEpoch

    def train_epoch(self, dataloader):
        raise NotImplementedError
    
    def save_models(self, num_epochs):

        # Create directory if it does not exist
        if not os.path.exists(CKPT_DIR):
            os.makedirs(CKPT_DIR)
        G_FN = 'g' + str(num_epochs) + '.pt'
        D_FN = 'd' + str(num_epochs) + '.pt'
        generatorPath = os.path.join(CKPT_DIR, G_FN)
        discriminatorPath = os.path.join(CKPT_DIR, D_FN)
        torch.save(self.generator.state_dict(), generatorPath)
        print("\tSaved generator: %s" % generatorPath)
        torch.save(self.discriminator.state_dict(), discriminatorPath)
        print("\tSaved discriminator: %s" % discriminatorPath)

    def plot_loss(self, output_dir=os.getcwd()):
        plt.plot(self.discrim_loss, c='red')
        plt.plot(self.gen_loss, c='blue')
        plt.title("GAN Loss per Epoch")
        plt.legend(['Discriminator', 'Generator'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        save_path = os.path.join(output_dir, 'GAN_Loss_per_Epoch_final.png')
        plt.savefig(save_path, transparent=True)
        plt.close()

# class DLoss(nn.Module):
#     ''' C-RNN-GAN discriminator loss
#     '''
#     def __init__(self, label_smoothing=False):
#         super(DLoss, self).__init__()
#         self.label_smoothing = label_smoothing

#     def forward(self, logits_real, logits_gen):
#         ''' Discriminator loss

#         logits_real: logits from D, when input is real
#         logits_gen: logits from D, when input is from Generator

#         loss = -(ylog(p) + (1-y)log(1-p))

#         '''
#         logits_real = torch.clamp(logits_real, EPSILON, 1.0)
#         d_loss_real = -torch.log(logits_real)

#         if self.label_smoothing:
#             p_fake = torch.clamp((1 - logits_real), EPSILON, 1.0)
#             d_loss_fake = -torch.log(p_fake)
#             d_loss_real = 0.9*d_loss_real + 0.1*d_loss_fake

#         logits_gen = torch.clamp((1 - logits_gen), EPSILON, 1.0)
#         d_loss_gen = -torch.log(logits_gen)

#         batch_loss = d_loss_real + d_loss_gen
#         return torch.mean(batch_loss)
