import abc
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import Variable
import glob

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

    def __init__(self, generator, discriminator, conditional_freezing=False):
        """
        Args:
            generator: Generator class
            discriminator: Discriminator class
        """
        self.discriminator = discriminator
        self.generator = generator
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
            self.generator.load_state_dict(torch.load(latest_g_model))
            print(f"Loaded generator model: {latest_g_model}")
        if latest_d_model is not None:
            self.discriminator.load_state_dict(torch.load(latest_d_model))
            print(f"Loaded discriminator model: {latest_d_model}")
        if latest_g_model is not None and latest_d_model is not None:
            startingEpoch = min(int(latest_g_model.split('/')[-1].split('.')[0][1:]), int(latest_d_model.split('/')[-1].split('.')[0][1:]))
            print(f"Starting from epoch {startingEpoch}")
        else:
            print("No pretrained models found. Starting from scratch.")
            startingEpoch = 0
        self.startingEpoch = startingEpoch

    def train_epoch(self, dataloader):
        g_loss_total, d_loss_total = 0.0, 0.0
        for i, dataTuple in enumerate(dataloader, 0): 
            mouse_trajectories, buttonTargets, trajectoryLengths = dataTuple
            mouse_trajectories, buttonTargets, trajectoryLengths = dataTuple
            # if len(mouse_trajectories) != BATCH_SIZE:
            #     continue
            mouse_trajectories = mouse_trajectories.to(self.device)
            buttonTargets = buttonTargets.to(self.device).squeeze(1)

            real_batch_size = mouse_trajectories.shape[0]

            g_states = self.generator.init_hidden(real_batch_size)
            d_state = self.discriminator.init_hidden(real_batch_size)

            raise NotImplementedError
            # train discriminator
            if self.conditional_freezing and self.freeze_d:
                real_data = Variable(real_data)
                real_prediction = self.discriminator(real_data)
                real_error = self.loss(real_prediction, Variable(torch.ones(batch_size, 1)))
                real_error.backward()

                fake_data = self.generator.generate_noise(batch_size).detach()
                fake_prediction = self.discriminator(fake_data)
                fake_error = self.loss(fake_prediction, Variable(torch.zeros(batch_size, 1)))
                fake_error.backward()

                self.optimizer_D.step()
                self.optimizer_D.zero_grad()

            # train generator
            fake_data = self.generator.generate_noise(batch_size)
            prediction = self.discriminator(fake_data)
            error = self.loss(prediction, Variable(torch.ones(batch_size, 1)))
            error.backward()

            self.optimizer_G.step()
            self.optimizer_G.zero_grad()

            gen_loss = error
            disc_loss = real_error + fake_error

            accuracy = (torch.sum(real_prediction > 0.5) + torch.sum(fake_prediction < 0.5)).float() / (2 * batch_size)

            if self.conditional_freezing:
                self.freeze_d = False
                if accuracy >= 95.0:
                    self.freeze_d = True
            g_loss_total += gen_loss.item()
            d_loss_total += disc_loss.item()

        return g_loss_total / len(dataloader), d_loss_total / len(dataloader)
    
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

    def train(self, dataloader, num_epochs, modelSaveInterval=None,
              sample_interval=None, num_plot_paths=10, output_dir=os.getcwd()):
        """
        Args:
            real_paths (np.ndarray): Total dataset
            num_epochs (int): number of batches to train for
            sample_interval (int): When to sample
            num_plot_paths (int): Number of paths to sample
            output_dir (str): where to save the models and sample images
            save_format (str): one of 'h5' or 'tf'
            initial_epoch (int): The initial epoch to start training from
        """
        self.freeze_d = False
        self.discrim_loss = []
        self.gen_loss = []
        self.generator.train()
        self.discriminator.train()

        for epoch in range(self.startingEpoch, num_epochs + self.startingEpoch):
            d_loss, g_loss = self.train_epoch(dataloader)
            if sample_interval and (epoch % sample_interval) == 0:
                raise NotImplementedError
                # Saving 3 predictions
                self.save_prediction(epoch, num_plot_paths,
                                     output_dir=output_dir)
            if modelSaveInterval and (epoch % modelSaveInterval) == 0 and epoch != 0:
                self.save_models(epoch)
            print("%d D avg loss: %.3f G avg loss: %.3f" % (epoch, d_loss, g_loss))
            self.discrim_loss.append(d_loss)
            self.gen_loss.append(g_loss)

        self.plot_loss(output_dir)
        self.save_models(epoch)

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
