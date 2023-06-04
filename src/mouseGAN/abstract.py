import abc
import os
import numpy as np
import matplotlib.pyplot as plt

from pymousegan.plot import plot_paths


class Generator(metaclass=abc.ABCMeta):
    """Abstract Generator class for the GAN.
    """

    def __init__(self, rand_noise_size=(100, 1), seq_shape=(100, 3),
                 build_kwargs={}):
        assert isinstance(rand_noise_size, (list, tuple)), \
            "rand_noise_size must be a list or tuple."
        self.noise_size = rand_noise_size
        self.seq_shape = seq_shape  # exists for testing
        self.model = self.build_model(**build_kwargs)

    def generate_noise(self, batch_size):
        gen_size = [batch_size] + list(self.noise_size)
        return np.random.normal(0, 1, size=gen_size)

    @abc.abstractmethod
    def build_model(self):
        """Main method for creating the generator model.

        Returns:
            tf.keras.Model or tf.Sequential()
        """
        return


class Discriminator(metaclass=abc.ABCMeta):
    """Abstract discriminator class for the GAN.
    """

    def __init__(self, seq_shape=(100, 3), build_kwargs={}):
        self.seq_shape = seq_shape
        self.model = self.build_model(**build_kwargs)

    @abc.abstractmethod
    def build_model(self):
        """Main method for creating the discriminator model.

        Returns:
            tf.keras.Model or tf.Sequential()
        """
        return


class MinibatchDiscriminator(Discriminator):
    """Abstract discriminator class for GANs that handle minibatch
    discrimination. Note: you still need to include the MinibatchDiscrimination
    layer in build_model.
    """

    def __init__(self, seq_shape=(100, 3), minibatch_discrim_units=None,
                 minibatch_discrim_row_size=None, build_kwargs={}):
        # will only act as kwargs if both args are integers
        if isinstance(minibatch_discrim_units, int) and \
                isinstance(minibatch_discrim_row_size, int):
            minibatch_discrim = {
                'units': minibatch_discrim_units,
                'row_size': minibatch_discrim_row_size,
            }
        elif minibatch_discrim_units is None or \
                minibatch_discrim_row_size is None:
            print('Warning: one of ' +
                  'minibatch_discrim_units/minibatch_discrim_row_size ' +
                  'is missing so no minibatch discrimination will be done.')
            minibatch_discrim = None

        else:
            minibatch_discrim = None
        # assumes that build_model has the minibatch_discrim kwargs argument
        build_kwargs.update({'minibatch_discrim': minibatch_discrim})

        super().__init__(seq_shape=seq_shape, build_kwargs=build_kwargs)

    @abc.abstractmethod
    def build_model(self, minibatch_discrim=None):
        """Main method for creating the discriminator model.

        Returns:
            tf.keras.Model or tf.Sequential()
        """
        return


class GAN(metaclass=abc.ABCMeta):
    """Abstract GAN class.
    """

    def __init__(self, discriminator, generator, d_opt, g_opt, model_paths={},
                 compile_kwargs={}):
        """
        Args:
            discriminator: Discriminator class, not the model
            generator: Generator class, not the model
            d_opt (tf.keras.optimizers.Optimizer): the discriminator optimizer
            g_opt (tf.keras.optimizers.Optimizer): the generator optimizer
            model_paths (dictionary): with keys 'discrim_path', 'gen_path',
                'combined_path' to represent the paths to each of the weights
                to load.
            compile_kwargs (dictionary): of arguments to feed into the
                `compile_models` method.
        """
        self.discriminator = discriminator
        self.generator = generator
        self.d_opt = d_opt
        self.g_opt = g_opt
        self.load_models(**model_paths)
        self.compile_models(**compile_kwargs)

    @abc.abstractmethod
    def load_models(self, discrim_path, gen_path, combined_save_path):
        """Loads the model paths (assumes this is called before `compile_models`)
        """
        pass

    @abc.abstractmethod
    def compile_models(self):
        """Compiles the models and creates the `combined` model field.

        If you're loading weights, make sure to call this after
        `load_models` to create the `combined` model properly.
        """
        return

    @abc.abstractmethod
    def train_step(self, real_paths, gt, batch_size=128):
        """Single training step for GAN

        Args:
            real_paths (np.ndarray OR tf.tensor): The paths to model
                with shape (num_paths, path_count, 3)
            gt (np.ndarray): Tuple of the groundtruths.
                First element should be an array of all 1s (real).
                Second element should be an array of all 0s (fake).
            batch_size (int):

        Returns:
            d_loss: [Discriminator loss, accuracy]
            g_loss: generator loss
        """
        return

    def train(self, real_paths, num_epochs, batch_size=128,
              sample_interval=50, num_plot_paths=10, output_dir=os.getcwd(),
              save_format='h5', initial_epoch=0):
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
        if save_format == 'tf':
            raise Exception(
                'Saving as .tf is currently not recommended because it doesnt save the optimizer state as of tf v.2.3.1')

        self.discrim_loss = []
        self.gen_loss = []
        # Adversarial ground truths
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        for epoch in range(initial_epoch, num_epochs):
            d_loss, g_loss = self.train_step(real_paths, (real, fake),
                                             batch_size)
            if (epoch % sample_interval) == 0:
                # Saving 3 predictions
                self.save_prediction(epoch, num_plot_paths,
                                     output_dir=output_dir)

            # Print the progress and save into loss lists
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                  (epoch, d_loss[0], 100*d_loss[1], g_loss))
            self.discrim_loss.append(d_loss[0])
            self.gen_loss.append(g_loss)

        self.plot_loss(output_dir)

        self.save_models(output_dir, num_epochs, save_format=save_format)

    def save_prediction(self, curr_epoch, num_paths=3, output_dir=os.getcwd()):
        """Saves single generator prediction.
        """
        noise = self.generator.generate_noise(num_paths)
        fake_paths = self.generator.model.predict(noise)
        plot_paths(fake_paths)
        save_path = os.path.join(output_dir, f'path_{curr_epoch}.png')
        plt.savefig(save_path, transparent=True)
        plt.close()

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

    def save_models(self, save_dir, num_epochs, save_format='tf'):
        """Saves the discriminator and generator in the GAN.

        Args:
            save_dir (str): Output directory to save the models in.
            num_epochs (int): Number of epochs; used to name the weights
        """
        self.discriminator.model.trainable = False
        combined_save_path = os.path.join(save_dir,
                                          f'combined_{num_epochs}_weights.{save_format}')
        print(f'Saving combined at {combined_save_path}...')
        self.combined.save(combined_save_path, save_format=save_format)

        self.discriminator.model.trainable = True
        discrim_save_path = os.path.join(save_dir,
                                         f'discrim_{num_epochs}_weights.{save_format}')
        print(f'Saving discriminator at {discrim_save_path}...')
        self.discriminator.model.save(discrim_save_path,
                                      save_format=save_format)

        gen_save_path = os.path.join(save_dir,
                                     f'gen_{num_epochs}_weights.{save_format}')
        print(f'Saving generator at {gen_save_path}...')
        self.generator.model.save(gen_save_path, save_format=save_format)