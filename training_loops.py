import math
import pickle
from Generator import Generator
from Discriminator import Discriminator
from loss import gradient_penalty, wasserstein_loss_discriminator, non_saturating_logistic_loss, PathLengthPenalty
from loss import path_length_regularisation
from keras.optimizer_v2.adam import Adam
import tensorflow as tf
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import tqdm
from StyleGAN_UTILS import MappingNetwork, generate_noise
from utils import plot_images, plot_function
import gc


def load(checkpoint_path):
    with open(checkpoint_path, 'rb') as checkpoint_file:
        instance = pickle.load(checkpoint_file)
    checkpoint_file.close()
    return instance


class GAN:
    """
    Encapsulation of the Style GAN architecture, internal training function, and all the functionalities needed to
    perform plotting and saving the models
    """

    def __init__(self,
                 z_dim: int = 512,
                 mapping_network_depth: int = 8,
                 w_dim: int = 512,
                 dataset_path: str = os.path.join('Data', 'images (HQ)'),
                 batch_size: int = 32,
                 data_format: str = 'channels_first',
                 resolution_log2: int = 8,
                 steps: int = 100_000,
                 learning_rate: float = 1e-3,
                 adam_betas: tuple = (.0, .99),
                 lazy_gradient_penalty_interval: int = 16,
                 lazy_path_length_penalty_interval: int = 16,
                 gradient_penalty_coefficient: float = 0.5,
                 storage_directory: str = os.path.join('models', 'ImageGenerator'),
                 resume: bool = True,
                 latest_checkpoint: str = None,
                 plotting_interval: int = 100,
                 checkpoint_interval: int = 1000,
                 style_mixing_probability: float = 0.1) -> None:
        """
        class initializer
        :param dataset_path: str, the path on which the dataset is stored, default is Data/images (HQ), which should
         work without any problem assuming you downloaded the dataset using the provided code in
         the file(data_downloader.py)
        :param batch_size: int, number of images per generator's step, default is 32 as in the Paper
        "Analyzing and Improving the Image Quality of StyleGAN"
        :param data_format: str, 'channels_first' or 'channels_last', defines on which axis are the color channels,
        default is 'channels_first'
        :param resolution_log2: int, the logarithm to the base 2 of the final desired resolution, default is 8, despite
        the paper suggested a good solution for high resolution images the computation power at hand might not be able
        to handle it
        :param steps: int, number of training steps, default 100_000
        :param learning_rate: float, default 1e-3
        :param adam_betas: tuple of two floats, beta1 and beta2 that are the parameters of adam optimizer,
        default (0.0, 0.99)
        :param lazy_gradient_penalty_interval: int, number of steps between each time the model adds the gradient
        penalty loss term, default 16
        :param lazy_path_length_penalty_interval: int, number of steps between each time the model adds the path length
        loss term, default 16
        :param gradient_penalty_coefficient: float, scaling factor for the gradient penalty, default .5
        :param storage_directory: str, path to the directory on which model checkpoints will be saved
        :param resume: bool, define the preference of starting training from scratch (False) or resume from the
        latest checkpoint (True), default True
        :param latest_checkpoint: str, the path to the latest checkpoint, MUST be provided if resume is True
        :param plotting_interval: int, number of steps to plot the results of the generator
        :param checkpoint_interval: int, number of steps to save a checkpoint of the model
        """
        # define the mapping network
        self.mapping_network = MappingNetwork(
            n_hidden_layers=mapping_network_depth,
            hidden_dim=w_dim,
            w_dim=w_dim)
        # define the Generator's model
        self.generator = Generator(data_format=data_format,
                                   resolution_log2=resolution_log2)
        # define the Discriminators Model
        self.discriminator = Discriminator(data_format=data_format,
                                           resolution=resolution_log2)
        # create a dataset generator from the specified directory
        self.dataset = ImageDataGenerator(rescale=1 / 255., data_format=data_format).flow_from_directory(
            dataset_path,
            target_size=[2 ** resolution_log2, 2 ** resolution_log2],
            batch_size=batch_size,
            class_mode=None)

        # save arguments as class attributes
        self.learning_rate = learning_rate
        self.steps = steps
        self.lazy_gradient_penalty_interval = lazy_gradient_penalty_interval
        self.lazy_path_length_penalty_interval = lazy_path_length_penalty_interval
        self.gradient_penalty_coefficient = gradient_penalty_coefficient
        optimizers_parameters = dict(learning_rate=self.learning_rate,
                                     beta_1=adam_betas[0],
                                     beta_2=adam_betas[1])
        self.generator_optimizer = Adam(**optimizers_parameters)
        self.discriminator_optimizer = Adam(**optimizers_parameters)
        self.mapping_network_optimizer = Adam(**optimizers_parameters)
        self.path_length_regularizer = PathLengthPenalty()
        self.last_step_index = 0
        self.storage_directory = storage_directory
        self.batch_size = batch_size
        self.variables_to_save = {
            'last_step_index': self.last_step_index,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'adam_betas': adam_betas,
            'lazy_gradient_penalty_interval': lazy_gradient_penalty_interval,
            'lazy_path_length_penalty_interval': lazy_path_length_penalty_interval,
            'gradient_penalty_coefficient': gradient_penalty_coefficient
        }
        self.resume = resume
        self.latest_checkpoint = latest_checkpoint
        self.plotting_interval = plotting_interval
        self.checkpoint_interval = checkpoint_interval

        # create a placeholder for the generator's output for plotting
        shape = [self.batch_size, 3, 2 ** resolution_log2, 2 ** resolution_log2]
        if data_format == 'channels_last':
            shape = [self.batch_size, 2 ** resolution_log2, 2 ** resolution_log2, 3]
        self.batch_to_plot = tf.random.normal(shape=shape)
        self.generator_loss_history = []
        self.discriminator_loss_history = []

        self.latest_inputs = {
            'z': tf.constant(0.),
            'w': tf.constant(0.)
        }
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.resolution_log2 = resolution_log2
        self.style_mixing_probability = style_mixing_probability
        self.data_format = data_format
        self.a = 10
        self.decay = 0.001

    def get_w(self):
        if np.random.uniform(0, 1) < self.style_mixing_probability:
            cross_over_point = int(np.random.uniform(0, 1) * (self.resolution_log2 - 2))
            z1 = generate_noise(self.batch_size, z_dim=self.z_dim)
            z2 = generate_noise(self.batch_size, z_dim=self.z_dim)
            w1 = self.mapping_network(z1)
            w2 = self.mapping_network(z2)
            w1 = tf.tile(w1[None, :, :], multiples=[cross_over_point, 1, 1])
            w2 = tf.tile(w2[None, :, :], multiples=[self.resolution_log2 - 2 - cross_over_point, 1, 1])
            w = tf.concat([w1, w2], axis=0)
            self.latest_inputs['z'] = z1
            self.latest_inputs['w'] = w
            return w
        else:
            z = generate_noise(self.batch_size, z_dim=self.z_dim)
            self.latest_inputs['z1'] = z
            w = self.mapping_network(z)
            self.latest_inputs['w1'] = w
            w = tf.tile(w[None, :, :], multiples=[self.resolution_log2 - 2, 1, 1])
            return w

    def step(self, ind, real_images):
        """
        take one training step
        :param ind: int, current step number, makes the function self-aware about where it is in the training process
        :return: None, to plot the current batch of images they will be saved to the self.batch_to_plot attribute
        """

        # create a Gradient Tape for each model (generator, discriminator)
        with tf.GradientTape(persistent=True) as discriminator_tape, tf.GradientTape(persistent=True) as generator_tape:
            discriminator_tape.watch(real_images)
            # generate a batch of fake images
            w = self.get_w()
            generated_images = self.generator(w)

            # calculate the discriminator's output for the generated batch
            fake_outputs = self.discriminator(generated_images, training=True)
            # calculate the discriminator's output for the real images
            real_outputs = self.discriminator(real_images, training=True)
            # calculate the discriminator's loss
            discriminator_loss = wasserstein_loss_discriminator(real_outputs, fake_outputs)
            # calculate the generator's loss
            generator_loss = non_saturating_logistic_loss(f_fake=fake_outputs)

            # in case the steps meets the gradient penalty interval
            if (ind + 1) % self.lazy_gradient_penalty_interval == 0:
                # calculate the gradients of the discriminator with respect to the real images
                gradients = discriminator_tape.gradient(target=real_outputs, sources=real_images)
                # compute the gradient penalty term
                gp = gradient_penalty(gradients)
                # add the scaled gradient penalty to the discriminator's loss term
                discriminator_loss += 0.5 * self.gradient_penalty_coefficient * gp * self.lazy_gradient_penalty_interval

            if (ind + 1) % self.lazy_path_length_penalty_interval == 0:
                # calculate the path length penalty
                plp, variation = path_length_regularisation(generator=self.generator, w=w, a=self.a)
                print(plp)
                self.a = self.a * (1 - self.decay) + variation * self.decay
                # path length penalty might be None in the warmup (first) iteration
                if plp is not None:
                    # add path length penalty term to the generator
                    generator_loss += plp

            # compute gradients of the discriminator's loss function w.r.t the discriminator's trainable variables
            grads_of_discriminator = discriminator_tape.gradient(discriminator_loss,
                                                                 self.discriminator.trainable_variables)

            # compute the gradients of the generator's loss function w.r.t the generator's trainable variables
            grads_of_generator = generator_tape.gradient(generator_loss, self.generator.trainable_variables)

        # take an optimization step for the discriminator
        self.discriminator_optimizer.apply_gradients(
            zip(grads_of_discriminator, self.discriminator.trainable_variables))
        # take an optimization step for the generator
        self.generator_optimizer.apply_gradients(zip(grads_of_generator, self.generator.trainable_variables))
        # self.batch_to_plot = generated_images
        self.discriminator_loss_history.append(discriminator_loss)
        self.generator_loss_history.append(generator_loss)

        del generated_images
        del generator_loss
        del generator_tape
        del fake_outputs
        del real_images
        del real_outputs
        del discriminator_tape
        del discriminator_loss
        del grads_of_generator
        del grads_of_discriminator
        gc.collect()


    def train(self) -> None:
        """
        putting all the training steps together in one function (iterate through desired number of steps,
        plot generated images each desired number of steps, save the model each desired number of steps
        :return: None
        """
        # create a place to save the checkpoints if it doesn't already exist
        if not os.path.isdir(self.storage_directory):
            os.mkdir(self.storage_directory)

        # loop through the entirety of the steps
        for i in tqdm.tqdm(range(self.last_step_index, self.steps, 1), desc="Training"):
            # take a training step
            # generate the next batch of training images
            real_images = tf.convert_to_tensor(self.dataset.next())
            self.step(i, real_images)

            # check checkpoint interval
            if i % self.checkpoint_interval == 0 and i != 0:
                self.last_step_index = i
                # save the current work
                self.save()

            # check plotting interval
            if i % self.plotting_interval == 0 and i != 0:
                self.last_step_index = i
                # plot the current generated images batch
                self.plot()

    def save(self) -> None:
        """
        a helper function to save the current instance(self) of the class
        :return: None
        """
        # set the current directory (w.r.t. the current step)
        current_checkpoint_directory = os.path.join(self.storage_directory, f'checkpoint {self.last_step_index:04}')
        self.mapping_network.save(os.path.join(current_checkpoint_directory, 'Mapping Network'))
        self.generator.save(os.path.join(current_checkpoint_directory, 'Generator'))
        self.discriminator.save(os.path.join(current_checkpoint_directory, 'Discriminator'))

    def plot(self):
        # make sure all the image to plot are in the valid float range for plotting
        self.batch_to_plot = tf.clip_by_value((self.batch_to_plot + 1) / 2., 0, 1)
        # in case the batch to plot contains only one image
        plot_images(self.batch_to_plot,
                    total_title=f'A Batch Of Images after {self.last_step_index} Steps',
                    data_format=self.data_format)
        plot_function(tf.convert_to_tensor([self.discriminator_loss_history, self.generator_loss_history]),
                      legend=['Discriminator Loss', 'Generator Loss'])


def main(params: dict = None):
    if params is None:
        params = {
            'batch_size': 1,
            'resolution_log2': 6,
            'data_format': 'channels_first',
            'lazy_gradient_penalty_interval': 2,
            'lazy_path_length_penalty_interval': 2,
            'plotting_interval': 10,
            'checkpoint_interval': 1000,
        }

    total_gan = GAN(**params)
    total_gan.train()


if __name__ == '__main__':
    with tf.device('/gpu:0'):
        main()
