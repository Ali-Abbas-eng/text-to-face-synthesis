import os
from abc import ABC
import torch
from typing import Tuple
from utils import download_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from keras.models import Model
from keras.layers import Layer
from StyleGAN_UTILS import MappingNetwork, NoiseInjector, ModulatedConv2D, ToRGB, ConvUpDown
from StyleGAN_UTILS import generate_noise, fused_bias_activation, num_filters
from dnnlib_tf.tflib.ops.upfirdn_2d import upsample_conv_2d, upsample_2d
from utils import print_colored_text as c_print  # fancy print
from utils import plot_images  # fancy plot
import numpy as np
import warnings
import pickle
from utils import block_print, enable_print
channels_first = ['channels_first', 'CHANNELS_FIRST', 'NCHW']
channels_last = ['channels_last', 'CHANNELS_LAST', 'NHWC']


class GeneratorLayer(Layer):
    """
    an Encapsulation of the Generator Block (i.e., put enough of theses on top of each other, and you got yourself
    a nice state-of-the-art StyleGAN1 or StyleGAN2 generator)
    NOTE: This work is dedicated only for generating 1:1 images (with an arbitrary number of filters) on each block, and
    thus the entire Generator.
    """

    def __init__(self,
                 input_channels: int,
                 resolution: int,
                 w_dim: int or tuple,
                 kernel_size: int = 3,
                 upsample: bool = True,
                 resample_kernel=None,
                 data_format: str = 'channels_first',
                 feature_maps_decay: float = 1.0,
                 **kwargs) -> None:
        """
        initializer of the generator layer (for later use as a component in the generator Block and final architecture)
        :param input_channels: int, the number of filters produced from previous Generator block or the input itself (3)
        :param output_channels: int, the desired number of output filters
        :param w_dim: int, or tuple, the dimensions of the intermediate noise vector
        :param starting_size: int, the width (and height) of the image
        :param kernel_size: int, the width (and height) of the kernel
        :param upsample: bool, double the images' size or not
        :param data_format: str, specify at which axis the channels are, could be one of ['NCHW', 'NHWC',
                                        'channels_first' or 'channels_last']
        :param version: int, set to the version of StyleGAN you would like to use (currently 1 or 2)
        :param feature_maps_decay: float, parameters to add to get different number of filters (specifically in case of
        tests where we want to plot RGB output).
        """
        super(GeneratorLayer, self).__init__(*kwargs)
        self.use_upsample = upsample  # in case the output size (height and width) is double the input size
        self.data_format = data_format  # needed to know in which dimension are the channels

        # modulated convolution layer
        self.mod_conv = ModulatedConv2D(input_channels=input_channels,
                                        output_channels=num_filters(resolution - 1,
                                                                    feature_maps_decay=feature_maps_decay),
                                        w_dimensions=w_dim,
                                        up=self.use_upsample,
                                        kernel_size=kernel_size,
                                        data_format=self.data_format,
                                        resample_kernel=resample_kernel,
                                        demodulate=True)
        # style stochasticity layer (multiply the output of the convolutional layer with random noise channel-wise)
        self.noise_injector = NoiseInjector()

    def call(self, inputs: [tf.Tensor, tf.Tensor], **kwargs) -> tf.Tensor:
        """
        override the call function to create a callable instance of the layer
        :param inputs: iterable, any iterable with two elements (the input images and the W vector)
        :return: tf.Tensor
        """
        super(GeneratorLayer, self).call(kwargs)
        x = self.mod_conv(inputs)
        x = self.noise_injector(x)
        x = fused_bias_activation(x, self.mod_conv.bias, data_format=self.data_format)
        return x


class GeneratorBlock(Layer):
    """
    an Encapsulation of one block of the generator (which corresponds to a specific resolution)
    """

    def __init__(self,
                 input_channels: int,
                 resolution: int,
                 w_dim: int = 512,
                 kernel_size: int = 3,
                 data_format: str = 'channels_first',
                 feature_maps_decay: float = 1.0) -> None:
        """
        initializer of the generator block, creates a modulated convolution layer (with upsampling) followed by a
        modulated convolution layer (without upsampling) and an input/output skip layer
        :param input_channels: int, number of channels from previous layer
        :param resolution: int, the stage resolution of the block
        :param w_dim: int, the desired dimension of intermediate noise vector W
        :param kernel_size: int, height and width of the convolutional kernels
        :param data_format: str, could be one of 'channels_first', 'channels_last', 'NCHW', 'NHWC'
        """
        super(GeneratorBlock, self).__init__(**{'name': f'Generator_Block_{resolution * 2 - 5}'})
        self.data_format = data_format
        # first layer in the block upsamples to the next resolution
        self.layer_1_up = GeneratorLayer(input_channels=input_channels,
                                         resolution=resolution,
                                         w_dim=w_dim,
                                         kernel_size=kernel_size,
                                         upsample=True,
                                         data_format=self.data_format,
                                         feature_maps_decay=feature_maps_decay)

        # second layer in the block (modulated convolution + noise) without upsampling
        self.layer_2 = GeneratorLayer(input_channels=num_filters(resolution),
                                      resolution=resolution,
                                      w_dim=w_dim,
                                      kernel_size=kernel_size,
                                      upsample=False,
                                      data_format=self.data_format,
                                      feature_maps_decay=feature_maps_decay)

        # skip layer (lets the input skip through a convolution with upsampling to the final output)
        self.layer_3_skip = ConvUpDown(input_filters=input_channels,
                                       resolution=resolution - 1,
                                       up=True,
                                       feature_maps_decay=feature_maps_decay,
                                       kernel_size=1,
                                       data_format=self.data_format)

    def call(self, inputs: [tf.Tensor, tf.Tensor], **kwargs) -> tf.Tensor:
        """
        override call function to create a callable instance of the class
        :param inputs: [tf.Tensor, tf.Tensor], output of the last channel (or input constant) and the intermediate noise
        :param kwargs: dict, argument to call super with
        :return: tf.Tensor
        """
        super(GeneratorBlock, self).call(kwargs)
        x_shortcut = inputs[0]
        x = self.layer_1_up(inputs)
        x = self.layer_2([x, inputs[1]])
        x_shortcut = self.layer_3_skip(x_shortcut)
        x = (x + x_shortcut) * (1 / np.sqrt(2))
        return x


class Generator(Model, ABC):
    """
    Putting it all together, compose the Generator using blocks and layers defined earlier, to build a model that gives
    the final output which is an image with the desired size
    """

    def __init__(self,
                 w_dim: int = 512,
                 data_format: str = 'channels_first',
                 resolution_log2: int = 10,
                 resample_kernel=None) -> None:
        """
        initializer of the generator, takes the desired dimensions of input noise vector Z and
        :param z_dim: int, the dimensions of the noise vector
        :param w_dim: int, the dimensions of the intermediate noise vector
        :param data_format: str, identifier of the channels' axis
        :param resolution_log2: int, the logarithm to base 2 of the final desired resolution
        :param resample_kernel: list, resample kernel in the FIR operation
        """
        super(Generator, self).__init__()
        if resample_kernel is None:
            resample_kernel = [1, 3, 3, 1]
        self.resample_kernel = resample_kernel
        self.data_format = data_format

        starting_constant_shape = [1, num_filters(1), 4, 4]
        if self.data_format == 'channels_last':
            starting_constant_shape = starting_constant_shape[0], starting_constant_shape[2], \
                                      starting_constant_shape[3], starting_constant_shape[1]
        self.starting_constant = tf.Variable(name='StartingConstant',
                                             initial_value=tf.random.normal(shape=starting_constant_shape),
                                             shape=starting_constant_shape,
                                             trainable=True)

        self.conv_layer = GeneratorLayer(input_channels=num_filters(1),
                                         resolution=1,
                                         w_dim=w_dim,
                                         kernel_size=3,
                                         upsample=False,
                                         data_format=self.data_format)

        self.to_rgb_1 = ToRGB(w_dimensions=w_dim,
                              input_channels=num_filters(1),
                              data_format=self.data_format)
        self.resolution_log2 = resolution_log2

        for res in range(3, self.resolution_log2 + 1):
            current_main_block = GeneratorBlock(
                input_channels=num_filters(res - 1),
                resolution=res,
                data_format=self.data_format
            )
            current_shortcut_block = ToRGB(input_channels=num_filters(res - 1),
                                           w_dimensions=w_dim,
                                           data_format=self.data_format)

            self.__setattr__(name=f'block_{res * 2 - 5}', value=current_main_block)
            self.__setattr__(name=f'to_RGB_{res * 2 - 5}', value=current_shortcut_block)

    def call(self, w: tf.Tensor, **kwargs) -> tf.Tensor or Tuple[tf.Tensor, dict]:
        """
        override the call function to create a callable instance of the class
        :param w: tf.Tensor, the intermediate noise vector w
        :param kwargs: dict, inputs to super
        :return:
        """
        # super(Generator, self).call(kwargs)
        x = tf.tile(self.starting_constant, [w.shape[1], 1, 1, 1])
        x = self.conv_layer([x, w[0]])
        y = self.to_rgb_1([x, w[0]])
        for res in range(3, self.resolution_log2 + 1):
            current_main_block = self.__getattribute__(f'block_{res * 2 - 5}')
            x = current_main_block([x, w[res - 3]])
            current_shortcut_block = self.__getattribute__(f'to_RGB_{res * 2 - 5}')
            t = current_shortcut_block([x, w[res - 3]])
            y = upsample_2d(y,
                            k=self.resample_kernel,
                            data_format='NCHW' if self.data_format == 'channels_first' else 'NHWC')
            y = y + t

        return y

    def get_latest_inputs(self):
        return self.latest_inputs


def import_generator(pickle_path: str = None, url: str = None) -> torch.nn.Module:
    """
    given a path to the pickled file the function reads the file at the specified path (downloads it first if it does
    not exist) and then returns the generator model
    :param pickle_path: str, the path to the file from/to which the function will read/write the model
    :param url: str, link to pickle file to be downloaded in case the pickle file does not exist
    :return: torch.nn.Module
    """
    if pickle_path is None:
        pickle_path = os.path.join('models', 'ImageGenerator', 'StyleGANPT', 'ffhq.pkl')
    if url is None:
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl'
    block_print()
    # in case the file does not exist
    if not os.path.isfile(pickle_path):
        # download it
        download_data(path=pickle_path,
                      url=url)
    # disable warnings for the next step
    with warnings.catch_warnings():
        # read the pickled file as binary
        with open(pickle_path, 'rb') as f:
            # load the generator model
            generator = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module

    # return the generator model
    return generator


if __name__ == '__main__':
    from time import sleep


    def test_style_gan_generator_layer(data_format: str = 'channels_first'):
        c_print('Testing Generator Layer...', 'red')
        params = {
            'input_channels': 512,
            'resolution': 4,
            'w_dim': 512,
            'batch_size': 4,
            'data_format': data_format
        }

        c_print(params, 'blue')

        gen_layer = GeneratorLayer(input_channels=params['input_channels'],
                                   resolution=params['resolution'],
                                   w_dim=params['w_dim'],
                                   upsample=True,
                                   data_format=params['data_format'])
        shape = params['batch_size'], params['input_channels'], 2 ** params['resolution'], 2 ** params['resolution']
        if data_format == 'channels_last':
            shape = shape[0], shape[2], shape[3], shape[1]
        rand_image = tf.random.normal(shape=shape)

        rand_w = tf.random.normal(shape=[params['batch_size'], params['w_dim']])

        c_print(f'Variables of Interest Shapes:'
                f'\n\tRandom Image: {rand_image.shape}'
                f'\n\tRandom W: {rand_w.shape}',
                'blue')

        result_full = gen_layer([rand_image, rand_w])

        rgb_rand_w = tf.random.normal(shape=[params['batch_size'], params['w_dim']])
        rgb_generator_layer = GeneratorLayer(input_channels=num_filters(params['resolution']),
                                             resolution=12,
                                             feature_maps_decay=2.01,
                                             w_dim=params['w_dim'],
                                             upsample=True,
                                             data_format=params['data_format'])
        rgb_result = rgb_generator_layer([result_full, rgb_rand_w])
        plot_images(rgb_result, labels=[f'GenLayer Output #{i}' for i in range(params['batch_size'])],
                    data_format=params['data_format'])
        c_print(f'Output of the Layer (Shape): {rgb_result.shape}', 'blue')
        del rgb_result
        del rgb_generator_layer
        del result_full
        del rand_w
        del rand_image
        c_print('Done!\n')


    def test_generator_block(data_format: str = 'channels_first'):
        c_print('Testing Generator Block...', 'red')
        params = {
            'batch_size': 4,
            'input_channels': 512,
            'resolution': 5,
            'w_dim': 512,
            'data_format': data_format
        }
        c_print(f'Parameters: {params}')

        shape = [params['batch_size'], params['input_channels'], 2 ** params['resolution'], 2 ** params['resolution']]
        if data_format == 'channels_last':
            shape = shape[0], shape[2], shape[3], shape[1]
        rand_image = tf.random.normal(shape=shape)
        rand_w = tf.random.normal(shape=[params['batch_size'],
                                         params['w_dim']])
        c_print(f'Input Parameters (Shapes):'
                f'\n\tRandom Image: {rand_image.shape}'
                f'\n\tRandom W: {rand_w.shape}',
                'blue')
        test_gen_block = GeneratorBlock(input_channels=params['input_channels'],
                                        w_dim=params['w_dim'],
                                        resolution=12,
                                        feature_maps_decay=2.001,
                                        data_format=params['data_format'])
        result = test_gen_block([rand_image, rand_w])
        plot_images(result, labels=[f'GenBlock Output #{i}' for i in range(params['batch_size'])],
                    data_format=params['data_format'])
        c_print(f'Shape of the Output of the Generator Block: {result.shape}', 'blue')
        del result
        del test_gen_block
        del rand_w
        del rand_image
        del shape
        c_print('Done!\n')


    def test_generator_architecture(data_format: str = 'channels_first'):
        c_print('Testing Generator...', 'red')
        params = {
            'batch_size': 2,
            'z_dim': 512,
            'w_dim': 512,
            'map_hidden_dim': 512,
            'resample_kernel': [1, 3, 3, 1],
            'data_format': data_format,
            'resolution_log_2': 9,
            'style_mixing_probability': 1.
        }

        c_print(f'Input Parameters: '
                f'{params}',
                'blue')

        mapping_network = MappingNetwork(w_dim=params['w_dim'],
                                         hidden_dim=params['map_hidden_dim'],
                                         n_hidden_layers=8)
        z = generate_noise(n_samples=params['batch_size'], z_dim=params['z_dim'])
        test_generator = Generator(
            w_dim=params['w_dim'],
            resample_kernel=params['resample_kernel'],
            data_format=params['data_format'],
            resolution_log2=params['resolution_log_2'],
        )
        w = mapping_network(z)
        w = tf.tile(w[None, :, :], [params['resolution_log_2'] - 2, 1, 1])
        result = test_generator(w)
        test_generator.summary()
        c_print(f'Shape of the Output of the Generator: {result.shape}', 'blue')
        mean = tf.reduce_mean(result,
                              axis=[2, 3] if params['data_format'] == 'channels_first' else [1, 2],
                              keepdims=True)
        std = tf.math.reduce_std(result,
                                 axis=[2, 3] if params['data_format'] == 'channels_first' else [1, 2],
                                 keepdims=True)
        result = (result - mean) / std
        result = tf.clip_by_value(result, 0, 1)
        plot_images(result, [f'Output Image #{i}' for i in range(params['batch_size'])],
                    data_format=params['data_format'])
        # [c_print(f"\t{key}'s Shape: {value.shape}\n", 'blue') for key, value in latents.items()]
        del result
        del std
        del mean
        del test_generator
        c_print('Done!\n')


    test_style_gan_generator_layer()
    sleep(1)
    test_style_gan_generator_layer('channels_last')
    sleep(1)
    test_generator_block()
    sleep(1)
    test_generator_block('channels_last')
    sleep(1)
    test_generator_architecture()
    sleep(1)
    test_generator_architecture('channels_last')
