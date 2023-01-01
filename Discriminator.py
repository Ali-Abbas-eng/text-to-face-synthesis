import gc
from abc import ABC
import numpy as np
from keras.layers import Layer, Dense
from keras.models import Model
from StyleGAN_UTILS import ConvUpDown, FromRGB
from StyleGAN_UTILS import fused_bias_activation, num_filters, minibatch_stddev_layer
from StyleGAN_UTILS import channels_last, channels_first
import tensorflow as tf
from utils import print_colored_text as c_print


class DiscriminatorBlock(Layer):
    """
    encapsulate one block of the Discriminator (input -> same size conv -> downsample conv -> output 1)
                                                                                                [+] -----> output
                                               (input -------------------> downsample conv -> output 2)
    """

    def __init__(self,
                 resolution: int,
                 feature_maps_decay: float = 1.0,
                 resample_kernel=None,
                 data_format: str = 'channels_first',
                 **kwargs: dict) -> None:
        """
        Initializer of the Discriminator's Block, sets up the three required convolutional blocks
        :param resolution: int, stage (the log2 of the current resolution)
        :param feature_maps_decay: float, a factor to manipulate number of filters in the output image
        :param resample_kernel: list or list-like, the kernel used in the resample kernel default [1, 3, 3, 1]
        :param data_format: str, determines the axis of the channels
        :param kwargs: dict, keyword arguments for the super class initializer
        """
        super(DiscriminatorBlock, self).__init__(trainable=kwargs.get('trainable', True))
        if resample_kernel is None:
            # in case the resample kernel is none (use the default value of [1, 3, 3, 1])
            resample_kernel = [1, 3, 3, 1]
        # set class's attributes
        self.data_format = data_format
        self.resolution = resolution
        self.resample_kernel = resample_kernel
        self.feature_maps_decay = feature_maps_decay
        self.input_filters = num_filters(self.resolution, feature_maps_decay=self.feature_maps_decay)

        # first convolutional layer
        self.conv_1 = ConvUpDown(input_filters=self.input_filters,
                                 resolution=self.resolution - 1,
                                 feature_maps_decay=self.feature_maps_decay,
                                 kernel_size=3,
                                 resample_kernel=self.resample_kernel,
                                 data_format=self.data_format)

        # second convolution layer
        self.conv_2 = ConvUpDown(input_filters=num_filters(self.resolution - 1),
                                 resolution=self.resolution - 2,
                                 feature_maps_decay=self.feature_maps_decay,
                                 kernel_size=3,
                                 up=False,
                                 down=True,
                                 resample_kernel=resample_kernel,
                                 data_format=self.data_format)

        # skip connection (input feed to a convolutional layer with same output shape as the second one (for adding))
        self.input_skip = ConvUpDown(input_filters=self.input_filters,
                                     resolution=self.resolution - 2,
                                     feature_maps_decay=self.feature_maps_decay,
                                     up=False,
                                     down=True,
                                     resample_kernel=self.resample_kernel,
                                     data_format=self.data_format)

        self.bias_1 = None
        self.bias_2 = None

    def build(self, input_shape: tuple) -> None:
        # specify the number of filters in the first convolutional layer
        first_layer_filters = num_filters(self.resolution - 1, feature_maps_decay=self.feature_maps_decay)

        # build the bias of the first convolutional layer
        self.bias_1 = tf.Variable(initial_value=tf.zeros(shape=(first_layer_filters,)),
                                  shape=(first_layer_filters,),
                                  trainable=True)

        # same operations as above but for the second convolutional layer
        second_layer_filters = num_filters(self.resolution - 2, feature_maps_decay=self.feature_maps_decay)
        self.bias_2 = tf.Variable(initial_value=tf.zeros(shape=(second_layer_filters,)),
                                  shape=(second_layer_filters,),
                                  trainable=True)

    def call(self, inputs: tf.Tensor, **kwargs: dict) -> tf.Tensor:
        """
        create a callable instance
        :param inputs: tf.Tensor, the input to the block (a batch of images with an arbitrary number of channels)
        :param kwargs: dict, keyword arguments to the super class call function call
        :return: tf.tensor, convolved * 2 + plus skip conv output of the block
        """
        # super class call function call with keyword arguments
        super(DiscriminatorBlock, self).call(kwargs)

        # save the inputs (to be passed through the skip connection)
        x_shortcut = inputs

        # apply the first convolution
        x = self.conv_1(inputs)
        # add bias and activate
        x = fused_bias_activation(x, b=self.bias_1, data_format=self.data_format)

        # apply the second convolution
        x = self.conv_2(x)
        # add second bias and activate
        x = fused_bias_activation(x, b=self.bias_2, data_format=self.data_format)

        # convolve the inputs again to be of the same shape of the output of the second convolution
        x_shortcut = self.input_skip(x_shortcut)

        # combine the outputs (x + x_skip) / square_root(2)
        return (x + x_shortcut) * (1. / tf.sqrt(2.))


class Discriminator(Model, ABC):
    """
    Encapsulation of the whole Discriminator's Architecture (as discussed in the paper)
    """

    def __init__(self,
                 resolution: int = 10,
                 resample_kernel: list = None,
                 feature_maps_decay: float = 1.0,
                 data_format: str = 'channels_first',
                 negative_slop: float = .2,
                 **kwargs) -> None:
        """
        initializer of the discriminator, dynamically builds the network with respect to the desired resolution output
        :param resolution: int, log_2 of the resolution (resolution of the generator's output (e.g., 10 - > 1024 ** 2)
        :param resample_kernel: list or list-like, the resample kernel to be used by the up/downsample operations
        :param feature_maps_decay: float, a factor to manipulate the number of filters in each stage
        :param data_format: str, determines which axis holds the channels
        :param negative_slop: float, the alpha parameters of the relu activation function
        :param kwargs: dict, keyword argument to the super class init function
        """
        # call to the super class initializer
        super(Discriminator, self).__init__(kwargs)

        # in case the resample kernel is None
        if resample_kernel is None:
            # use the default value
            resample_kernel = [1, 3, 3, 1]

        # set class' attributes
        self.resample_kernel = resample_kernel
        self.data_format = 'NCHW' if data_format in channels_first else 'NHWC'
        self.resolution = resolution
        self.feature_maps_decay = feature_maps_decay
        self.from_rgb_name_base = "from_RGB res_"
        self.discriminator_block_name_base = 'DiscBlock res_'
        self.negative_slop = negative_slop

        # iterate over each resolution (from the highest resolution to the lowest)
        # and build discriminator's block accordingly
        for i in range(self.resolution, 2, -1):
            # the input is the RGB output of the Generator (we need to pass that through a FromRGB layer (only once))
            if i == self.resolution:
                # define the FromRGB Layer
                from_rgb = FromRGB(resolution=i,
                                   feature_maps_decay=self.feature_maps_decay,
                                   data_format=self.data_format)
                # set attribute dynamically
                self.__setattr__(f'{self.from_rgb_name_base}{2 ** i}', from_rgb)

            # define the current block of the discriminator at the current resolution
            current_block = DiscriminatorBlock(resolution=i,
                                               resample_kernel=self.resample_kernel,
                                               feature_maps_decay=self.feature_maps_decay,
                                               data_format=self.data_format)
            # set attribute dynamically
            self.__setattr__(f'{self.discriminator_block_name_base}{2 ** i}', current_block)

        # save us some memory
        del current_block
        del from_rgb
        gc.collect()

        # last convolutional block in the discriminator (no up/downsampling and no skips)
        self.last_conv = ConvUpDown(input_filters=num_filters(1) + 1,
                                    resolution=1,
                                    up=False,
                                    down=False,
                                    kernel_size=3,
                                    data_format=self.data_format)

        # create a dense layer with (num filters) units (neurons)
        self.dense_1 = Dense(units=num_filters(0))
        self.dense_1_bias = tf.Variable(initial_value=tf.zeros(shape=[num_filters(0)]),
                                        trainable=True,
                                        shape=[num_filters(0)])

        # create the final dense layer with one neuron (the one that we'll actually use to classify real/fake images)
        self.dense_0 = Dense(units=1)

    def call(self, inputs, **kwargs):
        x = None  # initialize the x with None (avoiding PEP8 errors)

        # iterate again over each resolution (highest to lowest)
        for i in range(self.resolution, 2, -1):
            # for the first block we need to convert from RGB to an input with a lot more channels
            if i == self.resolution:
                # get the one and only FromRGB layer
                from_rgb = self.__getattribute__(f'{self.from_rgb_name_base}{2 ** i}')
                # transform the input images
                x = from_rgb(inputs)

            # get the block that corresponds to the current resolution
            current_block = self.__getattribute__(f'{self.discriminator_block_name_base}{2 ** i}')

            # pass x into the current discriminator's block
            x = current_block(x)

        x = minibatch_stddev_layer(x, data_format=self.data_format)

        # convolve x one last time
        x = self.last_conv(x)

        # while x is the output of convolutions it should be flattened before being fed to a dense layer
        # flatten
        x = tf.reshape(x, [-1, np.prod(x.shape[1:])])

        # propagate x into the first dense layer
        x = self.dense_1(x)

        # activate
        x = fused_bias_activation(x, self.dense_1_bias)

        # the final one valued score (batch-wise)
        y = self.dense_0(x)
        return y


if __name__ == '__main__':
    from time import sleep


    def test_discriminator_block(data_format: str = 'channels_first'):
        c_print(f'Testing Discriminator Block...,                 data format = {data_format}', 'red')
        params = {
            'batch_size': 8,
            'resolution': 8,
            'feature_maps_decay': 1.0
        }
        in_filters = num_filters(params['resolution'], feature_maps_decay=params['feature_maps_decay'])
        input_shape = [params['batch_size'], in_filters, 2 ** params['resolution'], 2 ** params['resolution']]
        if data_format == 'channels_last':
            input_shape = [input_shape[0], *input_shape[2:], input_shape[1]]

        last_layer_output = tf.random.normal(shape=input_shape)
        c_print(f"Shape of the Previous Block's Output: {last_layer_output.shape}", 'blue')

        discriminator_block = DiscriminatorBlock(resolution=params['resolution'],
                                                 feature_maps_decay=params['feature_maps_decay'])
        result = discriminator_block(last_layer_output)
        c_print(f'Shape of the Output of the Discriminator Block: {result.shape}', 'blue')
        c_print('Done!\n')


    def test_discriminator_architecture(data_format: str = 'channels_first'):
        c_print(f"Testing Discriminator's Architecture...,        data format = {data_format}", 'red')
        params = {
            'batch_size': 8,
            'resolution': 8
        }

        input_shape = [params['batch_size'], 3, 2 ** params['resolution'], 2 ** params['resolution']]
        if data_format in channels_last:
            input_shape = [params['batch_size'], *input_shape[2:], 3]

        inputs = tf.random.normal(shape=input_shape)
        c_print(f"Shape of the Generator's Output: {inputs.shape}", 'blue')

        discriminator = Discriminator(data_format=data_format)
        results = discriminator(inputs)
        c_print(f"Shape of the Discriminator's Output: {results.shape}", 'blue')
        c_print('Done!\n')


    test_discriminator_block()
    sleep(1)
    test_discriminator_block(data_format='channels_last')

    test_discriminator_architecture()
    sleep(1)
    test_discriminator_architecture(data_format='channels_last')


