from abc import ABC
import gc
import tensorflow as tf
from keras.models import Model
from keras.layers import Layer, Dense
from scipy.stats import truncnorm
import numpy as np
from utils import print_colored_text as c_print
from utils import plot_images
from dnnlib_tf.tflib.ops.upfirdn_2d import upsample_conv_2d, conv_down_sample_2d
from dnnlib_tf.tflib.ops.fused_bias_act import fused_bias_act
from termcolor import colored

channels_first = ['channels_first', 'CHANNELS_FIRST', 'NCHW']
channels_last = ['channels_last', 'CHANNELS_LAST', 'NHWC']


def generate_noise(n_samples: int, z_dim: int, truncation: float = .7) -> np.ndarray:
    """
    generate truncated noise (i.e., get random values from a distribution that has a cut from '-truncation' value until
    the beginning of lower tail, and a cut from 'truncation' value until the end of the upper tail
    :param n_samples: int, number of random noise vectors to be generated (i.e., batch_size)
    :param z_dim: int, dimensions of the generated noise vector
    :param truncation: float, the cutting value
    :return: Tensor (pt, tf or even a numpy array) of shape (n_samples, z_dim) containing the noise vectors
    """
    noise = truncnorm.rvs(-truncation, +truncation, size=(n_samples, z_dim))
    return np.array(noise)


def instance_norm(x: tf.Tensor, epsilon: float = 1e-8, data_format: str = 'channels_first') -> tf.Tensor:
    """
    applies instance normalisation on a batch of images

                                            x - mean(x) [on width and height axis]
    instance norm of x = ------------------------------------------------------------------------------
                            square root of (standard deviation of (x) [on width and height axis] + epsilon)

    :param x: tf.Tensor, the batch of images to be normalised
    :param epsilon: float, a small parameter for numerical stability
    :param data_format: str, specify what axis holds the number of channels
    :return: tf.Tensor, the instance normalised batch
    """
    normalisation_axis = [2, 3] if data_format in channels_first else [1, 2]
    x -= tf.reduce_mean(x, axis=normalisation_axis, keepdims=True)
    epsilon = tf.constant(epsilon, dtype=x.dtype)
    x *= tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=normalisation_axis, keepdims=True) + epsilon)
    return x


def equalised_weights(input_shape: int or tuple or list,
                      output_shape: int or tuple or list = None,
                      use_w_scale: bool = True,
                      learning_rate_multiplier: float = 1.):
    fan_in = np.prod(input_shape)
    he_std = 1 / np.sqrt(fan_in)
    if output_shape is not None:
        if type(input_shape) == int:
            if type(output_shape) == int:
                shape = [input_shape, output_shape]
            else:
                shape = [input_shape, *output_shape]
        else:
            if type(output_shape) == int:
                shape = [*input_shape[1:], output_shape]  # discard batch size
            else:
                shape = [*input_shape[1:], *output_shape]  # discard batch size
    else:
        shape = input_shape
    if use_w_scale:
        init_std = 1.0 / learning_rate_multiplier
        runtime_coefficient = he_std * learning_rate_multiplier
    else:
        init_std = he_std / learning_rate_multiplier
        runtime_coefficient = learning_rate_multiplier
    return tf.Variable(initial_value=tf.random.normal(mean=0, stddev=init_std, shape=shape),
                       shape=shape,
                       trainable=True), runtime_coefficient


class EqualisedDense(Layer):
    def __init__(self, units: int, learning_rate_multiplier: float = 1., **kwargs):
        super(EqualisedDense, self).__init__(trainable=kwargs.get('trainable', True))
        self.units = units
        self.learning_rate_multiplier = learning_rate_multiplier
        self.weight = None
        self.bias = None
        self.init_std = None
        self.runtime_coeff = None

    def build(self, input_shape):
        self.weight, self.runtime_coeff = equalised_weights(input_shape=input_shape,
                                                            output_shape=self.units,
                                                            use_w_scale=True,
                                                            learning_rate_multiplier=self.learning_rate_multiplier)

        self.bias = tf.Variable(initial_value=tf.zeros(shape=[self.units, ]), trainable=True, shape=[self.units, ])

    def call(self, inputs, *args, **kwargs):
        x = tf.matmul(inputs, self.weight * self.runtime_coeff) + self.bias
        return tf.nn.leaky_relu(x)


class MappingNetwork(Model, ABC):
    """
    the mapping network that will learn to map the input noise vector Z to the intermediate noise vector W
    """

    def __init__(self, hidden_dim: int, w_dim: tuple or int, n_hidden_layers: int = 8) -> None:
        """
        initializer of the network, must have input dimensions, number of units in the hidden layers, and the output
        dimension
        :param hidden_dim: int, number of units in the hidden layers
        :param w_dim: tuple or int, dimensions of the output intermediate noise vector
        """
        super(MappingNetwork, self).__init__()
        self.depth = n_hidden_layers
        self.layer_base_name = 'MappingLayer__'
        # the network (as of the StyleGAN paper) is 7 dense layer with relu activation function followed by linear
        for i in range(self.depth):
            # define number of units in the current layer, number of units is the same for all hidden layers
            # but the output layer should know the number of dimensions in the desired intermediate noise vector w
            units = hidden_dim if i < n_hidden_layers - 1 else w_dim

            # input shape for all layers is the same output shape of all hidden layers but the input for the very first
            self.__setattr__(name=f'{self.layer_base_name}{i}', value=EqualisedDense(units=units))

    def call(self, z: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        override the call function (to make callable instances of the mapping network)
        :param z: tf.Tensor, the input noise vector
        :return:
        """
        # normalize z
        z = z / np.linalg.norm(z, axis=1, keepdims=True)
        # propagate through the network layers
        for i in range(self.depth):
            z = self.__getattribute__(f'{self.layer_base_name}{i}')(z)
        return z


class NoiseInjector(Layer):
    """
    a simple layer to inject another random noise vector (unit 'B' in the StyleGAN paper)
    """

    def __init__(self) -> None:
        """
        only sets the attributes of the layer (i.e., kernel_shape = 1
        """
        super(NoiseInjector, self).__init__()

        # create a trainable variable for noise injection with a shape broadcastable to the image shape (channel-wise)
        self.kernel_shape = 1

        self.kernel = None

    def build(self, input_shape: tuple = None) -> None:
        self.kernel = tf.Variable(
            initial_value=tf.random.normal(shape=[self.kernel_shape, ]),
            shape=[self.kernel_shape, ],
            trainable=True
        )

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        override the call function to create callable objects of the Noise Injector
        :param inputs: tf.Tensor, the batch of images to be injected
        :return:
        """
        super(NoiseInjector, self).call(kwargs)
        noise_shape = list(inputs.shape)
        noise = tf.random.normal(noise_shape)
        return noise + self.kernel * noise


class AdaIN(Layer, ABC):
    """
    a simple model to inject the intermediate noise vector W
    NOTE: this is a property for StyleGAN 1 and had been removed in StyleGAN2 paper
    """

    def __init__(self,
                 channels: int,
                 w_dim: int or tuple,
                 data_format: str = 'channels_first',
                 **kwargs) -> None:
        """
        initializer of the Adaptive Instance Normalisation
        :param channels: int, number of channels
        :param w_dim: int or tuple, shape of the intermediate noise vector
        :param data_format: str, specify what axis holds the number of channels
        """
        super(AdaIN, self).__init__(**kwargs)

        # dense layer to extract the style from (s) from the intermediate noise vector (W)
        self.style_scale_transform = Dense(units=channels)
        self.style_scale_transform.build(input_shape=(w_dim,))

        # dense layer to extract the bias (shift) from the intermediate noise vector (W)
        self.style_shift_transform = Dense(units=channels)
        self.style_shift_transform.build(input_shape=(w_dim,))

        self.data_format = data_format

    def call(self, inputs: [tf.Tensor, tf.Tensor], **kwargs) -> tf.Tensor:
        """
        override the call function to create callable objects of AdaIn Class
        :param inputs: list, two elements [tf.Tensor, tf.Tensor] the image to which we want to apply AdaIN and the
        intermediate noise vector
        :return:
        """
        super(AdaIN, self).call(kwargs)
        images_batch = inputs[0]
        w = inputs[1]
        normalized_image = instance_norm(images_batch, data_format=self.data_format)
        if self.data_format in channels_first:
            style_scale = self.style_scale_transform(w)[:, :, None, None]
            style_shift = self.style_shift_transform(w)[:, :, None, None]
        else:
            style_scale = self.style_scale_transform(w)[:, None, None, :]
            style_shift = self.style_shift_transform(w)[:, None, None, :]
        transformed_image = style_scale * normalized_image + style_shift
        return transformed_image


class ConvUpDown(Layer):
    """
    Convolutional Layer with optional up/down sampling
    """
    def __init__(self,
                 input_filters: int,
                 resolution: int,
                 up: bool = False,
                 down: bool = False,
                 feature_maps_decay: float = 1.,
                 kernel_size: int = 1,
                 resample_kernel=None,
                 data_format: str = 'channels_first',
                 **kwargs) -> None:
        """
        Initializer of the class, takes the parameters needed to define the layer (up/down sampling or just a
        convolution)
        :param input_filters: int, number of filters in the input
        :param resolution: int, the stage (the logarithm to base 2 of the current resolution)
        :param up: bool, perform upsampling, default False
        :param down: bool, perform downsampling, default False
        :param feature_maps_decay: float, decay to control number of output feature maps
        :param kernel_size: int, width and height of the convolutional filter, default 1
        :param resample_kernel: list, 4 integer specifying the resampling kernel to be used by FIR operations
        :param data_format: str, specify where to find the filters (final axis -channels_last-,
                                                                second axis -channels_first-, default channels_first)
        :param kwargs: dict, keyword arguments to call super __init__ function
        """
        super(ConvUpDown, self).__init__(trainable=kwargs.get('trainable', True))
        if resample_kernel is None:
            resample_kernel = [1, 3, 3, 1]
        self.conv_weights = None
        assert not (up and down)
        self.input_filters = input_filters
        self.output_filters = num_filters(resolution, feature_maps_decay=feature_maps_decay)
        self.up = up
        self.down = down
        self.data_format = 'NCHW' if data_format in channels_first else 'NHWC'
        self.weights_shape = [kernel_size, kernel_size, input_filters, self.output_filters]
        self.resample_kernel = resample_kernel
        self.runtime_coeff = None

    def build(self, input_shape) -> None:
        self.conv_weights, self.runtime_coeff = equalised_weights(input_shape=self.weights_shape)

    def call(self, images, **kwargs):
        super(ConvUpDown, self).call(kwargs)
        if self.up:
            images = upsample_conv_2d(images, self.conv_weights, self.resample_kernel, data_format=self.data_format)
        elif self.down:
            images = conv_down_sample_2d(images, self.conv_weights, self.resample_kernel, data_format=self.data_format)
        else:
            images = tf.nn.conv2d(images,
                                  self.conv_weights,
                                  data_format=self.data_format,
                                  strides=[1, 1, 1, 1],
                                  padding='SAME')
        return images


class ModulatedConv2D(Layer):
    """
    Implementation of Modulated Convolution produced in the StyleGAN2 paper (replacement of the Adaptive Instance
    Normalisation)
    """

    def __init__(self,
                 w_dimensions: int or tuple,
                 input_channels: int,
                 output_channels: int,
                 up: bool = True,
                 down: bool = False,
                 kernel_size: int = 3,
                 demodulate: bool = True,
                 resample_kernel=None,
                 padding: str = 'SAME',
                 data_format: str = 'channels_first',
                 **kwargs) -> None:
        """
        initializer of the Modulated Convolution operation
        :param w_dimensions: int or tuple, the dimensions of the intermediate noise vector W
        :param input_channels: int, number of channels in the input image
        :param output_channels: int, number of channels in the output image
        :param up: bool, specify whether to double the size of the image or not
        :param down: bool, specify whether to half the size of the image or not
        :param kernel_size: int, height and width of the kernel
        :param demodulate: bool, specify whether to use demodulation or not (ToRGB Layer)
        :param padding: str, padding value ['SAME', 'VALID', ...etc]
        :param data_format: str, specify the axis which holds the number of channels
        """
        super(ModulatedConv2D, self).__init__(**kwargs)
        if resample_kernel is None:
            resample_kernel = [1, 3, 3, 1]

        # set class instance's attributes
        self.data_format = 'NCHW' if data_format in ['channels_first', 'NCHW'] else 'NHWC'
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv_weights = None
        self.bias = None
        self.style_scale_transform = Dense(units=input_channels, input_shape=(w_dimensions,))
        self.epsilon = 1e-6  # for numerical stability
        self.padding = padding
        self.demodulate = demodulate
        self.resample_kernel = resample_kernel
        assert not (up and down), colored("You can't use up-sampling and down-sampling simultaneously in one layer",
                                          'red')
        self.up = up
        self.down = down
        del resample_kernel
        gc.collect()

    def build(self, input_shape) -> None:
        self.conv_weights = tf.Variable(
            initial_value=tf.random.normal(shape=(self.kernel_size,
                                                  self.kernel_size,
                                                  self.input_channels,
                                                  self.output_channels)),
            trainable=True,
            shape=[self.kernel_size,
                   self.kernel_size,
                   self.input_channels,
                   self.output_channels]
        )
        self.bias = tf.Variable(name='bias',
                                initial_value=tf.random.normal(shape=[self.output_channels]),
                                shape=self.output_channels)

    def call(self, inputs: [tf.Tensor, tf.Tensor], **kwargs) -> tf.Tensor:
        """
        override the call function to create callable objects of ModulatedConv2D class
        W' = s * w (the intermediate noise factor multiplied by the scale factor learned from the dense layer)

                                                        W'
        W'' =  ----------------------------------------------------------------------------------------
                            square root of [(sum of element-wise square of W') + epsilon)]
        NOTE: although this function supports both data formats ['channels_first', 'channels_last'], it's highly
        recommended to use the 'channels_first' data format (less operations)
        :param inputs: list, exactly two elements [batch of images to which we want to perform the Modulated Convolution
        and the intermediate noise vector W]
        :return:
        """
        super(ModulatedConv2D, self).call(kwargs)
        image, w = inputs[0], inputs[1]  # unpack
        style_scale = self.style_scale_transform(w)  # get style scale (s in the paper) from the dense layer

        # in case the channels are stored in the final axis
        if self.data_format in channels_last:
            # transpose the input tensor (instead of rewriting the whole code to account for that data format)
            image = tf.transpose(image, perm=[0, 3, 1, 2])

        # scale the parameters of W by the style scale learned values
        w = self.conv_weights[None] * style_scale[:, None, None, :, None]

        batch_size, input_channels, height, width = image.shape
        image_boarders = (image.shape[2], image.shape[3])

        # scaling factor (square root of the sum of element-wise square of w'
        denominator = tf.math.rsqrt(tf.reduce_sum(w ** 2, axis=[2, 3, 4]))

        if self.demodulate:
            # introduce extra dimensions to be broadcastable to the weights' shape
            denominator = denominator[:, :, None, None, None]
        else:
            # in case there's no demodulation, eliminate the denominator's effect
            denominator = 1  # do not use demodulation (only for the ToRGB Layer as the paper suggests)

        # get W'' (W' / sum(square(W')
        w = w * denominator

        # number of output channels is saved in the last axis of W''
        output_channels = w.shape[-1]

        # fuse input images into a single image with (#channels times batch size) channels
        efficient_image = tf.reshape(image, [1, -1, *image_boarders])

        # transpose W'' so that the batch size axis and output channels are the axis 4 and 5 respectively,
        # and then fuse the filters using reshape (like we fused the input images)
        efficient_filter = tf.reshape(
            # transpose
            tf.transpose(w, perm=[1, 2, 3, 0, 4]),
            # reshape
            shape=[w.shape[1], w.shape[2], w.shape[3], -1]
        )
        if self.up:
            efficient_output = upsample_conv_2d(efficient_image, efficient_filter, k=self.resample_kernel)
        elif self.down:
            efficient_output = conv_down_sample_2d(image, efficient_filter, k=self.resample_kernel)
        else:
            efficient_output = tf.nn.conv2d(efficient_image,
                                            efficient_filter,
                                            strides=[1, 1, 1, 1],
                                            padding=self.padding,
                                            data_format='NCHW')

        # calculate the output shape
        output_shape = (batch_size, output_channels, *efficient_output.shape[2:])

        # reshape output to the calculated shape
        efficient_output = tf.reshape(efficient_output, shape=output_shape)

        # in case the channels reside in the last axis
        if self.data_format in channels_last:
            # transpose the output images back to the expected data format
            efficient_output = tf.transpose(efficient_output, perm=[0, 2, 3, 1])

        del inputs
        gc.collect()
        return efficient_output


def minibatch_stddev_layer(x, group_size=4, num_new_features=1, data_format: str = 'NCHW'):
    group_size = tf.minimum(group_size, tf.shape(x)[0])
    if data_format == 'NHWC':
        x = tf.transpose(x, perm=[0, 3, 1, 2])
    s = x.shape
    y = tf.reshape(tensor=x,
                   shape=[group_size, -1, num_new_features, s[1] // num_new_features, s[2], s[3]])
    y -= tf.reduce_mean(y, axis=0, keepdims=True)  # [GMncHW] Subtract mean over group.
    y = tf.reduce_mean(tf.square(y), axis=0)  # [MncHW]  Calc variance over group.
    y = tf.sqrt(y + 1e-8)  # [MncHW]  Calc stddev over group.
    y = tf.reduce_mean(y, axis=[2, 3, 4], keepdims=True)  # [Mn111]  Take average over feature maps and pixels.
    y = tf.reduce_mean(y, axis=[2])  # [Mn11] Split channels into c channel groups
    y = tf.cast(y, x.dtype)  # [Mn11]  Cast back to original data type.
    # [NnHW]  Replicate over group and pixels.
    y = tf.tile(y, [group_size, 1, s[2], s[3]] if data_format == 'NCHW' else [group_size, s[2], s[3], 1])
    # [NCHW]  Append as new fmap.
    if data_format == 'NHWC':
        y = tf.transpose(y, perm=[0, 3, 1, 2])
    return tf.transpose(tf.concat([x, y], axis=1), perm=[0, 2, 3, 1] if data_format == 'NHWC' else [0, 1, 2, 3])


class ToRGB(Layer):
    """
    A layer that performs a convolution to obtain an image with three colour channels Red, Green, and Blue
    """

    def __init__(self,
                 w_dimensions: int or tuple,
                 input_channels: int,
                 data_format: str = 'channels_first',
                 **kwargs) -> None:
        """
        initializer of the class, it only needs to know with which data format its dealing
        :param data_format: str, can be on of 'channels_first', 'channels_last', 'NCHW', 'NHWC'
        :param kwargs: tuple, extra parameters for the parent class (like name)
        """
        super(ToRGB, self).__init__(**kwargs)
        self.mod_conv_2d = None
        self.rgb_bias = None
        self.w_dimensions = w_dimensions
        self.input_channels = input_channels
        self.data_format = data_format

    def build(self, input_shape: tuple = None, **kwargs) -> None:
        """
        initialize the Modulated Convolution layer with kernel size of 1 and no modulation
        :param input_shape: tuple, not used
        :return: None
        """
        super(ToRGB, self).build(kwargs)
        self.mod_conv_2d = ModulatedConv2D(output_channels=3,
                                           kernel_size=1,
                                           input_channels=self.input_channels,
                                           w_dimensions=self.w_dimensions,
                                           demodulate=False,
                                           up=False,
                                           data_format=self.data_format,
                                           name='ToRGB')
        self.rgb_bias = self.add_weight(name='ToRGB/bias',
                                        shape=(3,),
                                        initializer=tf.random_normal_initializer(0, 1),
                                        trainable=True)

    def call(self, inputs: [tf.Tensor, tf.Tensor], **kwargs) -> tf.Tensor:
        """
        override the call function to create a callable instance of the ToRGB layer
        :param inputs: list of exactly two elements, images and weights
        :param kwargs: tuple, keyword arguments for parent class call function
        :return:
        """
        super(ToRGB, self).call(kwargs)
        y = self.mod_conv_2d(inputs)  # perform convolution
        del inputs
        gc.collect()
        return y


def fused_bias_activation(x: tf.Tensor,
                          b: tf.Tensor or tf.Variable,
                          activation_function: str = 'lrelu',
                          alpha: float = 0.2,
                          gain: float = None,
                          data_format: str = 'channels_first') -> tf.Tensor:
    """
    apply bias addition and Activation
    :param x: tf.Tensor, convolutional layer output to be activated
    :param b: tf.Tensor, bias weights for the same convolutional layer
    :param activation_function: str, activation function to use could be one of ['relu', 'lrelu', 'linear', ...],
    default 'lrelu'.
    :param alpha: float, negative slope of the leaky relu function, default is 0.2
    :param gain: float, Scaling factor for the output tensor, or `None` to use default
                See `activation_funcs` for the default scaling of each activation function
                If unsure, consider specifying `1.0`
    :param data_format: str, can be on of 'channels_first', 'channels_last', 'NCHW', 'NHWC'
    :return: tf.Tensor, the activated outputs
    """
    return fused_bias_act(x,
                          b,
                          axis=1 if data_format in channels_first else 3,
                          act=activation_function,
                          alpha=alpha,
                          gain=gain)


def num_filters(stage: int,
                feature_maps_base: int = 16 << 10,
                min_feature_maps: int = 1,
                max_feature_maps: int = 512,
                feature_maps_decay: float = 1.0) -> int:
    """
    decide what is the number of filters in a given layer
    :param stage: int, the log2 of the resolution
    :param feature_maps_base: int, the base number of layers (numerator)
    :param min_feature_maps:  int, the minimum number of feature maps in any layer
    :param max_feature_maps: int, the maximum number of feature maps in any layer
    :param feature_maps_decay: float, extra parameter to decrease the number of feature maps even more
    :return: int, number of feature maps
    """
    result = np.clip(int(feature_maps_base / (2 ** (stage * feature_maps_decay))), min_feature_maps, max_feature_maps)
    return int(result)


class FromRGB(Layer):
    """
    Converts the input (which is the Generator's final output) to an input for the Discriminator (from 3 color channels
    to an arbitrary [number of filters in a specific stage] number of channels).
    """

    def __init__(self,
                 resolution: int,
                 feature_maps_decay: float = 1.0,
                 data_format: str = 'channels_first',
                 **kwargs) -> None:
        """
        initializer of the class, needs to know what was the final stage of the Generator (logarithm of the resolution
        to the base 2)
        :param resolution: int, the stage of generation (only the final resolution stage for StyleGAN2)
        :param feature_maps_decay: float, a parameter to manipulate the number of channels in the block
        :param data_format: str, I'm tired of explaining this one
        :param kwargs: dict, arguments to the super class initializer
        """
        # call the super class with the provided keyword arguments
        super(FromRGB, self).__init__(kwargs.get('trainable', True))

        # set class instances' variables
        self.resolution = resolution
        self.data_format = 'NCHW' if data_format in channels_first else 'NHWC'
        self.feature_maps = num_filters(resolution, feature_maps_decay=feature_maps_decay)

        # initialize a convolutional layer (no upsampling nor downsampling, and a default kernel size of 1)
        self.conv_layer = ConvUpDown(resolution=self.resolution,
                                     input_filters=3,  # since we know it's an RGB input
                                     data_format=self.data_format)

        self.bias_weights = None  # for fused bias activation

    def build(self, input_shape: tuple) -> None:
        """
        overriding the build method to make it possible to initialize the bias
        :param input_shape: tuple, the shape of the input images (without the batch size axis)
        :return: None
        """
        # call the super class build function
        super(FromRGB, self).build(input_shape)

        # initialize the bias
        self.bias_weights = tf.Variable(initial_value=tf.random.normal(shape=[self.feature_maps, ]),
                                        trainable=True,
                                        shape=self.feature_maps)

    def call(self, inputs: tf.Tensor, **kwargs: dict) -> tf.Tensor:
        """
        overriding the call function to create a callable instance of the class
        :param inputs: tf.Tensor, RGB image generated by synthesis network (Generator)
        :param kwargs: dict, keyword argument for the super class' call function
        :return: tf.Tensor, convolved image
        """
        # super class call function call
        super(FromRGB, self).call(kwargs)

        # convolve input tensor
        x = self.conv_layer(inputs)

        # activate the output of the convolution
        x = fused_bias_activation(x, self.bias_weights, data_format=self.data_format)

        return x


if __name__ == '__main__':
    from time import sleep


    def test_noise():
        c_print('Testing the implementation of the noise generation functionality', 'red')
        sample_noise = generate_noise(n_samples=1000, z_dim=10, truncation=0.2)
        max_noise = np.max(sample_noise)
        min_noise = np.min(sample_noise)
        std_noise = np.std(sample_noise)
        c_print(f'Shape of the generated noise array: {sample_noise.shape}', 'blue')
        c_print(f'Maximum value in the generated noise vectors: {max_noise}', 'blue')
        c_print(f'Minimum value in the generated noise vectors: {min_noise}', 'blue')
        c_print(f'St. dev value of the generated noise vectors: {std_noise}', 'blue')
        c_print(f'Done!\n')


    def test_mapping():
        c_print('Testing the implementation of the mapping network', 'red')
        params = {
            'w_dim': 512,
            'hidden_dim': 256,
            'n_hidden_layers': 8
        }
        map_obj = MappingNetwork(**params)
        noise = generate_noise(64, 128, 1)
        c_print(f'Shape of the noise vectors: {noise.shape}', 'blue')
        output = map_obj(noise)
        min_output = np.min(output)
        max_output = np.max(output)
        std_output = np.std(output)
        c_print(f'Minimum value in the resulted array of the forward propagation: {min_output}', 'blue')
        c_print(f'Maximum value in the resulted array of the forward propagation: {max_output}', 'blue')
        c_print(f'St. dev value in the resulted array of the forward propagation: {std_output}', 'blue')
        c_print(f'Shape of the resulted array of the forward propagation: {tuple(output.shape)}', 'blue')
        c_print(f'Done!\n')


    def test_injection(data_format: str = 'channels_first'):
        c_print(f'Testing the implementation of the Noise Injection Layer ({data_format})', 'red')
        test_noise_channels = 3000
        test_noise_samples = 20
        fake_images = tf.random.normal(shape=(test_noise_samples, test_noise_channels, 10, 10))
        if data_format == 'channels_last':
            fake_images = tf.transpose(fake_images, perm=[0, 2, 3, 1])

        noise_injector = NoiseInjector()
        noise_injector.build()
        abs_std = np.abs(np.std(noise_injector.kernel))

        c_print(f'St. Dev of the Noise Injector Weights: {abs_std}', 'blue')
        abs_mean = np.abs(np.mean(noise_injector.kernel))

        c_print(f'Mean of the Noise Injector Weights: {abs_mean}', 'blue')

        noise_injector_shape = tuple(noise_injector.kernel.shape)
        c_print(f'Shape of the Noise Injector Weights: {noise_injector_shape}', 'blue')
        noise_injector.kernel = tf.ones_like(noise_injector.kernel)
        difference = noise_injector(fake_images) - fake_images

        mean_std_ax_0 = np.mean(np.abs(np.std(difference, axis=0)))

        c_print(f'Mean of the Absolute Value of Std of the first axis: {mean_std_ax_0}', 'blue')

        mean_std_ax_1 = np.mean(np.abs(np.std(difference, axis=1)))
        c_print(f'Mean of the Absolute Value of Std Standard Deviation of the second axis: {mean_std_ax_1}', 'blue')

        mean_std_ax_2 = np.mean(np.abs(np.std(difference, axis=2)))
        c_print(f'Mean of the Absolute Value of Std of the third axis: {mean_std_ax_2}', 'blue')

        mean_std_ax_3 = np.mean(np.abs(np.std(difference, axis=3)))
        c_print(f'Mean of the Absolute Value of the Std of the fourth axis: {mean_std_ax_3}', 'blue')

        per_channel_change = np.std(np.mean(noise_injector(fake_images) - fake_images, axis=1))
        c_print(f'St. Dev of the Mean of the second axis (change per channel): {per_channel_change}', 'blue')

        noise_injector.kernel = tf.zeros_like(noise_injector.kernel)
        difference = noise_injector(fake_images) - fake_images
        difference = np.mean(np.abs(difference))
        c_print(f'fake images - fake images after the noise injection (with zero kernel): {difference}', 'blue')

        c_print('Done!\n')


    def test_adaptive_in(data_format: str = 'channels_first'):
        c_print(f'Testing Adaptive Instance Normalisation ({data_format})...', 'red')

        w_channels = 50
        image_channels = 20
        image_size = 30
        n_test = 10
        ada_in = AdaIN(image_channels, w_channels, data_format=data_format)
        test_w = tf.random.normal(shape=(n_test, w_channels))

        style_scale_shape = tuple(ada_in.style_scale_transform(test_w).shape)
        style_shift_shape = tuple(ada_in.style_shift_transform(test_w).shape)

        c_print(f'Shape of the Style Scale Transform output: {style_scale_shape}', 'blue')
        c_print(f'Shape of the Style Shift Transform output: {style_shift_shape}', 'blue')
        test_images = tf.random.normal(shape=(n_test, image_channels, image_size, image_size))
        if data_format == 'channels_last':
            test_images = tf.transpose(test_images, [0, 2, 3, 1])

        forward_pass = ada_in([test_images, test_w])
        c_print(f'Shape of the output of the forward pass {data_format}: {forward_pass.shape}',
                'blue')
        w_channels = 3
        image_channels = 2
        image_size = 3
        n_test = 1
        ada_in = AdaIN(image_channels, w_channels, data_format=data_format)
        ada_in.style_scale_transform.set_weights([np.ones((3, 2)) / 4, np.zeros(2)])
        ada_in.style_shift_transform.set_weights([np.ones((3, 2)) / 5, np.zeros(2)])

        test_input = np.ones(shape=(n_test, image_channels, image_size, image_size))
        if data_format == 'channels_last':
            test_input = np.transpose(test_input, (0, 2, 3, 1))
            test_input[:, 0] = 0
        else:
            test_input[:, :, 0] = 0
        test_input = tf.convert_to_tensor(test_input)
        test_w = tf.ones(shape=(n_test, w_channels))

        test_output = ada_in([test_input, test_w])
        result_1 = np.abs(test_output[0, 0, 0, 0] - 3 / 5 + tf.sqrt(tf.constant(9 / 8)))
        result_2 = np.abs(test_output[0, 0, 1, 0] - 3 / 5 - tf.sqrt(tf.constant(9 / 32)))
        c_print(f'Result of the first test  ({data_format}) = {result_1}', 'blue')
        c_print(f'Result of the second test ({data_format}) = {result_2}', 'blue')

        c_print('Done!\n')


    def test_modulated_conv(data_format: str = 'channels_first'):
        c_print(f'Testing Modulated Convolutional Layer ({data_format})...', 'red')
        example_modulated_conv = ModulatedConv2D(w_dimensions=128, input_channels=3, output_channels=3, kernel_size=3,
                                                 data_format=data_format, up=False)
        num_ex = 2
        image_size = 64
        rand_image = tf.random.normal(shape=(num_ex, 3, image_size, image_size))  # A 64x64 image with 3 channels
        if data_format == 'channels_last':
            rand_image = tf.transpose(rand_image, [0, 2, 3, 1])
        rand_w = tf.random.normal(shape=(num_ex, 128))
        new_image = example_modulated_conv([rand_image, rand_w])
        c_print(f'Shape of the new Modulated-Convoluted Image: {new_image.shape}', 'blue')

        second_modulated_conv = ModulatedConv2D(w_dimensions=128, input_channels=3, output_channels=3,
                                                kernel_size=3, data_format=data_format, up=False)
        second_image = second_modulated_conv([new_image, rand_w])
        c_print(f'Shape of the second Modulated-Convoluted Image: {second_image.shape}', 'blue')

        rand_image += 1
        rand_image /= 2
        rand_image = tf.clip_by_value(rand_image, 0, 1)
        plot_images(rand_image, labels=['Random Image #1', 'Random Image #2'], data_format=data_format)

        new_image = (new_image + 1) / 2
        new_image = tf.clip_by_value(new_image, 0, 1)
        plot_images(new_image,
                    labels=['Image #1 after Modulated Convolution #1', 'Image #2 after Modulated Convolution #1'],
                    data_format=data_format)
        second_image += 1
        second_image /= 2
        second_image = tf.clip_by_value(second_image, 0, 1)
        plot_images(second_image,
                    ['Image #1 after Modulated Convolution #2', 'Image #2 after Modulated Convolution #2'],
                    data_format=data_format)

        c_print('Done!\n')


    def test_to_rgb_layer(data_format: str = 'channels_first'):
        c_print(f'Testing ToRGB Layer ({data_format})...', 'red')
        test_image = tf.random.normal(shape=(16, 32, 64, 64))
        if data_format == 'channels_last':
            test_image = tf.transpose(test_image, perm=[0, 2, 3, 1])
        c_print(f'Shape of the Input to the ToRGB Layer: {test_image.shape}', 'blue')
        test_w = tf.random.normal(shape=(16, 128))
        to_rgb_instance = ToRGB(data_format=data_format, w_dimensions=128, input_channels=32)
        result = to_rgb_instance([test_image, test_w])
        if data_format == 'channels_first':
            result = tf.transpose(result, perm=[0, 2, 3, 1])
        c_print(f'Shape of the output of the ToRGB Layer: {result.shape}', 'blue')
        result = (result + 1) / 2
        result = tf.clip_by_value(result, 0, 1)
        plot_images(result, labels=[f'To RGB Image Output #{i}' for i in range(16)])
        c_print('Done!\n')


    def test_from_rgb_layer(data_format: str = 'channels_first'):
        c_print(f'Testing FromRGB Layer ({data_format})...', 'red')
        param = {
            'batch_size': 8,
            'resolution': 8
        }

        input_shape = [param['batch_size'],
                       3,
                       2 ** param['resolution'], 2 ** param['resolution']]
        if data_format == 'channels_last':
            input_shape = [input_shape[0],
                           *input_shape[2:],
                           3]

        random_generator_output = tf.random.normal(shape=input_shape)
        c_print(f'Shape of the Input: {random_generator_output.shape}', 'blue')
        from_rgb_layer = FromRGB(resolution=param['resolution'],
                                 data_format=data_format)

        rgb_images = from_rgb_layer(random_generator_output)
        c_print(f'Shape of the Output: {rgb_images.shape}', 'blue')

        c_print('Done!\n')

    test_noise()
    sleep(1)
    test_mapping()
    sleep(1)
    test_injection()
    sleep(1)
    test_injection(data_format='channels_last')
    sleep(1)
    test_adaptive_in()
    sleep(1)
    test_adaptive_in(data_format='channels_last')
    sleep(1)
    test_modulated_conv()
    sleep(1)
    test_modulated_conv(data_format='channels_last')
    sleep(1)
    test_to_rgb_layer()
    sleep(1)
    test_to_rgb_layer(data_format='channels_last')
    sleep(1)
    test_from_rgb_layer()
    sleep(1)
    test_from_rgb_layer(data_format='channels_last')
