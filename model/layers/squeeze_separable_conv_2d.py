import tensorflow as tf
from tensorflow.keras.layers import Layer, PReLU
from model.layers.conv2d import Conv2D
from model.layers.squeeze_block import SqueezeBlock
from model.layers.normalization import Normalization

class SqueezeSeparableConv2D(Layer):
    def __init__(self, out_channels, kernel_size, stride=1, divide=2.0, name: str = None, trainable: bool = True,
                 **kwargs):
        super(SqueezeSeparableConv2D, self).__init__(name=name, trainable=trainable, **kwargs)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = stride
        self.divide = divide

    def build(self, input_shape):
        if isinstance(input_shape[-1], int):
            self.height = input_shape[1]
            self.width = input_shape[2]
            self.channels = input_shape[3]
        else:
            self.height = input_shape[1].value
            self.width = input_shape[2].value
            self.channels = input_shape[3].value
        padding = int((self.kernel_size - 1) / 2)
        self.conv = tf.keras.Sequential([
            Conv2D(self.channels, self.kernel_size, stride=self.strides,
                   padding=padding, name='squeeze_separable_conv0',
                   groups=self.channels, bias=False),
            SqueezeBlock(self.channels, divide=self.divide),
            Conv2D(self.out_channels, kernel_size=1, stride=1, bias=False, name='squeeze_separable_conv1',
                   activation=Normalization(activation=PReLU(shared_axes=(1, 2))))]
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        output = self.conv(inputs)
        return output