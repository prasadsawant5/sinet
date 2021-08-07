import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Layer, PReLU
from model.layers.conv2d import Conv2D
from model.layers.normalization import Normalization
from model.layers.channel_shuffle import ChannelShuffle
from model.layers.upsampling_bilinear_2d import UpsamplingBilinear2D

class S2Block(Layer):
    def __init__(self, out_channels, config, name: str = None, trainable: bool = True, **kwargs):
        super(S2Block, self).__init__(name=name, trainable=trainable, **kwargs)
        kernel_size = config[0]
        pool_size = config[1]
        self.kernel_size = kernel_size
        self.resolution_down = False
        if pool_size > 1:
            self.resolution_down = True
            self.down_res = AveragePooling2D(pool_size, pool_size)
            self.up_res = UpsamplingBilinear2D(scale_factor=pool_size)
            self.pool_size = pool_size
        self.conv1x1 = Conv2D(out_channels, kernel_size=1, stride=1, bias=False)
        self.norm = Normalization()

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
        self.conv = Conv2D(self.channels, kernel_size=self.kernel_size, stride=1,
                           padding=padding, groups=self.channels, bias=False,
                           activation=Normalization(activation=PReLU(shared_axes=(1, 2))))
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        if self.resolution_down:
            inputs = self.down_res(inputs)
        output = self.conv(inputs)
        output = self.conv1x1(output)
        if self.resolution_down:
            output = self.up_res(output)
        return self.norm(output)