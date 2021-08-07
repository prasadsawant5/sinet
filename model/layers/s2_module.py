import tensorflow as tf
from tensorflow.keras.layers import Layer, PReLU
from model.layers.channel_shuffle import ChannelShuffle
from model.layers.conv2d import Conv2D
from model.layers.s2_block import S2Block
from model.layers.normalization import Normalization

class S2Module(Layer):
    def __init__(self, out_channels, add=True, config=None, name: str = None, trainable: bool = True, **kwargs):
        super(S2Module, self).__init__(name=name, trainable=trainable, **kwargs)
        if config is None:
            config = [[3, 1], [5, 1]]
        group_n = len(config)
        split_n = int(out_channels / group_n)
        split_patch = out_channels - group_n * split_n
        self.conv_split = Conv2D(split_n, kernel_size=1, stride=1, padding=0, bias=False, groups=group_n)
        self.s2_d1 = S2Block(split_n + split_patch, config[0])
        self.s2_d2 = S2Block(split_n, config[0])
        self.norm = Normalization(activation=PReLU(shared_axes=(1, 2)))
        self.add = add
        self.group_n = group_n
        self.channel_shuffle = ChannelShuffle(groups=group_n)

    def call(self, inputs, **kwargs):
        output = self.channel_shuffle(self.conv_split(inputs))
        
        combine = tf.concat([self.s2_d2(output), self.s2_d1(output)], -1)
        if self.add:
            combine = inputs + combine
        output = self.norm(combine)
        return output