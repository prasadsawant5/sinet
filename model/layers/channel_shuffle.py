import tensorflow as tf
from tensorflow.keras.layers import Layer

class ChannelShuffle(Layer):
    def __init__(self, groups=2, name: str = None, trainable: bool = True, **kwargs):
        super(ChannelShuffle, self).__init__(name=name, trainable=trainable, **kwargs)
        self.groups = groups

    def build(self, input_shape):
        if isinstance(input_shape[-1], int):
            self.height = input_shape[1]
            self.width = input_shape[2]
            self.channels = input_shape[3]
        else:
            self.height = input_shape[1].value
            self.width = input_shape[2].value
            self.channels = input_shape[3].value
        assert (self.channels % self.groups == 0)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        channels_per_group = self.channels // self.groups
        out = tf.reshape(inputs, shape=(-1, self.height, self.width, self.groups, channels_per_group))
        out = tf.transpose(out, perm=(0, 1, 2, 4, 3))
        out = tf.reshape(out, shape=(-1, self.height, self.width, self.channels))
        return out