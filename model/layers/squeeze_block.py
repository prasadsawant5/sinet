import tensorflow as tf
from tensorflow.keras.layers import Layer, PReLU, Dense

class SqueezeBlock(Layer):
    def __init__(self, exp_size, divide=4.0, name: str = None, trainable: bool = True, **kwargs):
        super(SqueezeBlock, self).__init__(name=name, trainable=trainable, **kwargs)
        if divide > 1:
            self.dense = tf.keras.Sequential([
                Dense(int(exp_size / divide)),
                PReLU(shared_axes=(1,)),
                Dense(exp_size),
                PReLU(shared_axes=(1,))]
            )
        else:
            self.dense = tf.keras.Sequential([
                Dense(exp_size),
                PReLU(shared_axes=(1,))]
            )

    def call(self, inputs, **kwargs):
        out = tf.reduce_mean(inputs, axis=(1, 2), keepdims=True)
        out = self.dense(out)
        return out * inputs