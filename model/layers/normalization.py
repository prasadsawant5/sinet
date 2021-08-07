import tensorflow as tf
from tensorflow.keras.layers import Layer, Lambda

class Normalization(Layer):
    def __init__(self, momentum: float = 0.1, epsilon: float = 1e-3, activation=None, is_sync: bool = True, name: str = None,
                 trainable: bool = True, **kwargs):
        super(Normalization, self).__init__(name=name, trainable=trainable, **kwargs)
        self.norm = tf.keras.layers.experimental.SyncBatchNormalization(momentum=momentum,
                                                                        epsilon=epsilon) if is_sync else tf.keras.layers.BatchNormalization(
            momentum=momentum, epsilon=epsilon)
        self.activation = activation if activation is not None else tf.keras.layers.Lambda(lambda x: x)


    def build(self, input_shape):
        super().build(input_shape)
        

    def call(self, inputs, **kwargs):
        return self.activation(self.norm(inputs))