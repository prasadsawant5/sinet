import tensorflow as tf
from tensorflow.keras.layers import Layer, Lambda

class UpsamplingBilinear2D(Layer):
    def __init__(self, name = None, scale_factor: int = 2, activation=None, trainable: bool = True, **kwargs):
        super(UpsamplingBilinear2D, self).__init__(trainable=trainable, name=name, **kwargs)
        self.scale_factor = scale_factor
        self.activation = activation if activation is not None else Lambda(lambda x: x)

    
    def build(self, input_shape):
        if isinstance(input_shape[-1], int):
            self.height = input_shape[1]
            self.width = input_shape[2]
        else:
            self.height = input_shape[1].value
            self.width = input_shape[2].value

        super().build(input_shape)


    def call(self, inputs, *args, **kwargs):
        return self.activation(tf.image.resize(inputs,
                                               (int(self.height * self.scale_factor),
                                                int(self.width * self.scale_factor))))