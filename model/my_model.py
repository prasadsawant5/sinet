import tensorflow as tf
from tensorflow.keras import Model
from model.layers.si_net import SINet

class MyModel:
    def build(self, x: tf.keras.layers.Input, num_classes: int = 3) -> tf.keras.Model:
        outputs = SINet(num_classes=num_classes, p=2, q=8)(x)
        model = Model(inputs=x, outputs=outputs, name='SINet')

        return model
