import tensorflow as tf
from tensorflow.keras.losses import Loss

class DiceLoss(Loss):
    def __init__(self, name='dice_loss'):
        super(DiceLoss, self).__init__(name=name)


    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.math.sigmoid(y_pred)
        numerator = 2 * tf.math.reduce_sum(y_true * y_pred)
        denominator = tf.math.reduce_sum(y_true + y_pred)
        
        return 1 - numerator / denominator