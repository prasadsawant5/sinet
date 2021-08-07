import tensorflow as tf
from tensorflow.keras.layers import Layer, PReLU
from model.layers.conv2d import Conv2D
from model.layers.s2_module import S2Module
from model.layers.normalization import Normalization
from model.layers.squeeze_separable_conv_2d import SqueezeSeparableConv2D

class Encoder(Layer):
    def __init__(self, classes=20, p=5, q=3, chnn=1.0, name: str = None, trainable: bool = True, **kwargs):
        super(Encoder, self).__init__(name=name, trainable=trainable, **kwargs)
        """
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        """
        config = [[[3, 1], [5, 1]], [[3, 1], [3, 1]],
                  [[3, 1], [5, 1]], [[3, 1], [3, 1]], [[5, 1], [3, 2]], [[5, 2], [3, 4]],
                  [[3, 1], [3, 1]], [[5, 1], [5, 1]], [[3, 2], [3, 4]], [[3, 1], [5, 2]]]

        print("SINet Enc bracnch num :  " + str(len(config[0])))
        print("SINet Enc chnn num:  " + str(chnn))

        with tf.name_scope('encoder'):
            out_channels = [16, 48 + 4 * (chnn - 1), 96 + 4 * (chnn - 1)]
            self.level1 = Conv2D(12, kernel_size=3, stride=2,
                                padding=1, bias=False, name='conv0',
                                activation=Normalization(activation=PReLU(shared_axes=(1, 2))))
            self.level2_0 = SqueezeSeparableConv2D(out_channels[0], 3, 2, divide=1, name='squeeze_separable_conv0')
            self.level2 = []
            for i in range(0, p):
                if i == 0:
                    self.level2.append(S2Module(out_channels[1], config=config[i], add=False, name='s2_module0'))
                else:
                    self.level2.append(S2Module(out_channels[1], config=config[i], name='s2_module{}'.format(i)))
            self.BR2 = Normalization(activation=PReLU(shared_axes=(1, 2)), name='normalization0')
            self.level3_0 = SqueezeSeparableConv2D(out_channels[1], 3, 2, divide=2, name='squeeze_separable_conv1')
            self.level3 = []
            for i in range(0, q):
                if i == 0:
                    self.level3.append(S2Module(out_channels[2], config=config[2 + i], add=False, name='s2_module2'))
                else:
                    self.level3.append(S2Module(out_channels[2], config=config[2 + i], name='s2_module{}'.format(q)))
            self.BR3 = Normalization(activation=PReLU(shared_axes=(1, 2)))
            self.classifier = Conv2D(classes, kernel_size=1, stride=1, padding=0, bias=False)

    def call(self, inputs, **kwargs):
        output1 = self.level1(inputs)  # 8h 8w
        output2_0 = self.level2_0(output1)  # 4h 4w
        output2 = None
        for i, layer in enumerate(self.level2):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)  # 2h 2w
        output3_0 = self.level3_0(self.BR2(tf.concat([output2_0, output2], - 1)))  # h w
        output3 = None
        for i, layer in enumerate(self.level3):
            if i == 0:
                output3 = layer(output3_0)
            else:
                output3 = layer(output3)
        output3_cat = self.BR3(tf.concat([output3_0, output3], -1))
        classifier = self.classifier(output3_cat)
        return classifier