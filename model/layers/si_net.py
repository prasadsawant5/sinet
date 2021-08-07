import tensorflow as tf
from tensorflow.keras.layers import Layer, PReLU
from model.layers.squeeze_separable_conv_2d import SqueezeSeparableConv2D
from model.layers.s2_module import S2Module
from model.layers.conv2d import Conv2D
from model.layers.normalization import Normalization
from model.layers.upsampling_bilinear_2d import UpsamplingBilinear2D

class SINet(Layer):
    def __init__(self, num_classes=20, p=2, q=8, chnn=1.0, name: str = None, trainable: bool = True, **kwargs):
        super(SINet, self).__init__(name=name, trainable=trainable, **kwargs)
        """
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier 
        """
        self.num_classes = num_classes
        self.p = p
        self.q = q
        self.chnn = chnn
        config = [[[3, 1], [5, 1]], [[3, 1], [3, 1]],
                  [[3, 1], [5, 1]], [[3, 1], [3, 1]], [[5, 1], [3, 2]], [[5, 2], [3, 4]],
                  [[3, 1], [3, 1]], [[5, 1], [5, 1]], [[3, 2], [3, 4]], [[3, 1], [5, 2]]]
        out_channels = [16, int(48 + 4 * (chnn - 1)), int(96 + 4 * (chnn - 1))]
        self.conv_down1 = Conv2D(12, kernel_size=3, stride=2, padding=1, bias=False, name='conv0',
                                 activation=Normalization(activation=PReLU(shared_axes=(1, 2))))
        self.conv_down2 = SqueezeSeparableConv2D(out_channels[0], kernel_size=3, stride=2, divide=1, name='squeeze_separable_conv0')
        self.encoder_level2 = []
        for i in range(0, p):
            if i == 0:
                self.encoder_level2.append(
                    S2Module(out_channels[1], config=config[i], add=False, name='encoded_level2_s2_module{}'.format(i)))
            else:
                self.encoder_level2.append(S2Module(out_channels[1], config=config[i], name='encoded_level2_s2_module{}'.format(i)))
        self.encoder_norm2 = Normalization(activation=PReLU(shared_axes=(1, 2)), name='normalization0')
        self.conv_down3 = SqueezeSeparableConv2D(out_channels[1], kernel_size=3, stride=2, divide=2, name='squeeze_separable_conv1')
        self.encoder_level3 = []
        for i in range(0, q):
            if i == 0:
                self.encoder_level3.append(
                    S2Module(out_channels[2], config=config[2 + i], add=False, name='encoded_level3_s2_module0'))
            else:
                self.encoder_level3.append(S2Module(out_channels[2], config=config[2 + i], name='encoded_level3_s2_module{}'.format(i)))
        self.encoder_norm3 = Normalization(activation=PReLU(shared_axes=(1, 2)), name='normalization1')
        self.level3_classifier = Conv2D(num_classes, kernel_size=1, stride=1, padding=0, bias=False, name='conv1')
        self.up1 = UpsamplingBilinear2D(scale_factor=2, activation=Normalization(), name='upsampling_bilinear0')
        self.up2 = UpsamplingBilinear2D(scale_factor=2, activation=Normalization(), name='upsampling_bilinear1')
        self.level2_classifier = Conv2D(num_classes, kernel_size=1, stride=1,
                                        padding=0, bias=False, 
                                        activation=Normalization(
                                            activation=PReLU(shared_axes=(1, 2))), name='conv2')
        self.up3 = UpsamplingBilinear2D(scale_factor=2, name='upsampling_bilinear2')
        self.classifier = Conv2D(num_classes, 3, 1, 1, bias=False,
                                 activation=tf.nn.sigmoid if num_classes == 1 else tf.nn.softmax, name='output')

    def get_config(self):
        config = {
            "num_classes": self.num_classes,
            "p": self.p,
            "q": self.q,
            "chnn": self.chnn,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def call(self, inputs, **kwargs):
        out_down1 = self.conv_down1(inputs)  # 8h 8w
        out_down2 = self.conv_down2(out_down1)  # 4h 4w
        out_level2 = None
        for i, layer in enumerate(self.encoder_level2):
            if i == 0:
                out_level2 = layer(out_down2)
            else:
                out_level2 = layer(out_level2)  # 2h 2w
        out_down3 = self.conv_down3(self.encoder_norm2(tf.concat([out_down2, out_level2], -1)))  # h w
        out_level3 = None
        for i, layer in enumerate(self.encoder_level3):
            if i == 0:
                out_level3 = layer(out_down3)
            else:
                out_level3 = layer(out_level3)
        output3_cat = self.encoder_norm3(tf.concat([out_down3, out_level3], -1))
        enc_final = self.level3_classifier(output3_cat)  # 1/8
        dnc_stage1 = self.up1(enc_final)  # 1/4
        stage1_confidence = tf.reduce_max(
            tf.nn.softmax(dnc_stage1) if self.num_classes != 1 else tf.nn.sigmoid(dnc_stage1), axis=-1, keepdims=True)
        dnc_stage2_0 = self.level2_classifier(out_level2)  # 2h 2w
        dnc_stage2 = self.up2(dnc_stage2_0 * (1. - stage1_confidence) + dnc_stage1)  # 4h 4w
        dnc_stage2 = self.up3(dnc_stage2)
        classifier = self.classifier(dnc_stage2)
        return classifier