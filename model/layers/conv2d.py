import tensorflow as tf
from tensorflow.keras.layers import Layer, Lambda, Conv2D, DepthwiseConv2D

class Conv2D(Layer):
    def __init__(self, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1,
                 bias: bool = True, padding_mode: str = 'CONSTANT', apply_padding: bool = False,
                 activation=None, name: str = None, trainable: bool = True, **kwargs):
        super(Conv2D, self).__init__(name=name, trainable=trainable, **kwargs)

        self.padding_type = 'valid'
        if not apply_padding and padding != 0:
            self.padding_type = 'same'
        self.grouped_conv = []
        self.padding_mode = padding_mode
        self.apply_padding = apply_padding
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        self.bias = bias
        self.is_groups = False

        self.pad = Lambda(lambda x: tf.pad(x, tf.constant([[0, 0], [padding, padding], [padding, padding], [0, 0]]),
                             padding_mode)) if (padding != 0 and apply_padding is True) else Lambda(
            lambda x: x)
        self.activation = activation if activation is not None else Lambda(lambda x: x)


    def _split_channels(self, total_filters, num_groups):
        split = [total_filters // num_groups for _ in range(num_groups)]
        split[0] += total_filters - sum(split)
        return split


    def build(self, input_shape):
        if isinstance(input_shape[-1], int):
            self.height = input_shape[1]
            self.width = input_shape[2]
            self.channels = input_shape[3]
        else:
            self.height = input_shape[1].value
            self.width = input_shape[2].value
            self.channels = input_shape[3].value
        if (self.groups == self.channels and self.channels == self.out_channels) or self.out_channels is None:
            self.conv = tf.keras.layers.DepthwiseConv2D(kernel_size=self.kernel_size, strides=self.stride,
                                                        dilation_rate=self.dilation, use_bias=self.bias,
                                                        kernel_regularizer=tf.keras.regularizers.l2(
                                                            4e-4),
                                                        padding=self.padding_type)
        elif self.groups == 1:
            self.conv = tf.keras.layers.Conv2D(filters=self.out_channels, kernel_size=self.kernel_size,
                                               kernel_regularizer=tf.keras.regularizers.l2(
                                                   4e-4),
                                               strides=self.stride, dilation_rate=self.dilation, use_bias=self.bias,
                                               padding=self.padding_type)
        else:
            self.is_groups = True
            splits = self._split_channels(self.out_channels, self.groups)
            for i in range(self.groups):
                self.grouped_conv.append(
                    Conv2D(splits[i], kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                           dilation=self.dilation, groups=1, bias=self.bias, padding_mode=self.padding_mode,
                           activation=None,
                           apply_padding=self.apply_padding, name="grouped_{}".format(i))
                )
        super().build(input_shape)


    def call(self, inputs, **kwargs):
        if self.is_groups:
            if len(self.grouped_conv) == 1:
                out = self.grouped_conv[0](inputs)
            else:
                splits = self._split_channels(self.channels, len(self.grouped_conv))
                out = tf.concat([c(x) for x, c in zip(tf.split(inputs, splits, -1), self.grouped_conv)], -1)
        else:
            out = self.conv(self.pad(inputs))
        return self.activation(out)