import tensorflow as tf
from tensorflow.keras import layers

class ReflectionPad2d(tf.keras.layers.Layer):
    def __init__(self, padding):
        super(ReflectionPad2d, self).__init__()
        self.padding = tf.constant([[0, 0], [padding, padding], [padding, padding], [0, 0]])

    def call(self, x):
        return tf.pad(x, self.padding, mode='REFLECT')

class ResnetBlock(tf.keras.Model):
    def __init__(self, in_channels):
        super(ResnetBlock, self).__init__()
        self.conv_block = tf.keras.Sequential([
            ReflectionPad2d(1),
            layers.Conv2D(in_channels, kernel_size=3, strides=1),
            layers.GroupNormalization(groups=32, epsilon=1e-5),
            layers.Activation('relu'),
            ReflectionPad2d(1),
            layers.Conv2D(in_channels, kernel_size=3, strides=1),
            layers.GroupNormalization(groups=in_channels, epsilon=1e-5, axis=-1),
        ])

    def call(self, x):
        return x + self.conv_block(x)

class my_ResnetGenerator_tf(tf.keras.Model):
    def __init__(self):
        super(my_ResnetGenerator_tf, self).__init__()
        self.model = tf.keras.Sequential([
            ReflectionPad2d(3),
            layers.Conv2D(64, kernel_size=7, strides=1),
            layers.GroupNormalization(groups=64, epsilon=1e-5, axis=-1),
            layers.Activation('relu'),
            layers.Conv2D(128, kernel_size=3, strides=2, padding='same'),
            layers.GroupNormalization(groups=128, epsilon=1e-5, axis=-1),
            layers.Activation('relu'),
            layers.Conv2D(256, kernel_size=3, strides=2, padding='same'),
            layers.GroupNormalization(groups=256, epsilon=1e-5, axis=-1),
            layers.Activation('relu'),
            ResnetBlock(256),
            ResnetBlock(256),
            ResnetBlock(256),
            ResnetBlock(256),
            ResnetBlock(256),
            ResnetBlock(256),
            ResnetBlock(256),
            ResnetBlock(256),
            ResnetBlock(256),
            layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', output_padding=1),
            layers.GroupNormalization(groups=128, epsilon=1e-5, axis=-1),
            layers.Activation('relu'),
            layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', output_padding=1),
            layers.GroupNormalization(groups=64, epsilon=1e-5, axis=-1),
            layers.Activation('relu'),
            ReflectionPad2d(3),
            layers.Conv2D(3, kernel_size=7, strides=1),
            layers.Activation('tanh')
        ])

    def call(self, x):
        return self.model(x)
