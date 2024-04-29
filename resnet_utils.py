import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt
import numpy as np

def _select_norm_layer(norm):
  if norm == 'none':
      return lambda: lambda x: x
  elif norm == 'batch_norm':
      return tf.keras.layers.BatchNormalization
  elif norm == 'instance_norm':
      return tf.keras.layers.InstanceNormalization
  elif norm == 'layer_norm':
      return tf.keras.layers.LayerNormalization

def residual_block(x, out_channels, norm):
  norm_layer = _select_norm_layer(norm)

  x_skip = x
  # Layer 1
  x = tf.keras.layers.Conv2D(out_channels, (3,3), padding = 'same')(x)
  x = norm_layer()(x)
  x = tf.keras.layers.Activation('relu')(x)
  # Layer 2
  x = tf.keras.layers.Conv2D(out_channels, (3,3), padding = 'same')(x)
  x = norm_layer()(x)
  # Add the residue
  x = tf.keras.layers.Add()([x,x_skip])
  x = tf.keras.layers.Activation('relu')(x)   # ci vuole la relu ???
  return x

def convolutional_block(x, out_channels, norm):
  norm_layer = _select_norm_layer(norm)

  x_skip = x

  # Layer 1
  x = tf.keras.layers.Conv2D(out_channels, (3,3), padding = 'same', strides = (2,2))(x)
  x = norm_layer()(x)
  x = tf.keras.layers.Activation('relu')(x)
  # Layer 2
  x = tf.keras.layers.Conv2D(out_channels, (3,3), padding = 'same')(x)
  x = norm_layer()(x)
  # Pass residue through conv(1,1)
  x_skip = tf.keras.layers.Conv2D(out_channels, (1,1), strides = (2,2))(x_skip)
  # Add the residue
  x = tf.keras.layers.Add()([x,x_skip])
  x = tf.keras.layers.Activation('relu')(x)
  return x


def resnet_generator(input_shape=(256, 256, 3), output_channels=3, dim=64, n_downsamplings=2, n_residual=9, norm='batch_norm'):
  '''
    input_shape = shape of the input image
    output_channels = output channels, it will be 3 because we need to produce an image
    dim = number of filters applied in the convolutions
    n_downsamplings = number of downsampling (and later upsampling) operations
    n_residual = number of residual blocks present in the network
    norm = normalization layer selected
  '''
  norm_layer = _select_norm_layer(norm)

  # initialize the tensor
  h = inputs = tf.keras.Input(shape=input_shape)

  # pad and reflect the borders
  h = tf.pad(h, [[0,0],[3,3],[3,3],[0,0]], mode='REFLECT')
  h = tf.keras.layers.Conv2D(dim, 7, padding='valid', use_bias=False)(h)
  h = norm_layer()(h)
  h = tf.keras.layers.Activation('relu')(h)

  # downsampling
  for _ in range(n_downsamplings):
    dim *= 2; #double the dimension of the input
    h = tf.keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)    ### ADD POOLING LAYER ???
    h = norm_layer()(h)
    h = tf.keras.layers.Activation('relu')(h)

  # residual blocks
  for _ in range(n_residual):
    h = residual_block(h, dim, norm)

  # upsampling
  for _ in range(n_downsamplings):
    dim //= 2
    h = tf.keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)
    h = norm_layer()(h)
    h = tf.keras.layers.Activation('relu')(h)

  h = tf.pad(h, [[0,0],[3,3],[3,3],[0,0]], mode='REFLECT')
  h = tf.keras.layers.Conv2D(output_channels, 7, padding='valid')(h)
  h = tf.keras.layers.Activation('tanh')(h)

  return tf.keras.Model(inputs=inputs, outputs=h)
  