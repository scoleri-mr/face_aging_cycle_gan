from training_utils import *
from utils import *
from resnet_utils import *
from pretrained_resnet_pytorch import *
from pretrained_resnet_tensorflow import *
from transfer_learning_utils import *
from tensorflow.keras import models

def get_tensorflow_model(model_name:str):
  import torch
  import tensorflow as tf
  import numpy as np

  def unwrap(model):
    # get the wrapper layer's name
    for layer in model.layers:
      if isinstance(layer, models.Sequential):
          wrapper = layer.name

    # unwrap the model: we have an external sequential we don't like
    sequential_block = model.get_layer(wrapper)
    # Get the layers inside the Sequential block
    inner_layers = sequential_block.layers
    # Create a new model with the layers inside the Sequential block
    new_model = models.Sequential(inner_layers)

    return new_model

  # Load the PyTorch model
  path = 'pretrained_models/' + model_name
  pytorch_model = my_ResnetGenerator()
  state_dict = torch.load(path)
  state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
  pytorch_model.load_state_dict(state_dict)
  pytorch_model.eval()

  # Extract PyTorch weights
  weights_pytorch = {key: value.numpy() for key, value in pytorch_model.state_dict().items()}

  # Build the TensorFlow model with the same layers and dimensions
  model_tf = my_ResnetGenerator_tf()

  # Move the weights from PyTorch to TensorFlow
  for layer in model_tf.layers:
    if layer.name in weights_pytorch:
      layer.set_weights([weights_pytorch[layer.name]])

  # Perform a forward pass to ensure TensorFlow model builds its structure
  dummy_input = np.random.randn(4, 256, 256, 3).astype(np.float32)
  model_tf(dummy_input)

  model_tf = unwrap(model_tf)
  model_tf(dummy_input)

  return model_tf

def freeze_layers(model):

  def find_last_residual_block(model):
    last_residual_block = None
    for layer in reversed(model.layers):
        # Check if the layer is a residual block
        if 'resnet_block' in layer.name:
            last_residual_block = layer.name
            break
    return last_residual_block

  last_residual_block = find_last_residual_block(model)
  freeze = True
  for layer in model.layers:
    if layer.name == last_residual_block:
      freeze = False

    if freeze == True:
      layer.trainable = False
    else:
      layer.trainable = True