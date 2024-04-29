import os, sys
import time
import random
from tqdm import tqdm
from os import listdir
from PIL import Image
from IPython.display import clear_output
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt
import pickle

def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def preprocess_image_train(image):
  # resizing to 256 x 256 x 3
  image = tf.image.resize(image, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  # random mirroring
  image = tf.image.random_flip_left_right(image)
  # normalize images (subtract mean, divide by std)
  image = normalize(image)
  return image

def select_random_batch(folder_path, BATCH_SIZE, dimx, dimy):
  import random

  # Check if folder exists
  if not os.path.exists(folder_path):
      print(f"The folder {folder_path} does not exist.")
      return None
  
  # Get the list of files in the folder
  files = os.listdir(folder_path)

  # Select random images and convert them to tensors
  random_images = []
  for i in range(BATCH_SIZE):
    path = folder_path + '/' + random.choice(files)
    image = Image.open(path)
    image = image.resize([dimx, dimy])
    image_tensor = tf.convert_to_tensor(image)
    random_images.append(image_tensor)

  batch_tensor = tf.stack(random_images, axis=0) 
  return batch_tensor

def preprocess_image_test(image):
  # normalize images (subtract mean, divide by std)
  image = normalize(image)
  return image

def generate_images(model, test_input):
  prediction = model(test_input)
  plt.figure(figsize=(12, 12))

  display_list = [test_input[0], prediction[0]]
  title = ['Input Image', 'Predicted Image']

  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()
  return prediction

def generate_test_images(model, test_input):
    # Generate all test predictions
    predictions = model(test_input)

    plt.figure(figsize=(15, 15))
    display_rows = 2
    display_cols = test_input.shape[0] // display_rows

    for i in range(test_input.shape[0]):
        plt.subplot(display_rows, 2 * display_cols, 2 * i + 1)
        plt.title('Input Image')
        plt.imshow(test_input[i] * 0.5 + 0.5)
        plt.axis('off')

        plt.subplot(display_rows, 2 * display_cols, 2 * i + 2)
        plt.title('Predicted Image')
        plt.imshow(predictions[i] * 0.5 + 0.5)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def save_data_py(dati,filename):
    with open(filename,'wb') as datifile:
        pickle.dump(dati,datifile)

def load_data_py(filename):
    with open(filename,'rb') as datifile:
        x = pickle.load(datifile)
    return x

def compare_models(model1, model2, model3, test_input):
    # Generate all test predictions
    predictions1 = model1(test_input)
    predictions2 = model2(test_input)
    predictions3 = model3(test_input)

    plt.figure(figsize=(15, 15))
    display_rows = 4
    display_cols = test_input.shape[0]

    for i in range(test_input.shape[0]):
        # Plot input images in the first row
        plt.subplot(display_rows, display_cols, i + 1)
        plt.title('Input Image')
        plt.imshow(test_input[i] * 0.5 + 0.5)
        plt.axis('off')

        # Plot predicted images in the second row
        plt.subplot(display_rows, display_cols, display_cols + i + 1)
        plt.title('Predicted Image')
        plt.imshow(predictions1[i] * 0.5 + 0.5)
        plt.axis('off')

        # Plot predicted images in the third row
        plt.subplot(display_rows, display_cols, 2*display_cols + i + 1)
        plt.title('Predicted Image')
        plt.imshow(predictions2[i] * 0.5 + 0.5)
        plt.axis('off')

        # Plot predicted images in the forth row
        plt.subplot(display_rows, display_cols, 3*display_cols + i + 1)
        plt.title('Predicted Image')
        plt.imshow(predictions3[i] * 0.5 + 0.5)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('models_comparison', format='pdf')
    plt.show()

