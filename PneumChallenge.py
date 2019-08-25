# -*- coding: utf-8 -*-
"""PneumChallenge.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13QfUhKqfw1EzuLIQpIl1iv12cBhgKtn7
"""

#!/usr/bin/env python
# coding: utf-8

import os
import math
import random
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.transform import resize
import pydicom
import pylab
import cv2

print("Tensorflow version " + tf.__version__)

!pip install pydicom

from google.colab import drive
drive.mount('/content/drive')

!unzip '/content/drive/My Drive/Exam NET/train_images.zip' -d /content
!unzip '/content/drive/My Drive/Exam NET/test_images.zip' -d /content



tf.enable_eager_execution()

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_data_path = '/content/train_images'
test_data_path = '/content/test_images'
labels = pd.read_csv('/content/stage_2_train_labels.csv')

train_samples_num = len(os.listdir(train_data_path))
test_samples_num = len(os.listdir(test_data_path))

BATCH_SIZE = 32
EPOCHS = 5
STEPS_PER_EPOCH = int(train_samples_num / BATCH_SIZE)

# making a dictionary containing label data with paths to files
# KEY is PATH to image, VALUE is list of boxes, empty if Target equals to 0

def parse_labels(labels_df, data_path):
    # key - filenames, values - boxes
    labels_parsed = {}
    for index, row in labels_df.iterrows():
        filename = data_path + '/' + row['patientId'] + '.dcm'
        if filename not in labels_parsed:
            labels_parsed[filename] = []
        if row['Target'] == 1:
            labels_parsed[filename].append([row['x'], row['y'], row['width'], row['height']])
    return labels_parsed

def train_test_labels(image_path):
    valid_filenames = image_path[int(len(image_path) * 0.95) :]
    train_filenames = image_path[: int(len(image_path) * 0.95)]

    valid_labels = {}
    train_labels ={}

    for key in parsed_labels:
        if key in valid_filenames:
            valid_labels[key] = parsed_labels[key]
        elif key in train_filenames:
            train_labels[key] = parsed_labels[key]
    return train_labels, valid_labels

parsed_labels = parse_labels(labels, train_data_path)
image_filenames = list(parsed_labels.keys())

train_labels, valid_labels = train_test_labels(image_filenames)
train_filenames = list(train_labels.keys())
valid_filenames = list(valid_labels.keys())

train_images_num = len(train_filenames)
valid_images_num = len(valid_filenames)

class DatasetGenerator(tf.keras.utils.Sequence):

  def __init__(self, labels, total_items, epochs, batch_size, shuffle=True, predict = False):
    self.labels = labels
    self.total_items = total_items
    self.predict = predict
    self.epochs = epochs
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.filenames = list(labels.keys())
    self.on_epoch_end()

  def on_epoch_end(self):
      if self.shuffle:
           np.random.shuffle(self.filenames)

  def __len__(self):
          if self.predict:
              return int(np.ceil(len(self.filenames) / self.batch_size))
          else:
              return int(len(self.filenames) / self.batch_size)

  def load_data(self, file, label):
      image = pydicom.dcmread(file).pixel_array

      if not self.predict: 
          image_mask = np.zeros(image.shape)
          if label:
              for box in label:
                  x, y, w, h = box
                  image_mask[int(x):int(x + w), int(y):int(y + h)] = 1
          image_mask = resize(image_mask, (256, 256)) > 0.5
          image_mask = np.expand_dims(image_mask, -1)

      image = resize(image, (256, 256))                      
      image = np.expand_dims(image, -1)

      if not self.predict:
          return image, image_mask
      else: return image

  def __getitem__(self, index):
          filenames = self.filenames[index * self.batch_size: (index + 1) * self.batch_size]
          if self.predict:
              images = [self.load_data(filename, self.labels[filename]) for filename in filenames]
              images = np.array(images)
              return images, filenames
          else:
              images_masks = [self.load_data(filename, self.labels[filename]) for filename in filenames]
              images, masks = zip(*images_masks)
              images = np.array(images)
              masks = np.array(masks)
              return images, masks

train_generator = DatasetGenerator(train_labels, train_images_num, EPOCHS, BATCH_SIZE, True, False)
valid_generator = DatasetGenerator(valid_labels, valid_images_num, EPOCHS, BATCH_SIZE, False, False)

# swish activation

# swish activation function
def swish(x):
    return (tf.keras.backend.sigmoid(x) * x)

tf.keras.utils.get_custom_objects().update({'swish': tf.keras.layers.Activation(swish)})

# define model
def create_downsample(channels, inputs):
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(inputs)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Conv2D(channels, 1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    return x

def create_resblock(channels, inputs):
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(inputs)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
    return tf.keras.layers.add([x, inputs])

def create_network(input_size, channels, n_blocks=2, depth=4):
    # input
    inputs = tf.keras.Input(shape=(input_size, input_size, 1))
    x = tf.keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(inputs)
    # residual blocks
    for d in range(depth):
        channels = channels * 2
        x = create_downsample(channels, x)
        for b in range(n_blocks):
            x = create_resblock(channels, x)
    # output
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
    outputs = tf.keras.layers.UpSampling2D(2**depth)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# mean iou as a metric
def mean_iou(y_true, y_pred):
    y_pred = tf.round(y_pred)
    intersect = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    smooth = tf.ones(tf.shape(intersect))
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))

# create network and compiler
model = create_network(input_size=256, channels=32, n_blocks=2, depth=4)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_iou])

# Commented out IPython magic to ensure Python compatibility.
!pip install -q tf-nightly-2.0-preview
# Load the TensorBoard notebook extension
# %load_ext tensorboard

# learning rate transformation
def learn_rate_decay(epoch):
  return 0.01 * math.pow(0.6, epoch)

learn_rate_decay_callback = tf.keras.callbacks.LearningRateScheduler(learn_rate_decay, verbose=True)

tb_callback = tf.keras.callbacks.TensorBoard(log_dir='./log', histogram_freq=1,
                         write_graph=True,
                         write_grads=True,
                         batch_size=BATCH_SIZE,
                         write_images=True)

# checkpoint
chkp_path = '/content/md.ckpt'
cp_callback = tf.keras.callbacks.ModelCheckpoint(chkp_path, save_weights_only=True)

model.fit_generator(train_generator, validation_data=valid_generator, 
                    callbacks=[learn_rate_decay_callback, cp_callback],
                    epochs=EPOCHS, steps_per_epoch=15, validation_steps = 15,
                    use_multiprocessing=True, workers = 4)







