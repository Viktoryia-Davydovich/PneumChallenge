#!/usr/bin/env python
# coding: utf-8

# In[114]:


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


# In[115]:


AUTOTUNE = tf.data.experimental.AUTOTUNE

train_data_path = 'D:\\PneumChallenge\\data\\train_images'
test_data_path = 'D:\\PneumChallenge\\data\\test_images'
labels = pd.read_csv('D:\\PneumChallenge\\data\\stage_2_train_labels.csv')

train_samples_num = len(os.listdir(train_data_path))
test_samples_num = len(os.listdir(test_data_path))

BATCH_SIZE = 32
EPOCHS = 15
STEPS_PER_EPOCH = int(train_samples_num / BATCH_SIZE)


# In[116]:


# making a dictionary containing label data with paths to files
# KEY is PATH to image, VALUE is list of boxes, empty if Target equals to 0

def parse_labels(labels_df, data_path):
    # key - filenames, values - boxes
    labels_parsed = {}
    for index, row in labels_df.iterrows():
        filename = data_path + '\\' + row['patientId'] + '.dcm'
        if filename not in labels_parsed:
            labels_parsed[filename] = []
        if row['Target'] == 1:
            labels_parsed[filename].append([row['x'], row['y'], row['width'], row['height']])
    return labels_parsed


# In[130]:


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


# In[131]:


parsed_labels = parse_labels(labels, train_data_path)
image_filenames = list(parsed_labels.keys())

train_labels, valid_labels = train_test_labels(image_filenames)
train_filenames = list(train_labels.keys())
valid_filenames = list(valid_labels.keys())

train_images_num = len(train_filenames)
valid_images_num = len(valid_filenames)


# In[132]:


def load_data(file, label, predict = False):
    image = pydicom.dcmread(file).pixel_array
        
    if not predict: 
        image_mask = np.zeros(image.shape)
        if label:
            for box in label:
                x, y, w, h = box
                image_mask[int(x):int(x + w), int(y):int(y + h)] = 1
        image_mask = resize(image_mask, (256, 256)) > 0.5
        image_mask = np.expand_dims(image_mask, -1)
               
    image = resize(image, (256, 256))                      
    image = np.expand_dims(image, -1)
        
    if not predict:
        return image, image_mask
    else: return image
                       
            
def generator(labels, total_items, predict = False):
    filenames = list(labels.keys())
    i = 0
    while i < total_items:
        if predict:
            image = load_data(filenames[i], labels[filenames[i]], predict)
            yield images
        else:
            images, masks = load_data(filenames[i], labels[filenames[i]])
            yield images, masks
        i += 1
        
def data_input_fn(labels, total_items, predict, epochs, batch_size):
    # creation of datasets
    types = (tf.float64, tf.float64)
    shapes = ((256, 256, 1),(256, 256, 1))
    dataset = tf.data.Dataset.from_generator(lambda: generator(labels, total_items, predict),
                                             types, shapes)

    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size, drop_remainder = True)

    iterator = dataset.make_one_shot_iterator()
    while True:
        batch_features, batch_labels = iterator.get_next()
        yield batch_features, batch_labels


# In[133]:


train_generator = data_input_fn(labels = train_labels, total_items = train_images_num, predict = False,
                                epochs = EPOCHS, batch_size = BATCH_SIZE)
valid_generator = data_input_fn(labels = valid_labels, total_items = valid_images_num, predict = False,
                                epochs = EPOCHS, batch_size = BATCH_SIZE)


# In[134]:


# swish activation

# swish activation function
def swish(x):
    return (tf.keras.backend.sigmoid(x) * x)

tf.keras.utils.get_custom_objects().update({'swish': tf.keras.layers.Activation(swish)})


# In[135]:


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


# In[136]:


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

# learning rate transformation
def learn_rate_decay(epoch):
  return 0.01 * math.pow(0.6, epoch)

learn_rate_decay_callback = tf.keras.callbacks.LearningRateScheduler(learn_rate_decay, verbose=True)



tb_callback = tf.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# checkpoint
chkp_path = '/content/md.ckpt'
cp_callback = tf.keras.callbacks.ModelCheckpoint(chkp_path, save_weights_only=True)


# In[137]:


model.fit_generator(train_generator, validation_data=valid_generator, 
                    callbacks=[learn_rate_decay_callback, tb_callback, cp_callback],
                    epochs=EPOCHS, steps_per_epoch=3, validation_steps = 3)


# In[91]:





# In[99]:





# In[105]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[98]:


def generator(labels, samples_size, predict=False, batch_size=32):
    filenames = list(labels.keys())
    for i in range( samples_size // batch_size ):
        batch_filenames = filenames[i * batch_size: (i+1) * batch_size]
        if predict:
            images = [load_data(file, labels[file], predict) for file in batch_filenames]
            images = np.array(images)
            yield images
        else:
            images_masks = [load_data(file, labels[file]) for file in batch_filenames]
            images, masks = zip(*images_masks)
            images = np.array(images)
            masks = np.array(masks)
            yield images, masks


# In[ ]:





# In[ ]:





# In[ ]:





# In[18]:





# In[16]:





# In[19]:





# In[ ]:





# In[ ]:




