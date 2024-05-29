# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:17:34 2024

@author: Rishika
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 11 20:46:47 2024

@author: Rishika
"""
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from IPython.display import HTML 

from tensorflow.keras.preprocessing.image import ImageDataGenerator



# -------------------------------------------- Augmentation ---------------------------


#---------------------------------------------training----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    horizontal_flip=True,
    )

IMAGE_SIZE = 256
CHANNELS = 3

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=32,
    class_mode="sparse",
    
)

train_generator.class_indices

class_names = list(train_generator.class_indices.keys())
class_names

dir(train_generator)

#-----------------------------------------validation--------------------------------
validation_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    horizontal_flip=True,
)

validation_generator = validation_datagen.flow_from_directory(
    'dataset/val',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=32,
    class_mode="sparse",
    
)

#-------------------------------------------test------------------------------------
test_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    horizontal_flip=True,
)

test_generator = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=32,
    class_mode="sparse",
    
)

#----------------------------------------------------model build and compile------------------
input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes=3

model = models.Sequential([
    layers.InputLayer(input_shape=input_shape),
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])


model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'],
)

model.fit(
    train_generator,
    steps_per_epoch = 47,
    batch_size = 32,
    validation_data = validation_generator,
    validation_steps = 6,
    verbose=1,  #to print in detail
    epochs=10,
    )

#--------------------------------evaluating the model for test dataset-------------

scores = model.evaluate(test_generator)
scores

#------------------------------------saving the model
# Assuming 'model' is your trained Keras model

import h5py
print(h5py.__version__)

# Save the model
model.save("../potatoes.h5")


import h5py
model.save("../potatoes.h5")

