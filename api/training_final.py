# -*- coding: utf-8 -*-
"""
Created on Sat May 11 20:46:47 2024

@author: Rishika
"""
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

# Defining variables
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
n_classes = 3
EPOCHS=50

# Loading dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage", 
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names = dataset.class_names
print(class_names)

# Exploring the dataset
plt.figure(figsize=(10, 10))
for image_batch, label_batch in dataset.take(1):
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))  # Corrected to image_batch[i]
        plt.title(class_names[label_batch[i].numpy()])  # Corrected to label_batch[i]
        plt.axis("off")

# Splitting the dataset
def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, shuffle=True, shuffle_size=10000):
    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

# Caching the datasets
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# Preprocessing on images for training
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),  # Corrected import path
    layers.Rescaling(1.0 / 255),
])

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),  # Corrected import path
    layers.RandomRotation(0.2),
])

# Convolutional Neural Network
input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)

model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
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

model.build(input_shape=(BATCH_SIZE,) + input_shape)  # Correct input shape
model.summary()

#------------------------------------------1st model building and then 2nd step - compiler building for model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'],
)


#------------------------------------------------3rd step- model.fit---to train the network and finding accuracy
EPOCHS=4
history=model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_ds,  # This data is used in each epoch to track the accuracy
)

#------measuring the accuracy on test dataset
scores=model.evaluate(test_ds)
scores

#listing accuracy from history
history.history['accuracy']


#-----------------------------------prediction------------------------
import numpy as np
for images_batch, labels_batch in test_ds.take(1):
    first_image = (images_batch[0].numpy().astype("uint8"))  #will print the actual img
    first_label = labels_batch[0].numpy()
    
    print("first image to predict")
    plt.imshow(first_image)
    print("actual label:", class_names[first_label])  #to know the class type of first img
    
    batch_prediction=model.predict(images_batch)  #prediction for 32 img
    print("predicted labels -" ,class_names[np.argmax(batch_prediction[0])])
    
#----------running my prediction on entire batch

def predict(model, img):
    img_array=tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array=tf.expand_dims(img_array, 0) #creating batch
    
    predictions = model.predict(img_array)
    
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100*(np.max(predictions[0])),2)
    return predicted_class, confidence
    


plt.figure(figsize=(10,10))
for images, labels in test_ds.take(1): 
    for i in range(9):   #ruuning for loop on first batch , doing prediction on only 9 img
        #showing all img-
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        
        #showing predicted class and actual class
        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class= class_names[labels[i]]
        
        plt.title(f"actual:{actual_class},\n Predicted: {predicted_class}. \n Confidence: {confidence}%")
        
        plt.axis("off")
    
#----------------------------------------------saving the model in directory
import os

#getting next model version automatically-
model_version=max([int(i) for i in os.listdir("../models") + [0]]) + 1
#saving-
# Create directory if it doesn't exist
save_dir = f"../models/{model_version}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save the model
model.save("../potatoes.h5")


