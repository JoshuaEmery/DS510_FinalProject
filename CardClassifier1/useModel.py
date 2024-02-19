# this uses the model we trained in ds1_classifier.py to classify the images in the test set.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import PIL
import PIL.Image

# load the model
model = tf.keras.models.load_model('ds1_classifier.keras')

# load the test set
data_dir = '../dataset1/test'
batch_size = 32

test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    seed=123,
    image_size=(180, 180),
    batch_size=batch_size)

# show the first 9 images
plt.figure(figsize=(10, 10))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(test_ds.class_names[labels[i]])
        plt.axis("off")
plt.show()

# make a prediction
predictions = model.predict(test_ds)
print(predictions)

# show the first 9 predictions
plt.figure(figsize=(10, 10))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f'Predicted: {test_ds.class_names[np.argmax(predictions[i])]}')
        plt.axis("off")
plt.show()