import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# lets try to load some of the images
data_dir = '../dataset1/train'
batch_size = 32
img_height = 180
img_width = 180

# Crete a new instance of the CardDataset
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  # using 70 30 split for training and validation
  validation_split=0.3,
  subset="training",
  # seed for randomization
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
# the dataset is a tensorflow image dataset
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  # using 70 30 split for training and validation
  validation_split=0.3,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

print(f'Training class names: {train_ds.class_names}')
print(f'Validation class names: {val_ds.class_names}')



plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(train_ds.class_names[labels[i]])
    plt.axis("off")
plt.show()