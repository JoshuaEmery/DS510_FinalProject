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


# show the first 9 images
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(train_ds.class_names[labels[i]])
#     plt.axis("off")
# plt.show()


for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

normalization_layer = tf.keras.layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
# print(np.min(first_image), np.max(first_image))

# AUTOTUNE = tf.data.AUTOTUNE

# train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# num_layers = 53
# with tf.device("/gpu:0"):
#   model = tf.keras.Sequential([
#     tf.keras.layers.Rescaling(1./255),
#     tf.keras.layers.Conv2D(32, 3, activation='relu'),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Conv2D(32, 3, activation='relu'),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Conv2D(32, 3, activation='relu'),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(num_layers)
#   ])

  # model.compile(
  #   optimizer='adam',
  #   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  #   metrics=['accuracy'])

  # model.fit(
  #   train_ds,
  #   validation_data=val_ds,
  #   epochs=10
  # )

  # model.save('ds1_classifier.keras')

# load the model
model = tf.keras.models.load_model('ds1_classifier.keras')

# predict the first 9 images
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(val_ds.take(1))
print(predictions)

# show the first 9 images
plt.figure(figsize=(10, 10))
for images, labels in val_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(val_ds.class_names[np.argmax(predictions[i])])
    plt.axis("off")
plt.show()

