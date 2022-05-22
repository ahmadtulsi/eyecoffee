from mobilevit import create_mobilevit
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')

import os
from dagshub.keras import DAGsHubLogger
from dagshub import dagshub_logger
from omegaconf import OmegaConf
import mlflow

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.experimental import CosineDecay
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB3, ResNet50V2, MobileNetV2, NASNetMobile, InceptionV3
from tensorflow.keras.layers.experimental.preprocessing import RandomCrop,CenterCrop, RandomRotation



ready = pd.read_csv("/content/drive/MyDrive/Python/data/dataset.csv", usecols=['File', 'Label', 'Filepath'])

training_percentage = 0.8
training_item_count = int(len(ready) * training_percentage)
validation_item_count = len(ready) - training_item_count
training_df = ready[:training_item_count]
validation_df = ready[training_item_count:]

training_data = tf.data.Dataset.from_tensor_slices((training_df.Filepath.values, training_df.Label.values))
validation_data = tf.data.Dataset.from_tensor_slices((validation_df.Filepath.values, validation_df.Label.values))

batch_size = 12
image_size = 256
input_shape = (image_size, image_size, 3)
dropout_rate = 0.4
classes_to_predict = sorted(training_df.Label.unique())

AUTOTUNE = tf.data.experimental.AUTOTUNE
def load_image_and_label_from_path(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (image_size, image_size))
    #img = layers.experimental.preprocessing.Rescaling(1./255)(img)
    return img, label
training_data = training_data.map(load_image_and_label_from_path, num_parallel_calls=AUTOTUNE)
validation_data = validation_data.map(load_image_and_label_from_path, num_parallel_calls=AUTOTUNE)

data_augmentation_layers = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.RandomCrop(height=image_size, width=image_size),
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.25),
        layers.experimental.preprocessing.RandomZoom((-0.2, 0)),
        layers.experimental.preprocessing.RandomContrast((0.2,0.2))
    ]
)

def prepare(ds, shuffle=False, augment=False):
  if shuffle:
    ds = ds.shuffle(100)

  # Batch all datasets.
  ds = ds.batch(batch_size)

  # Use data augmentation only on the training set.
  if augment:
    ds = ds.map(lambda x, y: (data_augmentation_layers(x, training=True), y), 
                num_parallel_calls=AUTOTUNE)

  # Use buffered prefetching on all datasets.
  return ds.prefetch(buffer_size=AUTOTUNE)

train_ds = prepare(training_data, shuffle=True, augment=True)
val_ds = prepare(validation_data)

training_data_batches = training_data.shuffle(buffer_size=100).batch(batch_size).prefetch(buffer_size=AUTOTUNE)
validation_data_batches = validation_data.shuffle(buffer_size=100).batch(batch_size).prefetch(buffer_size=AUTOTUNE)

baseline_model = MobileNetV2(include_top=False, input_shape=input_shape, drop_connect_rate=dropout_rate)
inputs = Input(shape=input_shape)
augmented = data_augmentation_layers(inputs)
model = baseline_model(augmented)
pooling = layers.GlobalAveragePooling2D()(model)
dropout = layers.Dropout(dropout_rate)(pooling)
outputs = layers.Dense(len(classes_to_predict), activation="softmax")(dropout)
model = Model(inputs=inputs, outputs=outputs)

plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5',
                                                          monitor='val_accuracy',
                                                          mode='max',
                                                          save_best_only=True)
callbacks = [plateau, model_checkpoint]

if __name__ == "__main__":

    mlflow.tensorflow.autolog()

    with mlflow.start_run():
        model.compile(loss="sparse_categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
        history = model.fit(train_ds,
                            validation_data=val_ds,
                            epochs=10,
                            callbacks=[callbacks])
        trained_model_loss, trained_model_accuracy = model.evaluate(val_ds)

        with dagshub_logger() as logger:
                logger.log_metrics(loss=trained_model_loss, accuracy=trained_model_accuracy)

