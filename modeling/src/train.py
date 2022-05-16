from mobilevit import create_mobilevit
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')

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
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers.experimental.preprocessing import RandomCrop,CenterCrop, RandomRotation



MODELING_CONST_PATH = os.path.join('modeling', 'src', 'const.yaml')
GENERAL_CONST_PATH = os.path.join('src', 'const.yaml')

data_augmentation_layers = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.RandomCrop(height=image_size, width=image_size),
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.25),
        layers.experimental.preprocessing.RandomZoom((-0.2, 0)),
        layers.experimental.preprocessing.RandomContrast((0.2,0.2))
    ]
)


def get_data(data_path, training_percentage):
    dataset = pd.read_csv(data_path, usecols=['File', 'Label', 'Filepath'])
    training_percentage = training_percentage
    training_item_count = int(len(dataset) * training_percentage)
    validation_item_count = len(dataset) - training_item_count
    training_df = dataset[:training_item_count]
    validation_df = dataset[training_item_count:]

    training_data = tf.data.Dataset.from_tensor_slices((training_df.Filepath.values, training_df.Label.values))
    validation_data = tf.data.Dataset.from_tensor_slices((validation_df.Filepath.values, validation_df.Label.values))

    return training_data, validation_data



def load_image_and_label_from_path(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (image_size, image_size))
    #img = layers.experimental.preprocessing.Rescaling(1./255)(img)
    return img, label

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

def ready(training_data, validation_data):
    
    training_data = training_data.map(load_image_and_label_from_path, num_parallel_calls=AUTOTUNE)
    validation_data = validation_data.map(load_image_and_label_from_path, num_parallel_calls=AUTOTUNE)

    train_ds = prepare(training_data, shuffle=True, augment=True)
    val_ds = prepare(validation_data)

    return train_ds, val_ds

def get_callbacks(csv_logger_path, checkpoint_filepath, model_const):
    csv_logger = tf.keras.callbacks.CSVLogger(csv_logger_path)

    plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

    db_logger = DAGsHubLogger(metrics_path=os.path.join(os.getcwd(), *model_const.METRICS_PATH),
                              hparams_path=os.path.join(os.getcwd(), *model_const.PARAMS_PATH))

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                          monitor='val_accuracy',
                                                          mode='max',
                                                          save_best_only=True)

    return csv_logger, plateau, model_checkpoint, db_logger


if __name__ == "__main__":
    general_const = OmegaConf.load(os.path.join(os.getcwd(), GENERAL_CONST_PATH))
    modeling_const = OmegaConf.load(os.path.join(os.getcwd(), MODELING_CONST_PATH))

    mlflow.set_tracking_uri(general_const.MLFLOW_TRACKING_URL)
    mlflow.tensorflow.autolog()

    with mlflow.start_run():
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        image_size = general_const.IMAGE_SIZE
        batch_size = general_const.BATCH_SIZE
        training_data, validation_data = get_data(general_const.DATA_FILE, modeling_const.TRAINING_PERCENTAGE)
        train_ds, val_ds = ready(training_data, validation_data)
        
        callbacks = list(get_callbacks(os.path.join(os.getcwd(), *modeling_const.CSV_LOG_PATH),
                                       os.path.join(os.getcwd(), *modeling_const.CHECKPOINT_PATH),
                                       modeling_const))

        model = create_mobilevit(num_classes= general_const.NUM_CLASS)

        model.compile(loss=modeling_const.LOSS,
        optimizer=tf.keras.optimizers.Adam(lr= modeling_const.LEARNING_RATE),
                                           metrics= modeling_const.METRICS)
        history = model.fit(
                        train_ds,
                        validation_data=val_ds,
                        epochs=modeling_const.INIT_EPOCHS,
                        callbacks=[callbacks])

        trained_model_loss, trained_model_accuracy = model.evaluate(val_ds)

        with dagshub_logger() as logger:
            logger.log_metrics(loss=trained_model_loss, accuracy=trained_model_accuracy)

        model.save(os.path.join(os.getcwd(), general_const.PROD_MODEL_PATH))
