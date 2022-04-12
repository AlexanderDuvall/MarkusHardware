import datetime
import os
import pathlib

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, regularizers, Sequential

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
print(tf.__version__)
width = 314  # 640
height = 235  # 480
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.allow_growth = True

tf.config.experimental.set_memory_growth(physical_devices[0], True)
model = Sequential(
    [
        keras.Input(shape=(width, height, 3)),
        layers.experimental.preprocessing.Rescaling(1. / 255),
        layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(.001)),
        layers.MaxPooling2D(pool_size=(3, 3)),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(.001)),
        layers.Dropout(.5),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(.001)),
        layers.Flatten(),
        layers.Dense(8),
    ]
)
print(model.summary())

dataset_url = "C:\\Markus\\"
data_dir = tf.keras.utils.get_file(fname='C:\\Markus\\Dataset', origin="")
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.png')))

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=.0001),
    metrics=["accuracy"]
)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.15,
    subset="training",
    image_size=(width, height),
    seed=234234,
    batch_size=64,
    shuffle=True)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.15,
    seed=980243,
    subset="validation",
    image_size=(width, height),
    batch_size=64,
    shuffle=True
)

log_dir = "C:\\Markus\\Tensorboard\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

train_ds = train_ds.cache()
val_ds = val_ds.cache()
model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1, callbacks=tensorboard_callback)
print("Saving model...")
# model.save("Saved Model Data/MarkusModel")
