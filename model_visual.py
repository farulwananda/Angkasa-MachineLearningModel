import tensorflow as tf
from tensorflow.keras.models import save_model
from tensorflow.keras import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.initializers import RandomNormal, Constant
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ann_visualizer.visualize import ann_viz

# tf.keras.models
model = Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid'),
    tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid'),
    tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')

    # #Arsitektur VGG16

    # tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(64, 64, 3)),
    # tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    # tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    # tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    # tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    # tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    # tf.keras.layers.Flatten(),
    # tf.keras.layers.Dropout(0.5),
    # tf.keras.layers.Dense(units=4096, activation='relu'),
    # tf.keras.layers.Dropout(0.5),
    # tf.keras.layers.Dense(units=3, activation='softmax'),

])
model.summary()
# model = model
#
# ann_viz(model, view=True, filename='construct_model', title='Cnn_Test_1')

#%%
