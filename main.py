import tensorflow as tf
import glob
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import save_model
from tensorflow.keras import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.initializers import RandomNormal, Constant
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])
    except RuntimeError as e:
        print(e)

base_dir = 'dataset'
normal_dir = os.path.join(base_dir, 'normal')
basal_dir = os.path.join(base_dir, 'basal')
melanoma_dir = os.path.join(base_dir, 'melanoma')
vascular_dir = os.path.join(base_dir, 'vascular')

total_image = len(list(glob.iglob("dataset/*/*.*", recursive=True)))
print("Total Data Image          : ", total_image)

total_normal = len(os.listdir(normal_dir))
total_basal = len(os.listdir(basal_dir))
total_melanoma = len(os.listdir(melanoma_dir))
total_vascular = len(os.listdir(vascular_dir))

# Mencetak jumlah data kanker kulit basal, melanoma, vascular
print("Total Data Normal         : ", total_normal)
print("Total Data Basal          : ", total_basal)
print("Total Data Melanoma       : ", total_melanoma)
print("Total Data Vascular       : ", total_vascular)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(64, 64),
    batch_size=16,
    shuffle=True,
    color_mode="rgb",
    class_mode="categorical",
    subset="training"

)

validation_generator = val_datagen.flow_from_directory(
    base_dir,
    target_size=(64, 64),
    batch_size=16,
    shuffle=True,
    color_mode="rgb",
    class_mode="categorical",
    subset="validation"

)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid'),
    tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid'),
    tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu'),
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

Adam(learning_rate=0.0015, name='Adam')
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir='logs', histogram_freq=0, write_graph=True, write_images=False,
    update_freq='epoch', embeddings_freq=0,
    embeddings_metadata=None
)

accuracy = 0.99


class func_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') >= accuracy):
            print("Akurasi Telah Mencapai %2.2f%% , Proses Training Dihentikan." % (accuracy * 100))
            self.model.stop_training = True


callback = func_callback()

batch_size = 16

history = model.fit(
    train_generator,
    steps_per_epoch=335 // batch_size,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=83 // batch_size,
    verbose=2,
    callbacks=[lr_schedule, callback])

# Mengambil Nilai Accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
# Mengambil Nilai Loss
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

# Plot Accruracy
plt.plot(epochs, acc, 'r', label='Train accuracy')
plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()

# Plot Loss
plt.plot(epochs, loss, 'r', label='Train loss')
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()
plt.show()

# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# serialize weights to HDF5
model.save("model_x3.h5")
print("Saved model to disk")
