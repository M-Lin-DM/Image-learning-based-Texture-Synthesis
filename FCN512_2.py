import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

import os
import matplotlib.pyplot as plt
import cv2
import wandb
from wandb.keras import WandbCallback

# tf.config.experimental.list_physical_devices('GPU')
gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

wandb.init(entity='potoo', project='FCN512_lichen')

K.clear_session()

def process_image(imgtensor):
    imgtensor *= (1/255)
    # imgtensor += 1
    return imgtensor

def deprocess_image(imgtensor):
    imgtensor *= 255
    imgtensor = np.clip(imgtensor, 0, 255)
    return imgtensor

datadir = 'D:/Datasets/Lichen_partitioned' #your images must be inside a subfolder in the last folder of datadir. if using multiple classes, put each class in one subfolder of the train_data folder. also put validation and test sets each in their own subfolder at the same level as the training folder
datagen = keras.preprocessing.image.ImageDataGenerator(data_format="channels_last", preprocessing_function=process_image)

# load and iterate training dataset
train_it = datagen.flow_from_directory(datadir,
                                       class_mode='input',
                                       batch_size=8,
                                       target_size=(512, 512),
                                       shuffle=True,
                                       save_to_dir=None)

FCN_input = keras.Input(shape=(512, 512, 3), name='FCN_input')
x = layers.Conv2D(32, 7, strides=(1, 1), activation='relu', padding='same')(FCN_input)
x = layers.Conv2D(32, 7, strides=(1, 1), activation='relu', padding='same')(x)
x = layers.BatchNormalization(axis=-1)(x)
x = layers.Conv2D(64, 5, strides=(1, 1), activation='relu', padding='same')(x)
x = layers.Conv2D(64, 5, strides=(1, 1), activation='relu', padding='same')(x)
x = layers.BatchNormalization(axis=-1)(x)
x = layers.Conv2D(64, 3, strides=(1, 1), activation='relu', padding='same')(x)
x = layers.Conv2D(64, 3, strides=(1, 1), activation='relu', padding='same')(x)
x = layers.BatchNormalization(axis=-1)(x)
x = layers.Conv2D(64, 1, strides=(1, 1), activation='relu', padding='same')(x)
x = layers.Conv2D(64, 1, strides=(1, 1), activation='relu', padding='same')(x)
x = layers.BatchNormalization(axis=-1)(x)
FCN_output = layers.Conv2D(3, 1, strides=(1, 1), activation='relu', padding='same')(x)
FCN = keras.Model(FCN_input, FCN_output, name='FCN')

FCN.summary()

class ReconstructionLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        # self.batch_size = batch_size
        super(ReconstructionLogger, self).__init__()

    def on_epoch_end(self, epoch, logs):
        Nimgs = 7 #must be lower than batch size
        batchX, batchY = train_it.next()
        sample_images = batchX[:Nimgs, :, :, :]

        images = []
        reconstructions = []

        for i in range(Nimgs):
            reconstruction = deprocess_image(self.model.predict(np.expand_dims(sample_images[i, :, :, :], axis=0))) #for image iversion
            images.append(np.expand_dims(sample_images[i, :, :, :], axis=0))
            reconstructions.append(reconstruction)

        # wandb.log({"out_conv1":[]})
        wandb.log({"images": [wandb.Image(image) for image in images]}, commit=False)
        wandb.log({"reconstructions": [wandb.Image(reconstruction) for reconstruction in reconstructions]})

FCN.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False))

config = wandb.config
config.epochs = 2

FCN.fit(train_it,
                steps_per_epoch=539, #must be < (#images in dataset/batch size) lichen::886 is limit when batch =32. 3544 when batch =8 | letters(576imgs):: 72 when batch=8, 18 when batch=32. lichenpartition 540 max when batch=8
                epochs=config.epochs,
                verbose=2,
                # batch_size=config.batch_size,
                validation_data=None, callbacks=[WandbCallback(), ReconstructionLogger()]) #, TensorBoard(log_dir=wandb.run.dir)

FCN.save('FCN512_2_savemodel.h5')
FCN.save_weights('FCN512_2_weights')