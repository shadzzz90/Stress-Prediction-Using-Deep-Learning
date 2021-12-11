import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf

print(tf.__version__)
physical_devices = tf.config.experimental.list_physical_devices('GPU')

# physical_devices = tf.test.gpu_device_name()

from tensorflow import keras

if len(physical_devices) > 0:
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import numpy as np

trainInput1 = np.load('trainInput1.npy')
trainInput2 = np.load('trainInput2.npy')
trainOutput = np.load('trainOutput.npy')

testInput1 = np.load('testInput1.npy')
testInput2 = np.load('testInput2.npy')
testOutput = np.load('testOutput.npy')

valInput1 = np.load('valInput1.npy')
valInput2 = np.load('valInput2.npy')
valOutput = np.load('valOutput.npy')



def modelCNN():

    input1 = keras.Input(shape=(trainInput1.shape[1], trainInput1.shape[2], 1), batch_size=5)
    input2 = keras.Input(shape=(trainInput1.shape[1], trainInput1.shape[2], 1), batch_size=5)

    merged_input = keras.layers.concatenate([input1, input2])

    encoded = keras.layers.Conv2D(filters=120, kernel_size=(11, 11), strides=1, activation='relu')(merged_input)
    encoded = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(encoded)

    encoded = keras.layers.Conv2D(filters=60, kernel_size=(7, 7), strides=1, activation='relu')(encoded)
    encoded = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(encoded)

    encoded = keras.layers.Conv2D(filters=30, kernel_size=(3, 3), strides=2, activation='relu')(encoded)
    encoded = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(encoded)

    # flatten = keras.layers.Reshape(target_shape=(5880, 1))(encoded)
    # merged = keras.layers.Concatenate(axis=1)([flatten, input2])

    dense = keras.layers.Flatten()(encoded)

    # dense1 = keras.layers.Dense(5880)(dense)

    reshape = keras.layers.Reshape(target_shape=(14, 14, 30))(dense)

    decoded = keras.layers.UpSampling2D(size=(2, 2))(reshape)
    decoded = keras.layers.Conv2DTranspose(filters=30, kernel_size=(3, 3), strides=2, activation='relu')(decoded)

    decoded = keras.layers.UpSampling2D(size=(2, 2))(decoded)
    decoded = keras.layers.Conv2DTranspose(filters=60, kernel_size=(7, 7), strides=1, activation='relu')(decoded)

    decoded = keras.layers.UpSampling2D(size=(2, 2))(decoded)
    decoded = keras.layers.Conv2DTranspose(filters=120, kernel_size=(11, 11), strides=1, activation='relu')(decoded)

    output = keras.layers.Conv2DTranspose(filters=1, kernel_size=(1, 1), strides=1)(decoded)

    model = keras.Model(inputs=[input1, input2], outputs=output)
    model.summary()
    tf.keras.utils.plot_model(model, 'Stress_AutoEncoder_CNN.png', show_shapes=True)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=opt, loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


model = modelCNN()

# checkpoint_path = "training_1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
#
# # Create a callback that saves the model's weights
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)
#
#
# model.fit(x = [trainInput1,trainInput2],
#                      y= trainOutput,batch_size=1,epochs=1000,
#                      validation_data=([valInput1,valInput2],valOutput),
#                      validation_batch_size=1, callbacks=cp_callback)
#
# model.save('Stress_AutoEncoder')
