# ============================================
# A script that trains a simple cnn on MNIST
# to check whether tensorflow uses gpu
# ============================================

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import datasets, layers, models


# print status
print(f'TensorFlow version: {tf.__version__}')
print(f'GPU: {tf.test.is_gpu_available()}')
print(f'Eager execution: {tf.executing_eagerly()}')

# GPU configuration
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices):
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('memory growth:', tf.config.experimental.get_memory_growth(device))
else:
    print('Not enough GPU hardware devices available')

# load dataset
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))
x_train, x_test = x_train / 255.0, x_test / 255.0

# build model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# fit
model.fit(x_train, y_train,
          batch_size=64,
          epochs=10,
          validation_split=0.2,
          verbose=1)

# test
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print(test_acc)
