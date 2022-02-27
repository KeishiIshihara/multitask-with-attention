import tensorflow as tf
from tensorflow.python.client import device_lib
print('TensorFlow=%s'% tf.__version__)
devices = device_lib.list_local_devices()
print('NUN_GPUS=%s'% sum([1 for d in devices if ':GPU:' in d.name]))
