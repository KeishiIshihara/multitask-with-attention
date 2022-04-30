from abc import ABCMeta, abstractmethod
import logging
import os
import sys
import tensorflow as tf


def load_weights(model, weight_path):
    checkpoint = tf.train.Checkpoint(model=model)
    try:
        if os.path.isdir(weight_path):
            latest_ckpt = tf.train.latest_checkpoint(weight_path)
            logging.info('Latest ckpt:', latest_ckpt)
            status = checkpoint.restore(latest_ckpt).expect_partial()
        else:
            status = checkpoint.restore(weight_path).expect_partial()
    except Exception as e:
        logging.exception(e)
        sys.exit()
    else:
        logging.info('Weights loaded successfully')

    status.assert_existing_objects_matched()


class Model(tf.keras.Model, metaclass=ABCMeta):
    @abstractmethod
    def call(self):
        return

    def summary(self):
        if hasattr(self, 'model'):
            self.model.summary()
        else:
            raise AttributeError('Object has no model attr.')

    def load_weights(self, weight_path):
        self.checkpoint = tf.train.Checkpoint(model=self)
        try:
            if os.path.isdir(weight_path):
                latest_ckpt = tf.train.latest_checkpoint(weight_path)
                print('[INFO] latest ckpt:', latest_ckpt)
                status = self.checkpoint.restore(latest_ckpt).expect_partial()
            else:
                status = self.checkpoint.restore(weight_path).expect_partial()
        except Exception as e:
            print(e)
            sys.exit()
        else:
            print('Weights loaded successfully')

        status.assert_existing_objects_matched()

    @tf.function
    def predict(self, inputs, training=False):
        outputs = self(**inputs, training=training)
        return outputs

    def plot_model(self, model=None, filename='model.png'):
        if model is None and hasattr(self, 'model'):
            tf.keras.utils.plot_model(self.model, to_file=filename, show_shapes=True)
        elif model is not None:
            tf.keras.utils.plot_model(model, to_file=filename, show_shapes=True)
        else:
            raise ValueError('model not known')
