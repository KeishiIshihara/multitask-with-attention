"""
Each model will be tested from its construction to loading their pretrained weights
and inference using dummy data.
"""

import os.path as osp
import sys

import numpy as np
import tensorflow as tf

PROJECT_ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
sys.path.append(PROJECT_ROOT_DIR)
print(PROJECT_ROOT_DIR)

from model.baseline import Baseline
from model.cilrs import CILRS
from model.mt import DrivingModule
from model.mta import MTA


def get_dummy_inputs():
    rand = np.random.rand
    return [np.float32(rand(32, 160, 384, 3)), np.float32(rand(32, 4)), np.float32(rand(32, 1))]


class UnitTestModelLoadAndPredict(tf.test.TestCase):
    def test_MTA(self):
        model = MTA((160, 384, 3), 1)
        model.build_model()
        model.summary()

        # loading checkpoints test
        weight_path = osp.join(PROJECT_ROOT_DIR, 'ckpts/MTA/ckpt')
        model.load_weights(weight_path=weight_path)

        # prediction test
        output = model(*get_dummy_inputs())
        print(output.keys())

        del model

    def test_MT(self):
        model = DrivingModule((160, 384, 3))
        model.build_model()
        model.summary()

        # loading checkpoints test
        weight_path = osp.join(PROJECT_ROOT_DIR, 'ckpts/MT/ckpt')
        model.load_weights(weight_path=weight_path)

        # prediction test
        output = model(*get_dummy_inputs())
        print(output.keys())

        del model

    def test_CILRS(self):
        model = CILRS((160, 384, 3))
        model.build_model()
        model.summary()

        # loading checkpoints test
        weight_path = osp.join(PROJECT_ROOT_DIR, 'ckpts/CILRS/ckpt')
        model.load_weights(weight_path=weight_path)

        # prediction test
        output = model(*get_dummy_inputs())
        print(output.keys())

        del model

    def test_CILRS(self):
        model = Baseline((160, 384, 3), 1)
        model.build_model()
        model.summary()

        # loading checkpoints test
        weight_path = osp.join(PROJECT_ROOT_DIR, 'ckpts/baseline/ckpt')
        model.load_weights(weight_path=weight_path)

        # prediction test
        output = model(*get_dummy_inputs())
        print(output.keys())

        del model


if __name__ == '__main__':
    tf.test.main()
