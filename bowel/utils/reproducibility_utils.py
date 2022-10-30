import os
import random
import tensorflow as tf
import numpy as np


def setup_seed():
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    random.seed(4)
    np.random.seed(4)
    tf.random.set_seed(4)
