# coding: utf-8
import numpy as np
from chainer import Variable

def make_batch(*xs):
    """return: list of batched xs."""
    return [x.reshape(1,-1) if type(x)==Variable else np.array(x, dtype=np.float32).reshape(1,-1) for x in xs]