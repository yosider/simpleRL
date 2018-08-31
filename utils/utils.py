# coding: utf-8
import numpy as np
from chainer import Variable
from chainer import functions as F

def make_batch(*xs):
    """return: list of batched xs."""
    return [x.reshape(1,-1) if type(x)==Variable else np.array(x, dtype=np.float32).reshape(1,-1) for x in xs]

def make_target_value(rewards, preds, gamma):
    N = len(rewards)
    preds = F.concat((preds, np.zeros((1,1), dtype=np.float32)), axis=0)
    target = F.stack([rewards[i] + gamma*preds[i+1] for i in range(N)]) #(N, 1)
    return target
