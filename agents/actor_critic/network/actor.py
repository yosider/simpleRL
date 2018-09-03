# coding: utf-8
import numpy as np
from chainer import links as L 
from chainer import functions as F
from chainer import Chain 
from constants import *

class Actor(Chain):
    def __init__(self):
        super(Actor, self).__init__(
            h1 = L.Linear(O_DIM, 4*O_DIM),
            h2 = L.Linear(4*O_DIM, 2*O_DIM),
            h3 = L.Linear(2*O_DIM, A_DIM),
        )
        
    def __call__(self, o):
        log_pi = F.relu(self.h1(o))
        log_pi = F.relu(self.h2(log_pi))
        log_pi = F.log_softmax(self.h3(log_pi))
        probs = F.exp(log_pi)[0]

        # avoid "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
        diff = sum(probs.data[:-1]) - 1
        if diff > 0:
            probs -= (diff + EPS) / A_DIM

        a = np.random.multinomial(1, probs.data).astype(np.float32)
        return log_pi, a