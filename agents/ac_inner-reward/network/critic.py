# coding: utf-8
from chainer import links as L 
from chainer import functions as F
from chainer import Chain 
from constants import *

class Critic(Chain):
    def __init__(self):
        super(Critic, self).__init__(
            h1 = L.Linear(O_DIM, 4*O_DIM),
            h2 = L.Linear(4*O_DIM, 2*O_DIM),
            h3 = L.Linear(2*O_DIM, 1),
        )
        
    def __call__(self, o):
        value = F.relu(self.h1(o))
        value = F.relu(self.h2(value))
        value = F.relu(self.h3(value))
        return value