# coding: utf-8
from chainer import links as L 
from chainer import functions as F
from chainer import Chain, optimizers, optimizer_hooks, grad, initializers

from utils.utils import *
from constants import O_DIM, A_DIM, A_BOUND, S_BOUND, ACTOR_LEARNING_RATE

class Actor(Chain):
    def __init__(self):
        super(Actor, self).__init__(
            h1 = L.Linear(O_DIM, 400),
            h2 = L.Linear(400, 300),
            h3 = L.Linear(300, A_DIM, initialW=initializers.Uniform(scale=0.003)),
        )
        self.optimizer = optimizers.Adam(alpha=ACTOR_LEARNING_RATE)
        self.optimizer.setup(self)
        self.optimizer.add_hook(optimizer_hooks.GradientClipping(2.0))
        
    def __call__(self, s):
        a = F.relu(self.h1(s))
        #a = F.normalize(a)
        a = F.relu(self.h2(a))
        #a = F.normalize(a)
        a = A_BOUND * F.tanh(self.h3(a)) # |a| < A_BOUND
        return a

    def update(self, loss):
        self.cleargrads()
        loss.backward()
        self.optimizer.update()
        loss.unchain_backward()     # no need?
    