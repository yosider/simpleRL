# coding: utf-8
from chainer import links as L 
from chainer import functions as F
from chainer import Chain, optimizers, optimizer_hooks, initializers, Variable

from constants import O_DIM, A_DIM, S_BOUND, CRITIC_LEARNING_RATE, CRITIC_WEIGHT_DECAY

class Critic(Chain):
    def __init__(self):
        super(Critic, self).__init__(
            h1 = L.Linear(O_DIM, 400),
            h2_s = L.Linear(400, 300, nobias=True),
            h2_a = L.Linear(A_DIM, 300),
            h3 = L.Linear(300, 1, initialW=initializers.Uniform(scale=0.003)),
        )
        self.optimizer = optimizers.Adam(alpha=CRITIC_LEARNING_RATE)
        self.optimizer.setup(self)
        #self.optimizer.add_hook(optimizer_hooks.WeightDecay(CRITIC_WEIGHT_DECAY))
        self.optimizer.add_hook(optimizer_hooks.GradientClipping(2.0))
        
    def __call__(self, s, a):
        s = F.relu(self.h1(s))
        #s = F.normalize(s)
        s = self.h2_s(s)
        #print(F.normalize(a))
        a = self.h2_a(a)
        q = F.relu(s + a)
        q = self.h3(q)
        q.name = 'Critic output'
        return q

    def update(self, loss):
        self.cleargrads()
        loss.backward()
        self.optimizer.update()
        loss.unchain_backward() # no need?