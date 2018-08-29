# coding: utf-8
from chainer import links as L 
from chainer import functions as F 
from chainer import Chain, optimizers 

from constants import * 
from utils import *
from networks.actor import Actor 
from networks.critic import Critic

class Actor_critic(Chain):
    def __init__(self):
        super(Actor_critic, self).__init__(
            actor = Actor(),
            critic = Critic(),
        )
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self)

    def reset(self):
        self.obs = []
        self.rewards = []
        self.log_pies = []

    def step(self, o, r):
        O, = make_batch(o)
        log_pi, a = self.actor(O)
        a_id = np.where(a==1)[0][0]   # action index
        log_pi = log_pi[0][a_id]
        self.obs.append(o)
        self.rewards.append(r)
        self.log_pies.append(log_pi)
        return a_id

    def update(self):
        obs = F.stack(np.array(self.obs, dtype=np.float32))
        Vs = self.critic(obs)
        log_pies = F.stack(self.log_pies).reshape(1, -1)[:,:-1] #(1,5)
        As = F.stack([self.rewards[i] + GAMMA*Vs[i+1] - Vs[i] for i in range(len(Vs)-1)]) #(5,1)
        loss = -F.matmul(log_pies, As)
        
        self.cleargrads()
        loss.backward()
        self.optimizer.update()
        loss.unchain_backward()
        return loss.data[0][0]

        
        
        
        
