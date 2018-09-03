# coding: utf-8
from chainer import links as L 
from chainer import functions as F 
from chainer import Chain, optimizers 

from constants import * 
from utils.utils import *
from network.actor import Actor 
from network.critic import Critic

class Actor_critic(Chain):
    def __init__(self):
        super(Actor_critic, self).__init__(
            actor = Actor(),
            critic = Critic(),
        )
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self)

    def reset(self, s):
        a_id, log_pi = self._get_action(s)
        self.states = []
        self.rewards = []
        self.log_pies = []
        return a_id

    def step(self, s, r):
        a_id, log_pi = self._get_action(s)
        self.states.append(s)
        self.rewards.append(r)
        self.log_pies.append(log_pi)
        return a_id

    def _get_action(self, s):
        S, = make_batch(s)
        log_pi, a = self.actor(S)
        a_id = np.where(a==1)[0][0]   # action index
        log_pi = log_pi[0][a_id]
        return a_id, log_pi

    def update(self):
        states = F.stack(np.array(self.states, dtype=np.float32))  #(*, 1)
        V_pred = self.critic(states)
        V_target = make_target_value(self.rewards, V_pred, GAMMA)
        critic_loss = F.mean_squared_error(V_pred, V_target)

        log_pies = F.stack(self.log_pies).reshape(1, -1)  #(1, *)
        As_stopgrad = (V_target - V_pred).data  #(*, 1)
        actor_loss = -F.matmul(log_pies, As_stopgrad)
        
        self.cleargrads()
        actor_loss.backward()
        critic_loss.backward()
        self.optimizer.update()
        actor_loss.unchain_backward()
        critic_loss.unchain_backward()

        return actor_loss.data[0][0], critic_loss.data

            