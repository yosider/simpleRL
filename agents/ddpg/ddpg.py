# coding: utf-8
import numpy as np
from copy import deepcopy

from chainer import links as L 
from chainer import functions as F 
from chainer import Chain, optimizers, Variable 

from constants import * 
from utils.utils import make_batch, OrnsteinUhlenbeckActionNoise, soft_copy_param, disable_train
from utils.buffer import ReplayBuffer
from networks.actor import Actor 
from networks.critic import Critic

class DDPG(Chain):
    def __init__(self):
        super(DDPG, self).__init__(
            actor = Actor(),
            critic = Critic(),
        )
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)
        disable_train(self.target_actor)
        disable_train(self.target_critic)

        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(A_DIM))
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.time = 0

    def reset(self, s):
        self.prev_s = s
        self.noise.reset()

    def step(self, s, r, done, trainable):
        self.time += 1
        self.buffer.add(self.prev_s, self.prev_a, r, done, s, self.prev_noise)
        self.prev_s = s
        if trainable and self.time % TRAIN_INTERVAL == 0:
            if len(self.buffer) > NUM_WARMUP_STEP:
                return self._update()

    def get_action(self):
        S, = make_batch(self.prev_s)
        a = self.actor(S)[0]    # (A_DIM, )
        noise = self.noise().astype(np.float32)
        self.prev_a = a 
        self.prev_noise = noise
        return (a+noise).data.reshape(-1)

    def _update(self):
        S, A, R, D, S2, N = self.buffer.sample_batch(BATCH_SIZE)  # (6, BATCH_SIZE)
        S = np.array(S, dtype=np.float32)   # (BATCH_SIZE, O_DIM)
        S2 = np.array(S2, dtype=np.float32)
        A = F.stack(A)    # (BATCH_SIZE, A_DIM)
        R = np.array(R, dtype=np.float32).reshape(-1, 1)
        N = np.array(N)

        # update critic
        A_ = self.target_actor(S2)
        Y = R + GAMMA * self.target_critic(S2, A_.data)
        Q_batch = self.critic(S, (A+N).data)
        critic_loss = F.mean_squared_error(Y.data, Q_batch)
        self.critic.update(critic_loss)

        # update actor
        A = self.actor(S) # why?? but essential!!
        Q = self.critic(S, A)
        actor_loss = -F.sum(Q) / BATCH_SIZE
        #from chainer import computational_graph as c
        #g = c.build_computational_graph([actor_loss])
        #with open('graph_actorloss.dot', 'w') as o:
        #    o.write(g.dump())
        #exit()
        self.actor.update(actor_loss)

        # update target
        soft_copy_param(self.target_critic, self.critic, TAU)
        soft_copy_param(self.target_actor, self.actor, TAU)

        return actor_loss.data, critic_loss.data