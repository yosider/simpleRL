# coding: utf-8
import numpy as np
from constants import *

class Trainer():
    """ 
    trainer for ddpg:
        models are updated inside the model.
        "get a -> exec a -> observe" loop
    """
    def __init__(self, agent, logger):
        self.agent = agent
        self.logger = logger

        self.time = 0
        self.ep_finished = 0
        self.num_ep = NUM_EP
    
    def run(self):
        for ep in range(self.ep_finished, self.num_ep):
            s = ENV.reset()
            self.agent.reset(s)
            ep_reward = 0
            batch_actor_loss = 0
            batch_critic_loss = 0
            update_count = 0

            for ep_time in range(NUM_EP_STEP):
                a = self.agent.get_action()
                s, r, done, info = ENV.step(a)
                losses = self.agent.step(s, r, done, TRAINING)
                if losses is not None:
                    update_count += 1
                    batch_actor_loss += losses[0]
                    batch_critic_loss += losses[1]

                ep_reward += r
                self.time += 1
                if done:
                    break

            if update_count > 1:
                batch_actor_loss /= update_count
                batch_critic_loss /= update_count
            if LOG_STATS:
                self.logger.add_data(reward=ep_reward, batch_actor_loss=batch_actor_loss, batch_critic_loss=batch_critic_loss)
            if LOG_PARAMS:
                self.logger.add_params(self.agent)
            
            self.ep_finished += 1
            if ep % 2 == 0:
                print('episode {}: reward={:.2e}, actor loss={:.2e}, critic loss={:.2e}'.format(ep, ep_reward, batch_actor_loss, batch_critic_loss))

    def render(self):
        print('rendering the training result...')
        for ep in range(NUM_EP_RENDER):
            s = ENV.reset()
            self.agent.reset(s)
            ep_reward = 0

            for ep_time in range(NUM_EP_STEP):
                ENV.render()
                a = self.agent.get_action()
                s, r, done, info = ENV.step(a)
                a = self.agent.step(s, r, done, False)
                ep_reward += r 
                if done:
                    break
                    
            print('episode {}: reward={}'.format(ep+1, ep_reward))