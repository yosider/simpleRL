# coding: utf-8
import numpy as np
from chainer import serializers 

from constants import *
from utils.logger import Logger 
from actor_critic import Actor_critic

class Trainer():
    def __init__(self, agent, logger):
        self.agent = agent
        self.logger = logger
        self.time = 0
        self.ep_finished = 0
        self.num_ep = NUM_EP
    
    def run(self):
        for ep in range(self.ep_finished, self.num_ep):
            s = ENV.reset()
            a = self.agent.reset(s)
            ep_reward = 0
            ep_actor_loss = 0
            ep_critic_loss = 0

            for ep_time in range(NUM_EP_STEP):
                s, r, done, info = ENV.step(a)
                a = self.agent.step(s, r)
                ep_reward += r 
                self.time += 1

                if done:
                    actor_loss, critic_loss = self.agent.update()
                    ep_actor_loss += actor_loss
                    ep_critic_loss += critic_loss
                    break
                #elif (ep_time+1) % TRAIN_INTERVAL == 0:
                #    actor_loss, critic_loss, done = self._bootstrap_update()
                #    ep_loss += loss
                #    self.agent.reset()
                #    if done:
                #        break

            if ep % 100 == 0:
                print('episode {}: reward={}, actor loss={}, critic loss={}'.format(ep, ep_reward, ep_actor_loss, ep_critic_loss))
            self.logger.add_data(reward=ep_reward, actor_loss=ep_actor_loss, critic_loss=ep_critic_loss)
            self.logger.add_params(self.agent)
            self.ep_finished += 1

    def _bootstrap_update(self):
        """add bootstrap step. (without time increment)"""
        a = self.agent.step(s, r)
        s, r, done, info = ENV.step(a)
        actor_loss, critic_loss = self.agent.update()
        return actor_loss, critic_loss, done
            
    def render(self):
        print('rendering the training result...')
        for ep in range(NUM_EP_RENDER):
            s = ENV.reset()
            a = agent.reset(s)
            ep_reward = 0

            for ep_time in range(NUM_EP_STEP):
                ENV.render()
                s, r, done, info = ENV.step(a)
                a = agent.step(s, r)
                ep_reward += r 
                if done:
                    break
                    
            print('episode {}: reward={}'.format(ep+1, ep_reward))


if __name__ == '__main__':
    agent = Actor_critic()
    logger = Logger(ENV_NAME, agent)
    trainer = Trainer(agent, logger)
    if LOAD_MODEL:
        logger.load_model(agent, LOAD_DIR_NAME)

    while(1):
        try:
            if TRAINING:
                trainer.run()
        except:
            # print the exceptions but don't exit
            import traceback
            traceback.print_exc()

        if TRAINING:
            logger.visualize_params()
            logger.visualize_data(avg_step=100)
        if SAVE_MODEL:
            logger.save_model(agent)
        if RENDER_AFTER_TRAINING:
            trainer.render()
        
        print('>> continue? (y/n)')
        if input() == 'n':
            break
        trainer.num_ep += NUM_EP

