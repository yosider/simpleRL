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
            s, r = ENV.reset(), 0
            self.agent.reset()
            ep_reward = 0
            ep_loss = 0

            for ep_time in range(NUM_EP_STEP):
                a = self.agent.step(s, r)
                s, r, done, info = ENV.step(a)
                ep_reward += r 
                self.time += 1

                if done:
                    loss = self.agent.update()
                    ep_loss += loss
                    break 
                #elif (ep_time+1) % TRAIN_INTERVAL == 0:
                #    loss, done = self._bootstrap_update()
                #    ep_loss += loss
                #    self.agent.reset()
                #    if done:
                #        break

            if ep % 100 == 0:
                print('episode {}: reward={}, loss={}'.format(ep, ep_reward, ep_loss))
            self.logger.add(reward=ep_reward, loss=ep_loss)
            self.ep_finished += 1

    def _bootstrap_update(self):
        """add bootstrap step. (without time increment)"""
        a = self.agent.step(s, r)
        s, r, done, info = ENV.step(a)
        loss = self.agent.update()
        return loss, done
            
    def render(self):
        print('rendering the training result...')
        for ep in range(NUM_EP_RENDER):
            s, r = ENV.reset(), 0
            agent.reset()
            ep_reward = 0

            for ep_time in range(NUM_EP_STEP):
                ENV.render()
                a = agent.step(s, r)
                s, r, done, info = ENV.step(a)
                ep_reward += r 
                if done:
                    break
                    
            print('episode {}: reward={}'.format(ep+1, ep_reward))


if __name__ == '__main__':
    agent = Actor_critic()
    logger = Logger(ENV_NAME)
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
            logger.visualize(avg_step=100)
        if SAVE_MODEL:
            logger.save_model(agent)
        if RENDER_AFTER_TRAINING:
            trainer.render()
        
        print('>> continue? (y/n)')
        if input() == 'n':
            break
        trainer.num_ep += NUM_EP

