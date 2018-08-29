# coding: utf-8
import numpy as np
from chainer import serializers 

from constants import *
from utils import Logger 
from actor_critic import Actor_critic

class Trainer():
    def __init__(self, agent, logger):
        self.time = 0
        self.ep_finished = 0
        self.num_ep = NUM_EP
    
    def run(self):
        for ep in range(self.ep_finished, self.num_ep):
            s, r = ENV.reset(), 0
            agent.reset()
            ep_reward = 0
            ep_loss = 0

            for ep_time in range(NUM_EP_STEP):
                a = agent.step(s, r)
                s, r, done, info = ENV.step(a)
                ep_reward += r 
                self.time += 1

                if done:
                    loss = agent.update()
                    ep_loss += loss
                    agent.reset()
                    break 
                elif (ep_time+1) % TRAIN_INTERVAL == 0:
                    loss = agent.update()
                    ep_loss += loss
                    agent.reset()

            if ep % 100 == 0:
                print('episode {}: reward={}, loss={}'.format(ep+1, ep_reward, ep_loss))
            logger.add(reward=ep_reward, loss=ep_loss)
            self.ep_finished += 1

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
