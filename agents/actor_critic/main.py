# coding: utf-8
import numpy as np
from chainer import serializers 

from constants import *
from utils.logger import Logger
from utils.trainer import Trainer
from actor_critic import Actor_critic


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

