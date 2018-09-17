# coding: utf-8
import numpy as np
from chainer import serializers

from constants import *
from utils.logger import Logger
from trainer import Trainer
from ddpg import DDPG


if __name__ == '__main__':
    agent = DDPG()
    logger = Logger(ENV_NAME, agent, LOG_PARAMS)
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

        if SAVE_MODEL:
            logger.save_model(agent)
        if LOG_PARAMS:
            logger.visualize_params()
        if LOG_STATS:
            logger.visualize_data(avg_step=20)
        if RENDER_AFTER_TRAINING:
            trainer.render()
        
        print('>> continue? (y/n)')
        if input() == 'n':
            break
        trainer.num_ep += NUM_EP    # again NUM_EP episodes
        print('training restarted')
        print('WARNING: if you KeyboardInterrupted, don\'t do it again. The training will be killed without any logs. why...:(')