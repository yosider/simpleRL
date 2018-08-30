# coding: utf-8
import gym 
import env

# --- Flags
LOAD_MODEL = False 
LOAD_DIR_NAME = '__2018-08-29T12'

TRAINING = True
SAVE_MODEL = True
RENDER_AFTER_TRAINING = True

# --- Environments
ENV_NAME = 'CartPole-v0'
ENV = gym.make(ENV_NAME)
NUM_EP = 10000
NUM_EP_STEP = ENV._max_episode_steps
NUM_EP_RENDER = 5

# --- Dimensions
O_DIM = ENV.observation_space.shape[0]  # 2
A_DIM = ENV.action_space.n  # 1

# --- Learning params
GAMMA = 0.99
TRAIN_INTERVAL = NUM_EP_STEP
EPS = 1e-6
