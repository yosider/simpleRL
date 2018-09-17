# coding: utf-8
import gym 
import envs

# --- Flags
LOAD_MODEL = True
LOAD_DIR_NAME = "__2018-09-17T15-52-33"

TRAINING = False
LOG_STATS = False   # log reward, loss etc.
LOG_PARAMS = False   # log weights timelapse: too slow because of the big networks
SAVE_MODEL = False
RENDER_AFTER_TRAINING = True

# --- Environments
ENV_NAME = 'Pendulum-v0'
ENV = gym.make(ENV_NAME)
NUM_EP = 10000
NUM_EP_STEP = ENV._max_episode_steps+1
NUM_EP_RENDER = 5

# --- Dimensions
O_DIM = ENV.observation_space.shape[0]  # 3
A_DIM = ENV.action_space.shape[0]  # 1
A_BOUND = ENV.action_space.high  # 2
S_BOUND = ENV.observation_space.high

# --- Learning params
# from the paper
TAU = 0.001
BATCH_SIZE = 64
BUFFER_SIZE = 1000000
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
CRITIC_WEIGHT_DECAY = 0.01 # currently disabled in critic.py
GAMMA = 0.99
EPS = 1e-6
TRAIN_INTERVAL = 1 #FIXME
NUM_WARMUP_STEP = BATCH_SIZE*10 #FIXME

