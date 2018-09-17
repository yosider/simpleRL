# coding: utf-8
import numpy as np
from chainer import Variable
from chainer import functions as F
from chainer import links as L

def make_batch(*xs):
    """return: list of batched xs."""
    return [x.reshape(1,-1) if type(x)==Variable else np.array(x, dtype=np.float32).reshape(1,-1) for x in xs]

def make_sample_input(*size):
    r = np.random.randn(*size).astype(np.float32)
    r = np.expand_dims(r, axis=0)
    return Variable(r)

def make_target_value(rewards, preds, gamma):
    N = len(rewards)
    preds = F.concat((preds, np.zeros((1,1), dtype=np.float32)), axis=0)
    target = F.stack([rewards[i] + gamma*preds[i+1] for i in range(N)]) #(N, 1)
    return target


# From https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# From https://github.com/chainer/chainerrl/blob/master/chainerrl/misc/copy_param.py
def soft_copy_param(target_link, source_link, tau):
    """Soft-copy parameters of a link to another link."""
    target_params = dict(target_link.namedparams())
    for param_name, param in source_link.namedparams():
        if target_params[param_name].data is None:
            raise TypeError(
                'target_link parameter {} is None. Maybe the model params are '
                'not initialized.\nPlease try to forward dummy input '
                'beforehand to determine parameter shape of the model.'.format(
                    param_name))
        target_params[param_name].data[:] *= (1 - tau)
        target_params[param_name].data[:] += tau * param.data

    # Soft-copy Batch Normalization's statistics
    target_links = dict(target_link.namedlinks())
    for link_name, link in source_link.namedlinks():
        if isinstance(link, L.BatchNormalization):
            target_bn = target_links[link_name]
            target_bn.avg_mean[:] *= (1 - tau)
            target_bn.avg_mean[:] += tau * link.avg_mean
            target_bn.avg_var[:] *= (1 - tau)
            target_bn.avg_var[:] += tau * link.avg_var


# From https://github.com/chainer/chainerrl/blob/master/chainerrl/agents/ddpg.py
def disable_train(chain):
    call_orig = chain.__call__

    def call_test(self, x):
        with chainer.using_config('train', False):
            return call_orig(self, x)

    chain.__call__ = call_test