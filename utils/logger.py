# coding: utf-8
from pathlib import Path
from datetime import datetime
from more_itertools import chunked

import numpy as np
from matplotlib import pyplot as plt 
from chainer import serializers

class Logger():
    def __init__(self, env_name, agent):
        self.logroot = Path('logs') / env_name
        self.logdir = self.logroot / datetime.now().isoformat()[:19].replace(':', '-')
        self.logdir.mkdir(parents=True, exist_ok=True)

        self.data = {}
        self.max_params_size = 10000 #TODO
        self.params_count = 0
        self.params = self._init_params(agent)

    def add_data(self, **args):
        for name, val in args.items():
            if name in self.data.keys():
                self.data[name].append(val)
            else:
                self.data[name] = [val]

    def visualize_data(self, avg_step=1):
        for name, array in self.data.items():
            plt.plot(array)
            if avg_step > 1:
                moving_avg = np.convolve(array, np.ones(avg_step)/avg_step, mode='valid')
                plt.plot(moving_avg)
            plt.xlabel('episodes')
            plt.ylabel(name.replace('_', ' '))
            plt.title(name.replace('_', ' '))
            plt.savefig(str(self.logdir / '{}.png'.format(name)))
            plt.show()

    def _init_params(self, agent):
        params = dict()
        for name, param in agent.namedparams():
            params[name] = np.zeros((self.max_params_size, param.size))
        return params

    def add_params(self, agent):
        for name, param in agent.namedparams():
            assert name in self.params.keys()
            self.params[name][self.params_count] = param.data.reshape(-1)
        self.params_count += 1
        if self.params_count == self.max_params_size - 1:
            raise(NotImplementedError)
            #TODO
            
    def visualize_params(self):
        sorted_params = sorted(self.params.items(), key=lambda x: x[0])
        #print(self.params.keys())
        for (b_name, b), (w_name, w) in chunked(sorted_params, 2):
            # 転置して同じ成分の時系列が行に並ぶようにする
            b = b[:self.params_count, :].T
            plt.subplot(121)
            for elm in b:
                plt.plot(elm)
            plt.title(b_name.replace('/', ' '))
            plt.xlabel('episodes')

            w = w[:self.params_count, :].T
            plt.subplot(122)
            for elm in w:
                plt.plot(elm)
            plt.title(w_name.replace('/', ' '))
            plt.xlabel('episodes')

            plt.savefig(str(self.logdir / '{}.png'.format(w_name.replace('/', '_'))))
            #plt.show()

    def save_model(self, agent):
        serializers.save_npz(str(self.logdir / 'model.npz'), agent)

    def load_model(self, agent, dir_name):
        path = self.logroot / dir_name / 'model.npz'
        serializers.load_npz(path, agent)