# coding: utf-8
from pathlib import Path
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt 
from chainer import serializers

class Logger():
    def __init__(self, name):
        self.data = {}
        self.logroot = Path('log') / name
        self.logdir = self.logroot / datetime.now().isoformat()[:19].replace(':', '-')
        self.logdir.mkdir(parents=True, exist_ok=True)

    def add(self, **args):
        for name, val in args.items():
            if name in self.data.keys():
                self.data[name].append(val)
            else:
                self.data[name] = [val]

    def visualize(self, avg_step=1):
        #if time_axis in self.data.keys():
        #   time = self.data.pop(time_axis)

        for name, array in self.data.items():
            if avg_step > 1:
                moving_avg = np.convolve(array, np.ones(avg_step)/avg_step, mode='same')
                plt.plot(array)
                plt.plot(moving_avg)
            plt.xlabel('timestep (*{} times)'.format(avg_step))
            plt.ylabel(name.replace('_', ' '))
            plt.title(name.replace('_', ' '))
            plt.savefig(str(self.logdir / '{}.png'.format(name)))
            plt.show()

    def save_model(self, agent):
        serializers.save_npz(str(self.logdir / 'model.npz'), agent)

    def load_model(self, agent, dir_name):
        path = self.logroot / dir_name / 'model.npz'
        serializers.load_npz(path, agent)