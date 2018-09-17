# coding: utf-8
from collections import deque
import random
import numpy as np

class ReplayBuffer():
    def __init__(self, buffer_size, random_seed=123):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, *args):
        if self.count < self.buffer_size:
            self.buffer.append(args)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(args)

    def __len__(self):
        return self.count

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            # return all contents
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        batch = zip(*batch)
        return batch

    def reset(self):
        self.buffer.clear()
        self.count = 0
