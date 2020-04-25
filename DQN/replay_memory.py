from collections import namedtuple

import numpy as np
import tensorflow as tf

Item = namedtuple("Item", ('state', 'action', 'reward', 'next_state', 'is_terminated'))


class ReplayMemory:

    def __init__(self, minibatch_size, memory_size, history_len):
        self.minibatch_size = minibatch_size
        self.memory_size = memory_size
        self.history_len = history_len

        self.memory = [None] * memory_size
        self.index = 0
        self.is_full = False

    def push(self, state, action, reward, next_state, is_terminated):
        item = Item(state, action, reward, next_state, is_terminated)
        if not self.is_full and self.index + 1 == self.memory_size:
            self.is_full = True
        self.memory[self.index] = item
        self.index = (self.index + 1) % self.memory_size

    def get_minibatch_indices(self):
        indices = []
        while len(indices) < self.minibatch_size:
            idx = np.random.randint(low=self.history_len,
                                    high=1 + (self.memory_size if self.is_full else self.index),
                                    dtype=np.int32)
            if not np.any([sample.is_terminated for sample in self.memory[idx - self.history_len:idx]]):
                indices.append(idx - 1)
        return indices

    def get_minibatch_sample(self, indices):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        is_terminated_batch = []
        for idx in indices:
            item = self.memory[idx]
            state_batch.append(tf.constant(item.state, dtype=tf.float32))
            action_batch.append(tf.constant(item.action, dtype=tf.float32))
            reward_batch.append(tf.constant(item.reward, dtype=tf.float32))
            next_state_batch.append(tf.constant(item.next_state, dtype=tf.float32))
            is_terminated_batch.append(tf.constant(item.is_terminated, dtype=tf.float32))
        return tf.stack(state_batch), \
               tf.stack(action_batch), \
               tf.stack(reward_batch), \
               tf.stack(next_state_batch), \
               tf.stack(is_terminated_batch)
