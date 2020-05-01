from collections import namedtuple

import numpy as np
import tensorflow as tf

Transition = namedtuple("transition", ('state', 'action', 'reward', 'next_state', 'terminated'))


class ReplayMemory:

    def __init__(self, minibatch_size, memory_size, history_len):
        self.minibatch_size = minibatch_size
        self.memory_size = memory_size
        self.history_len = history_len

        self.memory = [None] * memory_size
        self.index = 0
        self.is_full = False

        # state and next_state will use uint8 (8 bit = 1 Byte)
        # action uses int32 (32 bit = 4 Byte)
        # reward uses float32 (32 bit = 4 Byte)
        # terminal uses boolean (8 bit = 1 Byte (numpy))
        total_est_mem = self.memory_size * (84 * 84 * 4 * 2 + 4 + 4 + 1) / 1024.0 ** 3
        print("- Estimated memory usage for replay memory: {:.4f} GB.".format(total_est_mem))

    def push(self, state, action, reward, next_state, terminated):
        transition = Transition(state, action, reward, next_state, terminated)
        if not self.is_full and self.index + 1 == self.memory_size:
            self.is_full = True
            print("- Replay memory is full.")
        self.memory[self.index] = transition
        self.index = (self.index + 1) % self.memory_size

    def get_minibatch_indices(self):
        indices = []
        while len(indices) < self.minibatch_size:
            idx = np.random.randint(low=self.history_len,
                                    high=self.memory_size if self.is_full else self.index,
                                    dtype=np.int32)
            if not np.any([sample.terminated for sample in self.memory[idx - self.history_len : idx]]):
                indices.append(idx)
        return indices

    def get_minibatch_sample(self, indices):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminated_batch = []
        for idx in indices:
            transition = self.memory[idx]
            state_batch.append(tf.constant(transition.state, dtype=tf.uint8))
            action_batch.append(tf.constant(transition.action, dtype=tf.int32))
            reward_batch.append(tf.constant(transition.reward, dtype=tf.float32))
            next_state_batch.append(tf.constant(transition.next_state, dtype=tf.uint8))
            terminated_batch.append(tf.constant(transition.terminated, dtype=tf.float32))
        return tf.stack(state_batch), \
               tf.stack(action_batch), \
               tf.stack(reward_batch), \
               tf.stack(next_state_batch), \
               tf.stack(terminated_batch)
