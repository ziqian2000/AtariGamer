from collections import namedtuple

import numpy as np
import tensorflow as tf

Transition = namedtuple("transition", ('state', 'action', 'reward', 'next_state', 'terminated'))


def ls(v): return v * 2 + 1


def rs(v): return v * 2 + 2


def par(v): return (v - 1) >> 1


class SumTree:

    def __init__(self, capacity):
        assert capacity % 2 == 0  # claim a full binary tree for convenience
        self.capacity = capacity
        self.sum = np.zeros(self.capacity * 2 - 1, dtype=np.float32)
        self.data = [None] * self.capacity
        self.idx = 0

    def push(self, transition, p):
        self.data[self.idx] = transition
        self.push_up(self.idx + self.capacity - 1, p)
        self.idx = (self.idx + 1) % self.capacity

    def push_up(self, pos, new_p):
        """
        push up
        :param pos: index in tree, must be a leaf, (capacity - 1) ~ (capacity * 2 - 2)
        :param new_p: new abs error
        """
        delta = new_p - self.sum[pos]
        self.sum[pos] += delta
        while pos > 0:
            pos = par(pos)
            self.sum[pos] += delta

    def is_leaf(self, v):
        return ls(v) >= self.capacity * 2 - 1

    def get(self, v):

        assert v < self.sum_p

        cur_idx = 0
        while True:
            if self.is_leaf(cur_idx):
                idx = cur_idx
                break

            if v <= self.sum[ls(cur_idx)]:
                cur_idx = ls(cur_idx)
            else:
                v -= self.sum[ls(cur_idx)]
                cur_idx = rs(cur_idx)

        return idx, self.sum[idx], self.data[idx - self.capacity + 1]

    @property
    def max_p(self):
        return np.max(self.sum[-self.capacity:])

    @property
    def min_p(self):
        return np.min(self.sum[-self.capacity:])

    @property
    def sum_p(self):
        return self.sum[0]


class PrioritizedReplayMemory:

    def __init__(self, minibatch_size, memory_size, history_len):

        self.epsilon = 1e-2  # avoid 0 priority
        self.alpha = 0.6
        self.beta0 = 0.4
        self.beta = self.beta0
        self.delta_beta = (1 - self.beta0) / (1000000 / 4)
        self.abs_err_upper = 1  # clipping

        self.minibatch_size = minibatch_size
        self.memory_size = memory_size
        self.history_len = history_len
        self.tree = SumTree(memory_size)
        self.cnt = 0

        # state and next_state will use uint8 (8 bit = 1 Byte)
        # action uses int32 (32 bit = 4 Byte)
        # reward uses float32 (32 bit = 4 Byte)
        # terminal uses boolean (8 bit = 1 Byte (numpy))
        total_est_mem = (self.memory_size * (84 * 84 * 4 * 2 + 4 + 4 + 1) + (2 * self.memory_size - 1) * 4) / 1024.0 ** 3

        print("- Estimated memory usage for replay memory: {:.4f} GB.".format(total_est_mem))

    def push(self, state, action, reward, next_state, terminated):
        transition = Transition(state, action, reward, next_state, terminated)
        max_p = self.tree.max_p
        if max_p == 0:
            max_p = self.abs_err_upper

        self.tree.push(transition, max_p)

        self.cnt += 1
        if self.cnt == self.memory_size:
            print("- Replay memory is full.")

    def sample(self):

        assert self.cnt > 0

        idx_batch = []
        imp_samp_weight_batch = []
        p_batch = []
        trans_batch = []

        interval_len = self.tree.sum_p / self.minibatch_size
        for i in range(self.minibatch_size):
            v = np.random.uniform(interval_len * i, interval_len * (i + 1))
            idx, p, transition = self.tree.get(v)

            idx_batch.append(idx)
            p_batch.append(p)
            trans_batch.append(transition)

        min_p = np.min(p_batch)
        for p in p_batch:
            imp_samp_weight_batch.append(np.power(p / min_p, -self.beta))

        self.beta = np.minimum(self.beta + self.delta_beta, 1.0)
        return self.unpack(trans_batch), idx_batch, imp_samp_weight_batch

    def update(self, idx_batch, abs_error_batch):
        abs_error_batch += self.epsilon  # avoid 0 priority
        clipped_abs_error = np.minimum(abs_error_batch, self.abs_err_upper)
        powered_abs_error = np.power(clipped_abs_error, self.alpha)
        for idx, new_p in zip(idx_batch, powered_abs_error):
            self.tree.push_up(idx, new_p)

    def unpack(self, trans_list):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminated_batch = []
        for transition in trans_list:
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
