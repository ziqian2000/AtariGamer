from collections import deque
from datetime import datetime

import imageio
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop

from src.DQN.network import Network
from src.memory import PrioritizedReplayMemory
from src.memory.replay_memory import ReplayMemory
from src.common import AtariEnvironment


class AgentForTesting:

    def __init__(self, env_id, load_path):
        # fake environment
        self.env_id = env_id
        self.env = AtariEnvironment(self.env_id, 0)

        # parameters
        self.action_num = self.env.get_action_num()
        self.history_len = 4

        # network & structure
        self.main_network = Network(action_num=self.action_num, history_len=self.history_len)
        self.frames = deque([], maxlen=self.history_len)

        # load
        loaded_checkpoints = tf.train.latest_checkpoint(load_path)
        self.main_network.load_weights(loaded_checkpoints)

    @tf.function
    def get_action(self, state, exploration_rate):
        """
        get action by ε-greedy algorithm
        :param state: current state
        :param exploration_rate: current exploration rate
        :return: action, an integer
        """
        if tf.random.uniform((), minval=0, maxval=1, dtype=tf.float32) < exploration_rate:  # explore: randomly choose action
            action = tf.random.uniform((), minval=0, maxval=self.action_num, dtype=tf.int32)
        else:
            q_value = self.main_network(tf.cast(tf.expand_dims(state, axis=0), tf.float32))
            action = tf.cast(tf.squeeze(tf.argmax(q_value, axis=1)), dtype=tf.int32)
        return action

    def act(self, cur_state):
        self.frames.append(cur_state)

        if len(self.frames) < self.history_len:  # not full
            return 0

        state = tf.transpose(np.concatenate(self.frames, axis=0), [1, 2, 0])
        return self.get_action(tf.constant(state, dtype=tf.uint8), tf.constant(0.0, dtype=tf.float32))


