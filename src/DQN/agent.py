from collections import deque
from datetime import datetime

import imageio
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from src.DQN.network import Network
from src.memory import PrioritizedReplayMemory
from src.memory.replay_memory import ReplayMemory
from src.common import AtariEnvironment


class Agent:

    def __init__(self, env_id, debug):
        # hyper-parameters
        self.discount_factor = 0.99
        self.minibatch_size = 32
        self.update_frequency = 4
        self.target_network_update_frequency = 3000
        self.history_len = 4
        self.memory_size = 130000 if not debug else 1000
        self.init_explr = 1.0
        self.final_explr = 0.1
        self.final_explr_frame = 1000000
        self.replay_start_size = 50000 if not debug else 500
        self.training_frames = int(1e7)
        self.learning_rate = 0.00025

        # frames limit
        self.fps = 30
        self.max_playing_time = 10  # minutes
        self.total_frames_limit = self.fps * 60 * self.max_playing_time

        # environment
        self.env_id = env_id
        self.env = AtariEnvironment(self.env_id, self.total_frames_limit)

        # other parameters
        self.action_num = self.env.get_action_num()
        self.latest_record_num = 100
        self.print_info_interval = 10 if not debug else 1
        self.save_weight_interval = 100 if not debug else 10
        self.play_interval = 100 if not debug else 10

        # network
        self.memory = PrioritizedReplayMemory(minibatch_size=self.minibatch_size, memory_size=self.memory_size, history_len=self.history_len)
        self.main_network = Network(action_num=self.action_num, history_len=self.history_len)
        self.target_network = Network(action_num=self.action_num, history_len=self.history_len)
        self.optimizer = Adam(lr=self.learning_rate, epsilon=1e-6)
        self.loss = tf.keras.losses.Huber()
        self.loss_metric = tf.keras.metrics.Mean()
        self.q_metric = tf.keras.metrics.Mean()

        # other tools (log, summary)
        self.log_path = "./log/" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + self.env_id

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

    @tf.function
    def get_explr(self, frames):
        """
        get exploration rate using linear annealing
        todo: optimization with better annealing method
        :param frames: the number of frames passed
        :return: exploration rate, an integer
        """

        if frames < self.replay_start_size:
            explr = self.init_explr
        elif frames < self.final_explr_frame:
            explr = self.init_explr + (self.final_explr - self.init_explr) / (self.final_explr_frame - self.replay_start_size) * (frames - self.replay_start_size)
        else:
            explr = self.final_explr
        return explr

    @tf.function
    def update_main_network(self, state_batch, action_batch, reward_batch, next_state_batch, terminated_batch, weight_batch):
        """
        update main Q network by experience replay
        :param weight_batch: importance sampling weight
        :param state_batch: batch of states
        :param action_batch: batch of actions
        :param reward_batch: batch of rewards
        :param next_state_batch: batch of next states
        :param terminated_batch: batch of whether it is terminated
        :return: Huber loss
        """
        with tf.GradientTape() as tape:
            next_state_q = self.target_network(next_state_batch)
            next_state_max_q = tf.reduce_max(next_state_q, axis=1)
            expected_q = reward_batch + self.discount_factor * next_state_max_q * (1.0 - tf.cast(terminated_batch, tf.float32))

            main_q = tf.reduce_sum(self.main_network(state_batch) * tf.one_hot(action_batch, self.action_num, on_value=1.0, off_value=0.0), axis=1)
            loss = self.loss(tf.stop_gradient(expected_q), main_q, weight_batch)
        gradients = tape.gradient(loss, self.main_network.trainable_variables)
        clipped_gradients = [tf.clip_by_norm(grad, 10) for grad in gradients]
        self.optimizer.apply_gradients(zip(clipped_gradients, self.main_network.trainable_variables))

        self.loss_metric.update_state(loss)
        self.q_metric.update_state(main_q)
        return tf.abs(main_q - expected_q)

    @tf.function
    def update_target_network(self):
        """
        synchronize weights of target network with main network
        """
        main_weights = self.main_network.trainable_variables
        target_weights = self.target_network.trainable_variables
        for main_v, target_v in zip(main_weights, target_weights):
            target_v.assign(main_v)

    def train(self):
        frames = 0
        episodes = 0
        latest_scores = deque(maxlen=self.latest_record_num)

        while frames < self.training_frames:

            cur_state = self.env.reset()
            episode_reward = 0
            terminated = False

            while not terminated:
                explr = self.get_explr(tf.constant(frames, dtype=tf.float32))
                action = self.get_action(tf.constant(cur_state, dtype=tf.uint8), tf.constant(explr, dtype=tf.float32))

                next_state, reward, terminated, _ = self.env.step(action)
                episode_reward += reward

                self.memory.push(cur_state, action, reward, next_state, terminated)
                cur_state = next_state

                if frames > self.replay_start_size:
                    if frames % self.update_frequency == 0:
                        (state_batch, action_batch, reward_batch, next_state_batch, terminated_batch), \
                            ptr_batch, imp_samp_weight_batch = self.memory.sample()

                        abs_error_batch = self.update_main_network(state_batch, action_batch, reward_batch, next_state_batch, terminated_batch, tf.expand_dims(imp_samp_weight_batch, -1))
                        self.memory.update(ptr_batch, abs_error_batch)

                    if frames % self.target_network_update_frequency == 0:
                        self.update_target_network()

                frames += 1

                if terminated:
                    latest_scores.append(episode_reward)
                    episodes += 1

                    if episodes % self.print_info_interval == 0:
                        print("Episode: {}\t Latest {} average score: {:.2f}\t Progress: {} / {} ( {:.2f} % )"
                              .format(episodes,
                                      self.latest_record_num, np.mean(latest_scores),
                                      frames, self.training_frames, frames / self.training_frames * 100))
                    if episodes % self.save_weight_interval == 0:
                        print("Weights saving...", end="")
                        self.main_network.save_weights(self.log_path + "/episode_{}".format(episodes))
                        print("Done!")
                    if episodes % self.play_interval == 0:
                        self.play(self.log_path, 5)

    def play(self, load_path, trials):
        loaded_checkpoints = tf.train.latest_checkpoint(load_path)
        self.main_network.load_weights(loaded_checkpoints)

        env = AtariEnvironment(self.env_id, self.total_frames_limit)
        reward_list = []
        frame_list = []

        for t in range(trials):

            cur_state = env.reset()
            frames = []
            episode_reward = 0
            terminated = False

            while not terminated:
                frames.append(env.render())
                action = self.get_action(tf.constant(cur_state, dtype=tf.uint8), tf.constant(0.0, dtype=tf.float32))

                next_state, reward, terminated, _ = env.step(action)
                episode_reward += reward

                cur_state = next_state

            reward_list.append(episode_reward)
            frame_list.append(frames)

        print("Scores on {} trials: ".format(trials), reward_list)
        print("Highest score: ", np.max(reward_list))
        print("Average score: ", np.mean(reward_list))
        best_idx = int(np.argmax(reward_list))
        imageio.mimsave(self.env_id + ".gif", frame_list[best_idx], fps=self.fps)