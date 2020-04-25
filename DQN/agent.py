from collections import deque
from datetime import datetime

import imageio
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from DQN.network import Network
from DQN.replay_memory import ReplayMemory
from common.atari_env import AtariEnvironment


class Agent:

    def __init__(self, env_id):
        # hyper-parameters
        self.discount_factor = 0.99
        self.minibatch_size = 32
        self.update_frequency = 4
        self.target_network_update_frequency = 1000
        self.history_len = 4
        self.memory_size = 10000
        self.init_explr = 1.0
        self.final_explr = 0.1
        self.final_explr_frame = 1000000
        self.replay_start_size = 10000
        self.training_frames = int(1e7)

        # environment
        self.env_id = env_id
        self.env = AtariEnvironment(self.env_id)

        # common parameters
        self.action_num = self.env.get_action_num()
        self.save_weight_interval = 50
        self.latest_record_num = 100
        self.max_playing_time = 10

        # network
        self.memory = ReplayMemory(minibatch_size=self.minibatch_size, memory_size=self.memory_size, history_len=self.history_len)
        self.main_network = Network(action_num=self.action_num, history_len=self.history_len)
        self.target_network = Network(action_num=self.action_num, history_len=self.history_len)
        self.optimizer = Adam(lr=1e-4, epsilon=1e-6)
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
    def update_main_network(self, state_batch, action_batch, reward_batch, next_state_batch, terminated_batch):
        """
        update main Q network by experience replay
        :param state_batch: batch of states
        :param action_batch: batch of actions
        :param reward_batch: batch of rewards
        :param next_state_batch: batch of next states
        :param terminated_batch: batch of whether it is terminated
        :return: Huber loss
        """
        with tf.GradientTape as tape:
            next_state_q = self.target_network(next_state_batch)
            next_state_max_q = tf.reduce_max(next_state_q, axis=1)
            expected_q = reward_batch + self.discount_factor * next_state_max_q * (1.0 - tf.cast(terminated_batch, tf.float32))

            main_q = tf.reduce_sum(self.main_network(state_batch) * tf.one_hot(action_batch, self.action_num, on_value=1.0, off_value=0.0), axis=1)
            loss = self.loss(tf.stop_gradient(expected_q), main_q)
        gradients = tape.gradient(loss, self.main_network.trainable_variables)
        clipped_gradients = [tf.clip_by_norm(grad, 10) for grad in gradients]
        self.optimizer.apply_gradients(zip(clipped_gradients, self.main_network.trainable_variables))

        self.loss_metric.update_state(loss)
        self.q_metric.update_state(main_q)
        return loss

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
        episode = 0
        latest_scores = deque(maxlen=self.latest_record_num)

        while frames < self.training_frames:

            cur_state = self.env.reset()
            episode_reward = 0
            terminated = False

            while not terminated:
                explr = self.get_explr(tf.constant(frames, tf.float32))
                action = self.get_action(tf.constant(cur_state), tf.constant(explr, tf.float32))

                next_state, reward, terminated, _ = self.env.step(action)
                episode_reward += reward

                self.memory.push(cur_state, action, reward, next_state, terminated)
                cur_state = next_state

                if frames > self.replay_start_size:
                    if frames % self.update_frequency == 0:
                        indices = self.memory.get_minibatch_indices()
                        state_batch, action_batch, reward_batch, next_state_batch, terminated_batch = self.memory.get_minibatch_sample(indices)
                        self.update_main_network(state_batch, action_batch, reward_batch, next_state_batch, terminated_batch)
                    if frames % self.update_target_network == 0:
                        self.update_target_network()

                frames += 1

                if terminated:
                    latest_scores.append(episode_reward)
                    episode_reward += 1

                    if episode % self.save_weight_interval == 0:
                        print("Episode: {}\t Latest {} average score: {}\t Progress: {} / {} ( {:.2f} % )"
                              .format(episode,
                                      self.latest_record_num, np.mean(latest_scores),
                                      frames, self.training_frames, np.round(frames / self.training_frames, 3) * 100))
                        print("Weight saving...", end="")
                        self.main_network.save_weights(self.log_path + "/episode_{}".format(episode))
                        print("Done!")
                        self.play(self.log_path, 5)

    def play(self, save_path, trials):
        loaded_ckpt = tf.train.latest_checkpoint(save_path)
        self.main_network.load_weights(loaded_ckpt)

        env = AtariEnvironment(self.env_id)
        reward_list = []
        frame_list = []

        for t in range(trials):

            cur_state = self.env.reset()
            frames = []
            episode_reward = 0
            terminated = False

            while not terminated:
                frames.append(env.render())
                action = self.get_action(tf.constant(cur_state, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32))

                next_state, reward, terminated, _ = env.step(action)
                episode_reward += reward

                cur_state = next_state
                if len(frames) > 20 * 60 * self.max_playing_time:  # To prevent falling infinite repeating sequences.
                    print("Playing takes {} minutes. Force termination.".format(self.max_playing_time))
                    break

            reward_list.append(episode_reward)
            frame_list.append(frames)

        print("Score on {} trials: ".format(trials), reward_list)
        print("Highest score: ", np.max(reward_list))
        best_idx = int(np.argmax(reward_list))
        imageio.mimsave(self.env_id + ".gif", frame_list[best_idx], fps=20)
