from datetime import datetime

from tensorflow_core.python.keras.optimizers import Adam
import tensorflow as tf

from DQN.network import Network
from DQN.replay_memory import ReplayMemory
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch


def create_env(env_id):
    env = make_atari(env_id)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)
    return env


class Agent:

    def __init__(self, env_id):
        # hyper-parameters
        self.discount_factor = 0.99
        self.minibatch_size = 32
        self.update_frequency = 4
        self.target_network_update_frequency = 1000
        self.history_len = 4
        self.memory_size = 10000
        self.init_exp = 1.0
        self.final_exp = 0.1
        self.final_exp_frame = 1000000
        self.replay_start_size = 10000
        self.training_frames = int(1e7)
        self.print_log_interval = 10
        self.save_weight_interval = 10

        # environment
        self.env_id = env_id
        self.env = create_env(env_id)

        # network
        self.memory = ReplayMemory(minibatch_size=self.minibatch_size, memory_size=self.memory_size, history_len=self.history_len)
        self.main_network = Network(action_size=self.env.action_space.n, history_len=self.history_len)
        self.target_network = Network(action_size=self.env.action_space.n, history_len=self.history_len)
        self.optimizer = Adam(lr=1e-4, epsilon=1e-6)
        self.loss = tf.keras.losses.Huber()
        self.loss_metric = tf.keras.metrics.Mean()
        self.q_metric = tf.keras.metrics.Mean()

        # other tools
        self.log_path = "./log/" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + self.env_id
        self.summary_writer = tf.summary.create_file_writer(self.log_path + "/summary/")
