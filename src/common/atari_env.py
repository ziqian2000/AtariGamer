import tensorflow as tf

from src.common import make_atari, wrap_deepmind, wrap_pytorch


def process(state):
    return tf.transpose(state, [1, 2, 0])


class AtariEnvironment:

    def __init__(self, env_id, total_frames_limit):
        self.env = make_atari(env_id)
        self.env = wrap_deepmind(self.env, frame_stack=True)
        self.env = wrap_pytorch(self.env)
        self.total_frames_limit = total_frames_limit
        self.total_frames_passed = 0

    def reset(self):
        self.total_frames_passed = 0
        state = self.env.reset()
        return process(state)

    def render(self):
        return self.env.render(mode='rgb_array')

    def step(self, action):
        self.total_frames_passed += 1
        next_state, reward, done, info = self.env.step(action)
        if self.total_frames_passed > self.total_frames_limit:
            print("Killed. ", end="")
            done = True
        return process(next_state), reward, done, info

    def get_action_num(self):
        return self.env.action_space.n
