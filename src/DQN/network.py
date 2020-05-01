import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Lambda


class Network(Model):

    def __init__(self, action_num, history_len):
        super(Network, self).__init__()
        self.normalize = Lambda(lambda x: tf.cast(x, dtype=tf.float32) / 255.0)
        self.conv1 = Conv2D(name="conv1", filters=32, kernel_size=(8, 8), strides=(4, 4), kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="relu", input_shape=(None, 84, 84, history_len))
        self.conv2 = Conv2D(name="conv2", filters=64, kernel_size=(4, 4), strides=(2, 2), kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="relu")
        self.conv3 = Conv2D(name="conv3", filters=64, kernel_size=(3, 3), strides=(1, 1), kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="relu")
        self.flatten = Flatten()
        self.dense1 = Dense(units=512, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="relu")
        self.dense2 = Dense(action_num, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="linear")

    @tf.function
    def call(self, cur):
        cur = self.normalize(cur)
        cur = self.conv1(cur)
        cur = self.conv2(cur)
        cur = self.conv3(cur)
        cur = self.flatten(cur)
        cur = self.dense1(cur)
        cur = self.dense2(cur)
        return cur
