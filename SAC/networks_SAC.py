import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


class ActorNetwork(keras.Model):
    def __init__(self, n_actions, name, model, checkpoints_dir="ckp/"):
        super(ActorNetwork, self).__init__()
        if not os.path.exists(checkpoints_dir):
            os.mkdir(checkpoints_dir)
        self.checkpoints_file = os.path.join(checkpoints_dir + model, name + ".weights.h5")

        self.layer1 = Dense(256, activation="relu")
        self.layer2 = Dense(256, activation="relu")
        self.layer3 = Dense(256, activation="relu")  # Add third layer
        
        # SAC outputs mean and log_std for Gaussian policy
        self.mean = Dense(n_actions, activation=None)
        self.log_std = Dense(n_actions, activation=None)
    
    @tf.function()
    def call(self, state):
        x = self.layer1(state)
        x = self.layer2(x)
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = tf.clip_by_value(log_std, -20, 2)
        
        return mean, log_std


class CriticNetwork(keras.Model):
    def __init__(self, name, model, checkpoints_dir="ckp/"):
        super(CriticNetwork, self).__init__()
        if not os.path.exists(checkpoints_dir):
            os.mkdir(checkpoints_dir)
        self.checkpoints_file = os.path.join(checkpoints_dir + model, name + ".weights.h5")

        self.layer1 = Dense(256, activation="relu")
        self.layer2 = Dense(256, activation="relu")
        self.q = Dense(1, activation=None)

    @tf.function()
    def call(self, state, action):
        x = tf.concat([state, action], axis=1)
        x = self.layer1(x)
        x = self.layer2(x)
        q = self.q(x)
        return q
