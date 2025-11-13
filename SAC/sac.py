import tensorflow as tf
import tensorflow.keras as keras
from replay_memory.ReplayBuffer import ReplayBuffer
from utils.networks_SAC import ActorNetwork, CriticNetwork
import numpy as np


## Actor-critic networks parameters:

# actor learning rate
alpha = 0.0003

# critic learning rate (Q-functions)
beta = 0.0003

# temperature learning rate
alpha_lr = 0.0003


## SAC algorithm parameters

# discount factor
gamma = 0.99

# target networks soft update factor 
tau = 0.005

# replay buffer max memory size
max_size = 10**6

# training batch size 
batch_size = 256

# Start learning after more samples
min_samples_before_training = 1000

# automatic entropy tuning
auto_entropy_tuning = False


## SAC agent class 
class SACAgent:
    def __init__(self, env, input_dims):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.auto_entropy_tuning = auto_entropy_tuning

        self.env = env
        self.n_actions = env.action_space.shape[0]
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.memory = ReplayBuffer(max_size, input_dims, self.n_actions)

        self._initialize_networks(self.n_actions)
        
        # Target entropy for automatic tuning
        if self.auto_entropy_tuning:
            self.target_entropy = -tf.constant(self.n_actions, dtype=tf.float32)
            self.log_alpha = tf.Variable(0.0, dtype=tf.float32, trainable=True)
            self.alpha_optimizer = keras.optimizers.Adam(learning_rate=alpha_lr)
        else:
            self.log_alpha = tf.Variable(tf.math.log(0.2), dtype=tf.float32, trainable=False)
        
        # Build networks
        dummy_state = tf.ones((1, input_dims), dtype=tf.float32)
        _ = self.actor(dummy_state)
        dummy_action = tf.ones((1, self.n_actions), dtype=tf.float32)
        _ = self.critic_1(dummy_state, dummy_action)
        _ = self.critic_2(dummy_state, dummy_action)
        _ = self.target_critic_1(dummy_state, dummy_action)
        _ = self.target_critic_2(dummy_state, dummy_action)
        
        self.update_parameters(tau=1)

    # Choose action based on actor network with stochastic policy
    def choose_action(self, state, evaluate=False):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        
        if evaluate:
            # Use mean action for evaluation
            mean, _ = self.actor(state)
            action = tf.tanh(mean)
        else:
            # Sample action for training
            action, _, _ = self.sample_action(state)
        
        # # Scale action to environment bounds
        # action = action * self.max_action
        return action[0].numpy()
    
    def sample_action(self, state, reparameterize=True):
        """Sample action from policy with reparameterization trick"""
        mean, log_std = self.actor(state)
        log_std = tf.clip_by_value(log_std, -20, 2)
        std = tf.exp(log_std)
        
        # Sample from normal distribution
        normal_dist = tf.random.normal(mean.shape)
        
        if reparameterize:
            # Reparameterization trick
            action_sample = mean + std * normal_dist
        else:
            action_sample = mean + std * tf.stop_gradient(normal_dist)
        
        # Apply tanh squashing
        action = tf.tanh(action_sample)
        
        # Calculate log probability with correction for tanh squashing
        log_prob = -0.5 * (tf.math.log(2.0 * np.pi) + 2 * log_std + 
                           ((action_sample - mean) / (std + 1e-6)) ** 2)
        log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True)
        
        # Correction for tanh squashing
        log_prob -= tf.reduce_sum(
            tf.math.log(1 - action ** 2 + 1e-6), axis=1, keepdims=True
        )
        
        return action, log_prob, tf.tanh(mean)
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
    
    # Main SAC algorithm learning process
    def learn(self):
        if self.memory.counter < self.batch_size:
            return

        # Sample batch of experiences from replay buffer
        states, actions, rewards, new_states, dones = self.memory.sample(self.batch_size)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        alpha = tf.exp(self.log_alpha)

        # Update Q-functions (critics)
        with tf.GradientTape(persistent=True) as tape:
            # Sample actions for next states
            new_actions, new_log_probs, _ = self.sample_action(new_states, reparameterize=False)
            new_actions_scaled = new_actions * self.max_action
            
            # Target Q-values
            target_q1 = self.target_critic_1(new_states, new_actions_scaled)
            target_q2 = self.target_critic_2(new_states, new_actions_scaled)
            target_q = tf.minimum(target_q1, target_q2)
            
            # Compute target with entropy term
            target_value = target_q - alpha * new_log_probs
            y = rewards + self.gamma * (1 - dones) * tf.squeeze(target_value, 1)
            y = tf.expand_dims(y, 1)
            
            # Current Q-values
            current_q1 = self.critic_1(states, actions)
            current_q2 = self.critic_2(states, actions)
            
            # Q-function losses
            critic_1_loss = 0.5 * tf.reduce_mean((current_q1 - y) ** 2)
            critic_2_loss = 0.5 * tf.reduce_mean((current_q2 - y) ** 2)

        # Update critics
        critic_1_gradients = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_1_gradients = [tf.clip_by_norm(g, 1.0) for g in critic_1_gradients]
        self.critic_1.optimizer.apply_gradients(
            zip(critic_1_gradients, self.critic_1.trainable_variables)
        )
        
        critic_2_gradients = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
        self.critic_2.optimizer.apply_gradients(
            zip(critic_2_gradients, self.critic_2.trainable_variables)
        )
        
        del tape

        # Update policy (actor)
        with tf.GradientTape() as tape:
            # Sample new actions with reparameterization
            new_actions, log_probs, _ = self.sample_action(states, reparameterize=True)
            new_actions_scaled = new_actions * self.max_action
            
            # Q-values for new actions
            q1_new = self.critic_1(states, new_actions_scaled)
            q2_new = self.critic_2(states, new_actions_scaled)
            q_new = tf.minimum(q1_new, q2_new)
            
            # Actor loss (maximize Q - alpha * log_prob)
            actor_loss = tf.reduce_mean(alpha * log_probs - q_new)

        # Update actor
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(
            zip(actor_gradients, self.actor.trainable_variables)
        )

        # Update temperature (alpha) if auto-tuning
        if self.auto_entropy_tuning:
            with tf.GradientTape() as tape:
                # Sample actions for alpha update
                _, log_probs, _ = self.sample_action(states, reparameterize=False)
                alpha_loss = -tf.reduce_mean(
                    self.log_alpha * (log_probs + self.target_entropy)
                )

            alpha_gradients = tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(zip(alpha_gradients, [self.log_alpha]))

        # Soft update target networks
        self.update_parameters()

    # Update target networks parameters with soft update rule
    def update_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        # Update target critic 1
        weights = []
        targets = self.target_critic_1.weights
        for i, weight in enumerate(self.critic_1.weights):
            weights.append(tau * weight + (1 - tau) * targets[i])
        self.target_critic_1.set_weights(weights)

        # Update target critic 2
        weights = []
        targets = self.target_critic_2.weights
        for i, weight in enumerate(self.critic_2.weights):
            weights.append(tau * weight + (1 - tau) * targets[i])
        self.target_critic_2.set_weights(weights)

    def save_models(self):
        print("---- saving models ----")
        self.actor.save_weights(self.actor.checkpoints_file)
        self.critic_1.save_weights(self.critic_1.checkpoints_file)
        self.critic_2.save_weights(self.critic_2.checkpoints_file)
        self.target_critic_1.save_weights(self.target_critic_1.checkpoints_file)
        self.target_critic_2.save_weights(self.target_critic_2.checkpoints_file)

    def load_models(self):
        print("---- loading models ----")
        self.actor.load_weights(self.actor.checkpoints_file)
        self.critic_1.load_weights(self.critic_1.checkpoints_file)
        self.critic_2.load_weights(self.critic_2.checkpoints_file)
        self.target_critic_1.load_weights(self.target_critic_1.checkpoints_file)
        self.target_critic_2.load_weights(self.target_critic_2.checkpoints_file)

    def _initialize_networks(self, n_actions):
        model = "sac"
        self.actor = ActorNetwork(n_actions, name="actor", model=model)
        self.critic_1 = CriticNetwork(name="critic_1", model=model)
        self.critic_2 = CriticNetwork(name="critic_2", model=model)
        self.target_critic_1 = CriticNetwork(name="target_critic_1", model=model)
        self.target_critic_2 = CriticNetwork(name="target_critic_2", model=model)

        self.actor.compile(keras.optimizers.Adam(learning_rate=alpha))
        self.critic_1.compile(keras.optimizers.Adam(learning_rate=beta))
        self.critic_2.compile(keras.optimizers.Adam(learning_rate=beta))
        self.target_critic_1.compile(keras.optimizers.Adam(learning_rate=beta))
        self.target_critic_2.compile(keras.optimizers.Adam(learning_rate=beta))
