import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from replay_memory.ReplayBuffer import ReplayBuffer
from utils.networks_PPO import ActorNetwork, CriticNetwork


## Actor-critic networks parameters:

# actor learning rate
alpha = 0.0003

# critic learning rate
beta = 0.001


## PPO algorithm parameters

# discount factor
gamma = 0.99

# GAE lambda parameter
gae_lambda = 0.95

# PPO clipping parameter
policy_clip = 0.2

# number of epochs for policy updates
n_epochs = 10

# training batch size
batch_size = 64

# entropy coefficient for exploration
entropy_coef = 0.01


## PPO agent class
class PPOAgent:
    def __init__(self, env, input_dims):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        
        self.env = env
        self.n_actions = env.action_space.shape[0]
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        
        self.memory = PPOMemory(batch_size)
        
        self._initialize_networks(self.n_actions)
    
    # Choose action based on actor network with stochastic policy
    def choose_action(self, state, evaluate=False):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        
        # Get mean and log standard deviation from actor
        mean, log_std = self.actor(state)
        std = tf.exp(log_std)
        
        if not evaluate:
            # Sample action from normal distribution
            action = mean + std * tf.random.normal(shape=mean.shape)
        else:
            action = mean
        
        # Clip action to valid range
        action = tf.clip_by_value(action, self.min_action, self.max_action)
        
        # Calculate log probability for the action
        log_prob = self._calculate_log_prob(mean, std, action)
        
        # Get value estimate
        value = self.critic(state)
        
        return action[0].numpy(), log_prob[0].numpy(), value[0].numpy()
    
    def remember(self, state, action, reward, value, log_prob, done):
        self.memory.store_memory(state, action, reward, value, log_prob, done)
    
    # Main PPO algorithms learning process
    def learn(self):
        if not self.memory.ready():
            return
        
        # Get all stored experiences
        states, actions, old_log_prob_arr, vals_arr, reward_arr, done_arr, batches = \
            self.memory.generate_batches()
        
        # Convert to numpy for advantage calculation
        values = vals_arr
        rewards = reward_arr
        dones = done_arr
        
        # Calculate advantages using GAE (returns numpy array)
        advantages = self._calculate_advantages(rewards, values, dones)
        
        # Perform multiple epochs of updates
        for _ in range(self.n_epochs):
            for batch in batches:
                # Index using numpy arrays, then convert to tensors
                states_batch = tf.convert_to_tensor(states[batch], dtype=tf.float32)
                old_log_probs_batch = tf.convert_to_tensor(old_log_prob_arr[batch], dtype=tf.float32)
                actions_batch = tf.convert_to_tensor(actions[batch], dtype=tf.float32)
                advantage_batch = tf.convert_to_tensor(advantages[batch], dtype=tf.float32)
                old_values_batch = tf.convert_to_tensor(vals_arr[batch], dtype=tf.float32)
                returns = advantage_batch + old_values_batch
                
                # Update critic network
                with tf.GradientTape() as tape:
                    critic_value = tf.squeeze(self.critic(states_batch), 1)
                    critic_loss = keras.losses.MSE(returns, critic_value)
                
                critic_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
                
                # Check if gradients exist before applying
                if critic_gradient and all(g is not None for g in critic_gradient):
                    self.critic.optimizer.apply_gradients(zip(
                        critic_gradient, self.critic.trainable_variables
                    ))
                
                # Update actor network
                with tf.GradientTape() as tape:
                    mean, log_std = self.actor(states_batch)
                    std = tf.exp(log_std)
                    new_log_probs = self._calculate_log_prob(mean, std, actions_batch)
                    
                    # Calculate probability ratio
                    prob_ratio = tf.exp(new_log_probs - old_log_probs_batch)
                    
                    # Calculate clipped surrogate objective
                    weighted_probs = advantage_batch * prob_ratio
                    clipped_probs = advantage_batch * tf.clip_by_value(
                        prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip
                    )
                    actor_loss = -tf.reduce_mean(tf.minimum(weighted_probs, clipped_probs))
                    
                    # Add entropy bonus for exploration
                    entropy = tf.reduce_mean(0.5 * (tf.math.log(2.0 * 3.14159265359 * std**2) + 1))
                    actor_loss -= self.entropy_coef * entropy
                
                actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
                
                # Check if gradients exist before applying
                if actor_gradient and all(g is not None for g in actor_gradient):
                    self.actor.optimizer.apply_gradients(zip(
                        actor_gradient, self.actor.trainable_variables
                    ))
        
        # Clear memory after learning
        self.memory.clear_memory()


    # Calculate advantages using Generalized Advantage Estimation (GAE)
    def _calculate_advantages(self, rewards, values, dones):
        advantages = []
        advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * advantage
            advantages.insert(0, advantage)
        
        # Return numpy array, not tensor
        advantages = np.array(advantages, dtype=np.float32)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)
        return advantages
        
    # Calculate log probability of action under Gaussian policy
    def _calculate_log_prob(self, mean, std, action):
        var = std ** 2
        log_prob = -0.5 * ((action - mean) ** 2 / var + tf.math.log(2.0 * 3.14159265359 * var))
        return tf.reduce_sum(log_prob, axis=1, keepdims=True)
    
    def save_models(self):
        print("---- saving models ----")
        self.actor.save_weights(self.actor.checkpoints_file)
        self.critic.save_weights(self.critic.checkpoints_file)
    
    def load_models(self):
        print("---- loading models ----")
        self.actor.load_weights(self.actor.checkpoints_file)
        self.critic.load_weights(self.critic.checkpoints_file)
    
    def _initialize_networks(self, n_actions):
        model = "ppo"
        self.actor = ActorNetwork(n_actions, name="actor", model=model)
        self.critic = CriticNetwork(name="critic", model=model)
        
        self.actor.compile(keras.optimizers.Adam(learning_rate=alpha))
        self.critic.compile(keras.optimizers.Adam(learning_rate=beta))


## PPO Memory class (replaces ReplayBuffer)
class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.batch_size = batch_size
    
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = list(range(0, n_states, self.batch_size))
        indices = np.arange(n_states, dtype=np.int32)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        # Convert to numpy arrays first to ensure consistent shapes
        states_arr = np.array(self.states, dtype=np.float32)
        actions_arr = np.array(self.actions, dtype=np.float32)
        log_probs_arr = np.array(self.log_probs, dtype=np.float32).reshape(-1)
        values_arr = np.array(self.values, dtype=np.float32).reshape(-1)
        rewards_arr = np.array(self.rewards, dtype=np.float32)
        dones_arr = np.array(self.dones, dtype=np.float32)
        
        return states_arr, \
               actions_arr, \
               log_probs_arr, \
               values_arr, \
               rewards_arr, \
               dones_arr, \
               batches
    
    def store_memory(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        # Ensure scalar values
        self.values.append(float(value) if np.isscalar(value) else float(value[0]))
        self.log_probs.append(float(log_prob) if np.isscalar(log_prob) else float(log_prob[0]))
        self.dones.append(int(done))
    
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def ready(self):
        return len(self.states) >= self.batch_size
