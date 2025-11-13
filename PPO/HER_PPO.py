import numpy as np
import tensorflow as tf


# Perform HER memory augmentation for PPO
def her_augmentation(agent, obs_array, actions, new_obs_array):
    # Hyperparameter for future goals sampling
    k = 4

    # Augment the replay buffer
    size = len(actions)
    for index in range(size):
        for _ in range(k):
            future = np.random.randint(index, size)
            _, future_achgoal, _ = new_obs_array[future].values()

            obs, _, _ = obs_array[index].values()
            state = np.concatenate((obs, future_achgoal, future_achgoal))

            new_obs, new_achgoal, _ = new_obs_array[index].values()
            next_state = np.concatenate((new_obs, future_achgoal, future_achgoal))

            action = actions[index]
            
            # Fix: Access the unwrapped environment's compute_reward method
            reward = agent.env.unwrapped.compute_reward(new_achgoal, future_achgoal, None)

            # Recompute log_prob and value with current policy for PPO
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
            action_tensor = tf.convert_to_tensor([action], dtype=tf.float32)
            
            # Get policy distribution parameters
            mean, log_std = agent.actor(state_tensor)
            std = tf.exp(log_std)
            
            # Calculate log probability for the action
            log_prob = agent._calculate_log_prob(mean, std, action_tensor)
            
            # Get value estimate
            value = agent.critic(state_tensor)

            # Store augmented experience in PPO memory
            agent.remember(state, action, reward, value[0].numpy()[0], log_prob[0].numpy()[0], True)
