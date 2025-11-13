import numpy as np


# Perform HER memory augmentation for SAC
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
            
            # Compute reward using the unwrapped environment
            reward = agent.env.unwrapped.compute_reward(new_achgoal, future_achgoal, None)
            reward = np.clip(reward, -10, 0) 

            # Store augmented experience in replay buffer
            # SAC is off-policy, so we can directly store transitions
            agent.remember(state, action, reward, next_state, True)
