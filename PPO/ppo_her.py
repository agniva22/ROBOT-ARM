import numpy as np
import gymnasium as gym
import panda_gym
from agents.ppo import PPOAgent
from utils.HER_PPO import her_augmentation
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":

    n_games = 5000
    n_steps = 2048  # Number of steps to collect before update
    best_score = -np.inf
    score_history = []
    avg_score_history = []
    success_history = []  # Track success rate
    win_percentage_history = []  # Track win percentage

    os.makedirs('ckp/ppo', exist_ok=True)
    os.makedirs('plot/ppo', exist_ok=True)
    
    env = gym.make('PandaReach-v3')
    obs_shape = env.observation_space['observation'].shape[0] + \
                env.observation_space['achieved_goal'].shape[0] + \
                env.observation_space['desired_goal'].shape[0]

    agent = PPOAgent(env=env, input_dims=obs_shape)

    for i in range(n_games):
        done = False
        truncated = False
        score = 0
        step = 0
        success = 0

        obs_array = []
        actions_array = []
        new_obs_array = []

        observation, info = env.reset()

        while not (done or truncated):
            curr_obs, curr_achgoal, curr_desgoal = observation.values()
            state = np.concatenate((curr_obs, curr_achgoal, curr_desgoal))

            # Choose an action (PPO returns action, log_prob, and value)
            action, log_prob, value = agent.choose_action(state, False)

            # Execute the chosen action in the environment
            new_observation, reward, done, truncated, _ = env.step(np.array(action))
            next_obs, next_achgoal, next_desgoal = new_observation.values()
            new_state = np.concatenate((next_obs, next_achgoal, next_desgoal))

            # Store experience in PPO memory
            agent.remember(state, action, reward, value, log_prob, done)
        
            obs_array.append(observation)
            actions_array.append(action)
            new_obs_array.append(new_observation)

            observation = new_observation
            score += reward
            step += 1
            if info.get("is_success", False):
                success = 1
        
        # Augment memory with HER
        her_augmentation(agent, obs_array, actions_array, new_obs_array)

        # PPO learns when enough experiences are collected
        agent.learn()
            
        score_history.append(score)
        success_history.append(success)
        avg_score = np.mean(score_history[-100:])
        avg_score_history.append(avg_score)

        # Calculate win percentage over last 100 episodes
        win_percentage = np.mean(success_history[-100:]) * 100
        win_percentage_history.append(win_percentage)

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
            # print(f"*** New best score: {best_score:.1f} - Models saved ***")
        print(f"Episode {i} steps {step} score {score:.1f} avg score {avg_score:.1f}")

        # Save models periodically (every 1000 episodes)
        if (i + 1) % 1000 == 0:
            agent.save_models()
            # print(f"--- Checkpoint saved at episode {i + 1} ---")
            
            # Plot 1: Average Score
            plt.figure(figsize=(5, 4))
            episodes = list(range(len(avg_score_history)))
            plt.plot(episodes, avg_score_history, label='Avg Score', linewidth=2, markersize=8)
            plt.axhline(y=0, color='r', linestyle='--', label='Success threshold', linewidth=2)
            plt.xlabel('Episode', fontsize=10, fontweight='bold')
            plt.ylabel('Average Score', fontsize=10, fontweight='bold')
            legend = plt.legend(fontsize=10, loc='best', prop={'weight': 'bold', 'size': 10})
            plt.grid(True, alpha=0.2)
            plt.xticks(fontsize=10, fontweight='bold')
            plt.yticks(fontsize=10, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'plot/ppo/avg_score_episode_{i+1}.eps', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot 2: Win Percentage
            plt.figure(figsize=(5, 4))
            plt.plot(episodes, win_percentage_history,label='Win %', linewidth=2, markersize=8, color='green')
            plt.xlabel('Episode', fontsize=10, fontweight='bold')
            plt.ylabel('Win Percentage (%)', fontsize=10, fontweight='bold')
            legend = plt.legend(fontsize=10, loc='best', prop={'weight': 'bold', 'size': 10})
            plt.grid(True, alpha=0.2)
            plt.xticks(fontsize=10, fontweight='bold')
            plt.yticks(fontsize=10, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'plot/ppo/win_percentage_episode_{i+1}.eps', dpi=300, bbox_inches='tight')
            plt.close()
            
            # print(f"Progress plots saved in plot/ppo folder")
    
    # Final save
    agent.save_models()
    print("\n=== Training Complete ===")
    print(f"Final avg score: {avg_score:.1f}")
    print(f"Final win percentage: {win_percentage:.1f}%")
    print(f"Best score achieved: {best_score:.1f}")
    
    # Final Plot 1: Average Score
    plt.figure(figsize=(5, 4))
    episodes = list(range(len(avg_score_history)))
    plt.plot(episodes, avg_score_history, label='Avg Score', linewidth=2, markersize=8)
    plt.axhline(y=0, color='r', linestyle='--', label='Success threshold', linewidth=2)
    plt.xlabel('Episode', fontsize=10, fontweight='bold')
    plt.ylabel('Average Score', fontsize=10, fontweight='bold')
    legend = plt.legend(fontsize=10, loc='best', prop={'weight': 'bold', 'size': 10})
    plt.grid(True, alpha=0.2)
    plt.xticks(fontsize=10, fontweight='bold')
    plt.yticks(fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot/ppo/final_avg_score.eps', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Final Plot 2: Win Percentage
    plt.figure(figsize=(5, 4))
    plt.plot(episodes, win_percentage_history,  label='Win %', linewidth=2, markersize=8, color='green')
    plt.xlabel('Episode', fontsize=10, fontweight='bold')
    plt.ylabel('Win Percentage (%)', fontsize=10, fontweight='bold')
    legend = plt.legend(fontsize=10, loc='best', prop={'weight': 'bold', 'size': 10})
    plt.grid(True, alpha=0.2)
    plt.xticks(fontsize=10, fontweight='bold')
    plt.yticks(fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot/ppo/final_win_percentage.eps', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Final plots saved in plot/ppo folder")
    
    # Save training data
    np.save('plot/ppo/score_history.npy', score_history)
    np.save('plot/ppo/avg_score_history.npy', avg_score_history)
    np.save('plot/ppo/success_history.npy', success_history)
    np.save('plot/ppo/win_percentage_history.npy', win_percentage_history)
    print("Training data saved in plot/ppo folder")

