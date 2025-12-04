import warnings

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt


def display(env: gym.Env) -> None:
    env.reset()
    img = plt.imshow(env.render())
    plt.axis('off')
    plt.show()


def discretize_state(state, bins):
    """Discretize the continuous state into discrete bins."""
    if isinstance(state, tuple):
        state = state[0]
    discretized_state = tuple(np.digitize(s, bins[i]) - 1 for i, s in enumerate(state))
    return discretized_state
def q_learning(env, num_episodes=500, alpha=0.07, gamma=0.99, epsilon=0.3):
    num_bins = [20, 20] # discretize the state space
    state_bins = [np.linspace(-1.2, 0.6, num_bins[0]),
                  np.linspace(-0.07, 0.07, num_bins[1])]
    num_actions = env.action_space.n # initialize q-table
    Q = np.zeros((num_bins[0], num_bins[1], num_actions))
    rewards = [] # store rewards and number of actions taken
    num_actions_list = []
    for episode in range(num_episodes):
        state = discretize_state(env.reset(), state_bins)
        total_reward = 0
        num_actions_episode = 0

        total_step = 0
        while True:
            total_step +=1
            # epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # explore
            else:
                action = np.argmax(Q[state])  # exploit
            next_state, reward, done, _ = env.step(action)[:4]

            #next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_state(next_state, state_bins)
            Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action]) # q-value update ny the q-learning formula
            state = next_state
            total_reward += reward
            num_actions_episode += 1
            if done:
                print(f"Episode {episode+1}: Total Reward: {total_reward}, Steps: {total_step}")
                break
        rewards.append(total_reward)
        num_actions_list.append(num_actions_episode)
    np.save('q_table.npy', Q)
    return rewards, num_actions_list

env = gym.make('MountainCar-v0', render_mode="rgb_array")
initial_state = env.reset()
rewards, num_actions_list = q_learning(env, num_episodes=500) # run q-learning

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(rewards,color='r', label='Episode Reward')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Q-learning in MountainCar-v0')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(num_actions_list, label='Number of Actions') # number of actions taken
plt.xlabel('Episode')
plt.ylabel('Number of Actions')
plt.title('Q-learning in MountainCar-v0')
plt.grid(True)
plt.tight_layout()
plt.show()

env.close()