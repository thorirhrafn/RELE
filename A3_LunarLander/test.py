import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt

def discretize_state(env, state, n=10):
    disc_state = []
    # print(f'state length: {len(state[0])}')

    # Discretize all variables expect for the 2 last boolean values
    for i in range(len(state) - 2): 
        low = env.observation_space.low[i]
        high = env.observation_space.high[i]
        interval = (high - low) / n
        disc_value = int((state[i] - low) / interval)
        disc_state.append(disc_value)

    for i in range(len(state) - 2, len(state)):
        disc_state.append(state[i])

    return tuple(disc_state)



def on_policy_monte_carlo(env, episodes=1000, epsilon=0.1):
    num_actions = env.action_space.n
    Q = {}  # Q-table as a dictionary
    returns = {}  # Dictionary to store returns for state-action pairs
    episode_rewards = []

    # for state in range(env.observation_space.n):
    #     Q[state] = np.zeros(num_actions)
    #     returns[state] = {action: [] for action in range(num_actions)}

    def generate_episode():
        episode = []
        state, _ = env.reset(seed=42)
        state = discretize_state(env, state)
        done = False
        while not done:
            if state not in Q:
                # Initialize Q-values for all actions with zeros
                Q[state] = np.zeros(num_actions)

            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(Q[state])  # Exploit
            next_state, reward, done, _, _ = env.step(action)
            next_state = discretize_state(env, next_state)
            episode.append((state, action, reward))
            state = next_state
        return episode

    for episode_num in range(episodes):
        episode = generate_episode()
        G = 0
        for t, (state, action, reward) in enumerate(episode):
            G += reward
            if state not in returns:
                # Initialize Q-values for all actions with zeros
                returns[state] = np.zeros(num_actions)
            returns[state][action] += G
            Q[state][action] = np.mean(returns[state][action])
        episode_rewards.append(G)

    return Q, episode_rewards

# Create the Lunar Lander environment
env = gym.make("LunarLander-v2")

# Run on-policy Monte Carlo control
Q_values, rewards = on_policy_monte_carlo(env)

# Plot rewards over time
plt.figure(figsize=(10, 6))
plt.plot(range(len(rewards)), rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('On-Policy Monte Carlo Control for Lunar Lander')
plt.grid(True)
plt.show()
