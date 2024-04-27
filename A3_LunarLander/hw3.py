import gymnasium as gym
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output

def analyse_env(env):
    print('Information about the environment: ')
    print(f'action space: {env.action_space}')
    print(f'observation space: {env.observation_space}')
    print(f'observation space - low: {env.observation_space.low}')
    print(f'observation space - high: {env.observation_space.high}')
    
def run_random(env, episodes=1000, timesteps=1000):
    episode_list = []
    episode_reward = []
    state, _ = env.reset()
    print('#############################')
    print('### Training Random Agent ###')
    print('#############################')
    for episode in range(episodes):
        # episode_list.append(episode)
        G = 0
        for t in range(timesteps):
            action = env.action_space.sample()
            # print(f'random action: {action}')
            state, reward, terminated, truncated, _ = env.step(action)
            state = discretize_state(env, state)
            # if t % 100 == 0:
            #    print(f'Observation: {observation}')
            #    print(f'Info: {info}')
            G += reward

            if terminated or truncated:
                state, _ = env.reset()
                break

        # print(f"Episode number: {episode + 1}, Reward: {G}")
        # episode_reward.append(G)

        if episode % 100 == 0:
            avg_reward = test_policy(env, None)
            episode_list.append(episode)
            episode_reward.append(avg_reward)
            print(f'### Episode: {episode} - Avg-Policy Reward: {avg_reward} ###')   

     # Calculate and print the average reward per 10 episodes
    # reward_per_10_episodes = np.split(np.array(episode_reward), episodes/10)
    # count = 10
    # print('**** Average Reward per 10 episodes for random agent **** \n')
    # for r in reward_per_10_episodes:
    #     print(count, ': ', str(sum(r/10)))
    #     count += 10    
       
    return episode_list, episode_reward


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

'''
def init_q_table(env, n=10):
    num_states = 4 * np.power(n, 6)
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))
    return Q

def epsilon_greedy(state, eps, Q_table, num_actions):
    # explore if random value is below epsilon, otherwise exploite using a greedy policy
    if random.uniform(0,1) < eps:
        return np.random.choice(num_actions)
    else:
        return np.argmax(Q_table[state])
'''        
    
def q_learning(env, episodes=10000, timesteps=1000):

    lr = 0.5
    gamma = 0.99
    eps = 0.1
    # Q_table = init_q_table(env)
    Q_table = {} 
    # print(Q_table)
    num_actions = env.action_space.n
    episode_list = []
    episode_reward = []
    print('#################################')
    print('### Training Q-learning Agent ###')
    print('#################################')
    # Q-learning loop
    for episode in range(episodes):
        # episode_list.append(episode)
        state, _ = env.reset()
        # print(f'state from gym: {state}')
        state = discretize_state(env, state)
        # print(f'state after discretize: {state}')

        # state_key = tuple(state)
        done = False
        G = 0

        for step in range(timesteps):
            # Check if state exists in the Q dictionary and initialize if not
            # var_type = type(state_key)
            # print(f'state variable type: {var_type}')
            if state not in Q_table:
                Q_table[state] = np.zeros(num_actions)
                # Q_table[state] = np.random.uniform(low=-0.1, high=0.1, size=num_actions)
            # Exploration-Explotation
            # explore if random value is below epsilon, otherwise exploite using a greedy policy
            if random.uniform(0,1) < eps:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_table[state])
                # if state not in Q_table:
                #    Q_table[state] = np.zeros(num_actions)
                #    action = env.action_space.sample()
                # else:    
                #    action = np.argmax(Q_table[state])
    
            # print(f'action selected: {action}')
            # new_state, reward, done, _ = env.step(action)
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            new_state = discretize_state(env, new_state)
            # new_state_key = tuple(new_state)
            # Check if new state exists in the Q dictionary and initialize if not
            if new_state not in Q_table:
                Q_table[new_state] = np.zeros(num_actions)
                # Q_table[new_state] = np.random.uniform(low=-0.1, high=0.1, size=num_actions)

            # old_value = Q_table[state][action]
            # print(f'Old Q(S,A): {old_value}')
            # Update Q-table for Q(s,a)
            Q_table[state][action] = Q_table[state][action]  + lr * (reward + gamma * np.argmax(Q_table[new_state]) - Q_table[state][action])

            # new_value = Q_table[state][action]
            # print(f'New Q(S,A): {new_value}')

            state = new_state
            G += reward

            if done:
                # state, _ = env.reset()
                break       
        
        # print(f'Episode: {episode} ; Reward: {G}')
        # episode_reward.append(G)

        if episode % 100 == 0:
            avg_reward = test_policy(env, Q_table)
            episode_list.append(episode)
            episode_reward.append(avg_reward)
            print(f'### Episode: {episode} - Avg-Policy Reward: {avg_reward} ###')

    # Calculate and print the average reward per 10 episodes
    # reward_per_10_episodes = np.split(np.array(episode_reward), episodes/10)
    # count = 10
    # print('**** Average Reward per 10 episodes for a Q agent **** \n')
    # for r in reward_per_10_episodes:
    #    print(count, ': ', str(sum(r/10)))
    #    count += 10    

    return episode_list, episode_reward, Q_table    


def on_policy_monte_carlo(env, episodes=1000):
    gamma = 0.99
    eps = 0.1
    Q_table = {}  
    returns = {}  # Dictionary to store returns for state-action pairs
    policy = {}  # ε-soft policy as a dictionary
    num_actions = env.action_space.n    # A(s)

    # for state in range(env.observation_space.n):
    #     Q_table[state] = np.zeros(num_actions)
    #     returns[state] = {action: [] for action in range(num_actions)}
    #     policy[state] = np.ones(num_actions) * eps / num_actions

    episode_list = []
    episode_reward = []
    print('#################################')
    print('### Training Monte-Carlo Agent ###')
    print('#################################')
    
    # MC loop
    for episode in range(episodes):
        # initialize episode
        episode_sim = []
        state, _ = env.reset()
        state = discretize_state(env, state)
        done = False
        
        # simulate an episode using the ε-soft greedy policy
        while not done:
            if state not in policy:
                policy[state] = np.ones(num_actions) * eps / num_actions
            action = np.argmax(policy[state])
            # print(f'MC action: {action}')
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            new_state = discretize_state(env, new_state)
            episode_sim.append((state, action, reward))
            state = new_state

        G = 0
        # update the Returns(s,a) and Q(s,a) for each pair s,a in the episode
        for state, action, reward in episode_sim:
            if state not in Q_table:
                Q_table[state] = np.zeros(num_actions)
            if state not in returns:
                returns[state] = {action: [] for action in range(num_actions)}
            G += reward
            # Append G to Returns(s,a)
            returns[state][action].append(G)
            # Q(s,a) <- average(Returns(s,a))
            Q_table[state][action] = np.mean(returns[state][action])

        # Update the policy for each state in the episode
        # for state, _, _ in episode_sim:  
            a_star = np.argmax(Q_table[state])
            for a in range(num_actions):
                if a == a_star:
                    policy[state][a] = 1 - eps + eps / num_actions
                else:
                    policy[state][a] = eps / num_actions

        if episode % 100 == 0:
            avg_reward = test_policy(env, Q_table)
            episode_list.append(episode)
            episode_reward.append(avg_reward)
            print(f'### Episode: {episode} - Avg-Policy Reward: {avg_reward} ###')

    return episode_list, episode_reward, Q_table   


def test_policy(env, Q_table, n = 10):
    episode_reward = []
    num_actions = env.action_space.n
    avg_reward = 0

    for _ in range(n):
        state, _ = env.reset()
        state = discretize_state(env, state)
        done = False
        G = 0
        # Need to store (state, action, reward, new_state, terminal) trajectory for each episode
        episode_data = []  

        while not done:
            if Q_table is not None:
                if state not in Q_table:
                    Q_table[state] = np.zeros(num_actions)
                action = np.argmax(Q_table[state])
            else: 
                action = env.action_space.sample()   
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            new_state = discretize_state(env, new_state)
            state = new_state
            G += reward

        episode_reward.append(G)

    for reward in episode_reward:
        avg_reward += reward

    return avg_reward / n

def test_agent(env, Q):
    # Test agent after training
    for episode in range(3):
        state, _  = env.reset()
        state = discretize_state(env, state)
        done = False
        print(f'----Episode: {episode+1} ---- \n')
        # time.sleep(1)

        for _ in range(1000):
            clear_output(wait=True)
            env.render()
            # time.sleep(0.5)

            action =np.argmax(Q[state])
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            new_state = discretize_state(env, new_state)

            if done:
                # clear_output(wait=True)
                # env.render()
                # clear_output(wait=True)
                break

            # state = new_state


def plot_reward(results, labels):
    plt.figure(figsize=(10, 6))
    for i, result in enumerate(results):
        plt.plot(result[0], result[1], label=labels[i])
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Agent Performance')
    plt.grid(True)
    plt.legend(labels)
    plt.show()

def main():
    env = gym.make("LunarLander-v2")
    # analyse_env(env)
    results = []
    episodes, rewards = run_random(env)
    results.append((episodes, rewards))
    
    episodes, rewards, Q = on_policy_monte_carlo(env)
    results.append((episodes, rewards))
    labels = ['Random','Monte-Carlo']

    # episodes, rewards, Q = q_learning(env)
    # results.append((episodes, rewards))
    # labels = ['Random','Q-learning']
    
    plot_reward(results, labels)
    env.close()

    env = gym.make("LunarLander-v2", render_mode="human")
    # Q = monte_carlo(env)
    # test_agent(env, Q)

    env.close()

if __name__ == '__main__':
    main()