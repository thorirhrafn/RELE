#######################################################################
# Copyright (C)                                                       #
# 2021 Stephan Schiffel(stephans@ru.is)                               #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Artem Oboturov(oboturov@gmail.com)                             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')

class NonStationaryBanditEnvironment:
    # @k_arm: # of arms
    # @gradient_baseline: if True, use average reward as baseline for gradient based bandit algorithm
    def __init__(self, k_arm=10, true_reward=0. ,mean = 0 , std_dev = 0.01):
        self.k = k_arm
        self.true_reward = true_reward
        self.std_dev = std_dev
        self.mean = mean

    def reset(self):
        # real reward for each action
        self.q_true = np.ones(self.k)
        self.best_action = np.argmax(self.q_true)

    # take an action return the reward for this action
    def step(self, action):
        # take independent random walks by adding a normally distributed increment with mean 0 and standard deviation 0.01 to all the q*(a) on each step
        for i in range(0,self.k):
            self.q_true[i] += np.random.normal(self.mean, self.std_dev)
        # Since all the bandits randomly change in each time step the best action can also change in each step
        self.best_action = np.argmax(self.q_true)
        # generate the reward based on the action taken
        reward = self.q_true[action]
        return reward

class Agent:
    # @k_arm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @step_size: constant step size for updating estimations
    def __init__(self, k_arm=10, epsilon=0. ,step_size = 0.1):
        self.k = k_arm
        self.indices = np.arange(self.k) # list of all possible actions
        self.epsilon = epsilon
        self.alpha = step_size      #set constant step size 
        self.q_delta = 0

    def reset(self):
        # estimation for each action
        self.q_estimation = np.zeros(self.k)
        # # of chosen times for each action
        self.action_count = np.zeros(self.k)

    # get an action for this bandit
    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)

        q_best = np.max(self.q_estimation)
        return np.random.choice(np.where(self.q_estimation == q_best)[0])
        
    # observe the result of the action (and learn from it)
    def observe(self, action, reward):
        self.action_count[action] += 1
        self.q_delta = reward - self.q_estimation[action]
        # update estimation using step size
        self.q_estimation[action] += self.alpha * (reward - self.q_estimation[action])


def simulate(runs, time, agents, environment):
    # keep track of reward of each agent in each run at every time step
    rewards = np.zeros((len(agents), runs, time))
    q_deltas = np.zeros((len(agents), runs, time))
    # keep track of each time an agent did the best action
    best_action_counts = np.zeros(rewards.shape)
    n = 0
    for i, agent in enumerate(agents):
        n += 1
        for r in range(runs):
            print(f'agent: {n}, run: {r} ')
            agent.reset()
            environment.reset()
            for t in range(time):
                action = agent.act()
                reward = environment.step(action)
                rewards[i, r, t] = reward
                if action == environment.best_action:
                    best_action_counts[i, r, t] = 1
                agent.observe(action, reward)
                q_deltas[i, r, t] = agent.q_delta
    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    mean_q_deltas = q_deltas.mean(axis=1)
    return mean_best_action_counts, mean_rewards, mean_q_deltas

#look at how accurate the estimated values Qt (a) are compared to the true values q∗ (a) over time with either a small
# or a large step size (e.g. 0.01 vs. 0.3) in an environment that is slowly changing vs. one that is
# quickly changing (standard deviation for the change in true value 0.01 vs. 0.1).

def figure_2_2_exp(runs=2000, time=10000):
    epsilons = [0.1]
    k = 10
    alphas = [0.01, 0.3]
    agents = []
    for alpha in alphas: 
        agent = Agent(k_arm=k, epsilon=0.1, step_size=alpha)
        agents.append(agent)

    environment = NonStationaryBanditEnvironment(k_arm=k, std_dev=0.01)
    plt.figure(figsize=(10, 20))

    for n, agent in enumerate(agents):
        _, _, q_deltas = simulate(runs, time, [agent], environment)
        label = 'alpha = 0.01'
        if n > 0:
            label = 'alpha = 0.3' 
    
        plt.subplot(2, 1, 1)
        for q_deltas in q_deltas:
            plt.plot(q_deltas, label=label)
        plt.title('Slowly changing environment [std_dev=0.01]')
        plt.xlabel('steps')
        plt.ylabel('difference between estimated and true Q')
        plt.legend()

    environment = NonStationaryBanditEnvironment(k_arm=k, std_dev=0.1)
    for n, agent in enumerate(agents):
        _, _, q_deltas = simulate(runs, time, [agent], environment)
        label = 'alpha = 0.01'
        if n > 0:
            label = 'alpha = 0.3' 
    
        plt.subplot(2, 1, 2)
        for q_deltas in q_deltas:
            plt.plot(q_deltas, label=label)
        plt.title('Quickly changing environment [std_dev=0.1]')
        plt.xlabel('steps')
        plt.ylabel('difference between estimated and true Q')
        plt.legend()

    plt.tight_layout()
    plt.savefig('fig_experiment.png')
    plt.close()   

def main():
    figure_2_2_exp()


if __name__ == '__main__':
    main()
