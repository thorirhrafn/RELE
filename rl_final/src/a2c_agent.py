import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class Actor(nn.Module):

    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        x = F.softmax(x, dim = -1)
        return x

class Critic(nn.Module):

    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        value = self.linear3(x)
        return value    
    

class A2CAgent:
    """
    PyTorch Implementation of Advantage Actor-Critic Model
    """

    def __init__(self, state_size, action_size) -> None:
        super().__init__()
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size, action_size)

        # Actor and Critic Optimizers
        self.actor_optim = optim.Adam(self.actor.parameters())
        self.critic_optim = optim.Adam(self.critic.parameters())

        # Discount rate
        self.gamma = .99

    def action_sample(self, state):
        """
        Sampling action
        """
        state = torch.FloatTensor(state)
        dist = Categorical(self.actor(state))
        action = dist.sample().item()
        return action

    def update(self, final_state, buffer):
        # Unzipping episode experience
        rewards, states, actions, dones = buffer.unzip()

        # Converting to tensors
        states = torch.tensor(states).float()
        actions = torch.tensor(actions)

        # Calculating discounted cumulative rewards
        final_value = self.critic(torch.FloatTensor(final_state))

        sum_reward = final_value
        discnt_rewards = []
        for step in reversed(range(len(rewards))):
            sum_reward = rewards[step] + self.gamma * sum_reward * (1 - dones[step])
            discnt_rewards.append(sum_reward)
        discnt_rewards.reverse()

        # Calculating Advantage
        discnt_rewards = torch.cat(discnt_rewards).detach()
        values = self.critic(states).squeeze(1)
        advantage = discnt_rewards - values

        # Calculating Gradient
        probs = self.actor(states)
        sampler = Categorical(probs)

        # log(Pi(at|st))
        log_probs = sampler.log_prob(actions)
        E = (log_probs * advantage.detach()).mean()

        # Since our goal is to maximize E and optimizers are made for
        # minimization, we are changing the sign of E value
        actor_loss = -E

        # Executing gradient shift: theta = theta + a x Gradient
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Optimizing Critic with MSE loss
        critic_loss = advantage.pow(2).mean()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # returns `advantage` for debug purposes
        return advantage.detach().numpy()    