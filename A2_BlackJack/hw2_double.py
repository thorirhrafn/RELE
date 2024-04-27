import numpy as np
import matplotlib.pyplot as plt

# An abstract class representing a Markov Decision Process (MDP).
class MDP:
    def __init__(self):
        self.computeStates()

    # discount factor
    discountFactor = 1
    
    # Return the start state.
    def startState(self): raise NotImplementedError("Override me")

    # Return set of actions possible from |state|.
    def actions(self, state): raise NotImplementedError("Override me")

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    # Mapping to notation from class:
    #   state = s, action = a, newState = s', reward = r, prob = p(s', r | s, a)
    # Terminal states should transition to themselves with probability 1 and reward 0
    def succAndProbReward(self, state, action): raise NotImplementedError("Override me")

    # Compute set of states reachable from startState.  Helper function
    # to know which states to compute values and policies for.
    # This function sets |self.states| to be the set of all reachable states.
    def computeStates(self):
        self.states = set()
        queue = []
        self.states.add(self.startState())
        # run a simple BFS to enumerate the state space
        queue.append(self.startState())
        while len(queue) > 0:
            state = queue.pop()
            for action in self.actions(state):
                for newState, prob, reward in self.succAndProbReward(state, action):
                    if newState not in self.states:
                        self.states.add(newState)
                        queue.append(newState)
        
        
class BlackjackMDP(MDP):
    
    # the discount factor for future rewards
    discountFactor = 0.9 # TODO: set this to the correct value
    states = set()

    def __init__(self): 
        self.terminal_state = ('terminal', 0, False, 0, False)
        self.start_state = ('start', 0, False, 0, False)
        self.double = False

    def startState(self): 
        return self.start_state

    def get_inital_states(self):
        state_dict = {}
        succ_states = []
        phase = 'player'
        # get all possible states based on 3 cards drawn at the start
        for card1 in range(2,12):
            for card2 in range(2,12):
                for card3 in range(2,12):
                    # initialize aces and probabilities
                    dealer_ace = False
                    player_ace = False
                    prob1 = 1/13
                    prob2 = 1/13
                    prob3 = 1/13
                    # Check for dealer ace
                    if card3 == 11:
                        dealer_ace = True
                    # Set probabilities    
                    if card3 == 10:
                        prob3 = 4/13
                    if card2 == 10:
                        prob2 = 4/13
                    if card1 == 10:
                        prob1 = 4/13
                    # Check for player ace
                    if card1 == 11 or card2 == 11:
                        player_ace = True
                    # Check for 2 aces
                    if card1 == 11 and card2 == 11:
                        card2 = 1
                    
                    # collect initial states into a dictionary and update probiblity to get only unique states
                    state = (phase, card1+card2, player_ace, card3, dealer_ace)   
                    prob = prob1 * prob2 * prob3
                    if state in state_dict:
                        state_dict[state] += prob
                    else:
                        state_dict[state] = prob        

        # test_prob = 0
        # Unpack the dictionary into a list
        for state, prob in state_dict.items(): 
            succ_states.append((state, prob, 0))  
            # test_prob += prob

        # print(f'Number of possible inital states: {len(succ_states)}')
        # print(f'Total inital probability: {test_prob}')

        return succ_states                                
                

    def get_new_hand_value(self, hand, ace, card):
        #Default probability
        prob = 1/13
        # add card value 
        hand += card

       # get an ace, check if it is usable
        if card == 1 and not ace:
            if hand < 12:
                hand += 10
                ace = True

        # Check if value has gone above 21 and if there is a useable ace
        # if so then adjust the hand value and set useable ace to False
        if hand > 21:
            if ace :
                hand -= 10
                ace = False
        
        # adjust probability if card has a value of 10
        if card == 10:
            prob = 4/13
        
        return (hand, ace, prob)    
    
    def get_reward(self, player_value, dealer_value):
        reward = 0
        # compare the values and set the reward
        if dealer_value > 21:
            reward = 1
        else:    
            if player_value > dealer_value:
                reward = 1
            elif player_value < dealer_value: 
                reward = -1    

        if self.double:
            reward = 2 * reward

        return reward
    
    def get_succ_states(self, state):
        succ_states = []
        phase = state[0]
        # get new possible state based on cards drawn
        if phase == 'player':
            for n in range(1, 11):
                
                player_hand, player_ace, prob = self.get_new_hand_value(state[1], state[2], n)
        
                if player_hand > 21:
                    if self.double:
                        succ_states.append((self.terminal_state, prob, -2))
                    else:
                        succ_states.append((self.terminal_state, prob, -1))
                else:
                    if self.double:
                        new_state1 = ('player', player_hand, player_ace, state[3], state[4])
                        succ_states.append((new_state1, prob, 0))
                        new_state2 = ('dealer', player_hand, player_ace, state[3], state[4])
                        succ_states.append((new_state2, prob, 0))
                    else:        
                        new_state = ('player', player_hand, player_ace, state[3], state[4])
                        succ_states.append((new_state, prob, 0))
        
        if phase == 'dealer':
            for i in range(1, 11):
                # get new hand value based on the card drawn
                dealer_hand, dealer_ace, prob = self.get_new_hand_value(state[3], state[4], i)

                if dealer_hand > 16:
                    reward = self.get_reward(player_value=state[1], dealer_value=dealer_hand)    
                    succ_states.append((self.terminal_state, prob, reward))
                else:    
                    new_tate = ('dealer', state[1], state[2], dealer_hand, dealer_ace)
                    succ_states.append((new_tate, prob, 0))

        return succ_states

    # Return set of actions possible from |state|.
    def actions(self, state):
        actions = []
        # check which phase of the game it is
        phase = state[0]
        # player is free to either hit or stand of their value is 21 or less
        if phase == 'player':
            actions = ['hit', 'stand', 'double']
        # the dealer has to hit if their value is less then 17, otherwise he must stand                
        elif phase == 'dealer':
            if state[3] > 16:
                actions = ['stand']    
            else:
                actions = ['hit']
        # if in the start phase then setup the game        
        elif phase == 'start':
            actions = ['wait']  
        # if in terminal phase then end the game          
        else:
            actions=['wait']

        return actions

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    # Mapping to notation from class:
    #   state = s, action = a, newState = s', reward = r, prob = p(s', r | s, a)
    # Terminal states should transition to themselves with probability 1 and reward 0
    def succAndProbReward(self, state, action):
        succ_states = []
        phase = state[0]
        
        # If Start, setup the board and move to the player phase
        # Else end, move to a terminal state    
        if action == 'wait':
            if phase == 'start':
                succ_states = self.get_inital_states()
            else:
                succ_states.append((self.terminal_state, 1, 0))               
                
        # Else check if its the player or dealer phase    
        if phase == 'player':
            # if it is the players turn check if the actions are stand or hit
            # if stand then move to the dealers turn with probability of 1, else find possible new card values and their probabilities
            if action == 'stand':
                state = ('dealer', state[1], state[2], state[3], state[4])
                succ_states.append((state, 1, 0))
            else:
                if action == 'double':
                    self.double = True
                succ_states = self.get_succ_states(state)
                
        if phase == 'dealer':
            # if it is the dealers turn check if the actions are stand or hit
            # if stand then move to the terminal state with probability of 1, else find possible new card values and their probabilities
            if action == 'stand':
                # dealer stands, need to compare card values in the state to check for reward
                reward = self.get_reward(player_value=state[1], dealer_value=state[3])
                # game is over, got to the terminal state
                succ_states.append((self.terminal_state, 1, reward))
            else:
                succ_states = self.get_succ_states(state)

        return succ_states

def value_iteration(mdp, gamma=0.99, theta=0.001):
    # Initialze values for all states
    V = {}
    for state in mdp.states:
        V[state] = 0

    # run over the state space until delta < theta
    while True:
        delta = 0
        for state in mdp.states:
            value = V[state]
            new_value = -1000

            # Iterate over all possible actions from the current state
            for action in mdp.actions(state):
                q = 0
                # Calculate the expected values of the successor states created from the action
                for newState, prob, reward in mdp.succAndProbReward(state, action):
                    q += prob * (reward + gamma * V[newState])
                # Update the value function if there is an increase
                new_value = max(new_value, q)

            V[state] = new_value
            delta = max(delta, abs(value - new_value))

        if delta < theta:
            break    
    
    return V

def get_policy(mdp, V, gamma=0.99):
    tau = {}

    # iterate through all states in the mdp
    for state in mdp.states:
        opt_action = None
        opt_value = -1000

        # Iterate through all actions in the current state
        for action in mdp.actions(state):
            q = 0
            # Calculate the expected values of the successor states using the values found for each state
            for newState, prob, reward in mdp.succAndProbReward(state, action):
                q += prob * (reward + gamma * V[newState])

            # Check if the current action has a higher  value
            if q > opt_value:
                opt_value = q
                opt_action = action

        # Assign the optimal action to each state in the optimal policy
        tau[state] = opt_action

    return tau 

def visualize_value_function(V, tau):

    # Player's starting hand without ace and the dealer's card
    player_values = range(4, 21)
    dealer_values = range(2, 12)

    value_function = np.zeros((len(player_values), len(dealer_values)))
    actions = np.zeros((len(player_values), len(dealer_values)))
    
    for i, pv in enumerate(player_values):
        for j, dv in enumerate(dealer_values):
            if dv > 10:
                value_function[i,j] = V[('player', pv, False, dv, True)]
                if tau[('player', pv, False, dv, True)] == 'hit':
                    actions[i,j] = 1
                if tau[('player', pv, False, dv, True)] == 'double':
                    actions[i,j] = 2       
            else:    
                value_function[i,j] = V[('player', pv, False, dv, False)]
                if tau[('player', pv, False, dv, False)] == 'hit':
                    actions[i,j] = 1
                if tau[('player', pv, False, dv, False)] == 'double':
                    actions[i,j] = 2       

    # Visualize the value function
    plt.figure(figsize=(10, 6))
    plt.imshow(value_function, cmap='viridis', origin='lower', aspect='auto')

    plt.xticks(np.arange(len(dealer_values)), dealer_values)
    plt.yticks(np.arange(len(player_values)), player_values)

    plt.colorbar()
    plt.title('Value Function for Blackjack')
    plt.xlabel("Dealer's starting card")
    plt.ylabel("Player's starting  hand with no ace")

    plt.show()

    # Visualize optimal policy, 0 = stand, 1 = hit, 2 = double
    plt.figure(figsize=(10, 6))
    plt.imshow(actions, cmap='viridis', origin='lower', aspect='auto')

    plt.xticks(np.arange(len(dealer_values)), dealer_values)
    plt.yticks(np.arange(len(player_values)), player_values)

    plt.colorbar()
    plt.title('Optimal policy for Blackjack')
    plt.xlabel("Dealer's starting card")
    plt.ylabel("Player's starting  hand with no ace")

    plt.show()


    # Player's starting hand with an ace and the dealer's card
    player_values = range(12, 22)
    dealer_values = range(2, 12)

    value_function = np.zeros((len(player_values), len(dealer_values)))
    actions = np.zeros((len(player_values), len(dealer_values)))

    for i, pv in enumerate(player_values):
        for j, dv in enumerate(dealer_values):
            if dv > 10:
                value_function[i,j] = V[('player', pv, True, dv, True)]
                if tau[('player', pv, True, dv, True)] == 'hit':
                    actions[i,j] = 1
                if tau[('player', pv, True, dv, True)] == 'double':
                    actions[i,j] = 2       
            else:    
                value_function[i,j] = V[('player', pv, True, dv, False)]
                if tau[('player', pv, True, dv, False)] == 'hit':
                    actions[i,j] = 1  
                if tau[('player', pv, True, dv, False)] == 'double':
                    actions[i,j] = 2      
        

    # Visualize the value function
    plt.figure(figsize=(10, 6))
    plt.imshow(value_function, cmap='viridis', origin='lower', aspect='auto')

    plt.xticks(np.arange(len(dealer_values)), dealer_values)
    plt.yticks(np.arange(len(player_values)), player_values)

    plt.colorbar()
    plt.title('Value Function for Blackjack')
    plt.xlabel("Dealer's starting card")
    plt.ylabel("Player's starting  hand with an ace")

    # Show the plot.
    plt.show()

    # Visualize optimal policy, 0 = stand, 1 = hit, 2 = double
    plt.figure(figsize=(10, 6))
    plt.imshow(actions, cmap='viridis', origin='lower', aspect='auto')

    plt.xticks(np.arange(len(dealer_values)), dealer_values)
    plt.yticks(np.arange(len(player_values)), player_values)

    plt.colorbar()
    plt.title('Optimal policy for Blackjack')
    plt.xlabel("Dealer's starting card")
    plt.ylabel("Player's starting  hand with no ace")

    plt.show()


def main():
    mdp = BlackjackMDP()
    mdp.computeStates()
    print(f"The MDP has {len(mdp.states)} reachable state(s).")
    num_player = 0
    num_dealer = 0
    num_start = 0
    num_terminal = 0

    for state in mdp.states:
        if state[0] == 'player':
            num_player += 1
        if state[0] == 'dealer':
            num_dealer += 1
        if state[0] == 'start':
            num_start += 1
        if state[0] == 'terminal':
            num_terminal += 1    
    print(f'player states: {num_player}') 
    print(f'dealer states: {num_dealer}')
    print(f'starting states: {num_start}') 
    print(f'terminal states: {num_terminal}')
    print(f'Total states: {num_dealer+num_player+num_start+num_terminal}')           

    # Perform value iteration over the states
    V = value_iteration(mdp, 0.99, 0.001)

    # Use the optimal values found during the value iteration to get the optimal policy of actions at each state
    tau = get_policy(mdp, V, 0.99)
    
    print('-----------------------')
    print('expected outcome using optimal policy:')
    print(V[mdp.start_state])

    visualize_value_function(V, tau)

if __name__ == '__main__':
    main()