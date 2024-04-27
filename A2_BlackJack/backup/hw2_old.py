import numpy as np


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
        # print self.states
        
class BlackjackMDP(MDP):
    
    # the discount factor for future rewards
    discountFactor = 0.9 # TODO: set this to the correct value
    states = set()

    def __init__(self): 
        self.terminal_state = ('terminal', 0, False, 0, False)
        self.start_state = ('start', 0, False, 0, False)

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
                    dealer_ace = 0
                    player_ace = 0
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

        test_prob = 0
        # Unpack the dictionary into a list
        for state, prob in state_dict.items(): 
            succ_states.append((state, prob, 0))  
            test_prob += prob

        print(f'Number of possible inital states: {len(succ_states)}')
        print(f'Total inital probability: {test_prob}')

        return succ_states                                
                

    def get_new_hand_value(self, hand, ace, card):
        #Default probability
        prob = 1/13
        # add card value 
        hand += card

       # get an ace, check if it is usable
        if card == 1 and not ace:
            if hand < 11:
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

        return reward
    
    def get_succ_states(self, state):
        succ_states = []
        phase = state[0]
        # get new possible state based on cards drawn
        if phase == 'player':
            for n in range(1, 11):
                # get new hand values based on the card drawn
                player_hand, player_ace, prob = self.get_new_hand_value(state[1], state[2], n)
                # create a new state from the new values
                if player_hand > 21:
                    succ_states.append((self.terminal_state, prob, -1))
                else:    
                    state = ('player', player_hand, player_ace, state[3], state[4])
                    succ_states.append((state, prob, 0))
        
        if phase == 'dealer':
            for i in range(1, 11):
                # get new hand value based on the card drawn
                # print(i)
                dealer_hand, dealer_ace, prob = self.get_new_hand_value(state[3], state[4], i)

                if dealer_hand > 16:
                    reward = self.get_reward(player_value=state[1], dealer_value=dealer_hand)    
                    succ_states.append((self.terminal_state, prob, reward))
                else:    
                    state = ('dealer', state[1], state[2], dealer_hand, dealer_ace)
                    succ_states.append((state, prob, 0))

        return succ_states

    # Return set of actions possible from |state|.
    def actions(self, state):
        actions = []
        # check which phase of the game it is
        phase = state[0]
        # player is free to either hit or stand of their value is 21 or less
        if phase == 'player':
            actions = ['hit', 'stand']
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



def main():
    mdp = BlackjackMDP()
    # mdp.states = set()
    mdp.computeStates()
    for state in mdp.states:
        print(state)
    print(f"The MDP has {len(mdp.states)} reachable state(s).")
    num_player = 0
    num_dealer = 0
    num_start = 0
    num_terminal = 0

    '''
    for hand in range(11,17):
        count = 0
        for state in mdp.states:
            if state[0] == 'dealer' and state[3] == hand and state[4] > 0:
                count += 1
                # print(state)
        print('-------')
        print(f'states where dealer has hand value {hand}  and an ace: {count}')
        print('-------')
    '''    

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


if __name__ == '__main__':
    main()