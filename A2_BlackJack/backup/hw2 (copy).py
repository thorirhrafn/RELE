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

        # Unpack the dictionary into a list
        for state, prob in state_dict.items(): 
            succ_states.append((state, prob, 0))  

        print(f'Number of possible inital states: {len(succ_states)}')

        return succ_states                                
                

    def get_new_hand_value(self, hand, ace, card, phase):
        #Default probability
        prob = 1/13
        # if hand == 11 and card == 11:
        #    print(f'hand value: {hand} , ace: {ace}, card value: {card}')

        # If not ace add value, else check if there already is an useable ace
        if card < 11:
            hand += card
        else:
            if phase == 'player':
                if hand < 13:
                    print(f'phase: {phase}')
                    print(f'card: {card} , ace: {ace} , hand: {hand}')
            if not ace and (hand < 11):
                print(card)
                hand += card
                ace = True
            else:
                hand += 1

        # adjust probability if card has a value of 10
        if card == 10:
            prob = 4/13
                
        # Check if value has gone above 21 when there is a useable ace
        # if so then adjust the hand value and set the ace to False
        if ace and hand > 21:
            hand -= 10
            ace = False
        
        '''
        # need to view different possible cases if an ace is drawn
        if card_value == 11:
            # print(f'hand value: {hand_value} , ace: {ace}')
            
            # Already have an ace, can only add one
            if ace:
                hand_value += 1
            else:
                # if hand is below 11 then can add 11 and the ace is usable
                if hand_value < 11:
                    hand_value += card_value
                    ace = True          
                # otherwise can only add 1 to the hand and the ace is unusable    
                else:
                    hand_value += 1

        # adjust probability if card has a value of 10
        elif card_value == 10:
            prob = 4/13
            hand_value += card_value
        else:
            hand_value += card_value    

        # Check if value has gone above 21 when there is a useable ace
        # if so then adjust the hand value and set the ace to False
        if ace and hand_value > 21:
            hand_value -= 10
            ace = False
        '''
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
            for n in range(2,12):
                # get new hand values based on the card drawn
                hand_value, ace, prob = self.get_new_hand_value(state[1], state[2], n, phase)
                # create a new state from the new values
                if hand_value > 21:
                    succ_states.append((self.terminal_state, prob, -1))
                else:    
                    state = ('player', hand_value, ace, state[3], state[4])
                    succ_states.append((state, prob, 0))
        
        if phase == 'dealer':
            for i in range(2, 12):
                # get new hand value based on the card drawn
                # ace11 = False
                # if state[3] < 15:
                #    if state[4] == True:
                #        ace11 = True
                    # print(f'dealer hand value: {state[3]} , dealer ace: {state[4]}')
                #if ace11:
                    #print('-------------')
                    #print(f'before new hand: hand_value = {state[3]}, ace = {state[4]}, card_value = {i}')

                dealer_hand, dealer_ace, prob = self.get_new_hand_value(state[3], state[4], i, phase)

                # if ace11:
                #    print(f'new hand: hand_value = {hand_value}, ace = {ace}')
                #    print('-------------')

                # create a new state from the new values
                # if hand_value > 16:
                #    succ_states.append((self.terminal_state, prob, 1))
                if dealer_hand > 16:
                    reward = self.get_reward(player_value=state[1], dealer_value=dealer_hand)    
                    succ_states.append((self.terminal_state, prob, reward))
                else:    
                    # if hand_value == 12:
                    #     print(f'dealer hand value: {hand_value} , dealer ace: {ace}')
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
            if state[3] < 17:
                actions = ['hit']    
            else:
                actions = ['stand']
        # if in the start phase then setup the game        
        elif phase == 'start':
            actions = ['start']  
        # if in terminal phase then end the game          
        else:
            actions=['end']

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
        if action == 'start':
            succ_states = self.get_inital_states()
        
        # End, move to a terminal state    
        if action == 'end':
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
    print(f"The MDP has {len(mdp.states)} reachable state(s).")
    num_player = 0
    num_dealer = 0
    num_start = 0
    num_terminal = 0

    count = 0
    for state in mdp.states:
        if state[0] == 'dealer' and state[3] == 14 and state[4] == True:
            count += 1
            # print(state)
    print('-------')
    print(f'states where dealer has hand value 14 and an ace: {count}')
    print('-------')

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
               


if __name__ == '__main__':
    main()