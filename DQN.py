import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
from collections import namedtuple
import random
from n_puzzle import N_puzzle

class hidden_unit(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super(hidden_unit, self).__init__()
        self.activation = activation
        self.linear = nn.Linear(in_channels, out_channels)
    
    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)   
        return out
        
#Here I use the linear model as the network
class Q_learning(nn.Module):
    def __init__(self, in_channels, hidden_layers, out_channels = 4, unit = hidden_unit, activation = F.relu):
        super(Q_learning, self).__init__()
        assert type(hidden_layers) is list
        self.in_channels = in_channels
        self.hidden_units = nn.ModuleList()
        prev_layer = in_channels
        for hidden in hidden_layers:
            self.hidden_units.append(unit(prev_layer, hidden, activation))
            prev_layer = hidden
        self.final_unit = nn.Linear(prev_layer, out_channels)
    
    def forward(self, x):
        out = x.view(-1, self.in_channels).float()
        for unit in self.hidden_units:
            out = unit(out)
        out = self.final_unit(out)
        return out

# The transition and ReplayMemory are copied from the website:
# http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
Transition = namedtuple('Transition',
                        ('state', 'action', 'new_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        #"""Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)  
    
#Just follow the object-oriented coding style...use a class to train and test    
class RL_training():
    def __init__(self, N, difficulty, Q_learning, epsilon = 1, gamma = 0.5, lr = 0.1, buffer = 100, batch_size = 40):
        self.N = N
        self.difficulty = difficulty
        self.N_puzzle = N_puzzle(self.N, difficulty)
        self.model = Q_learning
        self.criterion = torch.nn.MSELoss()
        self.optimizer = optim.RMSprop(self.model.parameters(), lr = lr)
        self.epsilon = epsilon
        self.gamma = gamma
        self.buffer = buffer
        self.batch_size = batch_size
        self.memory = ReplayMemory(buffer) 
    def train(self, epochs = 5001, record_sections = 1000):
        #to count the success rate in one record_section
        sucess_count = 0
        for i in range(epochs):
            if i%record_sections ==0:
                print("epoch is %d" %(i))
                print("epsilon is %.2f" % self.epsilon)
                print("success rate: %.2f" % (sucess_count/record_sections*100))
                sucess_count = 0
            self.N_puzzle = N_puzzle(self.N, self.difficulty)
            step = 0
            while True:
                #get the current state
                state = Variable(self.N_puzzle.board.clone().view(1,-1))
                #get the current Q values
                qval = self.model.forward(state)
                if (np.random.random() < self.epsilon): 
                    #choose random action
                    action = np.random.randint(0,4)
                else: 
                    #choose best action from Q(s,a) values
                    action = np.argmax(qval.data)
                #Take action, get the reward
                reward = self.N_puzzle.move(action) - self.N_puzzle.manhattan() 
                #Acuqire new state
                new_state = Variable(self.N_puzzle.board.clone().view(1,-1))
                #push the state-pair into memory
                self.memory.push(state.data, action, new_state.data, reward)
                step += 1
                # If the memory is not full, skip the update part
                if (len(self.memory) < self.buffer): #if buffer not filled, add to it
                    state = new_state
                    if reward == 10: #if reached terminal state, update game status
                        break
                    elif step >= 20:
                        break
                    else:
                        continue
                #sample a batch of state-pairs
                transitions = self.memory.sample(self.batch_size)
                batch = Transition(*zip(*transitions))
                state_batch = Variable(torch.cat(batch.state))
                action_batch = Variable(torch.LongTensor(batch.action)).view(-1,1)
                new_state_batch = Variable(torch.cat(batch.new_state), volatile = True)
                reward_batch = Variable(torch.FloatTensor(batch.reward))
                # the non_final_mask is to determine the reward
                # If a state achieves the goal, it should only get the reward, not reward + QMax
                # Because there is no future state for it.
                non_final_mask = (reward_batch != 10)
                #Let's run our Q function on S to get Q values for all possible actions
                # we only update the qval[action], leaving qval[not action] unchanged
                state_action_values = self.model(state_batch).gather(1, action_batch)
                #new Q
                newQ = self.model(new_state_batch)
                maxQ = newQ.max(1)[0]
                #update the y
#                 if reward == 0: #non-terminal state
#                     update = (-1.0*distance + (self.gamma * maxQ.data))
#                 else: #terminal state
#                     update = reward
                y = reward_batch
                y[non_final_mask] += self.gamma * maxQ[non_final_mask]
                #the y does not need to be back-propogated, so set to be not volatile.
                y.volatile = False
                loss = self.criterion(state_action_values, y)
                # Optimize the model
                self.optimizer.zero_grad()
                loss.backward()
                for param in self.model.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()
                # if an agent walks so many times and does not achieve the goal, it is likely that it has lost
                # Therefore, let the agent re-try the puzzle 
                if reward == 10:
                    if i %record_sections == 0:
                        print("%d steps are trained and the reward is %d" %(step, reward))
                    sucess_count += 1
                    break
                if step >= 20:
                    if i %record_sections == 0:
                        print("%d steps are trained and stop" % (step))
                    break
            if self.epsilon > 0.1:
                self.epsilon -= (1/epochs)
                
    
    def test(self, difficulty = None):
            if type(difficulty) == int:
                self.difficulty = difficulty
            i = 0
            print("Initial State:")
            self.N_puzzle = N_puzzle(self.N, self.difficulty)
            print(self.N_puzzle.display())
            status = 1
            #while game still in progress
            while(status == 1):
                state = Variable(self.N_puzzle.board.clone(), requires_grad = False)
                qval = self.model(state)
                print(qval.data)
                print("distance: ",self.N_puzzle.manhattan())
                action = np.argmax(qval.data) #take action with highest Q-value
                print('Move #: %s; Taking action: %s' % (i, action))
                reward = self.N_puzzle.move(action)
                print(self.N_puzzle.display())
                if reward == 10:
                    status = 0
                    print("Reward: %s" % (reward,))
                i += 1 #If we're taking more than 10 actions, just stop, we probably can't win this game
                if (i > 15):
                    print("Game lost; too many moves.")
                    break