import pygame
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.optim import Adam
from collections import deque
import random
import copy
import pickle
import time
# Local imports
import environment
import supportFunctions as sF


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')           # Some global configs
torch.autograd.set_detect_anomaly(True)


REPLAY_MEMORY_SIZE = 500000          # Constants
DISCOUNT = 0.99
EPOCHS = 10000
MIN_REPLAY_MEMORY_SIZE = 200000       # fit after testing < 10000
MINIBATCH_SIZE = 64                  # maybe 32   
UPDATE_TARGET_EVERY = 10
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

RENDER = False
IMPORT_REPLAY_MEMORY = False
IMPORTED_EPSILON = 0.5

MANUAL = False


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=3, padding=2),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_features=2592, out_features=4)
        self.optimizer = Adam(self.parameters(), lr=0.001)
        self.loss = nn.L1Loss()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)   # Flatten
        x = self.drop(x)
        out = self.fc(x)     # replace in_features with size of oReshape
        return out

class Agent:
    def __init__(self):
        self.model = ConvNet()          # To do: move model parameters to other class
        self.target_model = ConvNet()
        self.model_fit_count = 0
        self.first_train_step = True
        # Move models to GPU
        if device != 'cpu':
            self.model = self.model.cuda()
            self.model.loss = self.model.loss.cuda()
            self.target_model = self.target_model.cuda()
            self.target_model.loss = self.target_model.loss.cuda()
        # init both with same weights                                # target model is what we predict against every step
        self.target_model.load_state_dict(self.model.state_dict())   # target model weights will be updated every few steps to keep sanity because of large epsilon
        
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        if IMPORT_REPLAY_MEMORY == True:
            with open(f"replaymemory.pickle", 'rb') as handle:
                self.replay_memory = pickle.load(handle)

        self.target_update_counter = 0
        self.traincount = 0

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state, step):
        return self.model(state)

    def train(self, terminal_state, step):                  # realy call every step? 
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        if self.traincount == 0:
                # Save replay memory at first training step
                with open(f"replaymemory.pickle", 'wb') as handle:
                    pickle.dump(self.replay_memory, handle)
                self.first_train_step = False
        self.traincount += 1
        if self.traincount % 100 == 0:
            torch.save(self.model, f'models/model_{time.time_ns()}.model')

        # get random sample of replay memory 
        self.minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        
        for index, (self.current_states, self.action, self.reward, self.new_current_state, self.done) in enumerate(self.minibatch):

            # Get q of current action with max q
            current_qs = self.model(self.current_states)
            chos_action = torch.argmax(current_qs).item()
            #self.current_q = torch.max(current_qs)
            # if not game over get q from target model else new_q = reward
            new_qs = self.target_model(self.new_current_state)
            new_current_qs = current_qs.detach().clone() # insert self.new_q here  
            if not self.done:
                self.max_future_q = torch.max(new_qs).item()
                self.new_q = self.reward + DISCOUNT * self.max_future_q
            else:
                self.new_q = self.reward

            new_current_qs.data[0,chos_action] = self.new_q

            # Calc loss and backprop
            self.l = self.model.loss(current_qs, new_current_qs)
            self.l.backward()

            if terminal_state:                  # To Do: check if this is the right criteria, probably not
                self.model.optimizer.step()
                self.model.optimizer.zero_grad()
                if self.model_fit_count %50 == 0:
                    print("Model fitted " + str(self.model_fit_count) + " times")
                    print("current_q: " + str(current_qs))
                    print("new_q: " + str(new_current_qs))
                    self.model_fit_count += 1   
            else:
                self.target_update_counter += 1
                
            if self.target_update_counter > UPDATE_TARGET_EVERY:
                self.target_model.load_state_dict(self.model.state_dict())
                self.target_update_counter = 0
    
    def run_game(self):

        list_earned_rewards = list ()

        epsilon = 1
        if IMPORT_REPLAY_MEMORY == True:
            epsilon = IMPORTED_EPSILON
        epscount = 0
        stepcountlist = list()
        for self.epoch in range(EPOCHS):
            self.episode_reward = 0
            self.game = environment.Game(RENDER)  # later threadable
            if self.epoch % 10 == 0:
                print(f"Gamecount:{self.epoch}")
                print(f"Replaymemory Size: {len(self.replay_memory)}")
            
            

            stepcount = 0
            while self.game.reward != -1:
                stepcount +=1
                self.done = False
                ### interesting Variables ####
                # game.food_pos              #
                # game.snake_pos             #
                # game.snake_body            #
                # game.reward                #
                ##############################

                # set manual True if you want to manual control the snake
                
                if MANUAL == True:  
                    # Input 0, 1, 2, 3
                    self.keypressed = input()
                    # Call to step ingame, all variables acessible through self.game
                    if self.keypressed != '':
                        self.keypressed = int(self.keypressed)
                    self.game.step(self.keypressed)
                    print(self.game.snake_pos)

                else:
                    ####### Training #######
                    self.current_state = sF.getStateAsVec(self.game.game_window, self.game.frame_size_x, device)
                    if np.random.random_sample() > epsilon:
                        
                        # predict q and select action
                        self.q_values = self.model(self.current_state)
                        self.action = torch.max(self.q_values)
                    else:
                        self.action = np.random.randint(0, 4)

                    # Exec next step
                    if type(self.action) == torch.tensor:
                        self.game.step(self.action.item())
                    else:
                        self.game.step(self.action)

                    # Get reward from environment
                    self.reward = self.game.reward

                    # Check if game over
                    if self.reward == -1:
                        self.done = True
                    self.episode_reward += self.reward

                    self.new_state = sF.getStateAsVec(self.game.game_window, self.game.frame_size_x, device)
                    self.update_replay_memory((self.current_state, self.action, self.reward, self.new_state, self.done))
                    self.train(self.done, 0)

                    
                    ###### /Training ######    

                # On Game Over:;       
                if self.game.reward == -1:
                    # Count steps taken per game
                    stepcountlist.append(stepcount)
                    if len(stepcountlist) % 10 == 0:
                        with open(f"steplist.pickle", 'wb') as handle:
                            pickle.dump(stepcountlist, handle)
                        
                    
                    # Count rewards earned per game
                    print(f"Reward of episode: {self.episode_reward}")
                    list_earned_rewards.append(self.episode_reward)
                    if len(list_earned_rewards) % 10 == 0:
                        with open(f"rewardslist.pickle", 'wb') as handle:
                            pickle.dump(list_earned_rewards, handle)

                    
                    # Decay epsilon
                    if epsilon > MIN_EPSILON:
                        epsilon *= EPSILON_DECAY
                        epsilon = max(MIN_EPSILON, epsilon)
                                
                        if epscount % 10 == 0:
                            print("Current Epsilon: " + str(epsilon))
                        epscount+=1
                    self.game.quit()


sF.seed_everything(1) 
agent = Agent()

agent.run_game()