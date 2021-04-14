import pygame
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.optim import Adam
from collections import deque
import random
import copy
# Local imports
import environment
import supportFunctions as sF


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')           # Some global configs
torch.autograd.set_detect_anomaly(True)


REPLAY_MEMORY_SIZE = 50000          # Constants
DISCOUNT = 0.99
EPOCHS = 1000
MIN_REPLAY_MEMORY_SIZE = 5000         # fit after testing < 10000
MINIBATCH_SIZE = 32                  # maybe 32   
UPDATE_TARGET_EVERY = 32
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001
RENDER = False

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=2),
            #nn.BatchNorm2d(32),
            nn.ReLU())
            #nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(in_features=93312, out_features=4)
        self.optimizer = Adam(self.parameters(), lr=0.001)
        self.loss = nn.L1Loss()

    def forward(self, x):
        x = self.conv(x)
        #x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = x.view(x.size(0), -1)
        out = self.fc(x)     # replace in_features with size of oReshape
        return out

class Agent:
    def __init__(self):
        self.model = ConvNet()          # To do: move model parameters to other class
        self.target_model = ConvNet()
        self.model_fit_count = 0
        if device != 'cpu':
            self.model = self.model.cuda()
            self.model.loss = self.model.loss.cuda()
            self.target_model = self.target_model.cuda()
            self.target_model.loss = self.target_model.loss.cuda()
        # init both with same weights
        #self.target_model.fc.weight = self.model.fc.weight      # target model is what we predict against every step
        self.target_model.load_state_dict(self.model.state_dict())   # target model weights will be updated every few steps to keep sanity because of large epsilon
        
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.target_update_counter = 0

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state, step):
        return self.model(state)

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        #self.current_qs_list = list()                                           # get predictions for current_state and future_state with same model
        #self.future_qs_list = list()

        # get random sample of replay memory 
        self.minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # predict current qs of states in minibatch
        # self.current_states = [transition[0] for transition in self.minibatch]
        # for state in self.current_states:
        #     self.current_qs_list.append(self.model(state))

        # predict qs with target model
        # self.new_current_states = [transition[3] for transition in self.minibatch]
        # for state in self.new_current_states:
        #     self.future_qs_list.append(self.target_model(state))
        
        for index, (self.current_states, self.action, self.reward, self.new_current_state, self.done) in enumerate(self.minibatch):
            self.current_q = torch.max(self.model(self.current_states))
            if not self.done:
                self.max_future_q = torch.max(self.target_model(self.new_current_state))
                self.new_q = self.reward + DISCOUNT * self.max_future_q
            else:
                self.new_q = torch.tensor(self.reward, device=device)

            self.l = self.model.loss(self.current_q, self.new_q)
            self.l.backward()

            if terminal_state:                  # To Do: check if this is the right criteria, probably not
                self.model.optimizer.step()
                self.model.optimizer.zero_grad()
                if self.model_fit_count %50 == 0:
                    print("Model fitted " + str(self.model_fit_count) + " times")
                    print("current_q: " + str(self.current_q))
                    print("new_q: " + str(self.new_q))
                    self.model_fit_count += 1

                
            else:
                self.target_update_counter += 1
                
            if self.target_update_counter > UPDATE_TARGET_EVERY:
                self.target_model.load_state_dict(self.model.state_dict())
                self.target_update_counter = 0
                

        

    def getStateAsVec(self):
        # get game window
        rawImg = pygame.surfarray.array3d(self.game.game_window)

        # game window downscaled to 1px per field
        scaledImg = cv2.resize(rawImg, (0,0), fx=0.1, fy=0.1)            

        # game window to input Vectors - for now just lists - should be optimized in future iters
        # replace RGB with just 1,0,-1 for the 3 different colors ingame
        pxrow = [-1]
        pxscaled = list()
        pxscaled.append([-1] * (int(self.game.frame_size_x/10)+2))
        for i in scaledImg:
            for j in i:
                if j[1] == 255:
                    pxrow.append(-1)
                elif j[2] == 255:
                    pxrow.append(1)
                else:
                    pxrow.append(0)
            pxrow.append(-1)
            pxscaled.append(pxrow)
            pxrow = [-1]
        pxscaled.append([-1] * (int(self.game.frame_size_x/10)+2))
        torchVec = torch.tensor(pxscaled, dtype= torch.float).unsqueeze(0).unsqueeze(0).to(device)
        #torchVec = torchVec.to(device)
        return torchVec
    
    def run_game(self):
        self.epoch_reward = 0
        self.episode_reward = 1
        self.max_epoch_reward = 0
        epsilon = 1
        epscount = 0
        for self.epoch in range(EPOCHS):
            self.game = environment.Game(RENDER)  # later threadable - To Do: Stop rendering of game if in for loop
            
            print(f"Reward of episode: {self.episode_reward}")
            self.epoch_reward += self.episode_reward
            if self.epoch_reward > self.max_epoch_reward:
                self.max_epoch_reward = self.epoch_reward
                self.model.save(f'model_{self.epoch}.model')
            self.episode_reward = 0
            while self.game.reward != -1:
                self.done = False
                ### interesting Variables ####
                # game.food_pos              #
                # game.snake_pos             #
                # game.snake_body            #
                # game.reward                #
                ##############################

                # set manual True if you want to manual navigate the snake
                self.manual = True
                if self.manual == False:  
                    # Input 0, 1, 2, 3
                    self.keypressed = input()
                    # Call to step ingame, all variables acessible through self.game
                    if self.keypressed != '':
                        self.keypressed = int(self.keypressed)
                    self.game.step(self.keypressed)
                    print(self.game.snake_pos)

                else:
                    ####### Training #######
                    self.current_state = self.getStateAsVec()
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

                    self.reward = self.game.reward

                    if self.reward == -1:
                        self.done = True
                    self.episode_reward += self.reward

                    self.new_state = self.getStateAsVec()
                    self.update_replay_memory((self.current_state, self.action, self.reward, self.new_state, self.done))
                    self.train(self.done, 0)
                    self.current_state = self.new_state

                    if epsilon > MIN_EPSILON and self.model_fit_count != 0:
                        epsilon *= EPSILON_DECAY
                        epsilon = max(MIN_EPSILON, epsilon)
                        
                        if epscount % 100 == 0:
                            print("Current Epsilon: " + str(epsilon))
                        epscount+=1
                        
                    ###### /Training ######           
                if self.game.reward == -1:
                    self.game.quit()


sF.seed_everything(1) 
agent = Agent()

agent.run_game()