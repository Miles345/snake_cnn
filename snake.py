import pygame
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.optim import Adam
# Local imports
import environment
import supportFunctions as sF

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=2),
            #nn.BatchNorm2d(32),
            nn.ReLU())
            #nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(in_features=93312, out_features=4)

    def forward(self, x):
        x = self.conv(x)
        #x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = x.view(x.size(0), -1)
        out = self.fc(x)     # replace in_features with size of oReshape
        return out

class Agent:
    def __init__(self):
        self.model = ConvNet()          # To do: move model parameters to other class
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.loss = nn.L1Loss()
        self.LEARNING_RATE = 0.1
        self.DISCOUNT = 0.95
        self.EPOCHS = 20
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()

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
        torchVec = torch.tensor(pxscaled, dtype= torch.float).unsqueeze(0).unsqueeze(0)
        torchVec = torchVec.to(device)
        return torchVec
    
    def run_game(self):
        for self.epoch in range(self.EPOCHS):
            self.game = environment.Game()  # later threadable - To Do: Stop rendering of game if in for loop
            while self.game.reward != -1:
                ### interesting Variables ####
                # game.food_pos              #
                # game.snake_pos             #
                # game.snake_body            #
                # game.reward                #
                ##############################

                # set manual True if you want to manual navigate the snake
                self.manual = False
                if self.manual == True:  
                    # Input 0, 1, 2, 3
                    self.keypressed = input()
                    # Call to step ingame, all variables acessible through self.game
                    if self.keypressed != '':
                        self.keypressed = int(self.keypressed)
                    self.game.step(self.keypressed)
                    print(self.game.snake_pos)

                else:
                    ####### Training #######
                    self.stateVec = self.getStateAsVec()
                    # predict q and select action
                    self.q_values = self.model(self.stateVec)
                    self.np_q_values = self.q_values
                    self.np_q_values = self.np_q_values.detach().cpu().numpy()  # Try out more stuff here. This here rly necessary?
                    self.action = np.argmax(self.np_q_values)
                    print(self.action)
                    self.reward = self.game.reward
                    # Exec next step
                    self.game.step(self.action)
                    # Fixing out of bounds if gameborder is reached # not working 
                    if self.reward != -1:
                        self.future_stateVec = self.getStateAsVec()
                    else:
                        self.future_stateVec = self.stateVec
                    # Get maxQ' 
                    self.future_step = self.model(self.future_stateVec)
                    self.max_future_q = np.argmax(self.future_step.detach().cpu().numpy())
                    self.current_q = self.np_q_values[0][self.action]

                    self.new_q = (1- self.LEARNING_RATE) * self.current_q + self.LEARNING_RATE *(self.reward + self.DISCOUNT * self.max_future_q)
                    
                    # how to calc loss without changing whole tensor? Only part with maxq has to be changed
                    self.out = self.loss(self.new_q, self.current_q)
                    self.out.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    ###### /Training ######



            #print(pred)


            # updated_q_values = rewards_sample + 0.99 * tf.reduce_max(pred, axis=1)
            # updated_q_values = updated_q_values * (1 - done_sample) - done_sample*abs(self.PENALTY)
            # loss = self.loss(updated_q_values, )
            # cv2.imshow("sdf", scaledImg)
            # cv2.waitKey(0) # wait for ay key to exit window
            # cv2.destroyAllWindows() # close all windows
            
                if self.game.reward == -1:
                    self.game.quit()


sF.seed_everything(12341) 
agent = Agent()

agent.run_game()