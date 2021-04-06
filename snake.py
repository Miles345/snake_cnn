import pygame
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.optim import Adam
# Local imports
import snakegame
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
        self.fc = nn.Linear(in_features=86528, out_features=4)

    def forward(self, x):
        x = self.conv(x)
        #x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = x.view(x.size(0), -1)
        out = self.fc(x)     # replace in_features with size of oReshape
        return out

class Agent:
    def __init__(self):
        self.game = snakegame.Game()
        self.model = ConvNet()
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.loss = nn.L1Loss()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()

    def stateToVec(self):

        # get game window
        rawImg = pygame.surfarray.array3d(self.game.game_window)

        # game window downscaled to 1px per field
        scaledImg = cv2.resize(rawImg, (0,0), fx=0.1, fy=0.1)            

        # game window to input Vectors
        pxrow = [-1]
        pxscaled = list()
        pxscaled.append([-1] * int(self.game.frame_size_x/10))
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
        pxscaled.append([-1] * int(self.game.frame_size_x/10))

        torchVec = torch.tensor(pxscaled, dtype= torch.float).unsqueeze(0).unsqueeze(0)
        torchVec = torchVec.to(device)
        return torchVec
    
    def run_game(self):
        #while True:
        ### interesting Variables ####
        # game.food_pos              #
        # game.snake_pos             #
        # game.snake_body            #
        # game.reward                #
        ##############################

        # Input of commands
        keypressed = input()

        # Call to step ingame, all variables acessible through self.game
        self.game.step(keypressed)

        ###### Training ######
        stateVec = self.stateToVec()
        pred = self.model(stateVec)
        print(pred)


        # updated_q_values = rewards_sample + 0.99 * tf.reduce_max(pred, axis=1)
        # updated_q_values = updated_q_values * (1 - done_sample) - done_sample*abs(self.PENALTY)
        # loss = self.loss(updated_q_values, )
        # cv2.imshow("sdf", scaledImg)
        # cv2.waitKey(0) # wait for ay key to exit window
        # cv2.destroyAllWindows() # close all windows
        
        # if self.game.reward == -1:
        #     sys.exit()

        
seed = 12341
sF.seed_everything(seed) 
agent = Agent()

agent.run_game()