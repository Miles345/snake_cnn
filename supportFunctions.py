import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pygame
import cv2

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def getStateAsVec(game_window, frame_size):
    # get game window
    rawImg = pygame.surfarray.array3d(game_window)

    # game window downscaled to 1px per field
    scaledImg = cv2.resize(rawImg, (0,0), fx=0.1, fy=0.1)            

    # game window to input Vectors - for now just lists - should be optimized in future iters
    # replace RGB with just 1,0,-1 for the 3 different colors ingame
    pxrow = [-1]
    pxscaled = list()
    pxscaled.append([-1] * (int(frame_size/10)+2))
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
    pxscaled.append([-1] * (int(frame_size/10)+2))
    torchVec = torch.tensor(pxscaled, dtype= torch.float).unsqueeze(0).unsqueeze(0)
    #torchVec = torchVec.to(device)
    return torchVec

def print_stepcount():
    with open(f"steplist.pickle", "rb") as handle:
        steplist = pickle.load(handle)
    
    plt.scatter(range(len(steplist)), steplist, s= 5)
    plt.show()

print_stepcount()