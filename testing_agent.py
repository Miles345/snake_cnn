import pickle
import environment
import supportFunctions as sF
import torch
import torch.nn as nn
from torch.optim import Adam

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=1),
            #nn.BatchNorm2d(32),
            nn.ReLU())
            #nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=5, stride=1),
            #nn.BatchNorm2d(32),
            nn.ReLU())
            #nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, stride=1),
            #nn.BatchNorm2d(32),
            nn.ReLU())
            #nn.MaxPool2d(kernel_size=2, stride=2))
        #self.drop = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(in_features=12168, out_features=500)
        self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=4)
        self.optimizer = Adam(self.parameters(), lr=0.001)
        self.loss = nn.L1Loss()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)   # Flatten
        #x = self.drop(x)
        x = self.fc1(x)
        out = self.fc2(x)     # replace in_features with size of oReshape
        return out

RENDER = True
GAMES = 10
model = ConvNet()
model_path = f"models/model_1619379563735850300.model"

with open(model_path, 'rb') as handle:
    model = torch.load(handle)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

model = model.cuda()
for _ in range(GAMES):
    game = environment.Game(RENDER)
    while game.reward != -1:
        state = sF.getStateAsVec(game.game_window, game.frame_size_x).to(device=device)
        action = torch.argmax(model(state)).item()
        game.step(action)
        
