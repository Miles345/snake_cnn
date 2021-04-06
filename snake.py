import pygame, sys, time, random
import torch
import torch.nn as nn
import numpy as np
import cv2
import torchvision as tv
from torch.optim import Adam
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

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


class Game:
    def __init__(self):
        # Difficulty settings
        # Easy      ->  10
        # Medium    ->  25
        # Hard      ->  40
        # Harder    ->  60
        # Impossible->  120
        self.difficulty = 25

        # Window size
        self.frame_size_x = 500
        self.frame_size_y = 500

        # Checks for errors encountered
        self.check_errors = pygame.init()
        # pygame.init() example output -> (6, 0)
        # second number in tuple gives number of errors
        if self.check_errors[1] > 0:
            print(f'[!] Had {self.check_errors[1]} errors when initialising game, exiting...')
            sys.exit(-1)
        else:
            print('[+] Game successfully initialised')


        # Initialise game window
        pygame.display.set_caption('Snake Eater')
        self.game_window = pygame.display.set_mode((self.frame_size_x, self.frame_size_y))


        # Colors (R, G, B)
        self.black = pygame.Color(0, 0, 0)
        self.white = pygame.Color(255, 255, 255)
        self.red = pygame.Color(255, 0, 0)
        self.green = pygame.Color(0, 255, 0)
        self.blue = pygame.Color(0, 0, 255)


        # FPS (frames per second) controller
        self.fps_controller = pygame.time.Clock()


        # Game variables
        self.snake_pos = [100, 50]
        self.snake_body = [[100, 50], [100-10, 50], [100-(2*10), 50]]

        self.food_pos = [random.randrange(1, (self.frame_size_x//10)) * 10, random.randrange(1, (self.frame_size_y//10)) * 10]
        self.food_spawn = True

        self.keypressed = 'd'
        self.direction = 'RIGHT'
        self.change_to = self.direction

        self.score = 0
        self.reward = 0


    # Game Over
    def game_over(self):
        my_font = pygame.font.SysFont('times new roman', 90)                                            # Here -1 backprop
        game_over_surface = my_font.render('YOU DIED', True, self.red)
        game_over_rect = game_over_surface.get_rect()
        game_over_rect.midtop = (self.frame_size_x/2, self.frame_size_y/4)
        self.game_window.fill(self.black)
        self.game_window.blit(game_over_surface, game_over_rect)
        #self.show_score(0, self.red, 'times', 20)
        pygame.display.flip()
        time.sleep(3)
        pygame.quit()
        sys.exit()


    # Score
    def show_score(self, choice, color, font, size):
        score_font = pygame.font.SysFont(font, size)
        score_surface = score_font.render('Score : ' + str(self.score), True, color)
        score_rect = score_surface.get_rect()
        if choice == 1:
            score_rect.midtop = (self.frame_size_x/10, 15)
        else:
            score_rect.midtop = (self.frame_size_x/2, self.frame_size_y/1.25)
        self.game_window.blit(score_surface, score_rect)
        # pygame.display.flip()

    def step(self, keypressed):
        # Main logic
        self.reward = 0
        self.keypressed = keypressed
        for event in pygame.event.get():                                                                 # Here control of snake
        #     if event.type == pygame.QUIT:
        #         pygame.quit()
        #         sys.exit()
            # Whenever a key is pressed down
            # elif event.type == pygame.KEYDOWN:
            #     # W -> Up; S -> Down; A -> Left; D -> Right
            #     if event.key == pygame.K_UP or event.key == ord('w'):
            #         self.change_to = 'UP'
            #     if event.key == pygame.K_DOWN or event.key == ord('s'):
            #         self.change_to = 'DOWN'
            #     if event.key == pygame.K_LEFT or event.key == ord('a'):
            #         self.change_to = 'LEFT'
            #     if event.key == pygame.K_RIGHT or event.key == ord('d'):
            #         self.change_to = 'RIGHT'
            #     # Esc -> Create event to quit the game
            #     if event.key == pygame.K_ESCAPE:
            #         pygame.event.post(pygame.event.Event(pygame.QUIT))
        

            if self.keypressed == 'w':
                self.change_to = 'UP'
            if self.keypressed =='s':
                self.change_to = 'DOWN'
            if self.keypressed == 'a':
                self.change_to = 'LEFT'
            if self.keypressed == 'd':
                self.change_to = 'RIGHT'



        # Making sure the snake cannot move in the opposite direction instantaneously
        if self.change_to == 'UP' and self.direction != 'DOWN':
            self.direction = 'UP'
        if self.change_to == 'DOWN' and self.direction != 'UP':
            self.direction = 'DOWN'
        if self.change_to == 'LEFT' and self.direction != 'RIGHT':
            self.direction = 'LEFT'
        if self.change_to == 'RIGHT' and self.direction != 'LEFT':
            self.direction = 'RIGHT'

        # Moving the snake
        if self.direction == 'UP':
            self.snake_pos[1] -= 10
        if self.direction == 'DOWN':
            self.snake_pos[1] += 10
        if self.direction == 'LEFT':
            self.snake_pos[0] -= 10
        if self.direction == 'RIGHT':
            self.snake_pos[0] += 10

        # Snake body growing mechanism
        self.snake_body.insert(0, list(self.snake_pos))
        if self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]:
            self.food_spawn = False
            self.reward = 1
        else:
            self.snake_body.pop()

        # Spawning food on the screen
        if not self.food_spawn:
            self.food_pos = [random.randrange(1, (self.frame_size_x//10)) * 10, random.randrange(1, (self.frame_size_y//10)) * 10]
        self.food_spawn = True

        # GFX
        self.game_window.fill(self.black)
        for pos in self.snake_body:
            # Snake body
            # .draw.rect(play_surface, color, xy-coordinate)
            # xy-coordinate -> .Rect(x, y, size_x, size_y)
            pygame.draw.rect(self.game_window, self.green, pygame.Rect(pos[0], pos[1], 10, 10))

        # Snake food
        pygame.draw.rect(self.game_window, self.blue, pygame.Rect(self.food_pos[0], self.food_pos[1], 10, 10))

        # Game Over conditions
        # Getting out of bounds
        if self.snake_pos[0] < 0 or self.snake_pos[0] > self.frame_size_x-10:
            #self.game_over()
            self.reward = -1
        if self.snake_pos[1] < 0 or self.snake_pos[1] > self.frame_size_y-10:
            #self.game_over()
            self.reward = -1
        # Touching the snake body
        for block in self.snake_body[1:]:
            if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                #self.game_over()
                self.reward = -1

        #self.show_score(1, self.white, 'consolas', 20)
        # Refresh game screen
        pygame.display.update()
                                                                           # Here get current positions
        # Refresh rate
        self.fps_controller.tick(self.difficulty)
        if self.reward == -1:
            pygame.quit()

class Agent:
    def __init__(self):
        self.game = Game()
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
seed_everything(seed) 
agent = Agent()

agent.run_game()