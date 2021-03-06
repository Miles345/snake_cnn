import pygame, sys, time, random

# def game_over isn't used anymore. now it just sets the reward to -1

class Game:
    def __init__(self, render):
        self.render = render
        # Difficulty settings
        # Easy      ->  10
        # Medium    ->  25
        # Hard      ->  40
        # Harder    ->  60
        # Impossible->  120
        self.difficulty = 100000

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
        #pygame.display.set_caption('Snake Eater')
        self.game_window = pygame.display.set_mode((self.frame_size_x, self.frame_size_y))


        # Colors (R, G, B)
        self.black = pygame.Color(0, 0, 0)
        self.white = pygame.Color(255, 255, 255)
        self.red = pygame.Color(255, 0, 0)
        self.green = pygame.Color(0, 255, 0)
        self.blue = pygame.Color(0, 0, 255)


        # FPS (frames per second) controller
        self.fps_controller = pygame.time.Clock()

        #self.keypressed = random.randint(0,4)
        direct = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.direction = random.choice(direct)
        self.change_to = self.direction


        # Game variables
        self.snake_pos = [random.randint(3, 47)*10, random.randint(3, 47)*10]
        #self.snake_body = [[100, 50], [100-10, 50], [100-(2*10), 50]]
        if self.direction == 'UP':
            self.keypressed = 0
            self.snake_body = [self.snake_pos, [self.snake_pos[0], self.snake_pos[1]+10], [self.snake_pos[0], self.snake_pos[1]+20]]
        if self.direction == 'DOWN':
            self.keypressed = 1
            self.snake_body = [self.snake_pos, [self.snake_pos[0], self.snake_pos[1]-10], [self.snake_pos[0], self.snake_pos[1]-20]]
        if self.direction == 'LEFT':
            self.keypressed = 2
            self.snake_body = [self.snake_pos, [self.snake_pos[0]+10, self.snake_pos[1]], [self.snake_pos[0]-20, self.snake_pos[1]]]
        if self.direction == 'RIGHT':
            self.keypressed = 3
            self.snake_body = [self.snake_pos, [self.snake_pos[0]-10, self.snake_pos[1]], [self.snake_pos[0]-20, self.snake_pos[1]]] 

        ### moar food ###
        self.foodcount = 30
        self.foodposlist = list()
        for _ in range(self.foodcount):
            self.foodposlist.append([random.randrange(1, (self.frame_size_x//10)) * 10, random.randrange(1, (self.frame_size_y//10)) * 10])
        self.food_spawn = True

        

        self.score = 0
        self.reward = 0

    def step(self, keypressed):
        # Main logic
        self.fps_controller.tick(30)
        self.reward = 0
        #for event in pygame.event.get():                                                                 # Here control of snake

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
        
############ Here control from snake.py ###############
        self.keypressed = keypressed
        if self.keypressed == 0:                      
            self.change_to = 'UP'
        if self.keypressed == 1:
            self.change_to = 'DOWN'
        if self.keypressed == 2:
            self.change_to = 'LEFT'
        if self.keypressed == 3:
            self.change_to = 'RIGHT'
############         /changes           ###############


        # Making sure the snake cannot move in the opposite direction instantaneously
        if self.change_to == 'UP' and self.direction != 'DOWN':
            self.direction = 'UP'
        if self.change_to == 'DOWN' and self.direction != 'UP':
            self.direction = 'DOWN'
        if self.change_to == 'LEFT' and self.direction != 'RIGHT':
            self.direction = 'LEFT'
        if self.change_to == 'RIGHT' and self.direction != 'LEFT':
            self.direction = 'RIGHT'

        self.cont = self.snake_pos.copy()
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

        self.snake_body.insert(0, self.snake_pos)
        self.snake_body[1] = self.cont

        for foodpos in self.foodposlist:
            if self.snake_pos[0] == foodpos[0] and self.snake_pos[1] == foodpos[1]:
                self.food_spawn = False
                self.reward = 1
                self.foodposlist.remove(foodpos)
                break
        else:
            self.snake_body.pop()

        # Spawning food on the screen
        if not self.food_spawn:
            self.foodposlist.append([random.randrange(1, (self.frame_size_x//10)) * 10, random.randrange(1, (self.frame_size_y//10)) * 10])
        self.food_spawn = True

        # GFX
        self.game_window.fill(self.black)
        for pos in self.snake_body:
            # Snake body
            # .draw.rect(play_surface, color, xy-coordinate)
            # xy-coordinate -> .Rect(x, y, size_x, size_y)
            if pos == self.snake_pos:
                pygame.draw.rect(self.game_window, self.red, pygame.Rect(pos[0], pos[1], 10, 10))
            else:
                pygame.draw.rect(self.game_window, self.green, pygame.Rect(pos[0], pos[1], 10, 10))

        # Snake food
        for foodpos in self.foodposlist:
            pygame.draw.rect(self.game_window, self.blue, pygame.Rect(foodpos[0], foodpos[1], 10, 10))

        # Game Over conditions
        # Getting out of bounds
        if self.snake_pos[0] < 0 or self.snake_pos[0] > self.frame_size_x-10:
            self.reward = -1

        if self.snake_pos[1] < 0 or self.snake_pos[1] > self.frame_size_y-10:

            self.reward = -1
        # Touching the snake body
        for block in self.snake_body[1:]:
            if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                self.reward = -1


        # Refresh game screen
        if self.render == True:
            pygame.display.update()

    def quit(self):
        pygame.quit()