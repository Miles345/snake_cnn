import torch
import os
import random
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

#########backup############

def train(self, terminal_state, step):
    if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
        return
    # get random sample of replay memory 
    self.minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
    # predict current qs of states in minibatch
    self.current_states = np.array([transition[0] for transition in self.minibatch])
    self.current_qs_list = self.model(self.current_states)
    # predict qs with target model
    self.new_current_states = np.array([transition[3] for transition in self.minibatch])
    self.future_qs_list = self.target_model(self.new_current_states)

    self.X = []
    self.Y = []
    
    for index, (self.current_states, self.action, self.reward, self.new_current_state, self.done) in enumerate(self.minibatch):
        if not self.done:
            self.max_future_q = torch.max(self.future_qs_list[index])
            self.new_q = self.reward + DISCOUNT * self.max_future_q
        else:
            self.new_q = torch.tensor(self.reward)

        self.current_qs = self.current_qs_list[index]
        if terminal_state:
            self.new_q_tensor = self.current_qs.index_fill(0,self.action, self.newq)
            self.l = self.model.loss(self.current_qs, self.new_q_tensor)
            self.l.backward()
            self.model.optimizer.step()
            self.train_loss += self.l.intem() * self.current_states.size()
        else:
            self.target_update_counter += 1
            self.model.eval()
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_Weights())
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
                #self.np_q_values = self.q_values
                #self.np_q_values = self.np_q_values.detach().cpu().numpy()  # Try out more stuff here. This here rly necessary?
                #self.action = np.argmax(self.np_q_values)
                # torch.max returns max value of tensor
                # torch.argmax return indice of max value of tensor
                self.action = torch.max(self.q_values)
                print(self.q_values)
                print(self.action.item())
                self.reward = self.game.reward
                # Exec next step
                self.game.step(self.action.item())
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


