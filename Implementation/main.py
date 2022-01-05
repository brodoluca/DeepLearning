import numpy as np
import tensorflow.python.keras.backend as backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2
from tensorflow import keras

import pygame
import time


DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 128  # How many steps (samples) to use for training


UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = 100  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 7500

# Exploration settings
epsilon = 2 # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.8

#  Stats settings
AGGREGATE_STATS_EVERY = 10  # episodes
SHOW_PREVIEW = True

PY_GAME = True
OPEN_CV = False

WIDTH = 500
HEIGHT = 500
WHITE = (255,255,255) #RGB
RED = (255,0,0) #RGB
BLUE = (0,0,255) #RGB
GREEN = (0,255,0) #RGB
BLACK = (0,0,0) #RGB

RADIUS = 5
COLLISION_RADIUS = 100
WALL = 50
WALLB = False
class Wall:
    def __init__(self, x1, y1, width, height):
        self.sx = x1
        self.sy = y1
        self.width = width
        self.height = height

        self.x1 = x1
        self.y1 = y1+height
        self.x2 = x1+width
        self.y2 = y1
    def __str__(self):
        return f"WALL ({self.x1}, {self.y1},{self.x2}, {self.y2})"
class Blob:
    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)
        self.id = 0


    def __str__(self):
        return f"Blob ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __in__ (self, other):
        for element in other:
            if(self == element):
                return True
        return False


    def action(self, choice):
        '''
        Gives us 9 total movement options. (0,1,2,3,4,5,6,7,8)
        '''
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=-1, y=0)

        elif choice == 6:
            self.move(x=0, y=1)
        elif choice == 7:
            self.move(x=0, y=-1)

        elif choice == 8:
            self.move(x=0, y=0)

    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.size-1:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size-1:
            self.y = self.size-1

def find_point(x1, x2, y1, y2, x, y):
            if (x > x1 and x < x2 and y > y2 and y < y1):
                return True
            return False
class newBlobEnv:
    SIZE = 10
    RETURN_IMAGES = True
    MOVE_PENALTY = 1
    ENEMY_PENALTY = 300
    FOOD_REWARD = 25
    ROBOT_CLOSE_REWARD = 20
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    ACTION_SPACE_SIZE = 9
    NUMBER_OF_ROBOTS = 4
    COMLPETE_REWARD = NUMBER_OF_ROBOTS*ROBOT_CLOSE_REWARD
    COMMUNICATION_RANGE = 3
    PLAYER_N = 1
    TRAINING_ROOM = 500
    robots = []
    graph = []
    screen_list = []

    PLAYER_N = 1  # player key in dict
    # the dict! (colors)
    d = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 0)}
    complete = False
    def __init__(self):
        self.foundGraphs = 0
    def reset(self):
        self.wall = Wall(1,1,2,4)
        self.complete = False
        if(SHOW_PREVIEW):
            pygame.init()
            pygame.display.set_caption('Drone Swarm Simulation')
            self.screen_list.append(pygame.display.set_mode((WIDTH, HEIGHT), 0, 32))
            self.screen_list.append(pygame.display.set_mode((WIDTH, HEIGHT), 0, 32))
            self.screen_list[0].fill(WHITE)
            self.screen_list[1].fill(WHITE)
            self.robots.clear()
            self.graph.clear()
        for robot_id in range(0, self.NUMBER_OF_ROBOTS):
            temp = Blob(self.SIZE)
            finished = False
            is_in_list = False
            while not finished:
                for robot in self.robots:
                    if(temp == robot):
                        is_in_list = True
                    if (find_point(self.wall.x1, self.wall.x2 , self.wall.y1 , self.wall.y2 ,
                                   temp.x, temp.y )):
                        is_in_list = True

                if(is_in_list):
                    is_in_list = False
                    temp = Blob(self.SIZE)
                else:
                    finished= True

            temp.id = robot_id
            self.robots.append(temp)


        self.episode_step = 0

        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())

        return observation

    def calculate_distance(self, robot1, robot2):
        import math
        return math.sqrt((robot1.x - robot2.x)*(robot1.x - robot2.x) + (robot1.y-robot2.y)*(robot1.y-robot2.y))

    def step(self, action, id =0):
        self.coll = False
        self.episode_step += 1
        reward = 600
        old_distances = [0 for i in range(0,self.NUMBER_OF_ROBOTS)]
        for robot in self.robots:
            if(robot.id == id):
                pass
            old_distances[robot.id] = self.calculate_distance(robot, self.robots[id])


        old_pos = [self.robots[id].x, self.robots[id].y]
        #print(old_distances)
        self.robots[id].action(action)
        new_pos = [self.robots[id].x, self.robots[id].y]
        if(new_pos[0] == self.SIZE or new_pos[1] == self.SIZE or new_pos[0] == 0 or new_pos[1] == 0):
            reward -= 300

        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())


        if(find_point(self.wall.x1 , self.wall.x2 , self.wall.y1 , self.wall.y2 , new_pos[0], new_pos[1])):
                reward = -10000
                self.coll = True
                done = True
                self.cll_id = id
                return new_observation, reward, done


        reward += 10 * max( abs(self.robots[id].x - self.wall.x1),abs(self.robots[id].x - self.wall.x2) )
        reward += 10 * max(abs(self.robots[id].y - self.wall.y1), abs(self.robots[id].y - self.wall.y2))



        #if(new_pos[0] < self.wall.x1+1 or new_pos[0] > self.wall.x2+1 ):
         #   reward += 40
        #if (new_pos[0] > self.wall.y1 +2or new_pos[0]  <self.wall.y2 +2):
         #   reward += 40

        for robot in self.robots:
            if(robot.id != id):
                if(self.robots[id] == robot):
                    reward -=50
        #else:
         #   new_observation = (self.player-self.food) + (self.player-self.enemy)
        if (self.robots[id] in self.graph):
            for robot in self.graph:
                dist = self.calculate_distance(robot, self.robots[id])
                if (dist > self.COMMUNICATION_RANGE):
                    reward += -70
                elif (dist <= self.COMMUNICATION_RANGE):
                    reward += 20
            #if(reward == -((len(self.graph)-1)*10)):

                    #self.graph.remove(self.robots[id])
                    #if(len(self.graph) == 1):
                        #self.graph.clear()
        else:
            for robot in self.robots:
                if(robot.id != id):
                    dist = self.calculate_distance(robot, self.robots[id])
                    if (dist > old_distances[robot.id]):
                        reward += -30
                        if (robot in self.graph):
                            reward += -500
                    elif (dist <= old_distances[robot.id]):
                        reward += 40
                        if (robot in self.graph):
                            reward += 90
                    if(robot in self.graph and dist<=self.COMMUNICATION_RANGE):
                        if(self.robots[id] not in self.graph):
                            self.graph.append(self.robots[id])
                            reward += 90
                    if(len(self.graph)==0 and dist <= self.COMMUNICATION_RANGE):
                        self.graph.append(self.robots[id])
                        self.graph.append(robot)
                        reward += 90

        self.check_graph()
        if (len(self.graph) >= self.NUMBER_OF_ROBOTS):
            self.foundGraphs +=1
            reward = 3000
        done = False
        if reward == 3000 or self.episode_step >= self.TRAINING_ROOM:
            self.complete = 1
            if(reward == 3000):
                self.complete = 2
            if(self.episode_step >= self.TRAINING_ROOM):
                reward = -10000
            done = True

        return new_observation, reward, done

    def check_graph(self):
        remove = 0
        for robot_first in self.graph:
            for robot in self.graph:
                if(robot_first.id != robot.id):
                    dist = self.calculate_distance(robot, robot_first)
                    if(dist > self.COMMUNICATION_RANGE):
                        remove +=1
                        #print(dist)
            if(remove == len(self.graph)-1):
                self.graph.remove(robot_first)
            remove = 0

    def print_end(self):
        for i in range(0, len(self.graph) - 1):
            print("Robot " + str(self.graph[i].id) +" and " +str(self.graph[i + 1].id))
            print(self.calculate_distance(self.graph[i], self.graph[i + 1]))

    def render(self):

        if(PY_GAME):
            self.screen_list[1].fill(WHITE)
            self.screen_list[0].fill(WHITE)
            myFont = pygame.font.SysFont("Times New Roman", 30)
            pygame.draw.rect(self.screen_list[0], BLACK, (self.wall.sx*WALL, self.wall.sy*WALL, self.wall.width*WALL, self.wall.height*WALL))
            for robot in self.robots:
                pygame.draw.circle(self.screen_list[0], BLUE, (robot.x*50, robot.y*50), RADIUS)
                self.randNumLabel = myFont.render(str(robot.id), 1, (0, 0, 255))
                self.screen_list[1].blit(self.randNumLabel, (robot.x*50, robot.y*50))
                pygame.draw.circle(self.screen_list[0], RED, (robot.x*50, robot.y*50), self.COMMUNICATION_RANGE*50/2, width=1)
            for i in range(0, len(self.graph) - 1):
                pygame.draw.line(self.screen_list[0], GREEN, (self.graph[i].x*50, self.graph[i].y*50),
                                 (self.graph[i + 1].x*50, self.graph[i + 1].y*50))
            if (self.coll):
                self.randNumLabel = myFont.render("WALL COLLISION", 1, (0, 0, 0))
                self.steps_nece = myFont.render("Steps necessary: " + str(self.episode_step), 1, (0, 0, 0))
                self.c = myFont.render("Collision: " + str(self.cll_id), 1, (0, 0, 0))
                self.screen_list[1].blit(self.randNumLabel, (100, 300))
                self.screen_list[1].blit(self.steps_nece, (100, 340))
                self.screen_list[1].blit(self.c, (100, 380))
            elif(self.complete == 1):
                self.randNumLabel = myFont.render("Done! Graph NOT Completed", 1, (0, 0, 255))
                self.screen_list[1].blit(self.randNumLabel, (100, 200))
            elif(self.complete == 2):
                self.randNumLabel = myFont.render("Done! Graph completed", 1, (0, 0, 0))
                self.steps_nece = myFont.render("Steps necessary: " + str(self.episode_step), 1, (0, 0, 0))
                self.screen_list[1].blit(self.randNumLabel, (100, 200))
                self.screen_list[1].blit(self.steps_nece, (150, 240))

            pygame.display.update()
        if(OPEN_CV):
            img = self.get_image()
            img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
            cv2.imshow("image", np.array(img))  # show it!
            cv2.waitKey(1)


    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        for player in self.robots:
            env[player.x][player.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
        for y in range(self.wall.y2, self.wall.y1):
            for x in range(self.wall.x1, self.wall.x2):
                env[x][y] = self.d[2]
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img


class BlobEnv:
    SIZE = 10
    RETURN_IMAGES = True
    MOVE_PENALTY = 1
    ENEMY_PENALTY = 300
    FOOD_REWARD = 25
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    ACTION_SPACE_SIZE = 9
    PLAYER_N = 1  # player key in dict
    FOOD_N = 2  # food key in dict
    ENEMY_N = 3  # enemy key in dict
    # the dict! (colors)
    d = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 0)}

    def reset(self):
        self.player = Blob(self.SIZE)
        self.food = Blob(self.SIZE)
        while self.food == self.player:
            self.food = Blob(self.SIZE)
        self.enemy = Blob(self.SIZE)
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Blob(self.SIZE)

        self.episode_step = 0

        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:
            observation = (self.player-self.food) + (self.player-self.enemy)
        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)

        #### MAYBE ###
        #enemy.move()
        #food.move()
        ##############

        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
        else:
            new_observation = (self.player-self.food) + (self.player-self.enemy)

        if self.player == self.enemy:
            reward = -self.ENEMY_PENALTY
        elif self.player == self.food:
            reward = self.FOOD_REWARD
        else:
            reward = -self.MOVE_PENALTY

        done = False
        if reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY or self.episode_step >= 200:
            done = True

        return new_observation, reward, done

    def render(self):
        img = self.get_image()
        img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
        cv2.imshow("image", np.array(img))  # show it!
        cv2.waitKey(1)

    # FOR CNN #
    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]  # sets the food location tile to green color
        env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]  # sets the enemy location to red
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img


env = newBlobEnv()

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
backend.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir
    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

        self._should_write_train_graph = False

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()
# Agent class
class DQNAgent:
    def __init__(self):

        self.id = 0
        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(256, (3, 3), input_shape=env.OBSERVATION_SPACE_VALUES))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))

        model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]


agents = []
for id in range(0,env.NUMBER_OF_ROBOTS):
    agent =DQNAgent()
    agent.id = id
    agents.append(agent)



env.reset()

def printMetrics(env):
    if(isinstance(env, newBlobEnv)):
        print("TOTAL EPISODES:    |      " + str(EPISODES))
        print("----------------------------------------")
        print("FOUND GRAPHS :     |      " + str(env.foundGraphs))
        print("----------------------------------------")
        per = env.foundGraphs * 100 / EPISODES
        print("Percentage :       |      " + str(per))
        print("----------------------------------------")




# Iterate over episodes
loaded = False
if(loaded == True):
    for episode in tqdm(range(1,  EPISODES + 1), ascii=True, unit='episodes'):
        # Update tensorboard step every episode
        #for agent in agents:
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()
        # Reset flag and start iterating until episode ends
        done = False
        while not done:
            for agent in agents:
                    # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random() > epsilon:
                    # Get action from Q table
                    action = np.argmax(agent.get_qs(current_state))
                else:
                    # Get random action
                    action = np.random.randint(0, env.ACTION_SPACE_SIZE)

                new_state, reward, done = env.step(action, agent.id)
                if SHOW_PREVIEW:  # and not episode % AGGREGATE_STATS_EVERY:
                        env.render()

                if(done == True):
                    env.print_end()
                    time.sleep(4)
                    break
                    # Transform new continous state to new discrete state and count reward
                episode_reward += reward

                    # Every step we update replay memory and train main network
                agent.update_replay_memory((current_state, action, reward, new_state, done))
                agent.train(done, step)
                current_state = new_state

                step += 1

                    # Append episode reward to a list and log stats (every given number of episodes)
                ep_rewards.append(episode_reward)
                if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                    average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                    min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                    max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                    agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

                if epsilon > MIN_EPSILON:
                    epsilon *= EPSILON_DECAY
                    epsilon = max(MIN_EPSILON, epsilon)
        for agent in agents:
            agent.model.save("Agent_" + str(agent.id))

else:
    print("Start")
    loaded_agents = []
    for id in range(0, env.NUMBER_OF_ROBOTS):
        temp = DQNAgent()
        temp.id = id
        temp.model = keras.models.load_model(
            "Agent_" + str(agent.id), custom_objects={"CustomModel": agent.create_model()}
        )
        loaded_agents.append(temp)

    print("Agents loaded")

    for episode in tqdm(range(1,  EPISODES + 1), ascii=True, unit='episodes'):
            # Update tensorboard step every episode
            for agent in loaded_agents:
                agent.tensorboard.step = episode

            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            step = 1

            # Reset environment and get initial state
            current_state = env.reset()
            #print("This is episode" + str(episode))
            # Reset flag and start iterating until episode ends
            done = False
            while not done:
                for agent in loaded_agents:
                        # This part stays mostly the same, the change is to query a model for Q values
                    if np.random.random() > epsilon:
                        # Get action from Q table
                        action = np.argmax(agent.get_qs(current_state))
                    else:
                        # Get random action
                        action = np.random.randint(0, env.ACTION_SPACE_SIZE)

                    new_state, reward, done = env.step(action, agent.id)
                    if SHOW_PREVIEW:  # and not episode % AGGREGATE_STATS_EVERY:
                            env.render()

                    if(done == True):
                        #env.print_end()
                        time.sleep(1)
                        break
                        # Transform new continous state to new discrete state and count reward
                    episode_reward += reward
                        # Every step we update replay memory and train main network
                    agent.update_replay_memory((current_state, action, reward, new_state, done))
                    agent.train(done, step)
                    current_state = new_state

                    step += 1

                        # Append episode reward to a list and log stats (every given number of episodes)
                    ep_rewards.append(episode_reward)

                    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

                            # Save model, but only when min reward is greater or equal a set value
                       # if min_reward >= MIN_REWARD:
                        #    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

                        # Decay epsilon
                    if epsilon > MIN_EPSILON:
                        epsilon *= EPSILON_DECAY
                        epsilon = max(MIN_EPSILON, epsilon)
    for agent in loaded_agents:
                agent.model.save("Agent_"+str(agent.id))
                print("Agents saved!")

    printMetrics(env)
