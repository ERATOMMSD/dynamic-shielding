import pygame, pygame.locals, sys, random, math
import numpy as np
from const import *


class Car(pygame.sprite.Sprite):
    # This class represents a car. It derives from the "Sprite" class in Pygame.

    def __init__(self, color, width, height):
        # Call the parent class (Sprite) constructor
        super(Car, self).__init__()

        # Pass in the color of the car, and its x and y position, width and height.
        # Set the background color and set it to be transparent
        self.image = pygame.Surface([width, height])
        self.image.fill(WHITE)
        self.image.set_colorkey(WHITE)

        # Draw the car (a rectangle!)
        pygame.draw.rect(self.image, color, [0, 0, width, height])

        # Instead we could load a proper pciture of a car...
        # self.image = pygame.image.load("car.png").convert_alpha()

        # Fetch the rectangle object that has the dimensions of the image.
        self.rect = self.image.get_rect()


def rot_center(image, angle):
    """rotate an image while keeping its center and size"""
    orig_rect = image.get_rect()
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = orig_rect.copy()
    rot_rect.center = rot_image.get_rect().center
    rot_image = rot_image.subsurface(rot_rect).copy()
    return rot_image


GREEN = (20, 255, 140)
GREY = (210, 210, 210)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
PURPLE = (255, 0, 255)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
# This will be a list that will contain all the sprites we intend to use in our game.
all_sprites_list = pygame.sprite.Group()

playerCar = Car(RED, 20, 30)
playerCar.rect.x = 100
playerCar.rect.y = 100


def is_vertical(obstacle):
    if obstacle.xmax - obstacle.xmin > obstacle.ymax - obstacle.ymin:
        return False
    else:
        return True


# Add the car to the list of objects
all_sprites_list.add(playerCar)


class obstacle(object):
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax


def argmax(b):
    maxVal = -100000000
    maxData = None
    for i, a in enumerate(b):
        if a > maxVal:
            maxVal = a
            maxData = i
    return maxData


def int_tup(tup):
    return int(tup)


def make_horizontal_wall(obs_left, obs_right, loc='top'):
    if loc == 'bot':
        return obstacle(obs_left.xmin, obs_right.xmax, min(obs_left.ymax, obs_right.ymax),
                        min(obs_left.ymax, obs_right.ymax) + 20)
    if loc == 'top':
        return obstacle(obs_left.xmin, obs_right.xmax, max(obs_left.ymin, obs_right.ymin) - 20,
                        max(obs_left.ymin, obs_right.ymin))


class Space(object):
    def __init__(self, n, shape):
        self.n = n
        self.shape = shape


class Env(object):
    def __init__(self, net=None, env_label='Learning Visualizer', big_neg=False):
        # CONSTANTS for how large the drawing window is.

        self.viz = VIZ

        self.XSIZE = 480
        self.YSIZE = 480
        # When visualizing the learned policy, in order to speed up things, we only a fraction of all pixels on a lower resolution. Here are the parameters for that.
        self.MAGNIFY = 2
        self.NOFPIXELSPLITS = 1

        # self.viz = True
        self.counter = 0
        self.accidents = 0
        # Obstacle definitions
        obs1 = obstacle(0, 20, 10, 450)
        obs3 = obstacle(460, 480, 10, 450)
        obs2 = make_horizontal_wall(obs1, obs3, 'top')
        obs4 = make_horizontal_wall(obs1, obs3, 'bot')

        obs5 = obstacle(140, 160, 140, 320)
        obs7 = obstacle(330, 350, 140, 320)
        obs6 = make_horizontal_wall(obs5, obs7, 'top')
        obs8 = make_horizontal_wall(obs5, obs7, 'bot')

        # area_1 = obstacle(0, self.XSIZE / 2, self.YSIZE / 2, self.YSIZE)
        # area_2 = obstacle(0, self.XSIZE / 2, 0, self.YSIZE / 2)
        # area_3 = obstacle(self.XSIZE / 2, self.XSIZE, 0, self.YSIZE / 2)
        # area_4 = obstacle(self.XSIZE / 2, self.XSIZE, self.YSIZE / 2, self.YSIZE)
        XSIZE = self.XSIZE
        YSIZE = self.YSIZE
        self.areas = [AREA_1, AREA_2, AREA_3, AREA_4]

        self.OBSTACLES = [obs1, obs2, obs3, obs4, obs5, obs6, obs7, obs8]
        self.net = net

        self.action_space = Space(8, (0,))
        self.observation_space = Space(0, (4,))

        self.obstaclePixels = [[False for i in range(0, self.YSIZE)] for j in range(0, self.XSIZE)]
        for obs in self.OBSTACLES:
            for i in range(obs.xmin, obs.xmax):
                for j in range(obs.ymin, obs.ymax):
                    self.obstaclePixels[i][j] = True
        self.CRASH_COST = 1
        self.GOAL_LINE_REWARD = 1
        self.TRAIN_EVERY_NTH_STEP = 10  # 6
        self.currentPos = (100.0, 100.0)
        self.currentDir = random.random() * math.pi * 2
        self.currentSpeedPerStep = CAR_SPEED  # 3 #2.5 #1.0
        self.currentRotationPerStep = (math.pi / 4) / self.TRAIN_EVERY_NTH_STEP  # 0.10
        self.displayBufferEmpty = True
        self.isLearning = True
        if self.viz:
            self.screen = pygame.display.set_mode((self.XSIZE, self.YSIZE))
            pygame.display.set_caption(env_label)
        self.clock = pygame.time.Clock()
        self.isPaused = False
        if self.viz:
            self.screenBuffer = pygame.Surface(self.screen.get_size())
            self.screenBuffer = self.screenBuffer.convert()
        self.predictionBuffer = pygame.Surface((self.XSIZE / self.MAGNIFY, self.YSIZE / self.MAGNIFY))
        self.predictionBuffer.fill((64, 64, 64))  # Dark Gray
        pygame.font.init()
        self.usedfont = pygame.font.SysFont("monospace", 15)
        self.clock = pygame.time.Clock()
        if self.viz:
            # filling background
            self.screenBuffer.fill((200, 200, 200))
            # filling obstacles
            for obs in self.OBSTACLES:
                if is_vertical(obs):
                    c_counter = 0
                    for y in range(obs.ymin, obs.ymax):
                        c_counter += 1
                        for x in range(obs.xmin, obs.xmax):
                            if c_counter % 40 > 50:
                                self.screenBuffer.set_at((x, y), YELLOW)
                            else:
                                self.screenBuffer.set_at((x, y), BLACK)
                else:
                    for x in range(obs.xmin, obs.xmax):
                        for y in range(obs.ymin, obs.ymax):
                            self.screenBuffer.set_at((x, y), YELLOW)

        self.displayDirection = 0
        self.iteration = 0

    def inr1(self, x, y):
        if self.OBSTACLES[0].xmax <= x <= self.OBSTACLES[4].xmin and self.OBSTACLES[0].ymin <= y <= self.OBSTACLES[
            0].ymax:
            return True

    def inr2(self, x, y):
        # if self.OBSTACLES[1].xmin <= x <= self.OBSTACLES[1].xmax and self.OBSTACLES[1].ymax <= y <= self.OBSTACLES[5].ymin:
        if self.OBSTACLES[0].xmax <= x <= self.OBSTACLES[2].xmin and self.OBSTACLES[1].ymax <= y <= self.OBSTACLES[
            5].ymin:
            return True

    def inr3(self, x, y):
        if self.OBSTACLES[6].xmax <= x <= self.OBSTACLES[2].xmin and self.OBSTACLES[2].ymin <= y <= self.OBSTACLES[
            2].ymax:
            return True

    def inr4(self, x, y):
        # if self.OBSTACLES[3].xmin <= x <= self.OBSTACLES[3].xmax and self.OBSTACLES[7].ymax <= y <= self.OBSTACLES[3].ymin:
        if self.OBSTACLES[0].xmax <= x <= self.OBSTACLES[2].xmin and self.OBSTACLES[7].ymax <= y <= self.OBSTACLES[
            3].ymin:
            return True

    def inside(self, xy):
        x = xy[0]
        y = xy[1]
        # R1
        if self.inr1(x, y):
            return True
        # R2
        elif self.inr2(x, y):
            return True
        # R3
        elif self.inr3(x, y):
            return True
        # R4
        elif self.inr4(x, y):
            return True
        else:
            return False

    def reset(self, loc=0):
        self.viz = VIZ

        playerCar = Car(RED, 20, 30)
        playerCar.rect.x = 100
        playerCar.rect.y = 100
        # Add the car to the list of objects
        all_sprites_list = pygame.sprite.Group()
        all_sprites_list.add(playerCar)

        self.counter = 0
        if loc == 0:
            rand_x = 100
            rand_y = 400
            self.currentDir = math.pi
        elif loc == 1:
            rand_x = 100
            rand_y = 80
            self.currentDir = math.pi / 2
        elif loc == 2:
            rand_x = 400
            rand_y = 400
            self.currentDir = math.pi + math.pi / 2
        elif loc == 3:
            rand_x = 400
            rand_y = 80
            self.currentDir = 0
        else:
            raise ValueError('loc not in 0,4')

        self.currentPos = (rand_x, rand_y)

        self.iteration += 1
        if pygame.display.get_active() and self.viz:
            # Scale up the "predictionBuffer"
            self.screenBuffer.blit(pygame.transform.smoothscale(self.predictionBuffer, (self.XSIZE, self.YSIZE)),
                                   (0, 0))
            self.displayBufferEmpty = False
        else:
            if not self.displayBufferEmpty:
                # When the buffer comes back up, we don't want 
                self.predictionBuffer.fill((64, 64, 64))
                self.displayBufferEmpty = True

                # ====================================
        # Draw origin of the motion
        if self.viz:
            pygame.draw.line(self.screenBuffer, (0, 0, 255), (self.currentPos[0] - 2, self.currentPos[1] - 2),
                             (self.currentPos[0] + 2, self.currentPos[1] + 2), 3)
            pygame.draw.line(self.screenBuffer, (0, 0, 255), (self.currentPos[0] + 2, self.currentPos[1] - 2),
                             (self.currentPos[0] - 2, self.currentPos[1] + 2), 3)

        if self.viz:
            for event in pygame.event.get():
                if event.type == pygame.locals.QUIT or (
                        event.type == pygame.locals.KEYDOWN and event.key == pygame.locals.K_ESCAPE):
                    sys.exit(0)
                if (event.type == pygame.locals.KEYDOWN and event.key == pygame.locals.K_SPACE):
                    self.displayQValue = not self.displayQValue
                if (event.type == pygame.locals.KEYDOWN and event.key == pygame.locals.K_l):
                    self.isLearning = not self.isLearning
                if (event.type == pygame.locals.KEYDOWN and event.key == pygame.locals.K_RIGHT):
                    self.displayDirection = (self.displayDirection + 1) % 8
                if (event.type == pygame.locals.KEYDOWN and event.key == pygame.locals.K_LEFT):
                    self.displayDirection = (self.displayDirection + 7) % 8
        return [np.array([self.currentPos[0] / self.XSIZE, self.currentPos[1] / self.YSIZE,
                          math.sin(self.currentDir), math.cos(self.currentDir)]), loc]

    def step(self, action, get_input_function, step_int, shield, episode=0, current_episode_type=0):
        self.viz = VIZ
        if self.viz:
            self.screenBuffer.fill((200, 200, 200))
            # area
            area = self.areas[0]
            for y in range(int(area[2]), int(area[3])):
                for x in range(int(area[0]), int(area[1])):
                    self.screenBuffer.set_at((x, y), (230, 100, 100))
            area = self.areas[1]
            for y in range(int(area[2]), int(area[3])):
                for x in range(int(area[0]), int(area[1])):
                    self.screenBuffer.set_at((x, y), (100, 230, 100))
            area = self.areas[2]
            for y in range(int(area[2]), int(area[3])):
                for x in range(int(area[0]), int(area[1])):
                    self.screenBuffer.set_at((x, y), (100, 100, 230))
            area = self.areas[3]
            for y in range(int(area[2]), int(area[3])):
                for x in range(int(area[0]), int(area[1])):
                    self.screenBuffer.set_at((x, y), (250, 250, 250))

            for obs in self.OBSTACLES:
                if is_vertical(obs):
                    c_counter = 0
                    for y in range(obs.ymin, obs.ymax):
                        c_counter += 1
                        for x in range(obs.xmin, obs.xmax):
                            if c_counter % 100 > 50:
                                self.screenBuffer.set_at((x, y), YELLOW)
                            else:
                                self.screenBuffer.set_at((x, y), BLACK)
                else:
                    c_counter = 0
                    for x in range(obs.xmin, obs.xmax):
                        c_counter += 1
                        for y in range(obs.ymin, obs.ymax):
                            if c_counter % 100 > 50:
                                self.screenBuffer.set_at((x, y), YELLOW)
                            else:
                                self.screenBuffer.set_at((x, y), BLACK)
            grid = GRID
            num_line = int(math.ceil(480 / grid))
            for x in range(num_line):
                for y in range(480):
                    # if int(round(x - grid/2, 0)) % grid == 0 or int(round(y - grid/2, 0)) % grid == 0:
                    # if x < 100:
                    self.screenBuffer.set_at((int(x * grid), y), YELLOW)
                    self.screenBuffer.set_at((y, int(x * grid)), YELLOW)

        stepStartingPos = self.currentPos
        # Simulate the cars for some steps. Also draw the trajectory of the car.
        crash = False
        for i in range(0, self.TRAIN_EVERY_NTH_STEP):
            if action == 0:
                targetDir = self.currentDir
            elif action == 1:
                targetDir = self.currentDir + self.currentRotationPerStep
            elif action == 2:
                targetDir = self.currentDir - self.currentRotationPerStep
            else:
                raise NotImplementedError()
            self.currentDir = targetDir

            self.oldPos = self.currentPos
            self.currentPos = (self.currentPos[0] + self.currentSpeedPerStep * math.sin(self.currentDir),
                               self.currentPos[1] + self.currentSpeedPerStep * math.cos(self.currentDir))
            if not self.inside(self.currentPos):
                crash = True
            if self.viz:
                if shield:
                    aCar = pygame.image.load('green.jpg')
                else:
                    temp = [self.currentPos[0] / self.XSIZE, self.currentPos[1] / self.YSIZE,
                            math.sin(self.currentDir), math.cos(self.currentDir)]
                    input_bits = get_input_function(temp, step_int)[0]
                    aCar = pygame.image.load('black.jpg')
                for x in range(aCar.get_width()):
                    for y in range(aCar.get_height()):
                        color = aCar.get_at((x, y))
                        if color[0] > 150 and color[1] > 150 and color[2] > 150:
                            aCar.set_at((x, y), (200, 200, 200))
                aCar = pygame.transform.scale(aCar, (5, 5))
                aCar = pygame.transform.rotate(aCar, 90)
                playerCar.image = pygame.transform.rotate(aCar, self.currentDir * (180. / math.pi))
                playerCar.rect = playerCar.image.get_rect()
                playerCar.rect.x = list(map(int_tup, self.currentPos))[0]
                playerCar.rect.y = list(map(int_tup, self.currentPos))[1]
                all_sprites_list = pygame.sprite.Group()
                all_sprites_list.add(playerCar)
                all_sprites_list.draw(self.screenBuffer)
        R = 0

        if crash:
            done = True
            R += 0
            if current_episode_type == 0:
                self.accidents += 1
        elif ((self.currentPos[1] > self.YSIZE / 2) and (self.currentPos[0] < self.XSIZE / 2) and (
                stepStartingPos[0] > self.XSIZE / 2)):
            R += 1
            done = False
        else:
            if self.inr1(self.currentPos[0], self.currentPos[1]):
                R += min(10 * (stepStartingPos[1] - self.currentPos[1]) / self.YSIZE,
                         20 * (stepStartingPos[1] - self.currentPos[1]) / self.YSIZE)

            if self.inr2(self.currentPos[0], self.currentPos[1]):
                R += min(10 * (self.currentPos[0] - stepStartingPos[0]) / self.XSIZE,
                         20 * (self.currentPos[0] - stepStartingPos[0]) / self.XSIZE)

            if self.inr3(self.currentPos[0], self.currentPos[1]):
                R += min(10 * (self.currentPos[1] - stepStartingPos[1]) / self.YSIZE,
                         20 * (self.currentPos[1] - stepStartingPos[1]) / self.YSIZE)

            if self.inr4(self.currentPos[0], self.currentPos[1]):
                R += min(10 * (stepStartingPos[0] - self.currentPos[0]) / self.XSIZE,
                         20 * (stepStartingPos[0] - self.currentPos[0]) / self.XSIZE)
            done = False
        if self.counter >= MAX_CAR_STEP:  # 200:  #1000:
            done = True
        else:
            self.counter += 1
        S_dash = np.array([self.currentPos[0] / self.XSIZE, self.currentPos[1] / self.YSIZE,
                           math.sin(self.currentDir), math.cos(self.currentDir)])
        if self.viz and pygame.display.get_active():
            self.screen.blit(self.screenBuffer, (0, 0))
            pygame.display.flip()

        if self.viz:
            font_s = pygame.font.Font('freesansbold.ttf', 12)
            inp_text = get_input_function(S_dash, step_int)[1]
            text = font_s.render(inp_text, True, BLACK, WHITE)
            textRect = text.get_rect()
            textRect.center = (250, 230)
            self.screen.blit(text, textRect)

            if action == 0:
                text_dir = 'fwd'
            elif action == 1:
                text_dir = 'left'
            elif action == 2:
                text_dir = 'right'
            else:
                text_dir = 'unknown'
            text_direction = font_s.render(text_dir, True, BLACK, WHITE)
            textRectDir = text_direction.get_rect()
            if self.currentPos[1] > 50:
                textRectDir.center = (self.currentPos[0], self.currentPos[1] - 20)
            else:
                textRectDir.center = (self.currentPos[0], self.currentPos[1] + 20)
            self.screen.blit(text_direction, textRectDir)

            font = pygame.font.Font('freesansbold.ttf', 16)
            text_1 = font.render('1', True, BLACK, WHITE)
            textRect_1 = text_1.get_rect()
            textRect_1.center = (10, 200)
            self.screen.blit(text_1, textRect_1)

            text_2 = font.render('2', True, BLACK, WHITE)
            textRect_2 = text_2.get_rect()
            textRect_2.center = (200, 8)
            self.screen.blit(text_2, textRect_2)

            text_3 = font.render('3', True, BLACK, WHITE)
            textRect_3 = text_3.get_rect()
            textRect_3.center = (470, 200)
            self.screen.blit(text_3, textRect_3)

            text_4 = font.render('4', True, BLACK, WHITE)
            textRect_4 = text_4.get_rect()
            textRect_4.center = (200, 460)
            self.screen.blit(text_4, textRect_4)

            text_5 = font.render('5', True, BLACK, WHITE)
            textRect_5 = text_5.get_rect()
            textRect_5.center = (150, 200)
            self.screen.blit(text_5, textRect_5)

            text_6 = font.render('6', True, BLACK, WHITE)
            textRect_6 = text_6.get_rect()
            textRect_6.center = (200, 130)
            self.screen.blit(text_6, textRect_6)

            text_7 = font.render('7', True, BLACK, WHITE)
            textRect_7 = text_7.get_rect()
            textRect_7.center = (340, 200)
            self.screen.blit(text_7, textRect_7)

            text_8 = font.render('8', True, BLACK, WHITE)
            textRect_8 = text_8.get_rect()
            textRect_8.center = (200, 330)
            self.screen.blit(text_8, textRect_8)
            pygame.display.update()
        return (S_dash, R, done, {'info': 'data'})

    def close(self):
        sys.exit(0)

    def render(self, mode=True):
        self.viz = mode
