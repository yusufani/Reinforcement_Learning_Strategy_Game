import random
import time
from collections import deque

import cv2
import numpy as np
import pygame
from PIL import Image


class Player:
    def __init__(self, is_enemy, player_health, player_attack, map_size, base_edge):
        self.map_size = map_size
        self.health = player_health
        self.is_enemy = is_enemy
        self.attack = player_attack
        self.y = np.random.randint(0, map_size)
        if (is_enemy):
            # If enemy create top of the map
            self.x = base_edge
        else:
            # If agent create bottom of the map
            self.x = map_size - base_edge

    def __str__(self):
        return f"Blob ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        '''
        Gives us 9 total movement options. (0,1,2,3,4,5,6,7,8)
        '''
        if choice == 0:
            return 1, 1
        elif choice == 1:
            return -1, -1
        elif choice == 2:
            return -1, 1
        elif choice == 3:
            return 1, -1
        elif choice == 4:
            return 1, 0
        elif choice == 5:
            return -1, 0
        elif choice == 6:
            return 0, 1
        elif choice == 7:
            return 0, -1
        elif choice == 8:
            return 0, 0

    def move(self, x=False, y=False):
        # print("Düşmanın önceki konumu " , self.x , self.y)
        # print("Dirler " , x,y)
        # If no value for x, move randomly
        self.x += x
        self.y += y

        # If we are out of bounds, dont move !
        if self.x < 0 or self.y < 0 or self.x > self.map_size - 1 or self.y > self.map_size - 1:
            self.x -= x
            self.y -= y
        # print("Düşmanın sonraki konumu ", self.x, self.y)


class Base(Player):
    def __init__(self, is_enemy, health, map_size, base_edge):
        # super().__init__(is_enemy, player_health, player_attack, map_size, base_edge)
        self.health = health
        self.is_enemy = is_enemy
        self.y = np.random.randint(0, map_size - base_edge)
        if (is_enemy):
            # If enemy create top of the map
            self.x = 0
        else:
            # If agent create bottom of the map
            self.x = map_size - 1


class Forest:
    def __init__(self, map_size, area, base_edge):
        self.map_size = map_size
        self.area = area
        self.base_edge = base_edge
        self.list_of_trees = []
        count = 0
        while (count < self.area):
            x = random.randint(self.base_edge + 2, self.map_size - self.base_edge - 2)
            y = random.randint(0, self.map_size - 1)
            if (x, y) not in self.list_of_trees:
                self.list_of_trees.append((x, y))
                count += 1


class Env:
    SIZE = 10  # MAP SIZE
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3),  # 4 ,
    ACTION_SPACE_SIZE = 9  # Number of Action
    RETURN_IMAGES = True  #
    MOVE_PENALTY = -1  #
    FOREST_PENALTY = -10  # If the agent goes to the tree
    p = {
        "FOREST_AREA": 8,

        "AGENT_TO": {"ENEMY_PLAYER": 25, "ENEMY_BASE": 100},  # rewards for If agent attack enemy
        "ENEMY_TO": {"AGENT_PLAYER": -25, "AGENT_BASE": -100},  # rewards for If enemy attack agent

        "LOSS": -500,
        "WIN": 500,

        "BASE_LEN": 1,

        "AGENT_PLAYER_HEALTH": 100,
        "AGENT_BASE_HEALTH": 1000,
        "AGENT_PLAYER_ATTACK_POINT": 150,

        "ENEMY_PLAYER_HEALTH": 100,
        "ENEMY_BASE_HEALTH": 1000,
        "ENEMY_PLAYER_ATTACK_POINT": 50
    }

    # the dict! (colors)
    d = {"agent_player": (255, 255, 255),
         "agent_base": (255, 255, 255),
         "enemy_player": (100, 100, 100),
         "enemy_base": (100, 100, 100),
         "forest": (0, 255, 0)}

    def __init__(self, show):
        self.show = show
        if show:
            pygame.init()
            # self.screen = pygame.display.set_mode((800, 600))
            self.scale_factor = 20
            self.screen = pygame.display.set_mode((self.SIZE * self.scale_factor, self.SIZE * self.scale_factor))
            self.agent_base_image = pygame.transform.scale(pygame.image.load("images\\agent_Base.png").convert(),
                                                           (self.scale_factor, self.scale_factor))
            self.enemy_base_image = pygame.transform.scale(pygame.image.load("images\\enemy_base.png").convert(),
                                                           (self.scale_factor, self.scale_factor))
            self.agent_player_image = pygame.transform.scale(pygame.image.load("images\\agent_player.png").convert(),
                                                             (self.scale_factor, self.scale_factor))
            self.enemy_player_image = pygame.transform.scale(pygame.image.load("images\\enemy_player.png").convert(),
                                                             (self.scale_factor, self.scale_factor))
            self.stone = pygame.transform.scale(pygame.image.load("images\\stone.png").convert(),
                                                (self.scale_factor, self.scale_factor))
            pygame.display.flip()

    def create_player(self, is_enemy=False):
        name = "ENEMY" if is_enemy else "AGENT"
        return Player(is_enemy, self.p[name + "_PLAYER_HEALTH"], self.p[name + "_PLAYER_ATTACK_POINT"], self.SIZE,
                      self.p["BASE_LEN"])

    def create_base(self, is_enemy=False):
        name = "ENEMY" if is_enemy else "AGENT"
        return Base(is_enemy, self.p[name + "_BASE_HEALTH"], self.SIZE, self.p["BASE_LEN"])

    def reset(self):
        # print("Env resetlendi")
        self.agent_base = self.create_base(False)
        self.enemy_base = self.create_base(True)
        self.agent_player = self.create_player(False)
        while self.agent_base == self.agent_player:
            self.agent_player = self.create_player(False)
        self.enemy_player = self.create_player(True)
        while self.enemy_base == self.enemy_player:
            self.agent_player = self.create_player(True)

        self.forest = Forest(self.SIZE, self.p["FOREST_AREA"], self.p["BASE_LEN"])
        self.episode_step = 0

        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:
            raise Exception("Not implemented yet ")
            # observation = (self.player-self.en) + (self.player-self.enemy)
        return observation

    def make_attack(self, player, x, y, other):
        return player.x == other.x - x and player.y == other.y - y

    def step(self, action):
        reward = 0
        self.episode_step += 1
        move_agent_player = True
        move_enemy_player = True
        x, y = self.agent_player.action(action)
        # print("Agent action " , x , y )
        if (self.agent_player.x + x, self.agent_player.y + y) in self.forest.list_of_trees:
            # print("CEZA YDİ MAL")
            reward -= self.FOREST_PENALTY
            move_agent_player = False
        elif (self.make_attack(self.agent_player, x, y, self.enemy_player)):
            # print("Agent player enemy playera saldırdı")
            move_agent_player = False
            reward += self.p["AGENT_TO"]["ENEMY_PLAYER"]
            self.enemy_player.health -= self.agent_player.attack
        elif (self.make_attack(self.agent_player, x, y, self.enemy_base)):
            # print("agent player enemy base'e saldırdı")
            move_agent_player = False
            reward += self.p["AGENT_TO"]["ENEMY_BASE"]
            self.enemy_base.health -= self.agent_player.attack
        else:
            reward += self.MOVE_PENALTY

        x2, y2 = self.find_best_action_for_enemy()
        # print("x2 , Y2  -> " ,x2,y2)
        if (self.make_attack(self.enemy_player, x2, y2, self.agent_player)):
            # print("Enemy player agent playera saldırdı")
            move_enemy_player = False
            reward += self.p["ENEMY_TO"]["AGENT_PLAYER"]
            self.agent_player.health -= self.enemy_player.attack
        elif (self.make_attack(self.enemy_player, x2, y2, self.agent_base)):
            # print("Enemy player agent base'e saldırdı")
            move_enemy_player = False
            reward += self.p["ENEMY_TO"]["AGENT_BASE"]
            self.enemy_base.health -= self.enemy_player.attack

        done = False
        if self.agent_player.health < 0:
            # print("Agent player öldürüldü")
            self.agent_player = self.create_player(False)
        if self.enemy_player.health < 0:
            # print("enemy player öldürüldü")
            self.enemy_player = self.create_player(True)
        if self.agent_base.health < 0:
            done = True
            reward += self.p["LOSS"]
        if self.enemy_base.health < 0:
            done = True
            reward += self.p["WIN"]

        #### MAYBE ###
        # enemy.move()
        # food.move()
        ##############
        if move_agent_player:
            self.agent_player.move(x, y)
        if move_enemy_player:
            self.enemy_player.move(x2, y2)

        if self.agent_player == self.enemy_player:  # If 2 player in same coordinat
            self.enemy_player.move(-x2, -y2)

        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
            # print("new_observation ", new_observation)
        else:
            new_observation = (self.enemy_player - self.agent_player) + (self.agent_player - self.enemy_base)

        return new_observation, reward, done

    def render(self, step):
        print(step, "Renderlandı")
        img = self.get_image()
        img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
        cv2.imshow("image", np.array(img))  # show it!
        cv2.waitKey(1)
        time.sleep(0.2)
        # self.screen = pygame.display.set_mode((self.SIZE, self.SIZE))
        self.screen.fill((0, 0, 0))
        print(self.enemy_player.y * self.scale_factor, self.enemy_player.x * self.scale_factor)
        self.screen.blit(self.agent_player_image,
                         (self.agent_player.y * self.scale_factor, self.agent_player.x * self.scale_factor))
        self.screen.blit(self.agent_base_image,
                         (self.agent_base.y * self.scale_factor, self.agent_base.x * self.scale_factor))
        self.screen.blit(self.enemy_player_image,
                         (self.enemy_player.y * self.scale_factor, self.enemy_player.x * self.scale_factor))
        self.screen.blit(self.enemy_base_image,
                         (self.enemy_base.y * self.scale_factor, self.enemy_base.x * self.scale_factor))
        for i in self.forest.list_of_trees:
            self.screen.blit(self.stone, (i[1] * self.scale_factor, i[0] * self.scale_factor))

        # time.sleep(5)
        pygame.display.flip()
        # pygame.display.update()
        # a = input("sad")
        # time.sleep(1)

    # FOR CNN #
    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        env[self.agent_player.x][self.agent_player.y] = self.d[
            "agent_player"]  # sets the food location tile to green color
        env[self.agent_base.x][self.agent_base.y] = self.d["agent_base"]
        env[self.enemy_player.x][self.enemy_player.y] = self.d["enemy_player"]
        env[self.enemy_base.x][self.enemy_base.y] = self.d["enemy_base"]
        for i in self.forest.list_of_trees:
            env[i] = self.d["forest"]
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img

    class BFS():

        # To store matrix cell cordinates
        class Point:
            def __init__(self, x: int, y: int, parent):
                self.x = x
                self.y = y
                self.parent = parent

            def __str__(self):
                return f"Point (xy:{self.x}-{self.y}"

            def __eq__(self, other):
                if other == None:
                    return False
                return self.x == other.x and self.y == other.y
            # A data structure for queue used in BFS

        # Check whether given cell(row,col)
        # is a valid cell or not
        def isValid(self, row: int, col: int, map_size):
            return (row >= 0) and (row < map_size) and (col >= 0) and (col < map_size)

        # These arrays are used to get row and column
        # numbers of 4 neighbours of a given cell

        # Function to find the shortest path between
        # a given source cell to a destination cell.
        def BFS_algo(self, mat, src: Point, dest: Point, map_size):
            # print("Bilgiler" , src,dest,map_size)
            COL, ROW = map_size, map_size
            # check source and destination cell
            # of the matrix have value 1
            if mat[src.x][src.y] != 1 or mat[dest.x][dest.y] != 1:
                return -1
            visited = [[False for i in range(COL)] for j in range(ROW)]
            # Mark the source cell as visited
            visited[src.x][src.y] = True
            # Create a queue for BFS
            q = deque()
            # Distance of source cell is 0
            q.append(src)  # Enqueue source cell
            # Do a BFS starting from source cell
            while q:
                curr = q.popleft()  # Dequeue the front cell
                if curr.x == dest.x and curr.y == dest.y:
                    # print("Dest" , curr)
                    while (curr.parent != None and curr.parent != src and curr != src):
                        curr = curr.parent
                        # print(curr)
                    return curr.x - src.x, curr.y - src.y
                rowNum = [-1, 0, 0, 1]
                colNum = [0, -1, 1, 0]
                # Otherwise enqueue its adjacent cells
                for i in range(4):
                    row = curr.x + rowNum[i]
                    col = curr.y + colNum[i]
                    # if adjacent cell is valid, has path
                    # and not visited yet, enqueue it.
                    if (self.isValid(row, col, map_size) and mat[row][col] == 1 and not visited[row][col]):
                        visited[row][col] = True
                        q.append(self.Point(row, col, curr))

                        # Return -1 if destination cannot be reached
            # print("Yol bulunamadı")
            return 0, 0

    def find_best_action_for_enemy(self):
        # print("********************************************************")
        if self.enemy_player.is_enemy:
            x, y = self.agent_base - self.enemy_player
            base_dist = pow(x, 2) + pow(y, 2)
            x, y = self.agent_player - self.enemy_player
            player_dist = pow(x, 2) + pow(y, 2)
            if base_dist > player_dist:
                other = self.agent_player
                # print("Hedef agent player ")
            else:
                # print("Hedef agent base")
                other = self.agent_base

            map = np.ones((self.SIZE, self.SIZE))
            # print("Ağaçlar" , self.forest.list_of_trees)
            # print("Kaynak " ,self.enemy_player.x ,self.enemy_player.y )
            # print("Hedef" ,  other.x, other.y)
            for i in self.forest.list_of_trees:
                map[i] = 0

            bfs = self.BFS()
            src = bfs.Point(self.enemy_player.x, self.enemy_player.y, None)
            end = bfs.Point(other.x, other.y, None)
            return bfs.BFS_algo(map, src, end, self.SIZE)
        else:
            raise Exception("Selected player is not enemy there is a problem")
