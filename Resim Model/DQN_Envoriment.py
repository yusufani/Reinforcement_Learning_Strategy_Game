import math
import random
import time
from collections import deque

import cv2
import numpy as np
import pygame
from PIL import Image


class Player:
    def __init__(self, is_enemy, player_health, player_attack, map_size, base_edge, players_coordinats):
        self.map_size = map_size
        self.health = player_health
        self.is_enemy = is_enemy
        self.attack = player_attack
        # Controlling same x ,y cordinat problem
        if (is_enemy):
            # If enemy create top of the map
            self.x = base_edge
        else:
            # If agent create bottom of the map
            self.x = map_size - base_edge
        while (True):
            y = np.random.randint(0, map_size)
            if (self.x, y) not in players_coordinats:
                self.y = y
                break

    def __str__(self):
        return f"Blob ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def get_coordinats(self):
        return (self.x, self.y)

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

    def move(self, x, y):
        # print("Düşmanın önceki konumu " , self.x , self.y)
        # print("Dirler " , x,y)
        # If no value for x, move randomly
        self.x += x
        self.y += y

        # If we are out of bounds, dont move !

        if self.x < 0 or self.y < 0 or self.x > self.map_size - 1 or self.y > self.map_size - 1:
            self.x -= x
            self.y -= y
            if not self.is_enemy:
                return -10
        return 0
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
    NUMBER_OF_AGENT_PLAYER = 3
    NUMBER_OF_ENEMY_PLAYER = 3
    SIZE = 10
    # OBSERVATION_SPACE_VALUES = SIZE*SIZE+(NUMBER_OF_AGENT_PLAYER+1)*(NUMBER_OF_ENEMY_PLAYER+1)+NUMBER_OF_AGENT_PLAYER+NUMBER_OF_ENEMY_PLAYER+2  # +2 for base health
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)

    ACTION_SPACE_SIZE = 9

    MOVE_PENALTY_AGENT_AREA = -2
    MOVE_PENALTY_ENEMY_AREA = 1

    FOREST_PENALTY = -20
    p = {
        "FOREST_AREA": 8,

        "AGENT_TO": {"ENEMY_PLAYER": 40, "ENEMY_BASE": 80},
        "ENEMY_TO": {"AGENT_PLAYER": -20, "AGENT_BASE": -50},

        "LOSS": -500,
        "WIN": 1000,

        "PLAYER_N": 1,  # player key in dict
        "FOOD_N": 2,  # food key in dict
        "ENEMY_N": 3,  # enemy key in dict
        "BASE_LEN": 1,

        "AGENT_PLAYER_HEALTH": 100,
        "AGENT_BASE_HEALTH": 1000,
        "AGENT_PLAYER_ATTACK_POINT": 150,

        "ENEMY_PLAYER_HEALTH": 100,
        "ENEMY_BASE_HEALTH": 1000,
        "ENEMY_PLAYER_ATTACK_POINT": 50
    }

    # the dict! (colors)
    d = {"agent_player": (0, 0, 255),
         "agent_base": (40, 10, 56),
         "enemy_player": (255, 0, 0),
         "enemy_base": (255, 10, 56),
         "forest": (0, 255, 0)}

    def __init__(self, show):
        self.show = show
        if show:
            pygame.init()
            # self.screen = pygame.display.set_mode((800, 600))
            self.scale_factor = 40
            self.screen = pygame.display.set_mode((self.SIZE * self.scale_factor + self.SIZE * self.scale_factor,
                                                   self.scale_factor * int(
                                                       self.SIZE / 5) + 2 * self.SIZE * self.scale_factor))
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
            self.font = pygame.font.SysFont("comicsans", int(self.scale_factor / 2), True)
            pygame.display.flip()

    def create_player(self, is_enemy, players):
        name = "ENEMY" if is_enemy else "AGENT"
        return Player(is_enemy, self.p[name + "_PLAYER_HEALTH"], self.p[name + "_PLAYER_ATTACK_POINT"], self.SIZE,
                      self.p["BASE_LEN"], [i.get_coordinats for i in players])

    def create_base(self, is_enemy=False):
        name = "ENEMY" if is_enemy else "AGENT"
        return Base(is_enemy, self.p[name + "_BASE_HEALTH"], self.SIZE, self.p["BASE_LEN"])

    def reset(self):
        # print("Env resetlendi")
        self.agent_base = self.create_base(False)
        self.enemy_base = self.create_base(True)
        self.agent_players = []
        for i in range(self.NUMBER_OF_AGENT_PLAYER):
            self.agent_players.append(self.create_player(False, self.agent_players))
        self.enemy_players = []
        for i in range(self.NUMBER_OF_ENEMY_PLAYER):
            self.enemy_players.append(self.create_player(True, self.enemy_players))

        self.forest = Forest(self.SIZE, self.p["FOREST_AREA"], self.p["BASE_LEN"])
        self.episode_step = 0

        return np.array(self.get_observations())

    def make_attack(self, player, x, y, other):
        return player.x == other.x - x and player.y == other.y - y

    def step(self, actions, step, episode):
        RENDER_INFOS = []
        RENDER_INFOS.append(f"Agent Base Health : {self.agent_base.health}")
        RENDER_INFOS.append(f"Enemy Base Health : {self.enemy_base.health}")
        for idx, val in enumerate(self.agent_players):
            RENDER_INFOS.append(f"Agent Player {idx} Health = {val.health}")
        for idx, val in enumerate(self.enemy_players):
            RENDER_INFOS.append(f"Enemy Player {idx} Health = {val.health}")

        reward = 0
        self.episode_step += 1
        move_agent_players = [True, True, True]
        move_enemy_players = [True, True, True]

        agent_directions = []
        for idx, agent_player in enumerate(self.agent_players):
            agent_directions.append(agent_player.action(actions[idx]))
            if (agent_player.x + agent_directions[idx][0],
                agent_player.y + agent_directions[idx][1]) in self.forest.list_of_trees:
                RENDER_INFOS.append(f"Agent Player {idx} try to move on  stone  , REWARD = {self.FOREST_PENALTY}")
                # print("Taşa giteye çalıştı")
                reward += self.FOREST_PENALTY
                move_agent_players[idx] = False

            elif (self.make_attack(self.agent_players[idx], agent_directions[idx][0], agent_directions[idx][1],
                                   self.enemy_base)):
                RENDER_INFOS.append(
                    f"Agent Player {idx} Attack Enemy Base  REWARD = {self.p['AGENT_TO']['ENEMY_BASE']}")
                # print("agent player enemy base'e saldırdı")
                move_agent_players[idx] = False
                reward += self.p["AGENT_TO"]["ENEMY_BASE"]
                self.enemy_base.health -= self.agent_players[idx].attack
            for idx2, enemy_player in enumerate(self.enemy_players):
                if (self.make_attack(self.agent_players[idx], agent_directions[idx][0], agent_directions[idx][1],
                                     enemy_player)):
                    # print("Agent player enemy playera saldırdı")
                    RENDER_INFOS.append(
                        f"Agent Player {idx} Attack Enemy Player {idx2}  REWARD = {self.p['AGENT_TO']['ENEMY_PLAYER']}")
                    move_agent_players[idx] = False
                    reward += self.p["AGENT_TO"]["ENEMY_PLAYER"]
                    enemy_player.health -= self.agent_players[idx].attack
            if move_agent_players[idx]:
                if self.agent_players[idx].x > self.SIZE / 2:
                    RENDER_INFOS.append(
                        f"Agent Player in Agent Area x = {self.agent_players[idx].x} REWARD ={self.MOVE_PENALTY_AGENT_AREA}")
                    reward += self.MOVE_PENALTY_AGENT_AREA
                else:
                    RENDER_INFOS.append(
                        f"Agent Player in Enemy Area x = {self.agent_players[idx].x} REWARD ={self.MOVE_PENALTY_ENEMY_AREA}")
                    reward += self.MOVE_PENALTY_ENEMY_AREA
        '''
        Tried but not worked : (
        if episode and not test:
            if episode < 500:
                if episode %50 == 0:
                    print("The enemy completely Frozen episode = " , episode)
                enemy_directions = [(0, 0) for i in range(len(self.enemy_players))]
            elif episode < 5000 and episode%3 != 0 :
                if episode % 50 == 0:
                    print("The enemy freezes in 1 of every 3 steps , episode=", episode)

                enemy_directions = [(0, 0) for i in range(len(self.enemy_players))]
            else:
                enemy_directions = self.find_best_action_for_enemy()
        else:
            enemy_directions = self.find_best_action_for_enemy()
        '''
        if episode < 500:
            enemy_directions = [(0, 0) for i in range(len(self.enemy_players))]
        elif step % 3 == 0:
            enemy_directions = [(0, 0) for i in range(len(self.enemy_players))]
        else:
            enemy_directions = self.find_best_action_for_enemy()

        # print(enemy_directions)
        for idx, val in enumerate(self.enemy_players):
            if (self.make_attack(self.enemy_players[idx], enemy_directions[idx][0], enemy_directions[idx][1],
                                 self.agent_base)):
                # print("Enemy player agent base'e saldırdı")
                RENDER_INFOS.append(
                    f"Enemy Player {idx} Attack Agent Base  REWARD = {self.p['ENEMY_TO']['AGENT_BASE']}")
                move_enemy_players[idx] = False
                reward += self.p["ENEMY_TO"]["AGENT_BASE"]
                self.agent_base.health -= self.enemy_players[idx].attack
            for idx2, agent_player in enumerate(self.agent_players):
                if self.make_attack(self.enemy_players[idx], enemy_directions[idx][0], enemy_directions[idx][1],
                                    agent_player):
                    RENDER_INFOS.append(
                        f"Enemy Player {idx} Attack Agent Player {idx2}  REWARD = {self.p['ENEMY_TO']['AGENT_PLAYER']}")
                    # print("Enemy player agent playera saldırdı")
                    move_enemy_players[idx] = False
                    reward += self.p["ENEMY_TO"]["AGENT_PLAYER"]
                    agent_player.health -= self.enemy_players[idx].attack

        done = False
        for idx, agent_player in enumerate(self.agent_players):
            if agent_player.health < 0:
                RENDER_INFOS.append(f"*****AGENT Player {idx} is killed*****")
                # print("Agent player öldürüldü")
                self.agent_players[idx] = self.create_player(False, self.agent_players)
        for idx, enemy_player in enumerate(self.enemy_players):
            if enemy_player.health < 0:
                RENDER_INFOS.append(f"*****Enemy Player {idx} is killed******")
                # print("enemy player öldürüldü")
                self.enemy_players[idx] = self.create_player(True, self.enemy_players)

        if self.agent_base.health < 0:
            RENDER_INFOS.append(f"AGENT BASE DEFATED REWARD = {self.p['LOSS']}")
            print("Agent base öldü")
            done = True
            reward += self.p["LOSS"]
        if self.enemy_base.health < 0:
            RENDER_INFOS.append(f"ENEMY BASE DEFATED REWARD = {self.p['WIN']}")
            print("Enemy base öldü")
            done = True
            reward += self.p["WIN"]

        #### MAYBE ###
        # enemy.move()
        # food.move()
        ##############
        for idx, val in enumerate(move_agent_players):
            if val:
                reward += self.agent_players[idx].move(agent_directions[idx][0], agent_directions[idx][1])
        for idx, val in enumerate(move_enemy_players):
            if val:
                self.enemy_players[idx].move(enemy_directions[idx][0], enemy_directions[idx][1])

        for idx, i in enumerate(self.agent_players):
            for idx2, j in enumerate(self.enemy_players):
                if i == j:
                    j.move(-enemy_directions[idx][0], -enemy_directions[idx][1])

        new_observation = np.array(self.get_observations())
        # print("new_observation ", new_observation)
        RENDER_INFOS.append("******************************")
        RENDER_INFOS.append(f"STEP REWARD = {reward}")
        RENDER_INFOS.append("******************************")
        return new_observation, reward, done, RENDER_INFOS

    def render(self, total_reward, infos):
        self.render_raw()
        self.screen.fill((0, 0, 0))
        for enemy_player in self.enemy_players:
            self.screen.blit(self.enemy_player_image,
                             (enemy_player.y * self.scale_factor, enemy_player.x * self.scale_factor))
        for agent_player in self.agent_players:
            self.screen.blit(self.agent_player_image,
                             (agent_player.y * self.scale_factor, agent_player.x * self.scale_factor))
        self.screen.blit(self.agent_base_image,
                         (self.agent_base.y * self.scale_factor, self.agent_base.x * self.scale_factor))
        self.screen.blit(self.enemy_base_image,
                         (self.enemy_base.y * self.scale_factor, self.enemy_base.x * self.scale_factor))
        for i in self.forest.list_of_trees:
            self.screen.blit(self.stone, (i[1] * self.scale_factor, i[0] * self.scale_factor))
        text = self.font.render("Total reward: " + str(total_reward), 1,
                                (255, 255, 255))  # Arguments are: text, anti-aliasing, color
        self.screen.blit(text, (0, self.SIZE * self.scale_factor))
        i = 0
        for idx, info in enumerate(infos):
            text2 = self.font.render(info, 1, (255, 255, 255))  # Arguments are: text, anti-aliasing, color
            # print(info)
            self.screen.blit(text2,
                             (0, self.SIZE * self.scale_factor + self.scale_factor + idx * int(self.scale_factor / 2)))

        pygame.display.flip()
        time.sleep(0.5)
        # pygame.display.update()
        # a = input("sad")
        # time.sleep(1)

    # FOR CNN #
    def get_distance(self, player1, player2):
        return math.sqrt(pow(player1.x - player2.x, 2) + pow(player1.y - player2.y, 2))

    def get_observations(self, get_only_map=False):
        map = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        for i in self.agent_players:
            map[i.x][i.y] = self.d["agent_player"]
        for i in self.enemy_players:
            map[i.x][i.y] = self.d["enemy_player"]
        map[self.agent_base.x][self.agent_base.y] = self.d["agent_base"]
        map[self.enemy_base.x][self.enemy_base.y] = self.d["enemy_base"]
        for i in self.forest.list_of_trees:
            map[i] = self.d["forest"]

        img = Image.fromarray(map, "RGB")
        return img

    def render_raw(self):
        img = self.get_observations()
        img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
        cv2.imshow("image", np.array(img))  # show it!
        cv2.waitKey(1)

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
                return 0, 0
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
        '''

        :return list of directions of enemy players with bfs algorithm:
        '''
        # print("********************************************************")
        directions = []
        for enemy_player in self.enemy_players:
            x, y = self.agent_base - enemy_player
            base_dist = pow(x, 2) + pow(y, 2)
            players_dists = np.zeros(len(self.agent_players))
            for idx, agent in enumerate(self.agent_players):
                x, y = agent - enemy_player
                players_dists[idx] = pow(x, 2) + pow(y, 2)
            min = np.argmin(players_dists)

            if base_dist > players_dists[min]:
                other = self.agent_players[min]
            else:
                other = self.agent_base

            map = np.ones((self.SIZE, self.SIZE))
            for i in self.forest.list_of_trees:
                map[i] = 0
            ''''
            for i in self.agent_players:
                map[i.x][i.y]=0
            '''

            for i in self.enemy_players:
                map[i.x][i.y] = 1  # enemy players cordinats are not a

            bfs = self.BFS()
            src = bfs.Point(enemy_player.x, enemy_player.y, None)
            end = bfs.Point(other.x, other.y, None)
            directions.append(bfs.BFS_algo(map, src, end, self.SIZE))
        return directions
