import math
import random
import time
from collections import deque

import numpy as np
import pygame


class Player:
    def __init__(self, index, player_health, player_attack, map_size):
        self.live = True
        # Indx 0 -> bottom  | 1-> left | 2-> right | -1 enemy top
        self.index = index
        self.map_size = map_size
        self.health = player_health
        self.attack = player_attack
        # Controlling same x ,y cordinat problem
        if index == 0 or index == -1:
            val = int((map_size - 1) / 2)
            self.y = random.sample([val, val - 1, val + 1, val + 2, val - 2], 1)[0]
        elif index == 1:
            self.y = 1
        elif index == 2:
            self.y = map_size - 2
        else:
            raise Exception("Wrong index ->  ", index)

        if index == 1 or index == 2:
            val = int((map_size - 1) / 2)
            self.x = random.sample([val, val - 1, val + 1, val + 2, val - 2], 1)[0]
        elif index == -1:
            self.x = 1
        elif index == 0:
            self.x = map_size - 2
        else:
            raise Exception("Wrong index ->  ", index)

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

    def move(self, x, y, RENDER_INFOS=None):
        # print("Düşmanın önceki konumu " , self.x , self.y)
        # print("Dirler " , x,y)
        # If no value for x, move randomly
        self.x += x
        self.y += y

        # If we are out of bounds, dont move !
        if self.x < 0 or self.y < 0 or self.x > self.map_size - 1 or self.y > self.map_size - 1:
            self.x -= x
            self.y -= y
            RENDER_INFOS[1].append(f"Player {self.index} tried to cross map borders  , REWARD = -10")
            return -10
        return 0
        # print("Düşmanın sonraki konumu ", self.x, self.y)


class Base(Player):
    def __init__(self, index, health, map_size):
        # super().__init__(is_enemy, player_health, player_attack, map_size, base_edge)
        self.live = True
        self.index = index
        self.health = health
        if index == 0 or index == -1:
            self.y = int(map_size / 2)
        elif index == 1:
            self.y = 0
        elif index == 2:
            self.y = map_size - 1
        else:
            raise Exception("Wrong index ->  ", index)

        if index == 1 or index == 2:
            self.x = int(map_size / 2)
        elif index == -1:
            self.x = 0
        elif index == 0:
            self.x = map_size - 1
        else:
            raise Exception("Wrong index ->  ", index)


class Forest:
    def __init__(self, map_size, area, base_edge, get_forest_from_console):
        self.map_size = map_size
        self.area = area
        self.base_edge = base_edge
        self.list_of_trees = []
        count = 0
        if get_forest_from_console:
            print("You have chosen generate forest by user console.Please enter appropriate values ")
        while (count < self.area):
            if get_forest_from_console:
                x = int(input(
                    f"(Remaining {area - count} Tree ) Please enter Tree's X value in range [{3}] , [{self.map_size - 1 - 2}] : "))
                y = int(input(
                    f"(Please enter Tree's Y value in range [{3}] , [{self.map_size - 1 - 2}] : "))
            else:
                x = random.randint(3, self.map_size - 1 - 2)
                y = random.randint(3, self.map_size - 1 - 2)
            if (x, y) not in self.list_of_trees:
                self.list_of_trees.append((x, y))
                count += 1


class Env:
    N_AGENT = 3
    N_ENEMY = 1
    SIZE = 20
    OBSERVATION_SPACE_VALUES = SIZE * SIZE + (N_AGENT + N_ENEMY) * (4)  # +2 for base health
    ACTION_SPACE_SIZE = 8

    MOVE_PENALTY_AGENT_AREA = -4
    MOVE_PENALTY_ENEMY_AREA = -1

    FOREST_PENALTY = -20
    p = {
        "FOREST_AREA": 12,

        "AGENT_TO": {"ENEMY_PLAYER": 40, "ENEMY_BASE": 80},
        "ENEMY_TO": {"AGENT_PLAYER": -20, "AGENT_BASE": -50},

        "LOSE": -500,
        "WIN": 2000,

        "KILL_ENEMY": 120,
        "KILL_AGENT": -90,

        "BASE_LEN": 1,

        "AGENT_PLAYER_HEALTH": 100,
        "AGENT_BASE_HEALTH": 1000,
        "AGENT_PLAYER_ATTACK_POINT": 150,

        "ENEMY_PLAYER_HEALTH": 100,
        "ENEMY_BASE_HEALTH": 1000,
        "ENEMY_PLAYER_ATTACK_POINT": 50
    }

    # the dict! (colors)
    d = {"player_0": 1,
         "base_0": 2,
         "player_1": 3,
         "base_1": 4,
         "player_2": 5,
         "base_2": 6,
         "player_3": 7,
         "base_3": 8,
         "forest": 9}

    def __init__(self, show):
        self.show = show
        if show:
            pygame.init()
            # self.screen = pygame.display.set_mode((800, 600))
            self.scale_factor = 30
            self.screen = pygame.display.set_mode((self.SIZE * self.scale_factor + self.scale_factor * 10,
                                                   self.scale_factor * 8 + self.SIZE * self.scale_factor))
            self.agent_base_image = pygame.transform.scale(pygame.image.load("images\\agent_Base.png").convert(),
                                                           (self.scale_factor, self.scale_factor))
            self.agent_base_2_image = pygame.transform.scale(pygame.image.load("images\\agent_base_2.png").convert(),
                                                             (self.scale_factor, self.scale_factor))
            self.agent_base_3_image = pygame.transform.scale(pygame.image.load("images\\agent_base_3.png").convert(),
                                                             (self.scale_factor, self.scale_factor))
            self.enemy_base_image = pygame.transform.scale(pygame.image.load("images\\enemy_base.png").convert(),
                                                           (self.scale_factor, self.scale_factor))
            self.agent_player_2_image = pygame.transform.scale(
                pygame.image.load("images\\agent_player_2.png").convert(), (self.scale_factor, self.scale_factor))
            self.agent_player__3image = pygame.transform.scale(
                pygame.image.load("images\\agent_player_3.png").convert(),
                (self.scale_factor, self.scale_factor))
            self.agent_player_image = pygame.transform.scale(pygame.image.load("images\\agent_player.png").convert(),
                                                             (self.scale_factor, self.scale_factor))
            self.enemy_player_image = pygame.transform.scale(pygame.image.load("images\\enemy_player.png").convert(),
                                                             (self.scale_factor, self.scale_factor))
            self.stone = pygame.transform.scale(pygame.image.load("images\\pines.png").convert(),
                                                (self.scale_factor, self.scale_factor))
            self.font = pygame.font.SysFont("comicsans", int(self.scale_factor / 2), True)
            pygame.display.flip()

    def create_player(self, index):
        name = "ENEMY" if index == -1 else "AGENT"
        return Player(index, self.p[name + "_PLAYER_HEALTH"], self.p[name + "_PLAYER_ATTACK_POINT"], int(self.SIZE))

    def create_base(self, index):
        name = "ENEMY" if index == -1 else "AGENT"
        return Base(index, self.p[name + "_BASE_HEALTH"], int(self.SIZE))

    def reset(self, get_forest_from_console=False):
        # print("Env resetlendi")
        self.bases = []
        self.bases.append(self.create_base(0))
        self.bases.append(self.create_base(1))
        self.bases.append(self.create_base(2))
        self.bases.append(self.create_base(-1))
        self.players = []
        for i in range(self.N_AGENT):
            self.players.append(self.create_player(i))
        self.players.append(self.create_player(-1))

        self.forest = Forest(self.SIZE, self.p["FOREST_AREA"], self.p["BASE_LEN"], get_forest_from_console)
        self.episode_step = 0

        obs = []
        for player in self.players:
            if player.index != -1:
                obs.append(self.get_observations(player.index))
        return obs

    def make_attack(self, player, x, y, other):
        return player.x == other.x - x and player.y == other.y - y

    def step(self, actions, old_dones):
        RENDER_INFOS = [[], []]
        for base in self.bases:
            if base.live:
                RENDER_INFOS[0].append(f"Base {base.index} Health : {base.health}")
        for player in self.players:
            if player.live:
                RENDER_INFOS[0].append(f"Player {player.index} Health : {player.health}")
        rewards = {0: 0, 1: 0, 2: 0}
        self.episode_step += 1
        move_players = {}
        for i in self.players:
            move_players[i.index] = (True if i.live else False)

        for player in self.players:
            if player.live:
                if player.index != -1:
                    x, y = self.play_action(actions, player)
                    move = True
                    if (player.x + x, player.y + y) in self.forest.list_of_trees:
                        RENDER_INFOS[1].append(
                            f"Player {player.index} try to move on  stone  , REWARD = {self.FOREST_PENALTY}")
                        rewards[player.index] += self.FOREST_PENALTY
                        move = False

                    closest = self.SIZE * self.SIZE
                    for base in self.bases:
                        dist = self.get_distance(player, base)
                        if base.live and dist < closest:
                            closest = dist
                        if base.live and base.index != player.index and self.make_attack(player, x, y, base):
                            RENDER_INFOS[1].append(
                                f"Agent Player {player.index} Attack  Base {base.index}  REWARD = {self.p['AGENT_TO']['ENEMY_BASE']}")
                            rewards[player.index] += self.p["AGENT_TO"]["ENEMY_BASE"]
                            move = False
                            base.health -= player.attack
                            if base.health < 0 and not old_dones[base.index]:
                                rewards[player.index] += self.p["WIN"]
                                RENDER_INFOS[1].append(
                                    f"Agent Player {player.index} Attack  Base {base.index}  REWARD = {self.p['WIN']}")
                    for other_player in self.players:
                        if other_player.live and player.index != other_player.index and self.make_attack(player, x, y,
                                                                                                         other_player):
                            RENDER_INFOS[1].append(
                                f"Agent Player {player.index} Attack  Player {other_player.index}  REWARD = {self.p['AGENT_TO']['ENEMY_PLAYER']}")
                            move = False
                            rewards[player.index] += self.p["AGENT_TO"]["ENEMY_PLAYER"]
                            other_player.health -= player.attack

                    if closest == 0:
                        closest = 1
                    closest_reward = int(self.SIZE / closest)

                    rewards[player.index] += closest_reward
                    RENDER_INFOS[1].append(
                        f"Player {player.index} closest Base  REWARD = {closest_reward}")
                    if move:
                        if self.get_distance(player, self.bases[player.index]) > math.sqrt(self.SIZE):
                            rewards[player.index] += self.MOVE_PENALTY_ENEMY_AREA
                            RENDER_INFOS[1].append(
                                f"Player {player.index} NOT in own area  REWARD = {self.MOVE_PENALTY_ENEMY_AREA}")
                        else:
                            rewards[player.index] += self.MOVE_PENALTY_AGENT_AREA
                            RENDER_INFOS[1].append(
                                f"Player {player.index} in own area  REWARD = {self.MOVE_PENALTY_AGENT_AREA}")
                        rewards[player.index] += player.move(x, y, RENDER_INFOS=RENDER_INFOS)
                else:
                    x, y = self.find_best_action(player)
                    move = True
                    for base in self.bases:
                        if base.live and base.index != player.index and self.make_attack(player, x, y, base):
                            RENDER_INFOS[1].append(
                                f"Enemy Player {player.index} Attack  Base {base.index}  REWARD = {self.p['ENEMY_TO']['AGENT_BASE']}")
                            rewards[base.index] += self.p["ENEMY_TO"]["AGENT_BASE"]
                            base.health -= player.attack
                            move = False
                    for other_player in self.players:
                        if other_player.live and player.index != other_player.index and self.make_attack(player, x, y,
                                                                                                         other_player):
                            RENDER_INFOS[1].append(
                                f"Agent Player {player.index} Attack  Player {other_player.index}  REWARD = {self.p['ENEMY_TO']['AGENT_PLAYER']}")
                            rewards[other_player.index] += self.p["ENEMY_TO"]["AGENT_PLAYER"]
                            other_player.health -= player.attack
                            move = False
                    if move:
                        player.move(x, y)

        done = {}
        for b_idx, base in enumerate(self.bases):
            base.live = False if base.health < 0 else True
            done[base.index] = not base.live
            # print(old_dones)
            # print(base.index)
            if base.index != -1 and base.health < 0 and not old_dones[base.index]:  # Reward Cal
                print(f"Base {base.index} Dead-", end="")
                rewards[base.index] += self.p["LOSE"]
            elif base.index == -1 and not base.live and not old_dones[base.index]:
                print(f"Base Enemy Dead-", end="")
            if not base.live:  # Agent can learn game bug :D
                base.x = -1
                base.y = -1
                for idx, val in enumerate(self.players):
                    if val.index == base.index:
                        self.players[idx].live = False
                        self.players[idx].x = self.players[idx].y = -1

        for idx, player in enumerate(self.players):
            if not done[player.index]:
                if player.health < 0:
                    self.players[idx] = self.create_player(player.index)
                    if player.index != -1:
                        rewards[player.index] += self.p["KILL_ENEMY"] if player.index != -1 else self.p["KILL_AGENT"]

        new_observations = []
        for player in self.players:
            if player.index != -1:
                new_observations.append(self.get_observations(player.index))
        # print(rewards)
        # print(done)
        # print("new_observation ", new_observation)
        RENDER_INFOS[1].append("******************************")
        RENDER_INFOS[1].append(
            f"STEP REWARD Agent-1 = {rewards[0]}  Agent-2 = {rewards[1]} Agent-3 = {rewards[2]}       ")
        RENDER_INFOS[1].append("******************************")
        return new_observations, rewards, done, RENDER_INFOS

    def render(self, infos, RENDER_SPEED=0.2):
        self.screen.fill((0, 0, 0))
        if self.bases[0].live:
            self.screen.blit(self.agent_player_image,
                             (self.players[0].y * self.scale_factor, self.players[0].x * self.scale_factor))
            self.screen.blit(self.agent_base_image,
                             (self.bases[0].y * self.scale_factor, self.bases[0].x * self.scale_factor))
        if self.bases[1].live:
            self.screen.blit(self.agent_player_2_image,
                             (self.players[1].y * self.scale_factor, self.players[1].x * self.scale_factor))
            self.screen.blit(self.agent_base_2_image,
                             (self.bases[1].y * self.scale_factor, self.bases[1].x * self.scale_factor))
        if self.bases[2].live:
            self.screen.blit(self.agent_player__3image,
                             (self.players[2].y * self.scale_factor, self.players[2].x * self.scale_factor))
            self.screen.blit(self.agent_base_3_image,
                             (self.bases[2].y * self.scale_factor, self.bases[2].x * self.scale_factor))
        if self.bases[3].live:
            self.screen.blit(self.enemy_player_image,
                             (self.players[3].y * self.scale_factor, self.players[3].x * self.scale_factor))
            self.screen.blit(self.enemy_base_image,
                             (self.bases[3].y * self.scale_factor, self.bases[3].x * self.scale_factor))

        for i in self.forest.list_of_trees:
            self.screen.blit(self.stone, (i[1] * self.scale_factor, i[0] * self.scale_factor))
        i = 0
        for idx, info in enumerate(infos[1]):
            text2 = self.font.render(str(info), 1, (255, 255, 255))  # Arguments are: text, anti-aliasing, color
            # print(info)
            self.screen.blit(text2,
                             (0, self.SIZE * self.scale_factor + self.scale_factor + idx * int(self.scale_factor / 2)))
        for idx, info in enumerate(infos[0]):
            text2 = self.font.render(info, 1, (255, 255, 255))  # Arguments are: text, anti-aliasing, color
            # print(info)
            self.screen.blit(text2,
                             (self.SIZE * self.scale_factor + self.scale_factor, +idx * int(self.scale_factor / 2)))

        pygame.display.flip()
        time.sleep(RENDER_SPEED)
        # pygame.display.update()
        # a = input("sad")
        # time.sleep(1)

    # FOR CNN #
    def get_distance(self, player1, player2):
        return math.sqrt(pow(player1.x - player2.x, 2) + pow(player1.y - player2.y, 2))

    def get_observations(self, index, get_only_map=False):
        map = np.zeros((self.SIZE, self.SIZE), dtype=np.int8)
        for idx, val in enumerate(self.players):
            map[val.x][val.y] = self.d["player_" + str(idx)] if val.live else 10
        for idx, val in enumerate(self.bases):
            map[val.x][val.y] = self.d["base_" + str(idx)] if val.live else 10
        for i in self.forest.list_of_trees:
            map[i] = self.d["forest"]
        ## Map is ready
        if get_only_map:
            return map
        infos = np.zeros((len(self.players) * 4), dtype=np.int8)
        idx = 0
        for base in self.bases:
            infos[idx] = int(self.get_distance(self.players[index], base) if base.live else 10)
            infos[idx + 1] = int(base.health if base.live else -1)
            idx += 2
        for player in self.players:
            infos[idx] = int(self.get_distance(self.players[index], player) if player.live else 10)
            infos[idx + 1] = int(player.health if player.live else 10)
            idx += 2
        obs = np.concatenate([map.flatten(), infos.flatten()])

        return obs

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
                rowNum = [-1, 0, 0, 1, 1, -1, 1, -1]
                colNum = [0, -1, 1, 0, 1, -1, -1, 1]
                # Otherwise enqueue its adjacent cells
                for i in range(8):
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

    def find_best_action(self, player):
        '''

        :return list of directions of enemy players with bfs algorithm:
        '''
        # print("********************************************************")
        map = np.ones((self.SIZE, self.SIZE))
        for i in self.forest.list_of_trees:
            map[i] = 0

        alls = [*self.players, *self.bases]
        min_d = [alls[0], self.SIZE * self.SIZE]

        for other in alls:
            if other.live and other.index != player.index:
                x, y = player - other
                dist = pow(x, 2) + pow(y, 2)
                if dist < min_d[1]:
                    min_d[0] = other
                    min_d[1] = dist

        bfs = self.BFS()
        src = bfs.Point(player.x, player.y, None)
        end = bfs.Point(min_d[0].x, min_d[0].y, None)
        return bfs.BFS_algo(map, src, end, self.SIZE)

    def play_action(self, actions, player):
        return self.find_best_action(player) \
            if random.random() < 0.2 else player.action(actions[player.index])
