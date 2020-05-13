import pickle
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import style
from tqdm import tqdm

style.use("ggplot")

SIZE = 5  # Map Size
EPISODES = 1000000  # Number of episode

BASE_HEALTH = 10
PLAYER_ATTACK = 2  # Player attack power

MOVE_PENALTY = 1  # Agent player get penalty for each step
ENEMY_BASE_HIT_PENALTY = 5  # If enemy player hit to agent base
ENEMY_PLAYER_HIT_PENALTY = 2  # If  player  hit to agent
BASE_HIT_PENALTY = 5  # If AGENT hit to ENENMY base
PLAYER_HIT_PENALTY = 2  # If AGENT hit to enemy player
WIN_REWARD = 10

EPS_DECAY = 0.99999  # Epsilon greedy
SHOW_EVERY = 4000  # Show Game in every SHOW_EVERY STEP

start_q_table = "Q_table_modeli.pickle"  # For continue training | file name
epsilon = 1 if start_q_table is None else 0.1
SHOW_EVERY = 4000 if start_q_table is None else 1  # Show Game in every SHOW_EVERY STEP
LEARNING_RATE = 0.1
DISCOUNT = 0.95

AGENT_N = 1  # Number of agent
ENEMY_N = 1  # Number of enemy

d = {0: (255, 175, 0),
     1: (0, 255, 0),
     2: (0, 0, 255),
     3: (100, 100, 100)}


class Player:
    def __init__(self, is_enemy):
        self.attack = PLAYER_ATTACK
        self.y = np.random.randint(0, SIZE)
        if (is_enemy):
            # If enemy create top of the map
            self.x = 1
        else:
            # If agent create bottom of the map
            self.x = SIZE - 1

    def __str__(self):
        return f"X={self.x},Y={self.y}"

    def __sub__(self, other):  # Substraction operator overload
        # return int(math.sqrt(pow(x,2)+pow(y,2)))
        return (self.x - other.x, self.y - other.y)

    def find_best_action_for_enemy(self, player, base):
        if base - self > player - self:
            other = base
        else:
            other = player

        if other.x - self.x > 0:
            x = 1
        elif other.x == self.x:
            x = 0
        else:
            x = -1
        if other.y - self.y > 0:
            y = 1
        elif other.y == self.y:
            y = 0
        else:
            y = -1
        self.move(x, y)

    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)
        elif choice == 4:
            self.move(x=0, y=0)

    def move(self, x=False, y=False):
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > SIZE - 1:
            self.x = SIZE - 1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE - 1:
            self.y = SIZE - 1


class BASE(Player):
    def __init__(self, is_enemy):
        self.health = BASE_HEALTH
        self.y = np.random.randint(0, SIZE - 1)
        if (is_enemy):
            # If enemy create top of the map
            self.x = 0
        else:
            # If agent create bottom of the map
            self.x = SIZE - 1


print("Tablo oluşturuluyor")
if start_q_table is None:
    q_table = {}
    for x1 in tqdm(range(-SIZE + 1, SIZE)):
        for y1 in range(-SIZE + 1, SIZE):
            for x2 in range(-SIZE + 1, SIZE):
                for y2 in range(-SIZE + 1, SIZE):
                    for x3 in range(-SIZE + 1, SIZE):
                        for y3 in range(-SIZE + 1, SIZE):
                            q_table[((x1, y1), (x2, y2), (x3, y3))] = [np.random.uniform(-5, 0) for i in range(4)]
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)


def is_same_cord(x1, y1, x2, y2):
    return True if x1 == x2 and y1 == y2 else False


episode_rewards = []
print("Başlıyoruz")
for episode in tqdm(range(EPISODES)):
    agent = Player(is_enemy=False)
    agent_base = BASE(is_enemy=False)
    enemy = Player(is_enemy=True)
    enemy_base = BASE(is_enemy=True)

    if episode % SHOW_EVERY == 0:
        print(
            f"on # {episode} , epsilon {epsilon}  AGENT BASE HEALTH {agent_base.health} Enemy Base health {enemy_base.health}")
        print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False
    episode_reward = 0
    i = 0
    while (True):
        i += 1
        obs = (agent - enemy, agent - enemy_base, enemy - agent_base)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)
        agent.action(action)
        reward = 0
        if i % 5 == 0:
            enemy.find_best_action_for_enemy(agent, agent_base)
        if is_same_cord(agent.x, agent.y, enemy.x, enemy.y):
            reward = ENEMY_PLAYER_HIT_PENALTY
        if is_same_cord(agent.x, agent.y, enemy_base.x, enemy_base.y):
            enemy_base.health -= agent.attack
            reward = BASE_HIT_PENALTY
            if enemy_base.health < 0:
                reward = WIN_REWARD
        if is_same_cord(agent_base.x, agent_base.y, enemy.x, enemy.y):
            print("Rakibe saldırdı")
            agent_base.health -= enemy.attack
            reward = - ENEMY_PLAYER_HIT_PENALTY
            if agent_base.health < 0:
                reward = -  WIN_REWARD
        if reward == 0:
            reward = -MOVE_PENALTY

        new_obs = (agent - enemy, agent - enemy_base, enemy - agent_base)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == - ENEMY_PLAYER_HIT_PENALTY:
            new_q = - ENEMY_PLAYER_HIT_PENALTY
        elif reward == BASE_HIT_PENALTY:
            new_q = BASE_HIT_PENALTY
        elif reward == ENEMY_PLAYER_HIT_PENALTY:
            new_q = ENEMY_PLAYER_HIT_PENALTY
        elif reward == WIN_REWARD:
            reward == WIN_REWARD
        elif reward == - WIN_REWARD:
            reward == -WIN_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        q_table[obs][action] = new_q

        episode_reward += reward
        if reward == WIN_REWARD or reward == -WIN_REWARD:
            break
        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env[agent.x][agent.y] = d[0]
            env[agent_base.x][agent_base.y] = d[1]
            '''            
            env[agent_base.x+1][agent_base.y] =d[1]
            env[agent_base.x][agent_base.y+1] = d[1]
            env[agent_base.x+1][agent_base.y+1] =d[1]
            '''

            env[enemy.x][enemy.y] = d[2]
            env[enemy_base.x][enemy_base.y] = d[3]
            '''            
            env[enemy_base.x + 1][enemy_base.y] = d[3]
            env[enemy_base.x][enemy_base.y + 1] = d[3]
            env[enemy_base.x + 1][enemy_base.y + 1] = d[3]
            '''

            img = Image.fromarray(env, "RGB")
            img = img.resize((300, 300))
            cv2.imshow("", np.array(img))
            time.sleep(0.5)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}")
plt.xlabel(f"Episode # ")
plt.show()

with open(f"q-table-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
