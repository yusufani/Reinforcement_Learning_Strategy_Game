import os

from tqdm import tqdm

from DQNAgent import *
from DQN_Envoriment import *

os.environ['SDL_VIDEO_CENTERED'] = '1'


def train():
    # model_path="models\\Model_256_256__1144.00max_1144.00avg_1144.00min__1588194199.model"
    # model_path = "final_models\\new_Model_256_256__STEP=20000__3484.00max_2466.80avg_1980.00min__1588666438.model"
    model_path1 = "final_new_models\\new_Model_256_256__STEP=500__AgentNO 0__789.00max_-1518.80avg_-3005.00min__1589276304.model"
    model_path2 = "final_new_models\\new_Model_256_256__STEP=500__AgentNO 1_-802.00max_-1332.30avg_-1922.00min__1589276304.model"
    model_path3 = "final_new_models\\new_Model_256_256__STEP=500__AgentNO 2__270.00max_-868.20avg_-1547.00min__1589276304.model"

    log1 = ""
    log2 = ""
    log3 = ""
    '''    
    log1 = "logs\\"
    log2 = "logs\\"
    log3 = "logs\\"
    '''
    # log = "logs\\Model_256_256_1588152906"
    # HYPERPAREMETERS
    # Environment settings
    EPISODES = 20_000
    # Exploration settings

    epsilon = 1 if model_path1 == "" else 0.2
    # not a constant, going to be decayed
    EPSILON_DECAY = 0.99975
    MIN_EPSILON = 0.001

    #  Stats settings
    AGGREGATE_STATS_EVERY = 1  # episodes
    SHOW_PREVIEW = True
    SAVE_MODEL_FREQ = 500

    env = Env(SHOW_PREVIEW)

    # For stats
    ep_rewards = [[], [], []]

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)

    # Memory fraction, used mostly when training multiple agents
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    # backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')

    agents = []

    agents.append(DQNAgent(env, 0, model_path1, log1))
    agents.append(DQNAgent(env, 1, model_path2, log2))
    agents.append(DQNAgent(env, 2, model_path3, log3))

    # Iterate over episodes

    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

        # Update tensorboard step every episode
        for agent in agents:
            agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_rewards = [0, 0, 0]
        step = 1

        # Reset environment and get initial state
        current_states = env.reset()
        # Reset flag and start iterating until episode ends
        only_once = [True, True, True]
        dones = {0: False, 1: False, 2: False, -1: False}
        while is_finish(dones):
            # This part stays mostly the same, the change is to query a model for Q values
            actions = []
            if np.random.random() > epsilon:
                # Get action from Q table
                # print(all_acts)
                for i, agent in enumerate(agents):
                    if not dones[i]:
                        actions.append(np.argmax(agent.get_qs(current_states[i])))
                    else:
                        actions.append(-1)
            else:
                # Get random action
                for i in range(env.N_AGENT):
                    if not dones[i]:
                        actions.append(np.random.randint(0, env.ACTION_SPACE_SIZE))
                    else:
                        actions.append(-1)

            new_states, rewards, dones, RENDER_INFOS = env.step(actions, dones)

            # print(rewards)
            # print(dones)
            # Transform new continous state to new discrete state and count reward
            for idx in range(len(episode_rewards)):
                episode_rewards[idx] += rewards[idx]
            # print("\n\n")
            # print("Step rew " ,RENDER_INFOS[1])
            # print("Ep rew" , episode_rewards)
            if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                RENDER_INFOS[1].append(
                    f"TOTAL REWARDS AGENT-0 {str(episode_rewards[0])}  AGENT-1 {str(episode_rewards[1])}  AGENT-2 {str(episode_rewards[2])} ")
                env.render(RENDER_INFOS)
            for idx, agent in enumerate(agents):
                if dones[idx] and only_once[idx]:
                    agent.update_replay_memory(
                        (current_states[idx], actions[idx], rewards[idx], new_states[idx], dones[idx]))
                    agent.train(dones[idx])
                    only_once[idx] = False
                elif not dones[idx]:
                    agent.update_replay_memory(
                        (current_states[idx], actions[idx], rewards[idx], new_states[idx], dones[idx]))
                    agent.train(dones[idx])

            current_states = new_states
            step += 1
            if step > 10000:
                print("Adım sayısı aşıldı")
                break
            # Every step we update replay memory and train main network
        print("")
        for idx, agent in enumerate(agents):
            ep_rewards[idx].append(episode_rewards[idx])
            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(ep_rewards[idx][-AGGREGATE_STATS_EVERY:]) / len(
                    ep_rewards[idx][-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[idx][-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[idx][-AGGREGATE_STATS_EVERY:])
                print(f"Min reward {min_reward} Max reward {max_reward} avg {average_reward}")
                agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward,
                                               reward_max=max_reward,
                                               epsilon=epsilon, number_of_step=step)
            if not episode % SAVE_MODEL_FREQ or episode == 1:
                # Save model, but only when min reward is greater or equal a set value
                agent.model.save(
                    f'models/{MODEL_NAME}__STEP={episode}__AgentNO {idx}_{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)

        # Append episode reward to a list and log stats (every given number of episodes)


def is_finish(dones):
    count = 1
    for i, done in dones.items():
        count += 1 if done else 0
    return True if count < 4 else False


if __name__ == '__main__':
    train()
