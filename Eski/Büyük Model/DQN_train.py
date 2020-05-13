import os

from tqdm import tqdm

from DQNAgent import *
from DQN_Envoriment import *


def train():
    # model_path="models\\Model_256_256__1144.00max_1144.00avg_1144.00min__1588194199.model"
    # model_path = "final_models\\new_Model_256_256__STEP=20000__3484.00max_2466.80avg_1980.00min__1588666438.model"
    model_path = ""
    log = ""
    # log = "logs\\Model_256_256_1588152906"
    # HYPERPAREMETERS
    # Environment settings
    EPISODES = 20_000
    # Exploration settings

    epsilon = 1 if model_path == "" else 0.1
    # not a constant, going to be decayed
    EPSILON_DECAY = 0.99975
    MIN_EPSILON = 0.001

    #  Stats settings
    AGGREGATE_STATS_EVERY = 50  # episodes
    SHOW_PREVIEW = False
    SAVE_MODEL_FREQ = 1000

    env = Env(SHOW_PREVIEW)

    # For stats
    ep_rewards = [-200]

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

    agent = DQNAgent(env, model_path, log)

    # Iterate over episodes

    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()
        # Reset flag and start iterating until episode ends
        done = False
        while not done:

            # This part stays mostly the same, the change is to query a model for Q values
            actions = []
            if np.random.random() > epsilon:
                # Get action from Q table
                all_acts = agent.get_qs(current_state)
                # print(all_acts)
                for i in range(env.NUMBER_OF_AGENT_PLAYER):
                    actions.append(np.argmax(all_acts[i * env.ACTION_SPACE_SIZE: (i + 1) * env.ACTION_SPACE_SIZE]))
            else:
                # Get random action
                actions.append(np.random.randint(0, env.ACTION_SPACE_SIZE))
                actions.append(np.random.randint(0, env.ACTION_SPACE_SIZE))
                actions.append(np.random.randint(0, env.ACTION_SPACE_SIZE))

            new_state, reward, done, RENDER_INFOS = env.step(actions, step, episode)
            # Transform new continous state to new discrete state and count reward
            episode_reward += reward

            if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                env.render(episode_reward, RENDER_INFOS)

            # Every step we update replay memory and train main network
            agent.update_replay_memory((current_state, actions, reward, new_state, done))
            agent.train(done)

            current_state = new_state
            step += 1

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            print(f"Min reward {min_reward} Max reward {max_reward} avg {average_reward}")
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                           epsilon=epsilon, number_of_step=step)
        if not episode % SAVE_MODEL_FREQ or episode == 1:
            # Save model, but only when min reward is greater or equal a set value
            agent.model.save(
                f'models/{MODEL_NAME}__STEP={episode}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)


if __name__ == '__main__':
    train()
