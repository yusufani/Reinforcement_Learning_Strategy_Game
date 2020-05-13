from DQNAgent import *
from DQN_Envoriment import *


def test():
    model_path = "final_models\\new_Model_256_256__STEP=20000__-1688.00max_-2880.36avg_-8524.00min__1589009900.model"
    log = ""
    # log = "logs\\Model_256_256_1588152906"
    # HYPERPAREMETERS
    # Environment settings
    # Exploration settings

    env = Env(True)

    # For stats

    # Memory fraction, used mostly when training multiple agents
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    # backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    agent = DQNAgent(env, model_path, log, test=True)

    # Iterate over episodes

    while (True):

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        # Reset environment and get initial state
        current_state = env.reset()
        # Reset flag and start iterating until episode ends
        done = False
        step = 0
        while not done:
            step += 1
            # This part stays mostly the same, the change is to query a model for Q values
            actions = []
            # print(current_state)
            all_acts = agent.get_qs(current_state)
            print(all_acts)
            # print(all_acts)
            for i in range(env.NUMBER_OF_AGENT_PLAYER):
                actions.append(np.argmax(all_acts[i * env.ACTION_SPACE_SIZE: (i + 1) * env.ACTION_SPACE_SIZE]))

            print(actions)
            new_state, reward, done, RENDER_INFOS = env.step(actions, step, 500000)
            # Transform new continous state to new discrete state and count reward
            episode_reward += reward
            current_state = new_state
            env.render(episode_reward, RENDER_INFOS)

            # Every step we update replay memory and train main network


if __name__ == '__main__':
    test()
