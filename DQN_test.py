from DQNAgent import *
from DQN_Envoriment import *


def test():
    model_path1 = "final_new_models\\new_Model_256_256__STEP=500__AgentNO 0__789.00max_-1518.80avg_-3005.00min__1589276304.model"
    model_path2 = "final_new_models\\new_Model_256_256__STEP=2500__AgentNO 1_-397.00max_-1072.90avg_-1626.00min__1589308173.model"
    model_path3 = "final_new_models\\new_Model_256_256__STEP=2500__AgentNO 2_1781.00max_-480.80avg_-2080.00min__1589308173.model"
    log = ""
    # log = "logs\\Model_256_256_1588152906"
    # HYPERPAREMETERS
    # Environment settings
    # Exploration settings
    RENDER_SPEED = 0.2  # 0.2
    GET_FOREST_FROM_CONSOLE = False
    env = Env(True)

    # For stats

    # Memory fraction, used mostly when training multiple agents
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    # backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    agents = []
    agents.append(DQNAgent(env, 0, model_path1, test=True))
    agents.append(DQNAgent(env, 1, model_path2, test=True))
    agents.append(DQNAgent(env, 2, model_path3, test=True))

    # Iterate over episodes

    while (True):

        # Restarting episode - reset episode reward and step number
        episode_rewards = [0, 0, 0]

        # Reset environment and get initial state
        current_states = env.reset(GET_FOREST_FROM_CONSOLE)
        # Reset flag and start iterating until episode ends
        step = 1
        only_once = [True, True, True]
        dones = {0: False, 1: False, 2: False}
        while not dones[0] or not dones[1] or not dones[2]:
            step += 1
            # This part stays mostly the same, the change is to query a model for Q values
            actions = []
            # print(current_state)
            for i, agent in enumerate(agents):
                if not dones[i]:
                    if np.random.random() > 0.1:
                        print(agent.index, "-> ", agent.get_qs(current_states[i]))
                        actions.append(np.argmax(agent.get_qs(current_states[i])))
                    else:
                        actions.append(np.random.randint(0, env.ACTION_SPACE_SIZE))
                else:
                    actions.append(-1)
            print(actions)
            # print(all_acts)

            new_states, rewards, dones, RENDER_INFOS = env.step(actions, dones)
            for idx in range(len(episode_rewards)):
                episode_rewards[idx] += rewards[idx]
            RENDER_INFOS[1].append(
                f"TOTAL REWARDS AGENT-1 {str(episode_rewards[0])}  AGENT-2 {str(episode_rewards[1])}  AGENT-3 {str(episode_rewards[2])} ")
            env.render(RENDER_INFOS, RENDER_SPEED)
            # Transform new continous state to new discrete state and count reward
            # d
            current_states = new_states
            step += 1

            # Every step we update replay memory and train main network


if __name__ == '__main__':
    test()
