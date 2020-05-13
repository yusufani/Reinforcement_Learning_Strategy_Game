from _collections import deque

import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam

from DQN_Envoriment import *

MINIBATCH_SIZE = 64
import datetime

REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000

DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5

MODEL_NAME = r"new_Model_256_256"


class DQNAgent:
    def __init__(self, env, model_path="", log="", test=False):
        self.env = env
        if test:
            self.model = load_model(model_path)
        else:
            self.env = env
            self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
            self.target_update_counter = 0
            if model_path == "":
                # Main Model # gets trained every step
                self.model = self.create_model()
                # Target Model this is what we ..predict against every step

                log_dir = f"logs\\{MODEL_NAME}_{str(datetime.datetime.now()).replace(':', ' ').replace(' ', '_')}"
                self.tensorboard = ModifiedTensorBoard(log_dir=log_dir)
            else:
                self.model = load_model(model_path)
                self.tensorboard = ModifiedTensorBoard(log_dir=log)
            self.target_model = self.create_model()
            self.target_model.set_weights(self.model.get_weights())

    def create_model(self):
        model = Sequential()
        state_shape = self.env.OBSERVATION_SPACE_VALUES
        model.add(Dense(128, input_dim=state_shape,
                        activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.env.ACTION_SPACE_SIZE * self.env.NUMBER_OF_AGENT_PLAYER, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        # print(state.shape)
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def train(self, terminal_state):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states = np.array([transation[0] for transation in minibatch])
        current_qs_list = self.model.predict(current_states)
        new_current_states = np.array([transation[3] for transation in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []  # observations
        y = []  # Actions

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            current_qs = current_qs_list[index]
            for i in range(len(self.env.agent_players)):
                if not done:
                    max_future_q = np.max(
                        future_qs_list[index][i * self.env.ACTION_SPACE_SIZE:(i + 1) * self.env.ACTION_SPACE_SIZE])
                    new_q = reward + DISCOUNT * max_future_q
                else:
                    new_q = reward
                current_qs[i * self.env.ACTION_SPACE_SIZE + action[i]] = new_q
            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False,
                       callbacks=[self.tensorboard] if terminal_state else None)
        # Updating to determine if we want o-to update target_model yet
        if terminal_state:
            self.target_update_counter += 1
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        # print(**kwargs)
        super().__init__(**kwargs)
        print(self.log_dir)
        self.step = 1
        # self.writer = tf.summary.create_file_writer(self.log_dir)
        self.writer = tf.summary.FileWriter(self.log_dir)
        # self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

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
        # tf.summary.scalar('loss', stats['loss'], step=self.step)
        self._write_logs(stats, self.step)
