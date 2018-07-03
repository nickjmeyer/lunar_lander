import numpy as np
import scipy as sp
import scipy.interpolate as spi
import gym
import gym.wrappers
import random
import time
import keras

class QfnNN(object):
    def __init__(self, state_dim, action_dim, learning_rate, optimism, optimism_decay):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.models = [keras.models.Sequential([
            keras.layers.Dense(32, activation="relu", input_shape=(self.state_dim, )),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(1)])
                       for _ in range(self.action_dim)]

        for model in self.models:
            model.compile(loss='mse',
                          optimizer=keras.optimizers.Adam(lr=learning_rate))

        self.optimism = [optimism] * self.action_dim
        self.optimism_decay = optimism_decay

        self.data = [[] for _ in range(self.action_dim)]

    def get_input(self, action, state):
        input_array = np.zeros((1,self.state_dim))
        input_array[0, :] = state

        return input_array

    def eval(self, action, state):
        input_array = self.get_input(action, state)

        output = self.models[action].predict(input_array)

        return output

    def arg_max(self, state):
        values = [(self.models[action].predict(self.get_input(action, state))[0, 0] + self.optimism[action], action)
                  for action in range(self.action_dim)]

        best_action = max(values)[1]
        self.optimism[best_action] *= self.optimism_decay

        return best_action

    def store(self, state, action, target):
        self.data[action].append((state, target))

    def fit(self):
        for action, data in enumerate(self.data):
            if len(data) == 0:
                continue

            x = np.zeros((len(data), self.state_dim))
            y = np.zeros((len(data), ))

            for index, (d_state, d_target) in enumerate(data):
                x[index, :] = self.get_input(action, d_state)
                y[index] = d_target

            self.models[action].fit(x, y, epochs = 1, verbose = 0)

            data.clear()

    def replay(self, history, gamma):
        for data in self.data:
            data.clear()

        for state, action, new_state, reward in history:
            eval_arg_max_new_state = qfn.eval(qfn.arg_max(new_state), new_state)
            target = reward + gamma * eval_arg_max_new_state

            self.store(state, action, target)

        self.fit()


class QfnPoly(object):
    def __init__(self, state_dimension):
        self.n_terms = 1 + state_dimension + state_dimension**2 + state_dimension**3
        self.weights = [np.random.uniform(size = self.n_terms) for _ in range(4)]

    def create_terms(self, state):
        terms = np.zeros((self.n_terms,))

        terms[0] = 1.0

        beg = 1
        end = len(state) + beg
        terms[beg:end] = state

        beg = end
        end = beg + len(state)**2
        terms[beg:end] = np.outer(state, state).flatten()


        beg = end
        end = beg + len(state)**3
        terms[beg:end] = np.outer(np.outer(state, state).flatten(), state).flatten()

        return terms


    def eval(self, action, state):
        w = self.weights[action]
        return w.dot(self.create_terms(state))

    def derivative(self, action, state):
        terms = self.create_terms(state)

        return terms

    def arg_max(self, state):
        terms = self.create_terms(state)

        values = [(w.dot(terms), a) for a, w in enumerate(self.weights)]

        return max(values)[1]


gym.logger.set_level(40)

env = gym.make("LunarLander-v2")

eps = 0.05

qfn = QfnNN(8, 4, 0.001, 500, 0.999)

# for i in range(4):
#     qfn.models[i].load_weights('current_' + str(i) + '.h5')

alpha = 1e-4

gamma = 0.95

best_episodes = []

for i in range(250):
    observation = env.reset()
    reward = None
    done = False
    value = 0.0
    steps = 0

    history = []
    while not done:
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = qfn.arg_max(observation)

        new_observation, new_reward, done, info = env.step(action)

        # weights = qfn.weights[action]

        if reward is not None:
            eval_arg_max_new_state = qfn.eval(qfn.arg_max(new_observation), new_observation)
            # eval_state = qfn.eval(action, observation)
            # derivative = qfn.derivative(action, observation)
            # weights += alpha * (reward + gamma * eval_arg_max_new_state - eval_state) * derivative

            # reward += 50.0 / (np.linalg.norm(new_observation[:2]) + 1)

            if not done:
                target = reward + gamma * eval_arg_max_new_state
            else:
                target = reward

            qfn.store(observation, action, target)

            history.append((observation, action, new_observation, reward))


        observation = new_observation
        reward = new_reward

        value *= gamma
        value += reward

        steps += 1

    qfn.fit()

    if value >= 0.0:
        best_episodes.append((value, history))

        if len(best_episodes) > 25:
            best_episodes = sorted(best_episodes)[:25]

    print("Episode " + str(i) + " -> " + str(value) + " (" + str(steps) + ")")


for value, history in best_episodes:
    qfn.replay(history, gamma)
print("done")


# for i in range(4):
#     qfn.models[i].save_weights('demo_0' + str(i) + '.h5')

# video_count = 0

for i in range(10):
    recording_env = gym.wrappers.Monitor(env, './test' + str(video_count), force=True)
    video_count += 1
    observation = recording_env.reset()
    done = False
    value = 0.0
    while not done:
        recording_env.render()
        action = qfn.arg_max(observation)

        observation, reward, done, _ = recording_env.step(action)

        value *= gamma
        value += reward

    print("value: " + str(value))

    recording_env.render()
recording_env.close()
env.close()
