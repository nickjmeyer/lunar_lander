import numpy as np
import scipy as sp
import scipy.interpolate as spi
import gym
import random
import time
import keras

class QfnNN(object):
    def __init__(self, learning_rate):
        self.models = [keras.models.Sequential([
            keras.layers.Dense(32, activation="relu", input_shape=(8, )),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(1)])
                       for _ in range(4)]

        for model in self.models:
            model.compile(loss='mse',
                          optimizer=keras.optimizers.Adam(lr=learning_rate))

    def get_input(self, action, state):
        input_array = np.zeros((1,8))
        input_array[0, :] = state

        return input_array

    def eval(self, action, state):
        input_array = self.get_input(action, state)

        output = self.models[action].predict(input_array)

        return output

    def arg_max(self, state):
        values = [(self.models[action].predict(self.get_input(action, state))[0, 0], action)
                  for action in range(4)]

        print(values)

        return max(values)[1]

    def fit(self, state, action, target):
        self.models[action].fit(self.get_input(action, state), target, epochs = 1, verbose = 0)



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

eps = 0.1

qfn = QfnNN(0.1)

alpha = 1e-4

gamma = 0.95

for i in range(250):
    observation = env.reset()
    reward = None
    done = False
    value = 0.0
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

            target = reward + gamma * eval_arg_max_new_state

            qfn.fit(observation, action, target)

        observation = new_observation
        reward = new_reward

        value *= gamma
        value += reward

    print("Episode " + str(i) + " -> " + str(value))

observation = env.reset()
done = False
while not done:
    env.render()
    action = qfn.arg_max(observation)

    observation, _, done, _ = env.step(action)

env.render()
env.close()
