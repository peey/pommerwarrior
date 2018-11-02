import gym
import numpy as np
import time, pickle, os

env = gym.make('FrozenLake-v0')
env.reset()

eps = 0.9
epochs = 10000
max_steps = 100
lr_rate = 0.81
gamma = 0.96

Q = np.zeros((env.observation_space.n, env.action_space.n))

def choose_action(state):
	action = 0
	if np.random.uniform(0,1) < eps:
		# Random action from the space
		action = env.action_space.sample()
	else:
		action = np.argmax(Q[state, :])
	return action

def learn(state, state2, reward, action):
	predict = Q[state, action]
	target = reward + gamma * np.max(Q[state2, :])
	Q[state, action] = Q[state, action] + lr_rate * (target - predict)

for episode in range(epochs):
	state = env.reset()
	t = 0

	while t<max_steps:
		env.render()
		action = choose_action(state)
		state2, reward, done, info = env.step(action)
		learn(state, state2, reward, action)

		state = state2
		t+=1
		if done:
			break
		# Quite irritating, takes hell long even with 0.5 secs
		# time.sleep(0.5)

print(Q)

with open("frozenlakev0_qtable.pkl", 'wb') as f:
	pickle.dump(Q, f)
