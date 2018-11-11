import numpy as np
import pickle
import math
import random

max_heap = 20

heap = max_heap

Actions = [1, 2, 3]

"""
Citation: this code implements the q minimax algorithm from "Markov games as a framework for multi-agent reinforcement learning" by littman

Implementation is my own
"""
Q = np.ones((max_heap + 1, 3, 3))
V = np.ones(max_heap)
pi = np.full((max_heap + 1, 3), 1/3)
alpha = 1
decay = 0.99
gamma = 0.8

player_turn = True

def opponent_acts():
    global heap, Q, V, pi
    action = random.choice(Actions)
    heap -= action
    return action

def player_acts():
    global heap, Q, V, pi
    if random.uniform(0, 1) < 0.1: # exploring
        action = random.choice(Actions)
    else:
        r = random.uniform(0, 1)
        for a in Actions:
            if r <= pi[heap, a - 1]:
                action = a
            else:
                r -= pi[heap, a - 1]
    heap -= action
    return (heap + action, heap, action)

def learn():
    global player_turn, heap, Q, V, pi, alpha, decay, gamma
    # play multiple games

    heap = max_heap
    #player_turn = False # opponent goes first so we can learn
    game = [] # record game
    reward = -30
    while (heap > 0):
        o = opponent_acts()
        if (heap <= 0):
            reward = 30
            break
        s, sdash, a = player_acts()
        game.append((s, sdash, a, o))

    rew = reward/len(game)

    for move in game:
        s, sdash, a, o = move
        a -= 1 # adjust index
        o -= 1 # adjust index
        Q[s, a, o] = (1 - alpha) * Q[s, a, o] + alpha * (rew + gamma + V[sdash])
        # TODO figure out lpps
        # update policy for s, we can just do it manually because action space is sweet
        p = np.linspace(0, 1, 10)
        q = np.linspace(0, 1, 10)
        best_policy = pi[s]
        best_score = math.inf
        for p_i in p:
            for q_i in q:
                new_policy = [p_i, p_i + q_i, 1 - p_i - q_i]
                min_score, min_action = compute_vs(new_policy, s)
                if min_score < best_score:
                    best_policy = new_policy
                    best_score = min_score
                    

        #print(best_policy)

        pi[s, 0] = best_policy[0]
        pi[s, 1] = best_policy[1]
        pi[s, 2] = best_policy[2]


        min_score, min_action = compute_vs(pi[s], s)
        V[s] = min_score
        alpha = alpha * decay

def compute_vs(pi, s):
    global heap, Q, V
    min_score = math.inf
    min_action = 1
    for odash in Actions:
        score = 0
        for adash in Actions:
            score += (pi[adash - 1] * Q[s, adash - 1, odash - 1])

        if (score < min_score):
            min_action = odash
            min_score = score

    return (min_score, min_action)

for i in range(10):
    learn()

print("new pi")
print(pi)
