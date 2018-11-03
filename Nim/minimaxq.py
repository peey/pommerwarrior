import numpy as np
import pickle
import math
import random

max_heap = 100

Actions = [1, 2, 3]

"""
Citation: this code implements the q minimax algorithm from "Markov games as a framework for multi-agent reinforcement learning" by littman

Implementation is my own
"""
Q = np.ones((max_heap, 3, 3))
V = np.ones(max_heap)
pi = np.full((max_heap, 3), 1/3)
alpha = 1
decay = 0.99

player_turn = True

def opponent_acts():
    global heap, Q, V, pi
    action = Actions.sample()
    heap -= action
    return action

def act():
    global heap, Q, V, pi
    if random.uniform(0, 1) < 0.1: # exploring
        action = Actions.sample()
    else:
        r = random.uniform(0, 1)
        for a in Actions:
            if r <= pi[heap, a - 1]:
                action = a
            else:
                r -= pi[heap, a - 1]
    return (heap + action, heap, action)

def learn():
    global player_turn, heap, Q, V, pi, alpha, decay
    # play multiple games

    max_heap = 100
    #player_turn = False # opponent goes first so we can learn
    game = [] # record game
    reward = -30
    while (max_heap > 0):
        o = opponent_acts()
        if (max_heap <= 0):
            reward = 30
            break
        s, sdash, a = player_acts()
        game.append((s, sdash, a, o))

    rew = reward/len(game)

    for move in game:
        s, sdash, a, o = move
        Q[s, a, o] = (1 - alpha) * Q[s, a, o] + alpha * (rew + gamma + V[sdash])
        # update policy for s, we can just do it manually because action space is sweet
        p = np.linspace(0, 1, 0.1)
        q = np.linspace(0, 1, 0.1)
        best_policy = pi[s]
        best_score = pi[s]
        for p_i in p:
            for q_i in q:
                new_policy = [p_i, p_i + q_i, 1 - p_i - q_i]
                min_score, min_action = compute_vs(new_policy)
                if min_score < best_score:
                    best_policy = new_policy
                    best_score = min_score
                    


        pi[s, 1] = best_policy[1]
        pi[s, 2] = best_policy[2]
        pi[s, 3] = best_policy[3]


        min_score, min_action = compute_vs(pi, s)
        V[s] = min_score
        alpha = alpha * decay



def compute_vs(pi, s):
    global heap, Q, V
    min_score = math.inf
    min_action = 1
    for odash in Actions:
        score = 0
        for adash in Actions:
            score += pi[adash] * Q[s, adash, odash]

        if (score < min_score):
            min_action = odash

    return (min_score, min_action)
