import numpy as np
import pickle
import math
import random

max_heap = 100

Actions = [1, 2, 3]

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
        # update policy for s
        # pi[s, 1] = 
        # pi[s, 2]
        # pi[s, 3]


        compute_vs(pi)
        V[s] = min_score
        alpha = alpha * decay



def compute_vs(pi):
    global heap, Q, V
    min_score = math.inf
    min_action = 1
    for odash in Actions:
        score = 0
        for adash in Actions:
            score += pi[s, adash] * Q[s, adash, odash]

        if (score < min_score):
            min_action = odash

    return (min_score, min_action)


def minimax(p, s, memoization_enabled = True):
    minimax_called += 1
    if (memoization_enabled and str(s) in memoized):
        return memoized[str(s)]

    if (p.terminal_test(s)):
        newresult = p.utility(s)
    else:
        mmvals = [minimax(p, n, memoization_enabled) for n in p.actions(s)]
        if (p.player(s) == "X"): # max
            newresult = max(mmvals)
        else:
            newresult = min(mmvals)

    memoized[str(s)] = newresult
    return newresult
