import numpy as np
import pickle
import math
import random
from cvxopt import matrix, solvers

# citation: below two lines to supress solver output are taken from https://github.com/tulip-control/polytope/commit/4650577b3de2104f5483197ff53e90c0188fa3ac
solvers.options['show_progress'] = False
solvers.options['glpk'] = dict(msg_lev='GLP_MSG_OFF')

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

max_heap = 8

heap = max_heap - 1

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

def random_opponent_acts():
    global heap, Q, V, pi
    action = random.choice(Actions)
    heap -= action
    return action


def optimal_opponent_acts():
    global heap, Q, V, pi
    mod = heap % 4

    if (mod != 1): # then we're in a winning position, we put opponent in a losing position
        if mod == 0:
            action = 3
        if mod == 2:
            action = 1
        if mod == 3:
            action = 2
        heap -= action
        return action
    else:
        return random_opponent_acts() # disorient

def tired_optimal_opponent_acts():
    if (random.uniform(0, 1) < 0.3):
        return random_opponent_acts()
    else:
        return optimal_opponent_acts()

def player_acts(explore):
    global heap, Q, V, pi
    if random.uniform(0, 1) < explore: # exploring
        action = random.choice(Actions)
    else:
        r = random.uniform(0, 1)
        action = Actions[0]
        for a in Actions:
            if r <= pi[heap, a - 1]:
                action = a
            else:
                r -= pi[heap, a - 1]
    heap -= action
    return (heap + action, heap, action)

def cheating_player_acts(explore):
    global heap
    if (random.uniform(0, 1) < 0.3):
        action = optimal_opponent_acts()
        return (heap + action, heap, action)
    else:
        return player_acts(explore)

def play_game(explore, playerfunc, opponentfunc):
    global heap
    heap = max_heap - 1
    game = [] # record game
    while True:
        s, sdash, a = playerfunc(explore) # player acts first
        if (heap <= 0): # if we pick up last pebble, we lose
            game.append((s, 0, a, random.choice(Actions)))
            reward = -30
            break
        o = opponentfunc()
        game.append((s, sdash, a, o))
        if (heap <= 0): # if opponent picks up last pebble, we win
            reward = 30
            break

    rew = reward/len(game)
    return (game, rew)

def learn(explore, playerfunc, opponentfunc):
    global heap, Q, V, pi, alpha, decay, gamma
    # play multiple games

    game, rew = play_game(explore, playerfunc, opponentfunc) # high explore while learning

    for move in game:
        s, sdash, a, o = move
        a -= 1 # adjust index
        o -= 1 # adjust index
        Q[s, a, o] = (1 - alpha) * Q[s, a, o] + alpha * (rew + gamma + V[sdash])
        # TODO figure out lpps
        # we're going to use LPP to find pi[s, .]
        # The LPP variables will be x_i = pi[s, a_i] for i from 1 to |A|
        # Constraints will be
        #  1. x_i >= 0 forall i
        #  2. x_i <= 1 forall i
        #  3. \Sigma x_i = 1 

        # The last is an equality constraint, which basically reduces number of variables by 1 (third variable's policy value is 1 - sum of others)



        # Careful: cvxopt.matrix is in a column-major format and np.array is in row-major but we only need to worry about this when manually writing an array, otherwise the system knows internally how to handle a row-major or column-major matrix
        # One row represents one equation

        #Ax = b, maximize c
        # All equations must use <=, that's cvxopt convention
        lenA = len(Actions) - 1
        lp_A = np.vstack((-np.identity(lenA), np.identity(lenA), np.ones(lenA)))
        lp_b = np.vstack((np.zeros((lenA, 1)), np.ones((lenA, 1)), np.array([1])))

        # Note that we take actions at the same time, it's not alternating turn-based, so this way of minimizing o should work

        # assumption is that opponent will choose a move which give us the lowest possible payoff in our best policy against that move
        min_score = math.inf
        best_possible_policy = pi[s]

        print("%s: %s" % (s, Q[s]))
        for odash in Actions:
            # converting maximization function to 2 variables,
            # (p1*q1 + p2*q2 + p3*q3) becomes (p1*q1 + p2*q2 + (1 - p1 - p2)*q3)
            # becomes (p1* (q1 - q3) + p2*(q2 - q3) + q3) but we can drop the last q3 since it's a constant and will have no effect on maximization
            # by default it minimizes the given function, so we use - of the function we want to maximize
            lp_c = -np.array([[Q[s, adash - 1, odash - 1] - Q[s, lenA, odash - 1] for adash in Actions[:-1]]]).transpose() # gives us a numpy column vector

            sol = solvers.lp(matrix(lp_c), matrix(lp_A), matrix(lp_b))

            best_policy_for_odash = list(sol['x']) + [1 - sum(list(sol['x']))]

            score = action_policy_score(best_policy_for_odash, s, odash)

            print("%3.3f | %s " % (score, np.array(best_policy_for_odash)))

            if (score < min_score):
                best_possible_policy = best_policy_for_odash

        print()

        pi[s] = np.array(best_possible_policy).transpose()

        min_score, min_action = compute_vs(pi[s], s)
        V[s] = min_score
        alpha = alpha * decay

def action_policy_score(pi, s, o):
    global Q, V
    score = 0
    for adash in Actions:
        score += (pi[adash - 1] * Q[s, adash - 1, o - 1])
    return score
    

def compute_vs(pi, s):
    global Q, V
    min_score = math.inf
    min_action = 1
    for odash in Actions:
        score = action_policy_score(pi, s, odash)

        if (score < min_score):
            min_action = odash
            min_score = score

    return (min_score, min_action)

for i in range(200):
    learn(0.3, cheating_player_acts, random_opponent_acts)

for i in range(200):
    learn(0.3, cheating_player_acts, tired_optimal_opponent_acts)

for i in range(200):
    learn(0.3, player_acts, tired_optimal_opponent_acts)

for i in range(600):
    learn(0.1, player_acts, optimal_opponent_acts)

def pretty_print_policy(a):
    for (i, row) in enumerate(a):
        print("%3d: %s" % (i, row) )

def pretty_print_q(Q):
    for state in range(len(Q)):
        print("~~~~ state : %s ~~~~ " % state)
        print("Action 1: %s" % (Q[state][0]))
        print(Q[state])

def eval_player(n, opponent):
    wins = 0
    for i in range(n):
        game, rew = play_game(0, player_acts, opponent)
        if rew > 0:
            wins += 1
    return wins / n

print("new pi")
pretty_print_policy(pi)
print("========================")
pretty_print_q(Q)
print("========================")
pretty_print_policy(V)
print("========================")
print("Optimal opponent win rate: %f" % (eval_player(100, optimal_opponent_acts)))
print("Suboptimal oppn. win rate: %f" % (eval_player(100, tired_optimal_opponent_acts)))
