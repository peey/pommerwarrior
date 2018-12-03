import numpy as np
from util import Proximity, BlastStrength, DeadState, Actions
import enhanced_percepts as ep

"""
Enhanced actions could give control to a deterministic script for multiple time steps

Possible single-step actions:
    - Step toward (nearest) enemy
    - Step away from (nearest) enemy
    - Move toward (nearest) powerup 

Possible multi-step actions:
    - Chase a particular enemy
    - Attempt to get (different kinds of?) powerups 
    - Escape from a boxed-in situation

Possible states:
    - Suicide / Killed By Enemy (-100000000)
    - Won the game
    - No of enemies killed
"""

all_pairs_shortest_paths = np.full((11, 11, 11, 11), None, dtype=object)

def get_shortest_path_between(x1, y1, x2, y2):
    """
    if  coord_less_than(x1, y1, x2, y2):
        return all_pairs_shortest_paths[x1, y1, x2, y2]
    else:
        return reversed(all_pairs_shortest_paths[x2, y2, x1, y1])
    """
    return all_pairs_shortest_paths[x1, y1, x2, y2]

def coord_less_than(x1, y1, x2, y2): # left to right then top to bottom. Don't know if this is faster or coord to id would be faster.
    return y1 < y2 or (y1 == y2 and x1 < x2)

def floyd_warshall(board): # takes ((11^2)^3)/2 steps once in the begining of the game
    # citation: referenced wikipedia for pseudocode of algo, implementation is mine
    m, n = board.shape
    dist = np.full((m, n, m, n), 50, dtype=np.int8) # int8 since max manhattan distance won't exced 22

    for (x, y), value in np.ndenumerate(board):
        if x < m - 1 and board[x + 1, y] != 1:
            dist[x, y, x + 1, y] = 1
            all_pairs_shortest_paths[x, y, x + 1, y] = []
        if y < n - 1 and board[x, y + 1] != 1:
            dist[x, y, x, y + 1] = 1
            all_pairs_shortest_paths[x, y, x, y + 1] = []

    for (x_k, y_k), _ in np.ndenumerate(board):
        for (x_i, y_i), _ in np.ndenumerate(board):
            for (x_j, y_j), _ in np.ndenumerate(board):
                #if not coord_less_than(x_i, y_i, x_j, y_j):
                    #assert(coord_less_than(x_j, y_j, x_i, y_i) or (x_i == x_j and y_i == y_j))
                    #break # because undirected graph

                new_distance = dist[x_i, y_i, x_k, y_k] + dist[x_k, y_k, x_j, y_j]
                if new_distance <= dist[x_i, y_i, x_j, y_j]:
                    dist[x_i, y_i, x_j, y_j] = new_distance
                    all_pairs_shortest_paths[x_i, y_i, x_j, y_j] = all_pairs_shortest_paths[x_i, y_i, x_k, y_k] + [(x_k, y_k)] + all_pairs_shortest_paths[x_k, y_k, x_j, y_j]

class VirtualAction():
    def __init__(self, agent): # agent's object may contains richer info about state which doesn't concern the MDP
        self.agent = agent

    def is_valid(self, state, obs): 
        pass

    def is_active(self, state, obs): 
        pass

    def next_action(self, state, obs):
        pass


class ChaseNearestEnemy(VirtualAction):
    def __init__(self, agent):
        super(ChaseNearestEnemy, self).__init__(agent)
        self.currently_chasing = None # renew who you're chasing only after 4 steps to prevent oscillating aimlessly
        self.chase_left = 0
        self.chasee_last_pos = None

    def is_valid(self, state, obs): # can't be called again if we're already near an enemy
        board = obs["board"]
        if state.enemy_nearby != Proximity.NONE:
            #print("Invalid because enemy nearby")
            return False
        else:
            coords = self.next_coord(state, obs)
            if not coords:
                #print("Invalid because no next coords")
                return False
            x, y = coords
            if board[x, y] == 0 or board[x, y] == 6 or board[x, y] == 7 or board[x, y] == 8:
                return True
            else:
                #print("Invalid because blocked")
                return False

    def is_active(self, state, obs): # stops when we reach near enemy
        return self.is_valid(state, obs)

    def next_coord(self, state, obs):
        board = obs["board"]
        pos = obs["position"]

        if self.chase_left > 0:
            target_coords = ep.get_agent_new_pos(board, self.currently_chasing, self.chasee_last_pos)
            self.chase_left -= 1

        if self.chase_left == 0 or not target_coords:
            target_coords = ep.scanboard_richer(board, pos, self.agent.agent_value)
            self.currently_chasing = board[target_coords] 
            self.chase_left = 4

        self.chasee_last_pos = target_coords 
        shortest_path = get_shortest_path_between(*pos, *target_coords)
        return shortest_path[0] if shortest_path else False

    def next_action(self, state, obs):
        adj_coords = self.next_coord(state, obs)
        print("Chasing enemy %s " % self.currently_chasing)
        return action_to_reach_adj_coords(obs["position"], adj_coords)
    

def action_to_reach_adj_coords(ours, theirs): # the 4 coords directly adjacent to you
    x1, y1 = ours
    x2, y2 = theirs
    delx = x2 - x1
    dely = y2 - y1
    #print(ours, theirs, delx, dely)
    if dely == 1:
        return Actions.DOWN
    if dely == -1:
        return Actions.UP
    if delx == 1:
        return Actions.RIGHT
    if delx == -1:
        return Actions.LEFT

