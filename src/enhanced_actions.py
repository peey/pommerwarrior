import numpy as np

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

all_pairs_shortest_paths = [[[[None, None, None, 
                               None, None, None, 
                               None, None, None,
                               None, None] for i in range(11)] for j in range(11)] for k in range(11)]

def get_shortest_path_between(x1, y1, x2, y2):
    if  coord_less_than(x1, y1, x2, y2):
        return all_pairs_shortest_paths[x1, y1, x2, y2]
    else:
        return reversed(all_pairs_shortest_paths[x2, y2, x1, y1])

def coord_less_than(x1, y1, x2, y2): # left to right then top to bottom. Don't know if this is faster or coord to id would be faster.
    return y1 < y2 or (y1 == y2 and x1 < x2)

def floyd_warshall(board): # takes ((11^2)^3)/2 steps once in the begining of the game
    # citation: referenced wikipedia for pseudocode of algo, implementation is mine
    m, n = board.shape
    dist = np.full((m, n, m, n), np.int8.max, dtype=np.int8) # int8 since max manhattan distance won't exced 22

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
                if coord_less_than(x_i, y_i, x_j, y_j):
                    break # because undirected graph
                new_distance = dist[x_i, y_i, x_k, y_k] + dist[x_k, y_k, x_j, y_j]
                if new_distance > dist[x_i, y_i, x_j, y_j]:
                    dist[x_i, y_i, x_j, y_j] = new_distance
                    all_pairs_shortest_paths[x_i, y_i, x_j, y_j] = all_pairs_shortest_paths[x_i, y_i, x_k, y_k] + all_pairs_shortest_paths[x_k, y_k, x_j, y_j]

