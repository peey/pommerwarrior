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

all_pairs_shortest_paths = None 

def get_next_in_shortest_path(x1, y1, x2, y2):
    if not coord_less_than(x1, y1, x2, y2):
        x1, y1, x2, y2 = x2, y2, x1, y1
    return all_pairs_shortest_paths[x1, y1, x2, y2]

def coord_less_than(x1, y1, x2, y2): # left to right then top to bottom. Don't know if this is faster or coord to id would be faster.
    return y1 < y2 || (y1 == y2 && x1 < x2)

def floyd_warshall(board): # takes (11^2)^3 steps once in the begining of the game. Hopefully less since we have a directed graph
    # citation: referenced wikipedia for pseudocode of algo, implementation is mine
    global all_pairs_shortest_paths

    m, n = board.shape
    dist = np.full((m, n, m, n), np.int16.max, dtype=np.int16)
    all_pairs_shortest_paths = np.zeroes((m, n, m, n, m, n), dtype=(np.int16, 2)) # i, j: k. i -> k, k -> j

    for (x, y), value in np.ndenumerate(board):
        if x < m - 1 and board[x + 1, y] != 1:
            dist[x, y, x + 1, y] = 1
            all_pairs_shortest_paths[x, y, x + 1, y] = []
        if y < n - 1 and board[x, y + 1] != 1:
            dist[x, y, x, y + 1] = 1
            all_pairs_shortest_paths[x, y, x, y + 1] = []

    for (x_k, y_k), _ in np.ndenumerate(board)::
        for (x_i, y_i), _ in np.ndenumerate(board):
            for (x_j, y_j), _ in np.ndenumerate(board):
                if coord_less_than(x_i, y_i, x_j, y_j):
                    break # because undirected graph
                new_distance = dist[x_i, y_i, x_k, y_k] + dist[x_k, y_k, x_j, y_j]
                if new_distance > dist[x_i, y_i, x_j, y_j]:
                    dist[x_i, y_i, x_j, y_j] = new_distance
                    all_pairs_shortest_paths[x_i, y_i, x_j, y_j] = (x_k, y_k)
