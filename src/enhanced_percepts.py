import numpy as np

all_pairs_shortest_paths = np.full((11, 11, 11, 11), None, dtype=object)
dist = None 

def get_shortest_path_between(x1, y1, x2, y2):
    """
    if  coord_less_than(x1, y1, x2, y2):
        return all_pairs_shortest_paths[x1, y1, x2, y2]
    else:
        return reversed(all_pairs_shortest_paths[x2, y2, x1, y1])
    """
    return all_pairs_shortest_paths[x1, y1, x2, y2]

def get_shortest_distance_between(x1, y1, x2, y2):
    return dist[x1, y1, x2, y2]

def coord_less_than(x1, y1, x2, y2): # left to right then top to bottom. Don't know if this is faster or coord to id would be faster.
    return y1 < y2 or (y1 == y2 and x1 < x2)

def floyd_warshall(board): # takes ((11^2)^3)/2 steps once in the begining of the game
    global dist
    # citation: referenced wikipedia for pseudocode of algo, implementation is mine
    m, n = board.shape
    dist = np.full((m, n, m, n), 50, dtype=np.int8) # int8 since max manhattan distance won't exced 22

    for (x, y), value in np.ndenumerate(board):
        if x < m - 1 and board[x + 1, y] != 1:
            dist[x, y, x + 1, y] = 1
            dist[x + 1, y, x, y] = 1
            all_pairs_shortest_paths[x, y, x + 1, y] = []
            all_pairs_shortest_paths[x + 1, y, x, y] = []
        if y < n - 1 and board[x, y + 1] != 1:
            dist[x, y, x, y + 1] = 1
            dist[x, y + 1, x, y] = 1
            all_pairs_shortest_paths[x, y, x, y + 1] = []
            all_pairs_shortest_paths[x, y + 1, x, y] = []

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

def directions_from_coordinates(a, b):
    x1, y1 = a
    x2, y2 = b
    delx = x1 - x2
    dely = y1 - y2
    if delx > 0: # west (left)
        if dely > 0: # north (top)
            if delx > 1:
                if dely > 1:
                    return "nw"
                else:
                    return "west"
            else:
                return "north"
        else:
            if delx > 1:
                if dely > 1:
                    return "sw"
                else:
                    return "west"
            else:
                return "south"

    else: # east or flat
        if dely > 0: # north (top)
            if delx > 1:
                if dely > 1:
                    return "ne"
                else:
                    return "east"
            else:
                return "north"
        else:
            if delx > 1:
                if dely > 1:
                    return "ne"
                else:
                    return "east"
            else:
                return "south"

def manhattan_distance(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    return abs(x1 - x2) + abs(y1 - y2)

def closest_point(arr, pt): # arr is array of row and columns
    rows, cols = arr
    n = len(rows)
    distances = np.array([manhattan_distance((rows[i], cols[i]), pt) for i in range(n)])
    closest = np.argmin(distances)
    return (rows[closest], cols[closest])
    
def real_closest_point(arr, pt): # uses real distances 
    rows, cols = arr
    n = len(rows)
    distances = np.array([get_shortest_distance_between(rows[i], cols[i], *pt) for i in range(n)])
    closest = np.argmin(distances)
    return (rows[closest], cols[closest])

def scanboard(board, pos, teammate): # for now, just gives directions for nearest enemy and powerup
    enemies = np.where(board > 9)
    powerups = np.where((9 > board) & (board > 5) & (board != teammate))

    if enemies[0].size != 0:
        closest_enemy = directions_from_coordinates(pos, closest_point(enemies, pos))
    else:
        closest_enemy = "nowhere"

    if powerups[0].size != 0:
        closest_powerup = directions_from_coordinates(pos, closest_point(powerups, pos))
    else:
        closest_powerup = "nowhere"

    return closest_enemy , closest_powerup

def scanboard_richer(board, pos, self_id): 
    enemies = np.where((board > 9) & (board != self_id))
    closest_enemy_coords = real_closest_point(enemies, pos) # as long as we're alive, at least one will exist
    return closest_enemy_coords

def scanboard_closest_powerup(board, pos): 
    powerups = np.where((board >= 6) & (board <= 8)) # 6, 7, 8

    if powerups[0].size == 0:
        return None

    return real_closest_point(powerups, pos) 

def get_agent_new_pos(board, enemy_id, last_pos): #O(5), as if anyone cares though
    m, n = board.shape
    x, y = last_pos
    for xn, yn in [(x, y), (x + 1, y), (x - 1, y), (x, y -1), (x, y + 1)]:
        if 0 <= xn < m and 0 <= yn < n and board[xn, yn] == enemy_id:
            return (xn, yn)
    # else enemy is dead

"""
def get_agent_new_pos(board, enemy_id):
    enemy_coords = np.where((board == enemy_id))
    xs, ys = enemy_coords
    if len(xs) < 1:
        return None
    else:
        return xs[0], ys[0]
"""
