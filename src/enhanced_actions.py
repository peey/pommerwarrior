import numpy as np
from util import Proximity, BlastStrength, DeadState, Actions
import enhanced_percepts as ep
import BabyAgent
from pommerman import constants

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

class VirtualAction():
    def __init__(self, agent): # agent's object may contains richer info about state which doesn't concern the MDP
        self.agent = agent

    def is_valid(self, state, obs): 
        pass

    def is_active(self, state, obs): 
        pass

    def next_action(self, state, obs):
        pass

class EscapeBoxedIn(VirtualAction):
    def __init__(self, agent): # agent's object may contains richer info about state which doesn't concern the MDP
        self.agent = agent
        self.target = None

    def is_valid(self, state, obs): 
        #print("\t", state.is_surrounded, self.get_target(state, obs))
        return state.is_surrounded and self.get_target(state, obs)

    def is_active(self, state, obs): # once we aren't surrounded, no longer valid
        if self.is_valid(state, obs):
            return True
        else:
            self.target = None
            return False

    def get_target(self, state, obs): # for now, doesn't clear the passage or anything, just looks for a box nearby which isn't surrounded and embarks on a path to it
        pos    = obs["position"]
        x, y   = pos
        board  = obs["board"]
        m, n   = obs["board"].shape
        window = board[max(x - 3, 0):min(x + 4, m), max(y - 3, 0):min(y + 4, n)]

        clear_spots = np.where((board == 0) | ((board >= 6) & (board <= 8)))

        row_coord, col_coord = clear_spots
        distances = np.array([ep.get_shortest_distance_between(row_coord[i], col_coord[i], x, y) for i in range(n)])

        for ix in np.argsort(distances):
            xn, yn = row_coord[ix], col_coord[ix]

            if ep.is_pos_surrounded(board, pos, self.agent.agent_value):
                self.target = (xn, yn)
                break

        if not self.target: # this should be near impossible
            return False
        else:
            shortest_path = ep.get_shortest_path_between(x, y, *self.target)
            if not shortest_path:
                return False
            else:
                return True

    def next_action(self, state, obs):
        print("pos", obs["position"])
        print("target", self.target)
        shortest_path = ep.get_shortest_path_between(*obs["position"], *self.target)
        print("act", action_to_reach_adj_coords(obs["position"], shortest_path[0]))
        print("step",  shortest_path[0])
        return action_to_reach_adj_coords(obs["position"], shortest_path[0])


class ChaseNearestPowerup(VirtualAction):
    def __init__(self, agent): # agent's object may contains richer info about state which doesn't concern the MDP
        self.agent = agent
        self.timeout = 0

    def is_valid(self, state, obs): # if it's across the board, then we probably don't want it. Focus on nearby ones
        board = obs["board"]
        target_coords, next_coords = self.get_target_and_next_coords(state, obs)
        if target_coords:
            x, y = target_coords
            distance = ep.get_shortest_distance_between(*target_coords, *obs["position"])
            if distance > 7: # 7 or less steps to get it
                #print("invalid because large distance", distance)
                return False
            else:
                if board[x, y] == 0 or board[x, y] == 6 or board[x, y] == 7 or board[x, y] == 8:
                    #print('valid')
                    return True
                else:
                    #print("invalid because passage block")
                    return False
        else:
            #print("invalid because no path")
            return False

    def get_target_and_next_coords(self, state, obs):
        board, pos = obs["board"], obs["position"]
        target_coords = ep.scanboard_closest_powerup(board, pos)
        if target_coords:
            shortest_path = ep.get_shortest_path_between(*pos, *target_coords)
            if not shortest_path:
                return False, None
            else:
                return target_coords, shortest_path[0]
        else:
            return False, None

    def is_active(self, state, obs): 
        target_coords, next_coords = self.get_target_and_next_coords(state, obs)
        if target_coords:
            x, y = target_coords
            distance = ep.get_shortest_distance_between(*target_coords, *obs["position"])
            if distance > self.timeout: 
                #print("inactive because large distance", distance)
                self.timeout = 0
                return False # someone took the one we were planning to take. Or we took one earlier than planned. Disengage!
            else:
                if board[x, y] == 0 or board[x, y] == 6 or board[x, y] == 7 or board[x, y] == 8:
                    #print("active")
                    return True
                else:
                    #print("inactive because passage block")
                    self.timeout = 0
                    return False
        else:
            #print("inactive because no path")
            return False

    def next_action(self, state, obs):
        target_coords, next_coords = self.get_target_and_next_coords(state, obs)
        #print(self.is_active(state, obs))
        #print(target_coords, next_coords)
        self.timeout -= 1
        return action_to_reach_adj_coords(obs["position"], next_coords) 


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
            if coords:
                x, y = coords
                if board[x, y] == 0 or board[x, y] == 6 or board[x, y] == 7 or board[x, y] == 8:
                    #print("Valid")
                    return True
                else:
                    #print("Inalid because blocked")
                    return False
            else:
                #print("Inalid because no path")
                return False

    def is_active(self, state, obs): # stops when we reach near enemy
        return self.is_valid(state, obs)

    def next_coord(self, state, obs):
        board = obs["board"]
        pos = obs["position"]

        if self.chase_left > 0:
            target_coords = ep.get_agent_new_pos(board, self.currently_chasing, self.chasee_last_pos)

        if self.chase_left == 0 or not target_coords:
            target_coords = ep.scanboard_richer(board, pos, self.agent.agent_value)
            self.currently_chasing = board[target_coords] 
            self.chase_left = 4

        self.chasee_last_pos = target_coords 
        shortest_path = ep.get_shortest_path_between(*pos, *target_coords)
        return shortest_path[0] if shortest_path else False

    def next_action(self, state, obs):
        adj_coords = self.next_coord(state, obs)
        self.chase_left -= 1
        return action_to_reach_adj_coords(obs["position"], adj_coords)

def action_to_reach_adj_coords(ours, theirs): # the 4 coords directly adjacent to you
    y1, x1 = ours # y, x? yes indeed. That's how the pommer lords chose to do it
    y2, x2 = theirs 
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

