from pommerman.agents import BaseAgent
from pommerman import constants
from pommerman import utility

import collections
# from WarriorAgent import WarriorAgent
import itertools

import math
import numpy as np
import pickle
import crazy_util

"""
Agent's state will be an named tuple with following keys (to be changed later):
  - has_bomb
  - has_enemy
  - has_wood
  - los_bomb
  - has_ammo

Note: don't use nested tuples it for now!
"""
State = collections.namedtuple("State", ["has_bomb", "has_enemy", "has_wood", "los_bomb", "has_ammo"])

component_state_space = State([True, False], [True, False], [True, False], [True, False], [True, False])

# the following has a deterministic ordering
composite_state_space = [State(*record) for record in itertools.product(*component_state_space)]

# since dict lookup is O(1), let's cache it
state_to_index_map = {state: composite_state_space.index(state) for state in composite_state_space}

class CrazyAgent(BaseAgent):

    def __init__(self, *args, **kwargs):
        super(CrazyAgent, self).__init__(*args, **kwargs)
        self.eps_naught = 0.6      # randomness factor
        self.eps = 0.6      
        self.epochs = 10000
        self.max_steps = 100
        self.lr_rate_naught = 0.4
        self.lr_rate = 0.4
        self.gamma = 0.96
        self.prev_state = None
        self.cur_state = None # will be tuples, we convert to index when we have to do lookup
        self.last_reward = 0
        self.win = 0
        self.model_file = 'qtable_crazy-v0.pkl'
        self.unpickle_or_default()

    def update_params(self):
        multiplier = (1 / (1 + math.log(self.experience)))
        self.lr_rate =  multiplier * self.lr_rate_naught
        self.eps = multiplier * self.eps_naught 

    def pickle(self):
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.experience, f)
            pickle.dump(self.Q, f)
    
    def unpickle_or_default(self):
        try:
            with open(self.model_file, 'rb') as f:
                self.experience = pickle.load(f) # number of episodes we've been trained on so far
                self.Q = pickle.load(f)
        except Exception as e:
            self.experience = 0
            self.Q = np.zeros((len(composite_state_space), 6))

    def reward_for_state(self, s): # Assume reward for winning is 30, for losing is -30
        rewards = 0
        if s.los_bomb: # los bomb means we evaded or planted a bomb?
            rewards += 5
        if s.has_wood: # seek wood
            rewards += 1
        if not s.has_ammo: # learn to pick up ammo
            rewards -= 2
        if s.has_enemy: # avoid enemey ever so slightly
            rewards -= 1
        return rewards

    def get_possible_actions(self, board, pos, ammo):
        """
        0 : Pass
        1 : Up
        2 : Down
        3 : Left
        4 : Right
        5 : Bomb
        """
        valid_acts = []
        x, y = pos
        dirX = [-1,1, 0,0]
        dirY = [ 0,0,-1,1]
        for k in range(0, len(dirX)):
            newX = x + dirX[k]
            newY = y + dirY[k]
            # print((newX, newY), board.shape)
            if newX < board.shape[0] and newY < board.shape[1] and newX >=0 and  newY >= 0:
                if board[newX, newY] in [0, 6, 7, 8]:
                    valid_acts.append(k+1)
        if ammo > 0:
            valid_acts.append(5)

        valid_acts.append(0)
        return valid_acts

    def convert_bombs(self, bomb_map):
        '''Flatten outs the bomb array'''
        ret = []
        locations = np.where(bomb_map > 0)
        for r, c in zip(locations[0], locations[1]):
            ret.append(crazy_util.dotdict({
                'position': (r, c),
                'blast_strength': int(bomb_map[(r, c)])
            }))
        return ret

    def get_observation_state(self, board, pos, enemies, bomb_map, ammo):
        """
        Need just the board layout to decide everything
        board -> np.array
        pos   -> tuple
        enemies -> list
        """

        bombs = self.convert_bombs(np.array(bomb_map))

        has_bomb = False
        has_enemy = False
        has_wood = False
        los_bomb = False
        has_ammo = False

        if ammo > 0:
            has_ammo = True

        x, y = pos
        dirX = [-1,1, 0,0]
        dirY = [ 0,0,-1,1]
        for k in range(0, len(dirX)):
            newX = x + dirX[k]
            newY = y + dirY[k]
            # print((newX, newY), board.shape)
            if newX < board.shape[0] and newY < board.shape[1] and newX >=0 and  newY >= 0:
                if utility.position_is_bomb(bombs, (newX, newY)):
                    has_bomb = True
                if utility.position_is_wood(board, (newX, newY)):
                    has_wood = True
                if utility.position_is_enemy(board, pos, enemies):
                    has_enemy = True
                for bomb in bombs:
                    if ((abs(newX-bomb['position'][0]) <= bomb['blast_strength'] and newY == bomb['position'][1])
                        or (abs(newY-bomb['position'][1]) <= bomb['blast_strength'] and newX == bomb['position'][0])):
                        los_bomb = True

        if utility.position_is_bomb(bombs, (x,y)):
            has_bomb = True

        return State(has_bomb, has_enemy, has_wood, los_bomb, has_ammo)

    def learn(self, from_state, to_state, reward, action_taken):
        from_state_id = state_to_index_map[from_state]
        to_state_id = state_to_index_map[to_state]

        predict = self.Q[from_state_id, action_taken]
        target = reward + self.gamma * np.max(self.Q[to_state_id, :])
        self.Q[from_state_id, action_taken] = (1 - self.lr_rate) * predict + self.lr_rate * target 

        self.experience += 1
        self.update_params()

    def episode_end(self, reward):
        self.last_reward = reward * 30
        self.learn(self.prev_state, self.cur_state, reward, self.last_action)
        # print(self.Q)
        print('win status of last episode : ', reward)
        self.pickle()


    def act(self, obs, action_space):
        # print(action_space, obs)
        # print(obs['board'])
        state = self.get_observation_state(obs['board'],
                                           obs['position'],
                                           obs['enemies'],
                                           obs['bomb_blast_strength'],
                                           obs['ammo'])
        state_id = state_to_index_map[state]

        self.cur_state = state

        if self.prev_state != None:
            self.learn(self.prev_state, self.cur_state, self.last_reward, self.last_action)

        self.prev_state = state
        action = 0
        if np.random.uniform(0,1) < self.eps:
            # Random action from the space
            action = action_space.sample()
        else:
            actions = self.get_possible_actions(obs['board'],
                                                obs['position'],
                                                obs['ammo'])
            action = actions[0]
            for i in actions:
                if self.Q[state_id, i] > self.Q[state_id, action]:
                    action = i
            # action = np.argmax(self.Q[state, :])
        self.last_action = action
        # print(obs)
        self.eps -= 1/(obs['step_count']+100)
        self.last_reward = self.reward_for_state(self.cur_state)
        return action
