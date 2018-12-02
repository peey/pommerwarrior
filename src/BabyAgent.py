from pommerman.agents import BaseAgent
from pommerman import constants
from pommerman import utility

import collections
import itertools
from enum import Enum

import math
import numpy as np
import pickle
import crazy_util

import enhanced_percepts as ep
import enhanced_actions  as ea

AliveState = collections.namedtuple("AliveState", ["bomb_nearby", "enemy_nearby", 
                                                   "is_surrounded", "los_bomb", "ammo", 
                                                   "can_kick", "blast_strength", "enemies_alive", 
                                                   "nearby_enemy_has_bomb", "nearby_enemy_can_kick"])
class LosBomb(Enum):
    NO     = 1 # no bomb
    RED    = 2 # 2 or fewer ticks remiaining
    # ORANGE = 3 # 6 or fewer ticks remaining
    YELLOW = 4 # 6-10 ticks remaining

class BlastStrength(Enum):
    LOW  = 1 # 1 or 2
    HIGH = 2 # more than 2

class Proximity(Enum):
    NONE      = 1
    IMMIDIATE = 2 # in the 8 square surrounding agent
    CLOSE     = 3

alive_component_state_space = AliveState([Proximity.NONE, Proximity.IMMIDIATE, Proximity.CLOSE], [Proximity.NONE, Proximity.IMMIDIATE, Proximity.CLOSE],
                                         [True, False], [LosBomb.NO, LosBomb.RED, LosBomb.YELLOW], [0, 1, 2, 3], 
                                         [True, False], [BlastStrength.LOW, BlastStrength.HIGH], [1, 2, 3], 
                                         [True, False], [True, False]) # we don't have to worry about nearby enemy's blast strength because "loa_bomb" calculation will take care of it

class DeadState(Enum):
    SUICIDE = 1
    KILLED_BY_ENEMY = 2
    WON = 3

# events for reward - killed enemy, picked ammo, picked something. In these cases we should look back and give reward to sequence of states leading up to this.

# the following has a deterministic ordering
composite_state_space = [AliveState(*record) for record in itertools.product(*alive_component_state_space)] \
                        + [DeadState.SUICIDE, DeadState.KILLED_BY_ENEMY, DeadState.WON]

print(len(composite_state_space)) # about 20000 states

# since dict lookup is O(1), let's cache it
state_to_index_map = {state: composite_state_space.index(state) for state in composite_state_space}

class BabyAgent(BaseAgent):

    def __init__(self, *args, **kwargs):
        super(DiscoAgent, self).__init__(*args, **kwargs)
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
        self.model_file = 'qtable_disco-v0.pkl'
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
        if s.los_bomb: # los bomb means we will be killed by a bomb
            rewards -= 5
        if s.is_surrounded: # prevent from getting surrounded
            rewards += 3
        if not s.has_ammo: # learn to pick up ammo
            rewards -= 2
        if not s.can_kick:  # learn to pick up can kick
            rewards -= 1
        # if s.has_enemy: # avoid enemey ever so slightly
        #     rewards -= 1
        return rewards

    def get_possible_actions(self, board, pos, ammo, can_kick, bombs):
        """
        0 : Pass
        1 : Up
        2 : Down
        3 : Left
        4 : Right
        5 : Bomb
        """
        valid_acts = [0]
        x, y = pos
        dirX = [-1,1, 0,0]
        dirY = [ 0,0,-1,1]
        print(bombs)
        for k in range(0, len(dirX)):
            newX = x + dirX[k]
            newY = y + dirY[k]
            # print((newX, newY), board.shape)
            if newX < board.shape[0] and newY < board.shape[1] and newX >=0 and  newY >= 0:
                cbom = self.check_bomb((newX, newY), bombs)
                if ((board[newX, newY] in [0, 5, 6, 7, 8]) and (not cbom)):
                    valid_acts.append(k+1)
                elif board[newX, newY] in [3] and can_kick:
                    valid_acts.append(k+1)
                    print('contributed to suicide !')
                elif board[newX, newY] in [0, 6, 7, 8] and utility.position_is_bomb(bombs, (x,y)):
                    print('contributed to death !!!')
                    valid_acts.append(k+1)
                print('Appending ', k+1, newX, newY, cbom)
        if ammo > 0:
            valid_acts.append(5)

        if len(valid_acts) > 1 and utility.position_is_bomb(bombs, (pos[0], pos[1])) and self.check_bomb((pos[0], pos[1]), bombs):
            valid_acts.pop(0)

        return valid_acts

    def convert_bombs(self, bomb_map, bomb_life):
        '''Flatten outs the bomb array'''
        ret = []
        locations = np.where(bomb_map > 0)
        for r, c in zip(locations[0], locations[1]):
            ret.append(crazy_util.dotdict({
                'position': (r, c),
                'blast_strength': int(bomb_map[(r, c)]),
                'life': int(bomb_life[(r,c)])
            }))
        return ret

    def check_bomb(self, pos, bombs):
        (newX, newY) = pos
        for bomb in bombs:
            if (((bomb['blast_strength'] - bomb['life'] +1 >= abs(newX-bomb['position'][0])) and newY == bomb['position'][1])
                or ((bomb['blast_strength'] - bomb['life'] +1 >= abs(newY-bomb['position'][1])) and newX == bomb['position'][0])):
                # print(bomb, pos)
                return True
        return False

    def get_observation_state(self, board, pos, teammate, enemies, bomb_map, bomb_life, ammo, can_kick):
        """
        Need just the board layout to decide everything
        board -> np.array
        pos   -> tuple
        enemies -> list

        """

        bombs = self.convert_bombs(np.array(bomb_map), np.array(bomb_life))

        has_bomb = False
        has_enemy = False
        # is_surrounded = False
        is_surrounded = False
        los_bomb = False
        has_ammo = False
        # can kick is also a valid state

        if ammo > 0:
            has_ammo = True

        x, y = pos
        dirX = [-1,1, 0,0]
        dirY = [ 0,0,-1,1]
        blocks = 0
        for k in range(0, len(dirX)):
            newX = x + dirX[k]
            newY = y + dirY[k]
            # print((newX, newY), board.shape)
            if newX < board.shape[0] and newY < board.shape[1] and newX >=0 and  newY >= 0:
                if utility.position_is_bomb(bombs, (newX, newY)):
                    has_bomb = True
                if utility.position_is_rigid(board, (newX, newY)):
                    # is_surrounded = True
                    blocks += 1
                if utility.position_is_enemy(board, pos, enemies):
                    has_enemy = True

                los_bomb = self.check_bomb((newX, newY), bombs) or los_bomb

        if utility.position_is_bomb(bombs, (x,y)) or self.check_bomb((x,y), bombs):
            has_bomb = True

        if blocks > 2:
            is_surrounded = True

        enemy_direction, powerup_direction = ep.scanboard(board, pos, teammate.value) 

        return State(has_bomb, has_enemy, is_surrounded, los_bomb, has_ammo, can_kick, enemy_direction, powerup_direction)


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
        print(obs)
        state = self.get_observation_state(obs['board'],
                                           obs['position'],
                                           obs['teammate'],
                                           obs['enemies'],
                                           obs['bomb_blast_strength'],
                                           obs['bomb_life'],
                                           obs['ammo'],
                                           obs['can_kick'])
        state_id = state_to_index_map[state]

        self.cur_state = state

        if self.prev_state != None:
            self.learn(self.prev_state, self.cur_state, self.last_reward, self.last_action)

        self.prev_state = state
        action = 0
        actions = self.get_possible_actions(obs['board'],
                                            obs['position'],
                                            obs['ammo'],
                                            obs['can_kick'],
                                            self.convert_bombs(np.array(obs['bomb_blast_strength']), np.array(obs['bomb_life'])))
        if np.random.uniform(0,1) < self.eps:
            # Random action from the space
            action = np.random.choice(actions)
            # print('random')
        else:
            action = actions[0]
            indices = []
            for i in actions:
                if self.Q[state_id, i] > self.Q[state_id, action]:
                    action = i
            for act in actions:
                if self.Q[state_id, act] == self.Q[state_id, action]:
                    indices.append(act)
            action = np.random.choice(indices)
        self.last_action = action
        # print(obs)
        self.eps -= 1/(obs['step_count']+100)
        self.last_reward = self.reward_for_state(self.cur_state)

        print(obs['board'])
        print(actions, action, obs['can_kick'], self.cur_state)

        return np.asscalar(action)
