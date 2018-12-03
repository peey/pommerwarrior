from pommerman.agents import BaseAgent
from pommerman import constants
from pommerman import utility

import collections
import itertools
import random
from util import Proximity, BlastStrength, DeadState, Actions

import math
import numpy as np
import pickle
import crazy_util

import enhanced_percepts as ep
import enhanced_actions  as ea
import os

AliveState = collections.namedtuple("AliveState", ["bomb_nearby", "enemy_nearby", 
                                                   "is_surrounded", "los_bomb", "ammo", 
                                                   "can_kick", "blast_strength", "enemies_alive", 
                                                   "nearby_enemy_has_bomb", "nearby_enemy_can_kick"])

alive_component_state_space = AliveState([Proximity.NONE, Proximity.IMMEDIATE, Proximity.CLOSE], [Proximity.NONE, Proximity.IMMEDIATE, Proximity.CLOSE],
                                         [True, False], [True, False], [0, 1, 2, 3], 
                                         [True, False], [BlastStrength.LOW, BlastStrength.HIGH], [1, 2, 3], 
                                         [True, False], [True, False]) # we don't have to worry about nearby enemy's blast strength because "loa_bomb" calculation will take care of it


# events for reward - killed enemy, picked ammo, picked something. In these cases we should look back and give reward to sequence of states leading up to this.

# the following has a deterministic ordering
composite_state_space = [AliveState(*record) for record in itertools.product(*alive_component_state_space)] \
                        + [DeadState.SUICIDE, DeadState.KILLED_BY_ENEMY, DeadState.WON]

print(len(composite_state_space)) # about 20000 states

# since dict lookup is O(1), let's cache it
state_to_index_map = {state: composite_state_space.index(state) for state in composite_state_space}


class BabyAgent(BaseAgent):

    def __init__(self, *args, **kwargs):
        super(BabyAgent, self).__init__(*args, **kwargs)
        self.model_file = 'qtable_baby-v0.pkl'

        self.eps = 0.01

        self.lr_rate_naught = 0.8
        self.gamma = 0.96

        self.prev_state = None # will be tuples, we convert to index when we have to do lookup
        self.cur_state = None 

        self.accu_va_reward = 0
        self.accu_va_steps = 0
        self.virtual_action = None
        self.agent_value = None
        self.virtual_actions = {
          Actions.CHASE_NEAREST_ENEMY   : ea.ChaseNearestEnemy(self),
          Actions.CHASE_NEAREST_POWERUP : ea.ChaseNearestPowerup(self),
          Actions.ESCAPE_BOXED_IN       : ea.EscapeBoxedIn(self)
        }

        self.unpickle_or_default()

    def pickle(self):
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.experience, f)
            pickle.dump(self.Q, f)
            pickle.dump(self.N, f)
    
    def unpickle_or_default(self):
        try:
            with open(self.model_file, 'rb') as f:
                self.experience = pickle.load(f) # number of episodes we've been trained on so far
                self.Q = pickle.load(f)
                self.N = pickle.load(f)
        except Exception as e:
            self.experience = 0
            self.Q = np.zeros((len(composite_state_space), len(Actions)))
            self.N = np.zeros((len(composite_state_space), len(Actions)))

    # citation: exploration function taken from my own (2016254) submission to PA4
    def exploration_function(self, state_ix, action_ix):
        u = self.Q[state_ix, action_ix]
        n = self.N[state_ix, action_ix]

        #print(u, n)
        if (n < 100):
            return 0.8 + (np.random.uniform()/5)
        else:
            return u

    def reward_for_state(self, s): # Assume reward for winning is 1, for losing is -1
        rewards = 0

        if s.los_bomb: # los bomb means we will be killed by a bomb
            rewards -= 0.05
        if s.is_surrounded: # prevent from getting surrounded
            rewards -= 0.04
        if s.ammo == 0: # learn to pick up ammo when you have none. Not too negative because we don't wanna punish bomb planting
            rewards -= 0.01
        if not s.can_kick:  # learn to pick up can kick
            rewards -= 0.01
        if s.blast_strength != BlastStrength.HIGH:
            rewards -= 0.01

        return rewards

    def get_possible_actions(self, obs, board, pos, ammo, can_kick, bombs):
        valid_acts = [0]
        x, y = pos
        dirX = [-1, 1,  0, 0]
        dirY = [ 0, 0, -1, 1]
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
                    #print('contributed to suicide !')
                elif board[newX, newY] in [0, 6, 7, 8] and utility.position_is_bomb(bombs, (x,y)):
                    #print('contributed to death !!!')
                    valid_acts.append(k+1)
                #print('Appending ', k+1, newX, newY, cbom)
        if ammo > 0:
            valid_acts.append(5)

        if len(valid_acts) > 1 and utility.position_is_bomb(bombs, (pos[0], pos[1])) and self.check_bomb((pos[0], pos[1]), bombs):
            valid_acts.pop(0)

        for i in  range(6, len(Actions)):
            if self.virtual_actions[i].is_valid(self.cur_state, obs):
                #print("appending act ", i)
                valid_acts.append(i)

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

    def get_observation_state(self, obs):
        """
        Need just the board layout to decide everything
        board -> np.array
        pos   -> tuple
        enemies -> list

        Interesting keys: 
           obs['board'],
           obs['position'],
           obs['teammate'],
           obs['enemies'],
           obs['bomb_blast_strength'],
           obs['bomb_life'],
           obs['ammo'],
           obs['can_kick']
        """

        board = obs["board"]

        bombs = self.convert_bombs(np.array(obs["bomb_blast_strength"]), np.array(obs["bomb_life"]))

        d = collections.OrderedDict({
            "bomb_nearby": Proximity.NONE,
            "enemy_nearby": Proximity.NONE,
            "is_surrounded": False,
            "los_bomb": False,
            "ammo": 3 if obs['ammo'] > 3 else obs['ammo'],
            "can_kick": obs['can_kick'],
            "blast_strength": BlastStrength.LOW if obs['blast_strength'] <= 2 else BlastStrength.HIGH ,
            "enemies_alive": len(list(filter(lambda enemy: enemy.value in obs['alive'], obs['enemies']))),
            "nearby_enemy_has_bomb": False,
            "nearby_enemy_can_kick": False
        })


        x, y = obs['position']

        for del_x in range(-2, 3):
            for del_y in range(-2, 3):
                newX = x + del_x
                newY = y + del_y

                immediate_zone = abs(del_x) <= 1 and abs(del_y) <= 1

                if newX < board.shape[0] and newY < board.shape[1] and newX >=0 and  newY >= 0:
                    if utility.position_is_bomb(bombs, (newX, newY)):
                        d['bomb_nearby'] = Proximity.IMMEDIATE if immediate_zone else Proximity.CLOSE
                    if utility.position_is_enemy(obs['board'], obs['position'], obs['enemies']):
                        d['has_enemy'] = Proximity.IMMEDIATE if immediate_zone else Proximity.CLOSE

                    d['los_bomb'] = self.check_bomb((newX, newY), bombs) or d['los_bomb']

        if utility.position_is_bomb(bombs, (x,y)) or self.check_bomb((x,y), bombs): # TODO why two conditions?
            d["bomb_nearby"] = Proximity.IMMEDIATE

        d["is_surrounded"] = ep.is_pos_surrounded(obs["board"], obs["position"], self.agent_value)
        #print(d["is_surrounded"])

        return AliveState(**d)


    def learn(self, from_state, to_state, reward, action_taken):
        from_state_id = state_to_index_map[from_state]
        to_state_id = state_to_index_map[to_state]
        self.N[from_state_id, action_taken] += 1

        predict = self.Q[from_state_id, action_taken]
        target = reward + self.gamma * np.max(self.Q[to_state_id, :])

        n = min(self.experience, self.N[from_state_id, action_taken]/20)
        lr_rate = self.lr_rate_naught / (1 + math.log(1 + n))
        #print(lr_rate, self.N[from_state_id, action_taken])
        self.Q[from_state_id, action_taken] = (1 - lr_rate) * predict + lr_rate * target 


    def episode_end(self, reward):
        self.learn(self.prev_state, self.cur_state, reward, self.last_action)
        #print('win status of last episode : ', reward)
        self.experience += 1
        self.pickle()


    def act(self, obs, action_space):
        state = self.get_observation_state(obs)
        state_id = state_to_index_map[state]

        self.cur_state = state
        x, y = obs['position']

        if self.virtual_action != None: # virtual action is continuing
            self.accu_va_steps  += 1 
            if self.virtual_action.is_active(self.cur_state, obs):
                self.accu_va_reward += self.reward_for_state(self.cur_state)
                return self.virtual_action.next_action(self.cur_state, obs)
            else:
                #print("completed virtual action %d for %d steps" % (self.last_action, self.accu_va_steps))
                reward = self.accu_va_reward / self.accu_va_steps # rationale: this action is just a "faster" way to get to a desired state. If we let rewards accumulate, relative to other actions, it won't work (e.g. for rewards that are ctsly given like -ve for can't kick....)
                #print("concluded virtual action %s with reward %f for %d steps" % (self.last_action, reward, self.accu_va_steps))
                self.virtual_action = None # reset va variables and continue
                self.accu_va_reward = 0
                self.accu_va_steps  = 0
        else:
            reward = self.reward_for_state(self.cur_state)

        if self.prev_state == None: # can initialize stuff
            ep.floyd_warshall(np.array(obs['board']))

        if self.prev_state != None:
            self.learn(self.prev_state, self.cur_state, reward, self.last_action)
            self.agent_value = obs['board'][x, y]


        self.prev_state = state
        action = 0
        valid_actions = self.get_possible_actions(obs, obs['board'],
                                            obs['position'],
                                            obs['ammo'],
                                            obs['can_kick'],
                                            self.convert_bombs(np.array(obs['bomb_blast_strength']), np.array(obs['bomb_life'])))

        if np.random.uniform(0,1) < self.eps:
            action = random.choice(valid_actions)
        else:
            action_ix = np.argmax([self.exploration_function(state_id, action) for action in valid_actions])
            action = valid_actions[action_ix]

        self.last_action = action
        self.last_reward = self.reward_for_state(self.cur_state)

        if action >= 6: # it's a virtual action
            #print("picked virtual action", action, action in valid_actions, valid_actions)
            self.virtual_action = self.virtual_actions[action]
            return self.virtual_action.next_action(self.cur_state, obs)

        return action
