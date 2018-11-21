'''An example to show how to set up an pommerman game programmatically'''
from pommerman.agents import BaseAgent
from pommerman import constants
from pommerman import utility

import numpy as np
import pickle
import crazy_util

"""
Observation State Assumptions:
(Observations take precedence with high rank on highest priority)

0 - Adjacent to a bomb
4 - Line of sight of bomb
1 - Adjacent to enemy
2 - Adjacent to a wall
3 - Tiles on all four directions (essentially a fallback, which means a state different than the above three)

2* any of the above number == new state with bomb

Notes to future Self:

- obs['alive'] is the agent number of other alive agents
- obs['teammates'] is a list that either contains the dummy agent if no teammate, else the teammate agent Item
- obs['enemies'], same as above, except for enemies
- obs['ammo'] is the number of bombs

"""

class WarriorAgent(BaseAgent):
    """Customised agent for pommerman"""

    def __init__(self, *args, **kwargs):
        super(WarriorAgent, self).__init__(*args, **kwargs)
        self.eps = 0.15      # randomness factor
        self.epochs = 10000
        self.max_steps = 100
        self.lr_rate = 0.81
        self.gamma = 0.96
        self.prev_state = None
        self.cur_state = None
        self.last_reward = 0
        self.win = 0
        self.model_file = 'qtable_intelli_warrior-v1.pkl'
        try:
            with open(self.model_file, 'rb') as f:
                self.Q = pickle.load(f)
        except Exception as e:
            self.Q = np.zeros((10, 6))

    def get_possible_actions(self, board, pos, ammo, bombs):
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
                if board[newX, newY] in [0, 6, 7, 8] and not self.check_bomb((newX, newY), bombs):
                    valid_acts.append(k+1)
        if ammo > 0:
            valid_acts.append(5)

        valid_acts.append(0)
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
            if ((bomb['life'] <= bomb['blast_strength'] - (abs(newX-bomb['position'][0]) and newY == bomb['position'][1]))
                or (bomb['life'] <= bomb['blast_strength'] - (abs(newY-bomb['position'][1]) and newX == bomb['position'][0]))):
                return True
        return False

    def get_observation_state(self, board, pos, enemies, bomb_map, bomb_life, ammo):
        """
        Need just the board layout to decide everything
        board -> np.array
        pos   -> tuple
        enemies -> list
        """

        bombs = self.convert_bombs(np.array(bomb_map), np.array(bomb_life))

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

                los_bomb = self.check_bomb((newX, newY), bombs)

        if utility.position_is_bomb(bombs, (x,y)):
            has_bomb = True

        state = 3

        if has_bomb:
            state = 0
        elif los_bomb:
            state = 4
        elif has_enemy:
            state = 1
        elif has_wood:
            state = 2
        else:
            state = 3

        if has_ammo:
            state = 2 * state

        return state

    def learn(self, from_state, to_state, reward, action_taken):
        predict = self.Q[from_state, action_taken]
        target = reward + self.gamma * np.max(self.Q[to_state, :])
        self.Q[from_state, action_taken] = self.Q[from_state, action_taken] + self.lr_rate * (target - predict)

    def episode_end(self, reward):
        self.last_reward = reward
        self.learn(self.prev_state, self.cur_state, reward, self.last_action)
        # print(self.Q)
        print('reward for this episode : ', reward)
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.Q, f)


    def act(self, obs, action_space):
        # print(action_space, obs)
        # print(obs['board'])
        state = self.get_observation_state(obs['board'],
                                           obs['position'],
                                           obs['enemies'],
                                           obs['bomb_blast_strength'],
                                           obs['bomb_life'],
                                           obs['ammo'])
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
                                                obs['ammo'],
                                                self.convert_bombs(np.array(obs['bomb_blast_strength']), np.array(obs['bomb_life'])))
            action = actions[0]
            for i in actions:
                if self.Q[state, i] > self.Q[state, action]:
                    action = i
            # print(self.cur_state, actions, action)
            # action = np.argmax(self.Q[state, :])
        self.last_action = action
        # print(obs)
        self.eps -= 1/(obs['step_count']+100)
        if obs['step_count'] % 10 :
            self.last_reward = -obs['step_count']/10000
        else:
            self.last_reward = 0
        return action
