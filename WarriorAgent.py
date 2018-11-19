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
        self.eps = 0.1      # randomness factor
        self.epochs = 10000
        self.max_steps = 100
        self.lr_rate = 0.81
        self.gamma = 0.96
        self.prev_state = None
        self.cur_state = None
        self.last_reward = 0
        self.win = 0
        self.model_file = 'qtable_more_states-v0.pkl'
        try:
            with open(self.model_file, 'rb') as f:
                self.Q = pickle.load(f)
        except Exception as e:
            self.Q = np.zeros((10, 6))

    def get_observation_state(self, board, pos, enemies, bomb_map, ammo):
        """
        Need just the board layout to decide everything
        board -> np.array
        pos   -> tuple
        enemies -> list
        """
        def convert_bombs(bomb_map):
            '''Flatten outs the bomb array'''
            ret = []
            locations = np.where(bomb_map > 0)
            for r, c in zip(locations[0], locations[1]):
                ret.append(crazy_util.dotdict({
                    'position': (r, c),
                    'blast_strength': int(bomb_map[(r, c)])
                }))
            return ret

        bombs = convert_bombs(np.array(bomb_map))

        has_bomb = False
        has_enemy = False
        has_wood = False
        los_bomb = False
        has_ammo = False

        if ammo > 0:
            has_ammo = True

        x, y = pos
        dirX = [-1,0,1,0]
        dirY = [0,1,0,-1]
        for k1 in dirX:
            for k2 in dirY:
                newX = x+k1
                newY = y+k2
                # print((newX, newY), board.shape)
                if newX < board.shape[0] and newY < board.shape[1] and newX >=0 and  newY >= 0:
                    if utility.position_is_bomb(bombs, (newX, newY)):
                        has_bomb = True
                    if utility.position_is_wood(board, (newX, newY)):
                        has_wood = True
                    if utility.position_is_enemy(board, pos, enemies):
                        has_enemy = True

        for k1 in range(0, board.shape[0]):
            if utility.position_is_bomb(bombs, (k1, y)):
                los_bomb = True
            elif utility.position_is_bomb(bombs, (x, k1)):
                los_bomb = True

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
            action = np.argmax(self.Q[state, :])
        self.last_action = action
        # print(obs)
        if obs['step_count'] % 10 :
            self.last_reward = obs['step_count']/1000
        else:
            self.last_reward = 0
        return action
