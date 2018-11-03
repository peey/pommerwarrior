'''An example to show how to set up an pommerman game programmatically'''
from pommerman.agents import BaseAgent
from pommerman import constants
from pommerman import utility

import numpy as np
import pickle
import crazy_util


class MultiTypeTeamAgent(BaseAgent):
    """Customised agent for pommerman"""
    # Two classes - Agressor
    # Survivor
    eps = 0.1      # randomness factor
    epochs = 10000
    max_steps = 100
    lr_rate = 0.81
    gamma = 0.96
    prev_state = None
    cur_state = None
    last_reward = 0
    win = 0
    cur_class = 0

    def __init__(self, *args, **kwargs):
        super(MultiTypeTeamAgent, self).__init__(*args, **kwargs)
        self.model_file = 'class_qtable-v0.pkl'
        try:
            with open(self.model_file, 'rb') as f:
                self.Q = pickle.load(f)
        except Exception as e:
            self.Q = [np.zeros((5, 6)), np.zeros((5,6))]
        if MultiTypeTeamAgent.cur_class == 0:
            self.type = 0
            MultiTypeTeamAgent.cur_class += 1
        elif MultiTypeTeamAgent.cur_class == 1:
            self.type = 1
            MultiTypeTeamAgent.cur_class += 1
        else:
            print('Cannot have more than two instances !')
            exit(0)


    def get_observation_state(self, board, pos, enemies, bomb_map):
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

        if has_bomb:
            return 0
        elif los_bomb:
            return 4
        elif has_enemy:
            return 1
        elif has_wood:
            return 2
        else:
            return 3

    def learn(self, from_state, to_state, reward, action_taken, Q_val):
        predict = Q_val[from_state, action_taken]
        target = reward + self.gamma * np.max(Q_val[to_state, :])
        Q_val[from_state, action_taken] = Q_val[from_state, action_taken] + self.lr_rate * (target - predict)

    def episode_end(self, reward):
        MultiTypeTeamAgent.last_reward = reward
        self.last_reward = reward
        self.learn(self.prev_state, self.cur_state, reward, self.last_action, self.Q[self.type])
        # print(self.Q[self.type])
        print('reward for this episode : ', reward, 'Agent Type ', self.type)
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.Q, f)


    def act(self, obs, action_space):
        # print(action_space, obs)
        # print(obs['board'])
        state = self.get_observation_state(obs['board'],
                                           obs['position'],
                                           obs['enemies'],
                                           obs['bomb_blast_strength'])
        self.cur_state = state
        if self.prev_state != None:
            for warrior_type in [0,1]:
                self.learn(self.prev_state, self.cur_state, self.last_reward, self.last_action, self.Q[warrior_type])
        self.prev_state = state
        action = 0
        if np.random.uniform(0,1) < self.eps:
            # Random action from the space
            action = action_space.sample()
        else:
            action = np.argmax(self.Q[self.type][state, :])
        self.last_action = action
        # print(obs)
        if obs['step_count'] % 10 :
            if self.type == 0:
                # is a warrior. Needs to be punished for not finishing opponents
                self.last_reward = -obs['step_count']/10000
            else:
                # is a survivor, needs to learn to survive
                self.last_reward = 1/obs['step_count']
        else:
            self.last_reward = 0
        return action
