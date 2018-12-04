from BabyAgent import BabyAgent
from util import BlastStrength

class Fighter(BabyAgent):
    def __init__(self, *args, **kwargs):
        super(Fighter, self).__init__(*args, **kwargs)
        self.model_file = 'qtable_fighter-v0.pkl'

    def reward_for_state(self, s): # Assume reward for winning is 1, for losing is -1
        rewards = 0

        if s.los_bomb: # los bomb means we will be killed by a bomb
            rewards -= 0.05
        if s.is_surrounded: # prevent from getting surrounded
            rewards -= 0.08
        if s.ammo == 0: # learn to pick up ammo when you have none. Not too negative because we don't wanna punish bomb planting
            rewards -= 0.09
        if not s.can_kick:  # learn to pick up can kick
            rewards -= 0.01
        if s.blast_strength != BlastStrength.HIGH:
            rewards -= 0.01
        if s.enemy_nearby:
            rewards += 0.005

        return rewards

class Coward(BabyAgent):
    def __init__(self, *args, **kwargs):
        super(Coward, self).__init__(*args, **kwargs)
        self.model_file = 'qtable_coward-v0.pkl'

    def reward_for_state(self, s): # Assume reward for winning is 1, for losing is -1
        rewards = 0

        if not s.los_bomb: # los bomb means we will be killed by a bomb
            rewards += 0.001
        if s.is_surrounded: # prevent from getting surrounded
            rewards -= 0.05
        if s.enemy_nearby:
            rewards -= 0.05
        if s.nearby_enemy_has_bomb:
            rewards -= 0.08

        return rewards

class Chaser(BabyAgent):
    def __init__(self, *args, **kwargs):
        super(Chaser, self).__init__(*args, **kwargs)
        self.model_file = 'qtable_chaser-v0.pkl'

    def reward_for_state(self, s): # Assume reward for winning is 1, for losing is -1
        rewards = 0

        if s.los_bomb: # los bomb means we will be killed by a bomb
            rewards -= 0.1
        if s.is_surrounded: # prevent from getting surrounded
            rewards -= 0.1
        if s.ammo == 0: # learn to pick up ammo when you have none. Not too negative because we don't wanna punish bomb planting
            rewards -= 0.01
        if s.can_kick:  # learn to pick up can kick
            rewards += 0.001
        if s.blast_strength != BlastStrength.HIGH:
            rewards -= 0.05
        if s.enemy_nearby:
            rewards += 0.1

        return rewards
