'''An example to show how to set up an pommerman game programmatically'''
from pommerman.agents import BaseAgent
import numpy as np

class WarriorAgent(BaseAgent):
    """Customised agent for pommerman"""

    def act(self, obs, action_space):
        print(action_space, obs)
        return action_space.sample()
