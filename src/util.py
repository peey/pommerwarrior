from WarriorAgent import WarriorAgent
from CrazyAgent import CrazyAgent 
from HybridAgent import HybridAgent 
from DiscoAgent import DiscoAgent 
from enum import Enum, IntEnum

def get_agent(name):
    if name == 'WarriorAgent':
        return WarriorAgent
    if name == 'HybridAgent':
        return HybridAgent
    if name == 'CrazyAgent':
        return CrazyAgent
    if name == 'DiscoAgent':
        return DiscoAgent

    raise Exception("Specify a valid agent name. %s doesn't exist" % name)


class BlastStrength(Enum):
    LOW  = 1 # 1 or 2
    HIGH = 2 # more than 2

class Proximity(Enum):
    NONE      = 1
    IMMEDIATE = 2 # in the 8 square surrounding agent
    CLOSE     = 3


class DeadState(Enum):
    SUICIDE = 1
    KILLED_BY_ENEMY = 2
    WON = 3

class Actions(IntEnum):
    PASS = 0
    UP = 1
    DOWN = 2
    LEFT = 3 
    RIGHT = 4
    BOMB = 5
    CHASE_NEAREST_ENEMY   = 6 # virtual actions start at index 6
    CHASE_NEAREST_POWERUP = 7 
    ESCAPE_BOXED_IN       = 8 


