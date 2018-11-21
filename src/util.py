from WarriorAgent import WarriorAgent
from CrazyAgent import CrazyAgent 
from HybridAgent import HybridAgent 

def get_agent(name):
    if name == 'WarriorAgent':
        return WarriorAgent
    if name == 'HybridAgent':
        return HybridAgent
    if name == 'CrazyAgent':
        return CrazyAgent

    raise Exception("Specify a valid agent name. %s doesn't exist" % name)
