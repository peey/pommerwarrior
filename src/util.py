from WarriorAgent import WarriorAgent
from CrazyAgent import CrazyAgent 
from HybridAgent import HybridAgent 
from DiscoAgent import DiscoAgent 

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
