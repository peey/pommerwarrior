"""Implementation of a simple deterministic agent using Docker."""

from pommerman import agents
from pommerman.runner import DockerAgentRunner
import util 
import sys


class DockerSkinAgent(DockerAgentRunner):
    '''An example Docker agent class'''

    def __init__(self, SourceAgent):
        self._agent = SourceAgent()

    def init_agent(self, id, game_type):
        return self._agent.init_agent(id, game_type)

    def act(self, observation, action_space):
        return self._agent.act(observation, action_space)

    def episode_end(self, reward):
        return self._agent.episode_end(reward)

    def shutdown(self):
        return self._agent.shutdown()


def main(requested_agent_name = "HybridAgent"):
    '''Inits and runs a Docker Agent'''
    requested_agent = util.get_agent(requested_agent_name) 
    agent = DockerSkinAgent(requested_agent)
    agent.run()


if __name__ == "__main__":
    main(sys.argv[1])
