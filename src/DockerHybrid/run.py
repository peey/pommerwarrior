"""Implementation of a simple deterministic agent using Docker."""

from .. import HybridAgent
from pommerman.runner import DockerAgentRunner


class MyAgent(DockerAgentRunner):
    '''An example Docker agent class'''

    def __init__(self):
        self._agent = HybridAgent.HybridAgent()

    def act(self, observation, action_space):
        return self._agent.act(observation, action_space)


def main():
    '''Inits and runs a Docker Agent'''
    agent = MyAgent()
    agent.run()


if __name__ == "__main__":
    main()
