"""
TensorForce Agents - For benchmarking using PPO and DQN algorithms.

NOTE: The below code is heavily borrowed from pommerman cli script
https://github.com/MultiAgentLearning/playground/blob/master/pommerman/cli/train_with_tensorforce.py
since it was used only for benchmarking purposes.

Call this with a config, a game, and a list of agents, one of which should be a
tensorforce agent. The script will start separate threads to operate the agents
and then report back the result.

An example with all three simple agents running ffa:
python train_with_tensorforce.py \
 --agents=tensorforce::ppo,test::agents.SimpleAgent,test::agents.SimpleAgent,test::agents.SimpleAgent \
 --config=PommeFFACompetition-v0
"""
import atexit
import functools
import os

import argparse
import docker
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
import gym

from pommerman import helpers, make
from pommerman.agents import SimpleAgent, RandomAgent
from Tensorflow_Agents import *


def clean_up_agents(agents):
    """Stops all agents"""
    return [agent.shutdown() for agent in agents]


class WrappedEnv(OpenAIGym):
    '''An Env Wrapper used to make it easier to work
    with multiple agents'''

    def __init__(self, gym, visualize=False):
        self.gym = gym
        self.visualize = visualize

    def execute(self, action):
        if self.visualize:
            self.gym.render()

        actions = self.unflatten_action(action=action)

        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agent, actions)
        state, reward, terminal, _ = self.gym.step(all_actions)
        agent_state = self.gym.featurize(state[self.gym.training_agent])
        agent_reward = reward[self.gym.training_agent]
        return agent_state, terminal, agent_reward

    def reset(self):
        obs = self.gym.reset()
        agent_obs = self.gym.featurize(obs[3])
        return agent_obs


def main(max_steps=200, train_for=100, render=False):
    '''CLI interface to bootstrap taining'''
    parser = argparse.ArgumentParser(description="Playground Flags.")
    parser.add_argument(
        "--config",
        default="PommeFFACompetition-v0",
        help="Configuration to execute. See env_ids in "
        "configs.py for options.")
    parser.add_argument(
        "--render",
        default=False,
        action='store_true',
        help="Whether to render or not. Defaults to False.")
    args = parser.parse_args()

    config = args.config

    our_selection = DQN_TensorForce_Agent()
    # our_selection.restore_agent('PPO_Model.file')
    agents = [
        our_selection,
        SimpleAgent(),
        RandomAgent(),
        RandomAgent()
    ]

    env = make(config, agents)
    training_agent = None

    training_agent = our_selection
    env.set_training_agent(our_selection.agent_id)

    # Create a Proximal Policy Optimization agent
    agent = training_agent.initialize(env)

    atexit.register(functools.partial(clean_up_agents, agents))
    wrapped_env = WrappedEnv(env, visualize=(args.render or render))
    runner = Runner(agent=agent, environment=wrapped_env)
    runner.run(episodes=train_for, max_episode_timesteps=max_steps)
    won = len([x for x in runner.episode_rewards if x == 1])
    tie = len([x for x in runner.episode_timesteps if x == max_steps])
    lost = train_for - won - tie 
    # print("Stats: ", runner.episode_rewards, runner.episode_timesteps,
    #       runner.episode_times)
    print(won, tie, lost)
    # our_selection.save_model(file='PPO_Model.file')

    try:
        runner.close()
    except AttributeError as e:
        pass


if __name__ == "__main__":
    main(train_for=100, max_steps=500)
    main(train_for=5, render=True, max_steps=100)
