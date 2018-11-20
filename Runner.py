'''An example Q-Learning based warrior agent'''
import pommerman
from pommerman import agents
from WarriorAgent import WarriorAgent
from pommerman import constants
from ClassQL.MultiType import MultiTypeTeamAgent


def main():
    '''Simple function to bootstrap a game.
       
       Example training environment.
    '''
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)
    # exit(0)

    # Create a set of agents (exactly four)
    wa = WarriorAgent()
    agent_list = [
        # agents.RandomAgent(),
        agents.SimpleAgent(),
        # agents.WarriorAgent(),
        # MultiTypeTeamAgent(),
        agents.RandomAgent(),
        # MultiTypeTeamAgent(),
        agents.RandomAgent(),
        # agents.SimpleAgent(),
        wa,
        # agents.SimpleAgent(),
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]
    # Make the "TeamCompetition" environment using the agent list
    # env = pommerman.make('PommeFFACompetitionFast-v0', agent_list)
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    # Run the episodes just like OpenAI Gym
    num_wins = 0
    for i_episode in range(1):
        constants.MAX_STEPS = 200
        state = env.reset()
        done = False
        while not done:
            env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            # print(reward)
        print('Episode {} finished'.format(i_episode))
        if wa.last_reward == 1:
            num_wins+=1
        print("Win Ratio: ", num_wins, i_episode+1)
    env.close()


if __name__ == '__main__':
    main()
