'''An example Q-Learning based warrior agent'''
import pommerman
from pommerman import agents
from WarriorAgent import WarriorAgent
from CrazyAgent import CrazyAgent
from HybridAgent import HybridAgent
from DiscoAgent import DiscoAgent
from pommerman import constants
from ClassQL.MultiType import MultiTypeTeamAgent


def main():
    train_for = 100000 # episodes
    '''Simple function to bootstrap a game.
       
       Example training environment.
    '''
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)
    # exit(0)

    wa = DiscoAgent()
    agent_list = [
        agents.SimpleAgent(),
        agents.RandomAgent(),
        agents.SimpleAgent(),
        wa,
    ]

    # Make the "TeamCompetition" environment using the agent list
    env = pommerman.make('PommeFFACompetitionFast-v0', agent_list)
    # env = pommerman.make('PommeFFACompetition-v0', agent_list)

    # Run the episodes just like OpenAI Gym
    num_wins = 0
    for i_episode in range(train_for):

        constants.MAX_STEPS = 200 # doesn't seem to be working
        state = env.reset()
        done = False
        while not done:
            #env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            # print(reward)
        print('Episode {} finished'.format(i_episode))
        if wa.last_reward == 30:
            num_wins+=1
        print("Win Ratio: ", num_wins, i_episode+1)
    print("completed the episode")
    env.close()


if __name__ == '__main__':
    main()
