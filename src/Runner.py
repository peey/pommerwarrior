'''An example Q-Learning based warrior agent'''
import pommerman as pom
import os, sys

from BabyAgent import BabyAgent 

def main(train_for, to_render):

    '''Simple function to bootstrap a game.
       
       Example training environment.
    '''
    # Print all possible environments in the Pommerman registry
    print(pom.REGISTRY)

    wa = BabyAgent()
    agents = {
      "ours": [wa],
      "theirs": [pom.agents.SimpleAgent(), pom.agents.RandomAgent(), pom.agents.RandomAgent()]
    }


    # Make the "TeamCompetition" environment using the agent list
    # env = pommerman.make('PommeTeamCompetition-v0', agent_list)
    agents_list = agents["ours"] + agents["theirs"]
    assert(len(agents_list) == 4)
    env = pom.make('PommeFFACompetition-v0', agents_list)


    complete_game_count = 0
    draw_game_count = 0
    our_agents_wins             = [0     for agent in agents["ours"]]
    our_agents_draw_performance = [0     for agent in agents["ours"]]

    for i_episode in range(train_for):
        state = env.reset()

        our_agents_dead = [False for agent in agents["ours"]]
        done = False
        steps = 0

        while not done and not all(our_agents_dead) and steps < 300: # TODO hopefully no memory leak on abruptly ending and resetting an environment?
            if to_render:
                env.render()
            actions = env.act(state)
            #print("okay", actions)
            state, reward, done, info = env.step(actions) # done refers to if the whole game has ended or not
            our_agents_dead = [reward[i] == -1 for i in range(len(agents["ours"]))] # we may be dead and yet game isn't done as other players are competing. Little do we care.

        abrupt_end = not any(filter(lambda x: x == 1, reward)) # if no one won, then we abruptly ended the game

        print('Episode {} finished:'.format(i_episode))
        print("\t", reward)

        if not abrupt_end:
            print("\t complete game")
        else:
            print("\t abrupt end, may or may not have been a draw")

        for i in range(len(agents["ours"])):
            print("\t Our agent %d:" % i)

            our_agent = agents["ours"][i]

            if reward[i] == 1:
                our_agents_wins[i] += 1 # episode_end has already been called by the script
                print("\t\t it won")
                complete_game_count += 1
            elif reward[i] == -1 : 
                if abrupt_end:
                    our_agent.episode_end(-1)
                print("\t\t it lost")
                complete_game_count += 1
            elif abrupt_end:
                # Reward based on no of agents alive. Not -ve because our agent was alive, and if we hadn't intervened it "could've won". Each agent's objective is to kill others (directly or indirectly), even if they're ours. 
                # So 1 - 0.25 * alive is the reward (either 0.5, 0.25, or 0 if all alive). If they had killed agents in the game, they would recieve upto 0.5 more reward. 
                # So for optimal performance, reward goes up to 1.5 (combined with game end reward) for games that end properly, and up to 1 for games that end abruptly
                total_alive = len(filter(lambda x: x != -1, reward))
                our_agents_draw_performance[i] += total_alive
                our_agent.episode_end(1 - 0.25 * total_alive)
                draw_game_count += 1
                print("\t\t draw with %d survivors" % total_alive)

        with open("training_performance.txt", "a") as f:
            f.write("Win Ratio, %d, \"%s\"\n" % (complete_game_count, our_agents_wins))
            f.write("Draw Performance, %d, \"%s\"\n" % (draw_game_count, our_agents_draw_performance))

    print("completed the training")
    env.close()


if __name__ == '__main__':

    if len(sys.argv) > 1 and sys.argv[1] == "False":
        to_render = False
    else: 
        to_render = True

    if len(sys.argv) > 2:
        train_for = int(sys.argv[2])
    else: 
        train_for = 100000 # episodes


    main(train_for, to_render)
