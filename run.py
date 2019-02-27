from RL_brain import QLearningTable
from line_env import Line
import time


def update():
    for i_episode in range(12):
        observation = env.reset()
        count = 0

        while True:
            count += 1
            env.render()
            action = RL.choose_action(observation)
            observation_, reward, done = env.step(action)
            RL.learn(observation, action, reward, observation_)
            observation = observation_
            if done:
                break

        interaction = 'Episode %s: total_steps = %s' % (i_episode + 1, count)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')


# def update():
#     for episode in range(100):
#         # initial observation
#         observation = env.reset()
#
#         while True:
#             # fresh env
#             env.render()
#
#             # RL choose action based on observation
#             action = RL.choose_action(str(observation))
#
#             # RL take action and get next observation and reward
#             observation_, reward, done = env.step(action)
#
#             # RL learn from this transition
#             RL.learn(str(observation), action, reward, str(observation_))
#
#             # swap observation
#             observation = observation_
#
#             # break while loop when end of this episode
#             if done:
#                 break
#
#     # end of game
#     print('game over')
#     env.destroy()

if __name__ == "__main__":
    env = Line()
    RL = QLearningTable(actions=env.action_space, n_states=env.n_states)
    update()
    print('\r\nQ-table:\n')
    print(RL.table)
