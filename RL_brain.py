import numpy as np
import pandas as pd


class QLearningTable:

    def __init__(self, actions, n_states, learning_rate=0.01, reward_decay=0.9, e_greedy=0.95):
        super().__init__()
        self.epsilon = e_greedy
        self.gamma = reward_decay
        self.lr = learning_rate
        self.actions = actions
        self.table = pd.DataFrame(
                    np.zeros((n_states, actions.__len__())),  # q_table initial values
                    columns=actions,  # actions's name
        )

    def choose_action(self, state):
        # This is how to choose an action
        state_actions = self.table.iloc[state, :]
        if (np.random.uniform() > self.epsilon) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
            action_name = np.random.choice(self.actions)
        else:  # act greedy
            action_name = state_actions.idxmax()  # replace argmax to idxmax as argmax means a different function in newer version of pandas
        return action_name

    def learn(self, s, a, r, s_):
        q_predict = self.table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.table.loc[s_, :].max()
        else:
            q_target = r
        self.table.loc[s, a] += self.lr * (q_target - q_predict)

