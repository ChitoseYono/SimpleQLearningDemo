import time

N_STATES = 6  # the length of the 1 dimensional world
START_POINT = 0
FRESH_TIME = 0.4


class Line(object):
    def __init__(self):
        super().__init__()
        self.action_space = ['left', 'right']
        self.n_actions = len(self.action_space)
        self.n_states = N_STATES
        self.title = 'line'
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        done = False
        reward = 0
        if action == 'right':
            self.state += 1
            if self.state == self.n_states - 1:
                self.state = 'terminal'
                done = True
                reward += 1
        else:
            if self.state != START_POINT:
                self.state -= 1

        return self.state, reward, done

    def render(self):
        env_list = ['-'] * (self.n_states - 1) + ['#']  # '---------#' our environment
        env_list[self.state] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)
