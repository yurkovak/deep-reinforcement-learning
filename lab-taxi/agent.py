import numpy as np
from collections import defaultdict
import sys


class Agent:

    def __init__(self, nA=6, eps=.1, eps_decay=0.9, alpha=0.3, gamma=0.9, mode=0):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.num_episodes = 0

        self.eps = eps
        self.min_eps = 0.
        self.eps_decay = eps_decay

        self.alpha = alpha
        self.min_alpha = 0.
        self.alpha_decay = 1.

        self.gamma = gamma

        assert mode in range(2), 'Not valid mode'
        self.mode = ['q-learning', 'expected'][mode]

    def _get_probs(self, state):
        max_i = np.argmax(self.Q[state])
        return [(1 - self.eps + self.eps / self.nA) if i == max_i else self.eps / self.nA for i in range(self.nA)]

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return np.random.choice(range(self.nA), p=self._get_probs(state))

    def _update_params(self):
        self.eps = max(self.min_eps, self.eps*self.eps_decay)
        self.alpha = max(self.min_alpha, self.alpha*self.alpha_decay)
        # print("\rEpisode {} || Eps {} || Alpha {}".format(self.num_episodes, self.eps, self.alpha), end=" ")
        # sys.stdout.flush()

    def step(self, s, a, r, s_next, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if self.mode == 'q-learning':
            a_best = np.argmax(self.Q[s_next])
            exp_val = self.Q[s_next][a_best]
        elif self.mode == 'expected':
            exp_val = sum([x*y for x, y in zip(self.Q[s_next], self._get_probs(s_next))])
        else:
            raise RuntimeError('Unsupported mode')
        self.Q[s][a] = (1 - self.alpha) * self.Q[s][a] + self.alpha * (r + self.gamma * exp_val)

        if done:
            self.num_episodes += 1
            self._update_params()
