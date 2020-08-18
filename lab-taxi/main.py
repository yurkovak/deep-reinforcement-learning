import os
import logging

import gym
from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

from agent import Agent
from monitor import interact


class newJSONLogger(JSONLogger):

    def __init__(self, path):
        self._path = None
        super(JSONLogger, self).__init__()
        self._path = path if path[-5:] == ".json" else path + ".json"


class Logger(object):

    def __init__(self, optimizer, log_path):
        logging.basicConfig(filename=log_path,
                            format='%(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)

        self.optimizer = optimizer

    def log_state(self):
        logging.info("\nBest result: \n\t{}\n".format(self.optimizer.max))

        for i, res in enumerate(self.optimizer.res):
            logging.info("Iteration {}: \n\t{}".format(i, res))


env = gym.make('Taxi-v3')
def interact_wrapper(eps, eps_decay, alpha, gamma=0.9):
    agent = Agent(eps=eps, eps_decay=eps_decay, alpha=alpha, gamma=gamma)
    avg_rewards, best_avg_reward = interact(env, agent, 20000)
    return best_avg_reward


def resume(optimizer, log_path):
    if os.path.exists(log_path):
        load_logs(optimizer, logs=[log_path])
        print('Optimizer is aware of {} points.'.format(len(optimizer.space)))

    logger = newJSONLogger(path=log_path) # , reset=False
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    return optimizer


if __name__ == '__main__':

    pbounds = {'eps': (0.5, 1.), 'eps_decay': (0.5, 0.9), 'alpha': (0.1, 0.2)}
    optimizer = BayesianOptimization(
        f=interact_wrapper,
        pbounds=pbounds,
        random_state=47
    )
    optimizer = resume(optimizer, 'logs.json')
    # results of optimization can be viewed here
    logger = Logger(optimizer, 'result.log')

    optimizer.probe(
        params={'eps': .85, 'eps_decay': 0.6, 'alpha': 0.15},
        lazy=True,
    )
    try:
        optimizer.maximize(
            init_points=10,
            n_iter=100
        )
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    finally:
        logger.log_state()
