from environment import Environment
from pybrain3.rl.agents.learning import LearningAgent
from pybrain3.rl.learners.valuebased.nfq import NFQ
import pandas as pd

path = '/Users/arammoghaddassi/Google Drive/Projects/RL-Automated-Trading/data/'
aapl = pd.read_csv(path + 'AAPL.csv')
amzn = pd.read_csv(path + 'AMZN.csv')

n_trials = 1000 #Length of training loop.

env, agent = Environment(aapl), LearningAgent(NFQ)
