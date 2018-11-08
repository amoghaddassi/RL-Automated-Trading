from environment import Environment
from pybrain3.structure.modules.gaussianlayer import GaussianLayer
from pybrain3.rl.agents.learning import LearningAgent
from pybrain3.rl.learners.valuebased.nfq import NFQ
import pandas as pd

path = '/Users/arammoghaddassi/Google Drive/Projects/RL-Automated-Trading/data/'
aapl = pd.read_csv(path + 'AAPL.csv')
amzn = pd.read_csv(path + 'AMZN.csv')

#Model switches
n_episodes = 10
episdoe_length = 30

gaussian_layer = GaussianLayer(2)
nfq_learner = NFQ()

env, agent = Environment(aapl), LearningAgent(gaussian_layer, learner=nfq_learner)

agent.reset(), env.reset()

for ep in range(n_episodes):
    agent.newEpisode()
    for i in range(episdoe_length):
        state = env.state()
        agent.integrateObservation(state)
        action = agent.getAction() #Causing an error right now.
        state, reward = env.step(action)
        agent.giveReward(reward)
        print("Episode: {}, Trial: {}, Balance: {}".format(ep, i, env.account_value()))
        agent.learn() #When/how should I actually call this method?

