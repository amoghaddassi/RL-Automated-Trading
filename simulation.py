from environment import Environment
from pybrain3.rl.learners.valuebased import ActionValueNetwork
from pybrain3.rl.agents.learning import LearningAgent
from pybrain3.rl.learners.valuebased.nfq import NFQ
import pandas as pd

path = '/Users/arammoghaddassi/Google Drive/Projects/RL-Automated-Trading/data/'
aapl = pd.read_csv(path + 'AAPL.csv')
amzn = pd.read_csv(path + 'AMZN.csv')

#Model switches
n_episodes = 10
episdoe_length = 30

controller = ActionValueNetwork(dimState= 1, numActions= 3)#Maps states to actions.
learner = NFQ() #Does the actual learning, updates values in action value network.

env, agent = Environment(aapl), LearningAgent(controller, learner=learner)

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

