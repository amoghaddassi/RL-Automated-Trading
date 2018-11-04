from environment import Environment
from agent import Agent

n_trials = 1000 #Length of training loop.

env, agent = Environment(), Agent()
env.reset()
#Training loop
for _ in range(n_trials):
    observation = env.render()
    action = agent.take_action(observation)
    reward = env.step(action)
    agent.update_Q_vals(reward)