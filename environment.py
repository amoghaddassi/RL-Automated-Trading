import pandas as pd

path = '/Users/arammoghaddassi/Google Drive/Projects/RL-Automated-Trading/data/'
aapl = pd.read_csv(path + 'AAPL.csv')
amzn = pd.read_csv(path + 'AMZN.csv')
START_CASH = 1000

class Environment:
    def __init__(self, data):
        self.data = data
        self.index = 0
        self.position = 0 #The starting agent doesn't own anything
        self.cash = START_CASH

    def state(self):
        """Holds price of all assets that we're trading"""
        price = self.data.iloc[self.index]['Open']
        return price

    def step(self, action):
        """Takes a given action in the env, and returns:
        1. next_state: the price of all assets in state at time t+1
        2. reward: reward associated with the action"""
        self.index += 1
        next_state = self.state()

        past_balance = self.account_value()
        self.place_order(action)
        reward = self.eval_state(past_balance)

        return next_state, reward

    def place_order(self, action):
        if action > 0: #buy order
            price = self.state()
            cost = price * action
            if cost > self.cash:
                return #invalid trade
            self.cash -= cost
            self.position += action
        elif action < 0: #sell order
            if action > self.position: #invalid trade
                return
            price = self.state()
            proceeds = price * action
            self.cash += proceeds
            self.position += action

    def account_value(self):
        return self.cash + self.position * self.state()

    def eval_state(self, past_balance):
        """Takes a past balance and evaluates how well the most recent action impacted the balance."""
        return self.account_value() - past_balance #very simple measure, need to update.

    def reset(self):
        """Resets the environment to the starting state."""
        pass