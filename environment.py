#Environment switches
START_CASH = 1000

class Environment:
    """Represents the trading environment for an asset with price history data."""
    def __init__(self, data):
        self.data = data
        self.index = 0
        self.position = 0 #The starting agent doesn't own anything
        self.cash = START_CASH

    def state(self):
        """Returns the price of the asset at the current index"""
        return self.data.iloc[self.index]['Open']

    def step(self, action):
        """Takes a given action (action should be some integer) in the env, and returns:
        1. next_state: the price of the asset at time t+1
        2. reward: reward associated with the action"""
        self.index += 1
        next_state = self.state()

        past_balance = self.account_value()
        self.place_order(action)
        reward = self.reward(past_balance)

        return next_state, reward

    def place_order(self, amount):
        """Executes a trade that buys amount shares if amount > 0,
        and sells -amount shares otherwise."""
        amount = int(amount)
        if amount > 0: #buy order
            price = self.state()
            cost = price * amount
            if cost > self.cash:
                return #invalid trade
            self.cash -= cost
            self.position += amount
        elif amount < 0: #sell order
            if amount > self.position: #invalid trade
                return
            price = self.state()
            proceeds = price * amount
            self.cash += proceeds
            self.position += amount

    def account_value(self):
        """Returns the combined value of cash and stock in the account."""
        return self.cash + self.position * self.state()

    def reward(self, past_balance):
        """Takes a past balance and evaluates how well the most recent action impacted the balance."""
        return self.account_value() - past_balance #very simple measure, need to update.

    def reset(self):
        """Resets the environment to the starting state."""
        self.index = 0