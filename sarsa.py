import numpy as np
import torch
import torch.nn as nn

class Q_Function():
    def __init__(self, input_size, learning_rate, discount_rate, num_actions):
        self.model = nn.Sequential(
            nn.Linear(input_size, 35),
            nn.LeakyReLU(),
            nn.Linear(35, 50),
            nn.LeakyReLU(), #?
            nn.Linear(50, 1) #?
        )
        self.model = self.model.float()
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)

        self.discount_rate = discount_rate
        self.num_actions = num_actions

    def choose_action(self, curr_state):
        # choose optimal action given Q function
        best_val = -1000000
        best_action = -1
        for action in range(self.num_actions):
            curr_val = self.forward(curr_state, action)
            if curr_val > best_val:
                best_val = curr_val
                best_action = action

        return best_action

    def forward(self, state, action):
        if action < 0 or action >= self.num_actions:
            return -100
        action_onehot = np.zeros(self.num_actions)
        action_onehot[action] = 1

        input = state.copy()
        input.extend(action_onehot)
        input = torch.tensor(input).float()
        val = self.model(input)
        return val


    def fit(self, sars_tuples):
        print("fitting")
        for curr_tuple in sars_tuples:
            s, a, r, sp = curr_tuple

            curr_val = self.forward(s, a)
            ap = self.choose_action(sp)
            next_val = r + self.discount_rate * self.forward(sp, ap).detach()
            loss = self.loss(curr_val, next_val)
            loss.backward()
            self.optimizer.step()