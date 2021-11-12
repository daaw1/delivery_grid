import numpy as np
import torch
import torch.nn as nn
import copy

class Q_Function():
    def __init__(self, input_size, learning_rate, discount_rate, num_actions):
        self.model = nn.Sequential(
            nn.Linear(input_size, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 500),
            nn.LeakyReLU(), #?
            nn.Linear(500, 1) #?
        )
        self.model = self.model.float()
        self.target_model = copy.deepcopy(self.model)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)

        self.discount_rate = discount_rate
        self.num_actions = num_actions

        self.idx_to_delta = {}  # action index to action effect
        dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        for action in range(self.num_actions):
            curr_dir = dirs[int(action / 3)]
            self.idx_to_delta[action] = [elem * (int(action%3)+1) for elem in curr_dir]


    def choose_action(self, state, model="curr"):
        if np.random.rand() > 0.9:
            size = len(state) if type(state[0]) == list else 1
            return np.random.choice(range(self.num_actions), size=size)

        # choose optimal action given Q function
        all_vals = []
        for action in range(self.num_actions):
            if type(state[0]) == list:
                action = [action] * len(state)
            else:
                action = [action]
            # action = [action]
            if model == "curr":
                curr_vals = self.forward(state, action)
            else:
                curr_vals = self.target_forward(state, action)
            all_vals.append(curr_vals)
        best_actions = torch.stack(all_vals).T
        best_actions = torch.argmax(best_actions, dim=2)
        best_actions = best_actions.tolist()
        best_actions = best_actions[0]

        return best_actions



    def forward(self, state, actions):
        if type(state[0]) is not list:
            state = [state]
        if type(actions) is not list:
            actions = [actions]
        action_delta = [self.idx_to_delta[action] for action in actions]
        action_delta = torch.tensor(action_delta)
        # action_delta = self.idx_to_delta[actions]

        input = state.copy()
        input = torch.tensor(input)
        input = torch.cat((input, action_delta), dim=1)
        input = input.float()
        # print(input.shape)
        # print(input)
        val = self.model(input)
        return val


    def target_forward(self, state, actions):
        if type(state[0]) is not list:
            state = [state]
        action_delta = [self.idx_to_delta[action] for action in actions]
        action_delta = torch.tensor(action_delta)
        # action_delta = self.idx_to_delta[actions]

        input = state.copy()
        input = torch.tensor(input)
        input = torch.cat((input, action_delta), dim=1)
        input = input.float()
        # print(input.shape)
        # print(input)
        val = self.target_model(input)
        return val


    def fit(self, sars_tuples):
        all_loss = []
        all_loss_tensor = []
        for curr_tuple in sars_tuples:
            s, a, r, sp = curr_tuple

            curr_val = self.forward(s, a)
            with torch.no_grad():
                ap = self.choose_action(sp, model="target")
                next_val = r + self.discount_rate * self.target_forward(sp, ap)
            loss = self.loss(curr_val, next_val)
            all_loss.append(float(loss))
            all_loss_tensor.append(loss)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        # # sars_tuples = np.array(sars_tuples).T
        # sars_tuples = [list(x) for x in zip(*sars_tuples)]
        # # print(sars_tuples)
        # states, actions, rewards, next_states = sars_tuples
        # print(len(states), len(actions))
        # curr_vals = self.forward(states, actions)
        # with torch.no_grad():
        #     next_actions = self.choose_action(next_states)
        #     print(len(next_actions))
        #     rewards = torch.flatten(torch.tensor(rewards))
        #     pred = self.discount_rate * self.target_forward(next_states, next_actions)
        #     pred = torch.flatten(pred)
        #     next_vals = torch.sum(torch.stack((rewards, pred), dim=1), dim=1)
        # print(curr_vals.shape)
        # print(next_vals.shape)
        # losses = self.loss(curr_vals, next_vals)
        # losses.backward()
        # self.optimizer.step()
        # self.optimizer.zero_grad()
        # return torch.mean(all_loss_tensor)
        return np.mean(all_loss)