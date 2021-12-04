import numpy as np
import torch
import torch.nn as nn
import copy

class soft_Q_Function():
    def __init__(self, input_size, learning_rate, discount_rate, num_actions):
        self.model = nn.Sequential(
            nn.Linear(input_size, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 500),
            nn.LeakyReLU(),
            nn.Linear(500, 1)
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

        self.entropy_weight = 0.5



    def choose_action(self, state, model="curr"):
        # if np.random.rand() > 0.9:
        #     size = len(state) if type(state[0]) == list else 1
        #     actions = np.random.choice(range(self.num_actions), size=size)
        #     action = actions.tolist()[0]
        #     return action

        # choose action probabilistically given Q function
        all_vals = []
        for action in range(self.num_actions):
            if type(state[0]) == list:
                action = [action] * len(state)
                print("list")
            else:
                action = [action]
            if model == "curr":
                curr_vals = self.forward(state, action)
            else:
                curr_vals = self.target_forward(state, action)
            all_vals.append(curr_vals)
        all_vals = torch.tensor(all_vals)
        # print(all_vals)
        probs = nn.functional.softmax(all_vals, dim=0)
        optimal_action = np.argmax(all_vals)
        entropy = -np.log(probs)
        entropy = nn.functional.softmax(entropy, dim=0)
        weighted_probs = nn.functional.softmax(probs + self.entropy_weight*entropy, dim=0)

        weighted_probs = weighted_probs.numpy()
        # action = np.random.choice(a=range(len(weighted_probs)), p=weighted_probs)
        action = np.argmax(weighted_probs)

        explore = action != optimal_action
        # print(entropy)
        # print(max(all_vals) - min(all_vals))
        # print(action)
        # action = int(torch.argmax(all_vals, dim=0))

        return action, explore



    def forward(self, state, actions):
        if type(state[0]) is not list:
            state = [state]
        if type(actions) is not list:
            actions = [actions]
        action_delta = [self.idx_to_delta[action] for action in actions]
        action_delta = torch.tensor(action_delta)

        input = state.copy()
        input = torch.tensor(input)
        input = torch.cat((input, action_delta), dim=1)
        input = input.float()
        val = self.model(input)
        return val


    def target_forward(self, state, actions):
        if type(state[0]) is not list:
            state = [state]
        if type(actions) is not list:
            actions = [actions]
        action_delta = [self.idx_to_delta[action] for action in actions]
        action_delta = torch.tensor(action_delta)

        input = state.copy()
        input = torch.tensor(input)
        input = torch.cat((input, action_delta), dim=1)
        input = input.float()
        val = self.target_model(input)
        return val


    def fit(self, sars_tuples):
        all_loss = []
        all_loss_tensor = []
        for curr_tuple in sars_tuples:
            s, a, r, sp = curr_tuple

            curr_val = self.forward(s, a)
            with torch.no_grad():
                ap, _ = self.choose_action(sp, model="target")
                next_val = r + self.discount_rate * self.target_forward(sp, ap)
            loss = self.loss(curr_val, next_val)
            all_loss.append(float(loss))
            all_loss_tensor.append(loss)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        # if self.entropy_weight > 0.01:
        #     self.entropy_weight -= 0.01
        return np.mean(all_loss)