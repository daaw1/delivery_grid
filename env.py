from itertools import product
import numpy as np

INVALID_ACTION_REWARD = -1000000000 # 1B


class env():
    # the enviornment is represented as a matrix
    # each cell in the matrix corresponds to a block
    # blocks might have traffic, in which case the agent has limited movement
    def __init__(self, rows, cols, frac_traffic):
        self.num_rows = rows
        self.num_cols = cols
        self.loc = (int(rows/2), int(cols/2))
        self.max_delta = 3

        # generate initial traffic in random spots on the matrix
        row_list = list(range(self.num_rows))
        col_list = list(range(self.num_cols))
        unique_combinations = list(tuple(zip(row_list, element))
                                   for element in product(col_list, repeat=len(row_list)))
        num_traffic = frac_traffic * len(unique_combinations)
        self.traffic_idxs = list(np.random.sample(unique_combinations, num_traffic))

    def simulate_traffic(self):
        # updates traffic at each time step
        for curr_traffic_idx in self.traffic_idxs:
            r = np.random.rand()
            if r < 0.5:
                # do not change this traffic idx
                pass
            elif r < 0.75:
                # propogate this traffic to a random neighbor
                direction = np.random.choice([(1,0), (-1,0), (0,1), (0,-1)])
                self.traffic_idxs.append(curr_traffic_idx + direction)
            else:
                # remove traffic from this idx
                self.traffic_idxs.remove(curr_traffic_idx)

    def valid_actions(self):
        # returns the list of possible actions
        # this depends on the current location and traffic
        # you can only move horizontally or vertically
        action_list = []

        for i in range(self.max_delta+1):
            poss_loc = self.loc + (i,0)
            if poss_loc[0] >= self.num_rows or poss_loc[1] >= self.num_cols:
                # if out of bounds then ignore
                break
            action_list.append((i, 0))
            if poss_loc in self.traffic_idxs:
                # if a cell has traffic we can move into it but not out (in the same timestep)
                break

        for i in range(-self.max_delta, 0):
            # must split for traffic check
            poss_loc = self.loc + (i,0)
            if poss_loc[0] >= self.num_rows or poss_loc[1] >= self.num_cols:
                # if out of bounds then ignore
                break
            action_list.append((i, 0))
            if poss_loc in self.traffic_idxs:
                break

        for j in range(self.max_delta + 1):
            poss_loc = self.loc + (0,j)
            if poss_loc[0] >= self.num_rows or poss_loc[1] >= self.num_cols:
                # if out of bounds then ignore
                break
            action_list.append((0,j))
            if poss_loc in self.traffic_idxs:
                # if a cell has traffic we can move into it but not out (in the same timestep)
                break

        for j in range(-self.max_delta, 0):
            # must split for traffic check
            poss_loc = self.loc + (0,j)
            if poss_loc[0] >= self.num_rows or poss_loc[1] >= self.num_cols:
                # if out of bounds then ignore
                break
            action_list.append((0,j))
            if poss_loc in self.traffic_idxs:
                break
        pass

    def step(self, action):
        # update the state, simulate traffic, and return reward
        # expects an action as input in the form (delta i, delta j)
        if action not in self.valid_actions():
            return INVALID_ACTION_REWARD

        self.loc += action

        self.simulate_traffic()
