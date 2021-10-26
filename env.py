

INVALID_ACTION_REWARD = -1000000000 # 1B


class env():
    # the eniornment is represented as a matrix
    # each cell in the matrix corresponds to a block
    # blocks might have traffic, in which case the agent has limited movement
    def __init__(self, rows, cols, frac_traffic):
        self.num_rows = rows
        self.num_cols = cols
        self.loc = (int(rows/2), int(cols/2))
        self.max_delta = 3

        # generate initial traffic
        pass

    def simulate_traffic(self):
        # updates traffic at each time step
        pass

    def valid_actions(self):
        # returns the list of possible actions
        # this depends on the current location and traffic
        # you can only move horizontally or vertically
        action_list = []

        for i in range(self.max_delta+1):
            poss_loc = self.loc + (i,0)
            action_list.append((i, 0))
            if poss_loc in self.traffic:
                # if a cell has traffic we can move into it but not out (in the same timestep)
                break

        for i in range(-self.max_delta, 0):
            # must split for traffic check
            poss_loc = self.loc + (i,0)
            action_list.append((i, 0))
            if poss_loc in self.traffic:
                break

        for j in range(self.max_delta + 1):
            poss_loc = self.loc + (0,j)
            action_list.append((0,j))
            if poss_loc in self.traffic:
                # if a cell has traffic we can move into it but not out (in the same timestep)
                break

        for j in range(-self.max_delta, 0):
            # must split for traffic check
            poss_loc = self.loc + (0,j)
            action_list.append((0,j))
            if poss_loc in self.traffic:
                break
        pass

    def step(self, action):
        # update the state, simulate traffic, and return reward
        # expects an action as input in the form (delta i, delta j)
        if action not in self.valid_actions():
            return -10000000