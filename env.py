from itertools import product
import numpy as np

INVALID_ACTION_REWARD = -100
PACKAGE_REWARD = 10


class Environment():
    # the enviornment is represented as a matrix
    # each cell in the matrix corresponds to a block
    # blocks might have traffic, in which case the agent has limited movement
    def __init__(self, n, frac_traffic = 0.2, package_frac=0.05, package_prob=0.2):
        self.num_rows = n
        self.num_cols = n
        self.loc = np.array([int(n/2), int(n/2)])
        self.max_delta = 3

        # generate initial traffic in random spots on the matrix
        num_traffic = int(frac_traffic * n**2)
        self.traffic_idxs = np.random.randint(0, n, [num_traffic,2])

        # generate initial packages
        num_packages = int(package_frac * n**2)
        self.package_idxs = np.random.randint(0, n, [num_packages,2])
        self.package_prob = package_prob

    def simulate_packages(self):
        while np.random.rand() < self.package_prob:
            package_idx = tuple(np.random.randint(0, self.num_rows, size=2))
            if package_idx in self.package_idxs:
                self.simulate_packages()
            np.append(self.traffic_idxs, package_idx)


    def simulate_traffic(self):
        # updates traffic at each time step
        idx = 0
        for curr_traffic_idx in self.traffic_idxs:
            r = np.random.rand()
            if r < 0.5:
                # do not change this traffic idx
                idx += 1
                pass
            elif r < 0.75:
                # propogate this traffic to a random neighbor
                poss_directions = [(1,0), (-1,0), (0,1), (0,-1)]
                direction = poss_directions[np.random.choice(range(4))]
                np.append(self.traffic_idxs, curr_traffic_idx + direction)
                idx += 1
            else:
                # remove traffic from this idx
                self.traffic_idxs = np.delete(self.traffic_idxs, idx, 0)

    def valid_actions(self):
        # returns the list of possible actions
        # this depends on the current location and traffic
        # you can only move horizontally or vertically
        action_list = [(0,0)]

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

        return action_list

    def step(self, action):
        # update the state, simulate traffic, and return reward
        # expects an action as input in the form (delta i, delta j)
        if action not in self.valid_actions():
            return INVALID_ACTION_REWARD

        reward = 0
        self.loc += action
        if self.loc in self.package_idxs[0]:
            print("REWARD")
            reward = PACKAGE_REWARD
            idx = np.where(self.package_idxs == self.loc)
            self.package_idxs.remove(self.loc)
        self.simulate_traffic()

        new_state = [self.loc, self.package_idxs, self.traffic_idxs]

        return new_state, reward
