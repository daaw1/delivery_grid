from itertools import product
import numpy as np

INVALID_ACTION_REWARD = -100
PACKAGE_REWARD = 10


class Environment():
    # the enviornment is represented as a matrix
    # each cell in the matrix corresponds to a block
    # blocks might have traffic, in which case the agent has limited movement
    def __init__(self, n=10, max_delta=3, frac_traffic=0.2, package_frac=0.05, package_prob=0.2):
        self.num_rows = n
        self.num_cols = n
        self.loc = np.array([int(n/2), int(n/2)])
        self.max_delta = max_delta

        # generate initial traffic in random spots on the matrix
        self.max_traffic = int(n**2 / 5)
        self.traffic_idxs = np.zeros([self.max_traffic, 2])
        self.traffic_idxs[:int(self.max_traffic*2/3)] = np.random.randint(0, n, [int(self.max_traffic*2/3),2])

        self.num_traffic = int(self.max_traffic*2/3)

        # generate initial packages
        self.max_packages = int(n**2 / 15)
        self.package_idxs = np.zeros([self.max_packages, 2])
        self.package_idxs[:int(self.max_packages*2/3)] = np.random.randint(0, n, [int(self.max_packages*2/3),2])

        self.num_packages = int(self.max_packages*2/3)
        self.package_prob = package_prob

    def get_state(self):
        state = [self.loc, self.package_idxs, self.traffic_idxs]
        return state

    def simulate_packages(self):
        while np.random.rand() < self.package_prob:
            if self.num_packages >= self.max_packages:
                return
            package_idx = tuple(np.random.randint(0, self.num_rows, size=2))
            if package_idx in self.package_idxs:
                continue
            np.append(self.package_idxs, package_idx)
            self.num_packages += 1


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
                if self.num_traffic >= self.max_traffic:
                    idx += 1
                    continue
                # propogate this traffic to a random neighbor
                poss_directions = [(1,0), (-1,0), (0,1), (0,-1)]
                direction = poss_directions[np.random.choice(range(4))]
                np.append(self.traffic_idxs, curr_traffic_idx + direction)
                idx += 1
                self.num_traffic += 1
            else:
                # remove traffic from this idx
                self.traffic_idxs = np.delete(self.traffic_idxs, idx, 0)
                self.num_traffic -= 1

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
        last_state = self.get_state()

        reward = 0
        self.loc += action
        if self.loc in self.package_idxs[0]:
            print("REWARD")
            reward = PACKAGE_REWARD
            idx = np.where(self.package_idxs == self.loc)
            self.package_idxs.remove(self.loc)
        self.simulate_traffic()

        new_state = self.get_state()

        return last_state, action, reward, new_state
