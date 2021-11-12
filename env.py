from itertools import product
import numpy as np
import torch

INVALID_ACTION_REWARD = -20
PACKAGE_REWARD = 10


class Environment():
    # the enviornment is represented as a matrix
    # each cell in the matrix corresponds to a block
    # blocks might have traffic, in which case the agent has limited movement
    def __init__(self, n=10, max_delta=3, frac_traffic=0.2, package_frac=0.05, package_prob=0.2):
        self.num_rows = n
        self.num_cols = n
        self.loc = np.array([float(int(n/2)), float(int(n/2))])
        self.max_delta = max_delta

        # generate initial traffic in random spots on the matrix
        self.max_traffic = int(n**2 / 5)
        self.traffic_idxs = np.zeros([self.max_traffic, 2])
        self.traffic_idxs[:int(self.max_traffic*2/3)] = np.random.randint(0, n, [int(self.max_traffic*2/3),2])

        self.num_traffic = int(self.max_traffic*2/3)

        # generate initial packages
        self.max_packages = int(n**2 / 15)
        self.package_idxs = np.zeros([self.max_packages, 2], )
        self.package_idxs[:int(self.max_packages*2/3)] = np.random.randint(0, n, [int(self.max_packages*2/3),2])

        self.num_packages = int(self.max_packages*2/3)
        self.package_prob = package_prob

    def get_idx(self, array, value):
        idx = 0
        value = list(value.copy())
        valid = False
        for curr_val in array:
            if list(curr_val.copy()) == value:
                valid = True
                break
            idx += 1
        return valid, idx

    def get_state(self):
        state = list(self.loc)
        state.append(self.num_rows)
        state.extend(list(self.package_idxs.flatten()))
        state.extend(list(self.traffic_idxs.flatten()))
        return state

    def simulate_packages(self):
        while np.random.rand() < self.package_prob:
            if self.num_packages >= self.max_packages:
                return
            package_idx = tuple(np.random.randint(0, self.num_rows, size=2))
            if package_idx in self.package_idxs:
                continue
            valid, insert_idx = self.get_idx(self.package_idxs, [0,0])
            self.package_idxs[insert_idx] = package_idx
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

                valid, insert_idx = self.get_idx(self.traffic_idxs, [0,0])
                if not valid:
                    idx +=1
                    continue
                self.traffic_idxs[insert_idx] = curr_traffic_idx + direction
                idx += 1
                self.num_traffic += 1
            else:
                # remove traffic from this idx
                remove_idx = self.get_idx(self.traffic_idxs, curr_traffic_idx)
                self.traffic_idxs[remove_idx] = [0,0]
                self.num_traffic -= 1


    def check_position(self, new_loc):
        if new_loc[0] < -self.num_rows or new_loc[0] >= self.num_rows:
            return False
        if new_loc[1] < -self.num_rows or new_loc[1] >= self.num_rows:
            return False
        return True

    def compute_delta(self, action):
        idx = 0 if action[0] is not 0 else 1
        direction = np.sign(action[idx])
        change = np.zeros(2)

        while direction * change[idx] in range(0, direction * action[idx]):
            change[idx] += 1 * direction
            on_traffic, traffic_idx = self.get_idx(self.traffic_idxs, self.loc + change)
            if on_traffic:
                break
        return change

    def step(self, action):
        # update the state, simulate traffic, and return reward
        # expects an action as input in the form (delta i, delta j)
        last_state = self.get_state()

        reward = -1
        new_loc = self.loc + self.compute_delta(action)
        # new_loc = self.loc + action
        valid_loc = self.check_position(new_loc)
        if not valid_loc:
            reward = INVALID_ACTION_REWARD
            return last_state, action, reward, last_state
        self.loc = new_loc

        valid, package_remove_idx = self.get_idx(self.package_idxs, self.loc)
        if valid:
            # print("REWARD")
            reward = PACKAGE_REWARD
            self.package_idxs[package_remove_idx] = [0,0]
            # print(len(self.traffic_idxs.flatten()))
        self.simulate_traffic()

        new_state = self.get_state()

        return last_state, action, reward, new_state
