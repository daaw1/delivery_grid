from env import Environment
from sarsa import Q_Function

import argparse
import numpy as np

class run():
    def __init__(self, args):
        self.Q = Q_Function()

        self.n = args.n
        self.num_episodes = args.num_episodes
        self.num_steps = args.num_steps

        self.num_actions = 4 * 3 # 4 directions 3 max velocity
        self.idx_to_delta = {}
        dirs = [[1,0], [-1,0], [0,1], [0,-1]]
        for action in range(self.num_actions):
            curr_dir = dirs[int(action/3)]
            self.idx_to_delta[action] = curr_dir * (action+1)

        self.data = []
        self.log_file = open(f"experiments/{args.name}/log.txt")

    def log(self, loop, avg_reward):
        str = f"On the {loop}th iteration the reward was {avg_reward}"
        print(str)
        self.log_file.write(str)

    def choose_action(self, curr_state):
    # choose optimal action given Q function
        best_val = -1000000
        best_action = -1
        for action in range(self.num_actions):
            action_onehot = np.zeros(self.num_actions)
            action_onehot[action] = 1

            input = curr_state.copy()
            input.append(action_onehot)
            curr_val = self.Q(input)
            if curr_val > best_val:
                best_val = curr_val
                best_action = action

        return best_action


    def train(self):
        # train until convergence
        loop = 0
        while True:
            # reset data
            self.data = []
            all_rewards = []

            # execute policy, collect data
            for curr_episodes in range(self.num_episodes):
                self.curr_env = Environment(self.n)
                episode_data = []

                s = self.curr_env.get_state() # initial state
                for t in range(self.num_steps):
                    action = self.choose_action(s)
                    action_delta = self.idx_to_delta[action]
                    s, a, r, sp = self.curr_env.step(action_delta)
                    episode_data.append([s, a, r, sp])
                    s = sp
                    if loop % 10 == 0:
                        all_rewards.append(r)

            if loop % 10 == 0:
                avg_reward = np.mean(all_rewards)
                self.log(loop, avg_reward)

            # fit Q function
            pass

            # check convergence and reset if necessary
            loop += 1

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--num_episodes", default=3)
    arg_parser.add_argument("--num_steps", default = 50)
    arg_parser.add_argument("--n", default=10)

    arg_parser.add_argument(
        "--name", default="test_i",
        help="name to log experiments")
    arg_parser.add_argument(
        "-p", "--checkpoint", default=None,
        help="path to checkpoint directory to load from or None")
    arg_parser.add_argument(
        "-s", "--seed", default=0, help="random seed to use.", type=int)
    args = arg_parser.parse_args()

    main(args)