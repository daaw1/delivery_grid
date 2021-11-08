from env import Environment
from sarsa import Q_Function

import argparse
import os
import numpy as np

class run():
    def __init__(self, args):
        self.n = args.n # n by n matrix
        self.num_episodes = args.num_episodes # per training step
        self.num_steps = args.num_steps

        self.num_actions = 4 * 3 # 4 directions 3 max velocity
        self.idx_to_delta = {} # action index to action effect
        dirs = [[1,0], [-1,0], [0,1], [0,-1]]
        for action in range(self.num_actions):
            curr_dir = dirs[int(action/3)]
            self.idx_to_delta[action] = curr_dir * (action+1)

        self.data = []
        curr_directory = os.path.dirname(__file__)
        curr_directory = os.path.join(curr_directory, f"experiments/{args.name}")
        if not os.path.isdir(curr_directory):
            os.mkdir(curr_directory)
        self.log_file = open(curr_directory + "/log.txt", "w")


        input_size = int(args.n**2/15)*2 + int(args.n**2/5)*2 + 2 + self.num_actions
        self.Q = Q_Function(input_size, args.lr, args.dr, self.num_actions)

    def log(self, loop, avg_reward):
        str = f"On the {loop}th iteration the average reward was {avg_reward}"
        print(str)
        self.log_file.write(str)

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

                s = self.curr_env.get_state() # initial state
                for t in range(self.num_steps):
                    action = self.Q.choose_action(s)
                    action_delta = self.idx_to_delta[action]
                    s, a, r, sp = self.curr_env.step(action_delta)
                    self.data.append([s, action, r, sp])
                    s = sp
                    if loop % 10 == 0:
                        all_rewards.append(r)

            if loop % 10 == 0:
                avg_reward = np.mean(all_rewards)
                self.log(loop, avg_reward)

            # fit Q function
            self.Q.fit(self.data)

            # check convergence and reset if necessary
            loop += 1

        last_batch_rewards = all_rewards
        avg_reward = np.mean(last_batch_rewards)
        self.log(loop, avg_reward)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--num_episodes", default=3)
    arg_parser.add_argument("--num_steps", default = 50)
    arg_parser.add_argument("--n", default=10)
    arg_parser.add_argument("--lr", default=0.001)
    arg_parser.add_argument("--dr", default=0.95)

    arg_parser.add_argument(
        "--name", default="test_i",
        help="name to log experiments")
    arg_parser.add_argument(
        "-p", "--checkpoint", default=None,
        help="path to checkpoint directory to load from or None")
    arg_parser.add_argument(
        "-s", "--seed", default=0, help="random seed to use.", type=int)
    args = arg_parser.parse_args()

    r = run(args)
    r.train()