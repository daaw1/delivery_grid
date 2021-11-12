from env import Environment
from sarsa import Q_Function

import argparse
import os
import copy
import numpy as np
import torch

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
            self.idx_to_delta[action] = [elem * (int(action%3)+1) for elem in curr_dir]

        self.data = []
        curr_directory = os.path.dirname(__file__)
        self.curr_directory = os.path.join(curr_directory, f"experiments/{args.name}")
        if not os.path.isdir(self.curr_directory):
            os.mkdir(self.curr_directory)
        self.log_file = open(self.curr_directory + "/log.txt", "w")
        self.reward_file = open(self.curr_directory + "/reward.txt", "w")
        self.loss_file = open(self.curr_directory + "/loss.txt", "w")


        input_size = int(args.n**2/15)*2 + 2 + 2 + 1 + int(args.n**2/5)*2
        self.Q = Q_Function(input_size, args.lr, args.dr, self.num_actions)

    def log(self, loop, avg_reward, avg_loss):
        str = f"On the {loop}th iteration the average loss was {avg_loss}\
         and the average episodic reward was {avg_reward}"
        print(str)
        self.log_file.write(str + "\n")
        self.reward_file.write(f"{avg_reward}\n")
        self.loss_file.write(f"{avg_loss}\n")

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
                episode_rewards = []

                s = self.curr_env.get_state() # initial state
                for t in range(self.num_steps):
                    action = self.Q.choose_action(s)[0]
                    action_delta = self.idx_to_delta[action]
                    s, a, r, sp = self.curr_env.step(action_delta)
                    self.data.append([s, action, r, sp])
                    s = sp

                    episode_rewards.append(r)

                # print(episode_rewards)
                episode_rewards = np.sum(episode_rewards)
                # print(episode_rewards)
                all_rewards.append(episode_rewards)

            # fit Q function
            avg_loss = self.Q.fit(self.data)

            # check convergence and reset if necessary
            avg_reward = np.mean(all_rewards)
            if loop % 1 == 0:
                self.log(loop, avg_reward, avg_loss)
            if loop % 5 == 0:
                print("Updated Target Model")
                self.Q.target_model = copy.deepcopy(self.Q.model)
            loop += 1

            if loop > 500:
                break

        torch.save(self.Q.model.state_dict(), self.curr_directory)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--num_episodes", default=10, type=int)
    arg_parser.add_argument("--num_steps", default = 100, type=int)
    arg_parser.add_argument("-n", default=10, type=int)
    arg_parser.add_argument("-lr", default=0.0001, type=float)
    arg_parser.add_argument("-dr", default=0.95, type=float)

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