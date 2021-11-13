import os
import argparse
import matplotlib.pyplot as plt

def get_loss(loss_file):
    losses = loss_file.readlines()
    losses = [float(line.rstrip()) for line in losses[:500]]
    x = range(len(losses))
    return x, losses

def get_reward(reward_file):
    rewards = reward_file.readlines()
    rewards = [float(line.rstrip()) for line in rewards[:500]]
    x = range(len(rewards))
    return x, rewards

def run(all_names):
    file_directory = os.path.dirname(__file__)
    max_reward = 0
    for name in all_names:
        curr_directory = f"./experiments/{name}"

        reward_file = open(curr_directory + "/reward.txt", "r")
        curr_iter, curr_rewards = get_reward(reward_file)
        plt.plot(curr_iter, curr_rewards, label=f"{name}")

        if max(curr_rewards) > max_reward:
            max_reward = max(curr_rewards)
    plt.title("Environment Rewards")
    plt.xlabel("Iterations")
    plt.ylabel("Average Episodic Reward")
    # plt.yticks(range(int(max_reward)))
    plt.legend()
    plt.show()


    for name in all_names:
        curr_directory = f"./experiments/{name}"

        loss_file = open(curr_directory + "/loss.txt", "r")
        curr_iter, curr_losses = get_loss(loss_file)
        plt.plot(curr_iter, curr_losses, label=f"{name}")

        if max(curr_rewards) > max_reward:
            max_reward = max(curr_rewards)
    plt.title("SARSA Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Average Loss")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    all_names = ["baseline_sarsa", "penalized_sarsa", "traffic_sarsa"]
    run(all_names)

    all_names_2 = ["traffic_sarsa", "soft_sarsa"]
    # run(all_names_2)

