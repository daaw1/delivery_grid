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

def get_exploration(explore_file):
    explores = explore_file.readlines()
    explores = [float(line.rstrip()) for line in explores[:500]]
    x = range(len(explores))
    return x, explores

def run(all_names, type="loss"):
    file_directory = os.path.dirname(__file__)
    max_reward = 0
    for name in all_names:
        curr_directory = f"./experiments/{name}"

        reward_file = open(curr_directory + "/reward.txt", "r")
        curr_iter, curr_rewards = get_reward(reward_file)
        x,y = curr_iter, curr_rewards
        plt.plot(x, y, label=f"{name}")

    plt.title("Environment Rewards")
    plt.xlabel("Iterations")
    plt.ylabel("Average Episodic Reward")
    # plt.yticks(range(int(max_reward)))
    plt.legend()
    plt.show()


    for name in all_names:
        curr_directory = f"./experiments/{name}"

        loss_file = open(curr_directory + "/loss.txt", "r")


        if type == "Loss":
            curr_iter, curr_losses = get_loss(loss_file)
            x,y = curr_iter, curr_losses
        if type == "Explore":
            explore_file = open(curr_directory + "/explore.txt", "r")
            curr_iter, curr_explore = get_exploration(explore_file)
            x,y = curr_iter, curr_explore
        plt.plot(x, y, label=f"{name}")

    plt.title("SARSA Loss")
    plt.xlabel("Iterations")
    plt.ylabel(f"Average {type}")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    all_names = ["baseline_sarsa", "penalized_sarsa", "traffic_sarsa"]
    run(all_names)

    all_names_2 = ["soft_sarsa_p1", "soft_sarsa_p50"]
    run(all_names_2, type="Explore")

