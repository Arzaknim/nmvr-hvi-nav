import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from datetime import datetime
import numpy as np

# --------------------------------------------------------
# CONFIG
# --------------------------------------------------------
# INPUT_FILE = "../NeuronalNetwork/results.txt"
# INPUT_FILE = "../NeuronalNetwork/easy_medium.txt"
# INPUT_FILE = "../NeuronalNetwork/medium_hard.txt"
INPUT_FILE = "../NeuronalNetwork/medium_maze.txt"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
# --------------------------------------------------------

def parse_line(line):
    # Format:
    # 2018-09-13 19:31:54, -93, 10
    parts = line.strip().split(',')
    if len(parts) != 3:
        return None

    timestamp_str = parts[0].strip()
    reward = float(parts[1].strip())
    length = float(parts[2].strip())

    # convert timestamp
    ts = datetime.strptime(timestamp_str, TIME_FORMAT)

    return ts, reward, length


def load_and_group(filename):
    groups = {}  # key: datetime second, value: list of (reward, length)

    with open(filename, 'r') as f:
        for line in f:
            parsed = parse_line(line)
            if parsed is None:
                continue

            ts, reward, length = parsed

            if ts not in groups:
                groups[ts] = {"reward": [], "length": []}

            groups[ts]["reward"].append(reward)
            groups[ts]["length"].append(length)

    return groups


def compute_averages(groups):
    # Sort timestamps
    timestamps = sorted(groups.keys())

    avg_rewards = []
    avg_lengths = []

    for ts in timestamps:
        rewards = groups[ts]["reward"]
        lengths = groups[ts]["length"]

        avg_rewards.append(np.mean(rewards))
        avg_lengths.append(np.mean(lengths))

    return timestamps, avg_rewards, avg_lengths


def main():
    groups = load_and_group(INPUT_FILE)
    timestamps, avg_rewards, avg_lengths = compute_averages(groups)

    # Convert timestamps to numbers for plotting
    times_num = np.array([date2num(ts) for ts in timestamps])

    # Center time to improve conditioning
    t_centered = times_num - times_num.mean()

    degree = 4

    # Fit using centered time
    coeff_reward = np.polyfit(t_centered, avg_rewards, degree)
    trend_reward = np.poly1d(coeff_reward)

    coeff_length = np.polyfit(t_centered, avg_lengths, degree)
    trend_length = np.poly1d(coeff_length)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Rewards
    ax1.plot(times_num, avg_rewards, marker='o', label="Average Reward")
    ax1.plot(times_num, trend_reward(t_centered), 'r--', label="Reward Trend")
    ax1.set_ylabel("Average Reward")
    ax1.grid(True)
    ax1.legend()

    # Episode lengths
    ax2.plot(times_num, avg_lengths, marker='o', color='orange', label="Average Episode Length")
    ax2.plot(times_num, trend_length(t_centered), 'r--', label="Length Trend")
    ax2.set_ylabel("Average Episode Length")
    ax2.set_xlabel("Time")
    ax2.grid(True)
    ax2.legend()

    # Auto-format timestamps
    fig.autofmt_xdate()

    plt.tight_layout()

    plt.savefig("results_plot.png")
    print("Saved plot to results_plot.png")
    plt.show()


if __name__ == "__main__":
    main()
