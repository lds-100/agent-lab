import json


def save_trajectory(trajectory, reward, path="trajectories.jsonl"):
    trajectory["reward"] = reward

    with open(path, "a") as f:
        f.write(json.dumps(trajectory) + "\n")
