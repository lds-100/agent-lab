import json


def save_trajectory(trajectory, path="trajectories.jsonl"):
    with open(path, "a") as f:
        f.write(json.dumps(trajectory) + "\n")