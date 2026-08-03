import json


def save_trajectory(
    trajectory,
    evaluation,
    path="trajectories.jsonl",
):
    trajectory.update(evaluation)

    with open(path, "a") as f:
        f.write(json.dumps(trajectory) + "\n")