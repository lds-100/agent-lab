import json

import evaluate
from evaluate import evaluate_tool_trajectory

path = "baselines/baseline_multistep_trajectories_20260803_1.jsonl"
with open(path, "r") as f:
    trajectories = [json.loads(line) for line in f]


for item in trajectories:
    task = item["task"]

    task_data = next(
        task_data for task_data in evaluate.MULTISTEP_TASKS if task_data["task"] == task
    )

    diagnostics = evaluate_tool_trajectory(
        item["steps"],
        task_data["expected_tools"],
        task_data["reference_actions"],
    )

    print("\nTASK:", task)
    print("DIAGNOSTICS:", diagnostics)
