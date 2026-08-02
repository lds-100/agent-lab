from agent import agent
from reward import calculate_reward, calculate_multistep_reward
from dataset import save_trajectory
from evaluate import EVAL_TASKS, MULTISTEP_TASKS


def generate_baseline_dataset():
    for task, expected_tool in EVAL_TASKS:
        result = agent(task)

        reward = calculate_reward(
            result["tool"],
            expected_tool,
        )

        save_trajectory(
            result,
            reward,
            path="trajectories.jsonl",
        )


def generate_multistep_dataset():
    for task, expected_answer in MULTISTEP_TASKS:
        result = agent(task)

        reward = calculate_multistep_reward(
            result["answer"],
            expected_answer,
        )

        save_trajectory(
            result,
            reward,
            path="multistep_trajectories.jsonl",
        )