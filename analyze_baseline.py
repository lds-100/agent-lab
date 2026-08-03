import json
from collections import Counter

import evaluate


def print_trajectory_analysis(
    trajectory,
    tool_eval,
    reward,
    answer_correct,
):
    print(f"\nTASK: {trajectory['task']}")
    print("STEPS:")

    for i, step in enumerate(trajectory["steps"], 1):
        print(f"  {i}. {step['action']}")
        print(f"     → {step['observation']}")

    print(f"ANSWER CORRECT: {answer_correct}")
    print(f"TOOL SELECTION: {tool_eval['tool_selection_correct']}")
    print(f"ARGUMENT CORRECT: {tool_eval['argument_correct']}")
    print(f"MISSING STEPS: {tool_eval['missing_required_steps']}")
    print(f"INVALID CALLS: {tool_eval['invalid_calls']}")
    print(f"UNNECESSARY CALLS: {tool_eval['unnecessary_calls']}")
    print(f"REWARD: {reward:.2f}")
    print("-" * 80)


def analyze_baseline(path):
    with open(path, "r") as f:
        trajectories = [json.loads(line) for line in f]

    results = []

    for trajectory in trajectories:
        task_data = next(
            t
            for t in evaluate.MULTISTEP_TASKS
            if t["task"] == trajectory["task"]
        )

        # Ignore answer correctness for now.
        answer_correct = False

        tool_eval = evaluate.evaluate_tool_trajectory(
            trajectory["steps"],
            task_data["expected_tools"],
            task_data["reference_actions"],
        )

        reward = evaluate.calculate_multistep_efficiency_reward(
            answer_correct,
            tool_eval,
        )

        results.append(
            {
                "reward": reward,
                "answer_correct": answer_correct,
                **tool_eval,
            }
        )

        print_trajectory_analysis(
            trajectory,
            tool_eval,
            reward,
            answer_correct,
        )

    rewards = [r["reward"] for r in results]

    print("\nReward Summary")
    print("-" * 80)
    print(f"Num trajectories: {len(rewards)}")
    print(f"Min reward: {min(rewards):.2f}")
    print(f"Max reward: {max(rewards):.2f}")
    print(f"Average reward: {sum(rewards) / len(rewards):.2f}")

    print("\nReward Distribution")
    for reward, count in sorted(Counter(rewards).items()):
        print(f"{reward:>6.2f}: {count}")

    print("\nMetric Summary")
    print(
        f"Tool selection correct: "
        f"{sum(r['tool_selection_correct'] for r in results)}"
    )
    print(
        f"Argument correct: "
        f"{sum(r['argument_correct'] is True for r in results)}"
    )
    print(
        f"Total missing steps: "
        f"{sum(r['missing_required_steps'] for r in results)}"
    )
    print(
        f"Total invalid calls: "
        f"{sum(r['invalid_calls'] for r in results)}"
    )
    print(
        f"Total unnecessary calls: "
        f"{sum(r['unnecessary_calls'] for r in results)}"
    )


if __name__ == "__main__":
    analyze_baseline(
        "baselines/multistep_v2_trajectories_20260803_120535_profile_subject.jsonl",
    )