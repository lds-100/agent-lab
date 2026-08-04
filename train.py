import json
from datetime import datetime, timezone

import torch

from agent import (
    SYSTEM_PROMPT_TOOLS,
    execute_action,
    generate_action,
    sequence_log_probability,
)
from evaluate import (
    MULTISTEP_TASKS,
    evaluate_tool_trajectory,
    judge_answer,
)
from experiment_config import (
    CHECKPOINT_EVERY,
    CHECKPOINT_PREFIX,
    MAX_STEPS,
    NUM_UPDATES,
)
from model import model
from reward import calculate_multistep_efficiency_reward

# NUM_UPDATES = 1
# CHECKPOINT_EVERY = 1
# MAX_STEPS = 4

LEARNING_RATE = 1e-6

# CHECKPOINT_PREFIX = "lora_test_checkpoint"

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
PATH = f"training_trajectories_{RUN_ID}.jsonl"


trainable_parameters = [
    p for p in model.parameters()
    if p.requires_grad
]

optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=LEARNING_RATE,
)


def rollout(task_data):
    """
    Run one complete environment trajectory.

    Returns the trajectory and the generated-token log probability.
    """

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_TOOLS,
        },
        {
            "role": "user",
            "content": task_data["task"],
        },
    ]

    steps = []
    log_probability_terms = []

    answer = ""

    for _ in range(MAX_STEPS):
        action, generated_ids, input_ids = generate_action(messages)

        log_probability = sequence_log_probability(
            input_ids,
            generated_ids,
        )

        log_probability_terms.append(log_probability)

        tool, observation, is_final = execute_action(action)

        if is_final:
            answer = action
            break

        steps.append(
            {
                "action": action,
                "tool": tool,
                "observation": observation,
            }
        )

        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": action,
                },
                {
                    "role": "user",
                    "content": (
                        f"Tool result: {observation}\n\n"
                        "If you have enough information, "
                        "answer the original question. "
                        "Otherwise, make one more tool call."
                    ),
                },
            ]
        )

    if not answer:
        answer = ""

    total_log_probability = torch.stack(log_probability_terms).sum()

    return (
        {
            "task": task_data["task"],
            "expected_answer": task_data["expected_answer"],
            "steps": steps,
            "answer": answer,
        },
        total_log_probability,
    )


def calculate_trajectory_reward(
    task_data,
    trajectory,
):
    answer_correct = judge_answer(
        task_data["task"],
        task_data["expected_answer"],
        trajectory["answer"],
    )

    tool_eval = evaluate_tool_trajectory(
        trajectory["steps"],
        task_data["expected_tools"],
        task_data["reference_actions"],
    )

    reward = calculate_multistep_efficiency_reward(
        answer_correct,
        tool_eval,
    )

    return reward, answer_correct, tool_eval


def train_one_update(task_data, update):
    trajectory, log_probability = rollout(task_data)

    reward, answer_correct, tool_eval = calculate_trajectory_reward(
        task_data,
        trajectory,
    )

    # REINFORCE:
    #
    # maximize:
    #     reward * log_probability
    #
    # PyTorch minimizes loss, so negate it.

    loss = -reward * log_probability

    optimizer.zero_grad()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        trainable_parameters,
        max_norm=1.0,
    )

    optimizer.step()

    trajectory.update(
        {
            "run_id": RUN_ID,
            "update": update,
            "checkpoint": f"{CHECKPOINT_PREFIX}-{update}",
            "answer_correct": answer_correct,
            "tool_selection_correct": (tool_eval["tool_selection_correct"]),
            "argument_correct": (tool_eval["argument_correct"]),
            "missing_required_steps": (tool_eval["missing_required_steps"]),
            "invalid_calls": (tool_eval["invalid_calls"]),
            "unnecessary_calls": (tool_eval["unnecessary_calls"]),
            "reward": reward,
            "loss": loss.detach().item(),
        }
    )

    return trajectory


def train():
    model.train()

    with open(PATH, "w") as f:
        for update in range(
            1,
            NUM_UPDATES + 1,
        ):
            for task_data in MULTISTEP_TASKS:
                trajectory = train_one_update(
                    task_data,
                    update,
                )

                print(
                    json.dumps(
                        trajectory,
                        indent=2,
                    )
                )

                f.write(json.dumps(trajectory) + "\n")

            if update % CHECKPOINT_EVERY == 0:
                checkpoint = f"{CHECKPOINT_PREFIX}-{update}"

                model.save_pretrained(checkpoint)

                print(f"\nSaved {checkpoint}\n")
                


if __name__ == "__main__":
    train()
