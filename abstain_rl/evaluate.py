import json
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from .tasks import ABSTAIN_TASKS
from model import model, tokenizer


SYSTEM_PROMPT = """
You are an agent answering questions.

For questions you can answer from the provided context, output:
ANSWER: <answer>

If the information is not available in the context, output:
ABSTAIN

Output exactly one of these formats.
""".strip()


MAX_NEW_TOKENS = 32


def parse_action(action):
    action = action.strip()

    if action == "ABSTAIN":
        return {
            "action": "ABSTAIN",
            "answer": None,
        }

    if action.startswith("ANSWER:"):
        answer = action[len("ANSWER:"):].strip()

        if not answer:
            return {
                "action": "INVALID",
                "answer": None,
            }

        return {
            "action": "ANSWER",
            "answer": answer,
        }

    return {
        "action": "INVALID",
        "answer": None,
    }


def calculate_reward(
    expected_action,
    expected_answer,
    predicted_action,
    predicted_answer,
):
    if predicted_action == "INVALID":
        return -1.0

    if expected_action == "ANSWER":
        if predicted_action == "ABSTAIN":
            return -0.5

        if predicted_answer.strip() == expected_answer:
            return 1.0

        return -1.0

    if expected_action == "ABSTAIN":
        if predicted_action == "ABSTAIN":
            return 1.0

        return -1.0

    raise ValueError(
        f"Unknown expected action: {expected_action}"
    )


def generate_action(task):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": task["task"],
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
        )

    input_length = inputs["input_ids"].shape[-1]

    generated_ids = outputs[0][input_length:]

    raw_output = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    return (
        raw_output,
        generated_ids,
        inputs["input_ids"],
    )


def sequence_log_probability(
    input_ids,
    generated_ids,
):
    if generated_ids.numel() == 0:
        return torch.tensor(
            0.0,
            device=model.device,
            requires_grad=True,
        )

    full_ids = torch.cat(
        [
            input_ids,
            generated_ids.unsqueeze(0),
        ],
        dim=1,
    )

    outputs = model(
        input_ids=full_ids,
    )

    logits = outputs.logits

    prompt_length = input_ids.shape[1]

    generated_logits = logits[
        :,
        prompt_length - 1:-1,
        :,
    ]

    log_probs = torch.log_softmax(
        generated_logits,
        dim=-1,
    )

    token_log_probs = log_probs.gather(
        2,
        generated_ids.unsqueeze(0).unsqueeze(-1),
    ).squeeze(-1)

    return token_log_probs.sum()


def rollout(task):
    """
    Run one complete trajectory for one task.

    Returns:
        trajectory
        total_log_probability
    """

    (
        raw_output,
        generated_ids,
        input_ids,
    ) = generate_action(task)

    log_probability = sequence_log_probability(
        input_ids,
        generated_ids,
    )

    parsed = parse_action(raw_output)

    predicted_action = parsed["action"]
    predicted_answer = parsed["answer"]

    expected_action = task["expected_action"]
    expected_answer = task["expected_answer"]

    reward = calculate_reward(
        expected_action,
        expected_answer,
        predicted_action,
        predicted_answer,
    )

    if expected_action == "ANSWER":
        correct = (
            predicted_action == "ANSWER"
            and predicted_answer == expected_answer
        )

    elif expected_action == "ABSTAIN":
        correct = (
            predicted_action == "ABSTAIN"
        )

    else:
        raise ValueError(
            f"Unknown expected action: {expected_action}"
        )

    trajectory = {
        "task": task["task"],
        "expected_action": expected_action,
        "expected_answer": expected_answer,

        "raw_output": raw_output,

        "generated_ids": generated_ids.detach().cpu().tolist(),
        "input_ids": input_ids.detach().cpu().tolist(),

        "predicted_action": predicted_action,
        "predicted_answer": predicted_answer,

        "correct": correct,
        "reward": reward,

        "log_probability": log_probability.item(),
    }

    return (
        trajectory,
        log_probability,
    )

def evaluate(output_path="baseline_trajectories.json"):
    results = []

    correct = 0
    total_reward = 0.0

    for task in ABSTAIN_TASKS:

        trajectory, log_probability = rollout(task)

        results.append(trajectory)

        if trajectory["correct"]:
            correct += 1

        total_reward += trajectory["reward"]

        print("#" * 40)
        print("TASK:")
        print(task["task"])
        print("EXPECTED:", task["expected_action"])
        print("MODEL:", trajectory["raw_output"])
        print("PARSED:", trajectory["predicted_action"])
        print("ANSWER:", trajectory["predicted_answer"])
        print("CORRECT:", trajectory["correct"])
        print("REWARD:", trajectory["reward"])
        print(
            "TOKENS:",
            trajectory["generated_ids"].shape[0],
        )
        print(
            "LOG PROB:",
            log_probability.item(),
        )

    print(
        f"\nScore: {correct}/{len(ABSTAIN_TASKS)}"
    )

    print(
        f"Average reward: "
        f"{total_reward / len(ABSTAIN_TASKS):.2f}"
    )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results

