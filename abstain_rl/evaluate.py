import json
import re

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

def calculate_reward(
    expected_action,
    expected_answer,
    predicted_action,
    predicted_answer,
    correct,
):
    if predicted_action == "INVALID":
        return -1.0

    if expected_action == "ANSWER":

        if predicted_action == "ABSTAIN":
            return -0.5

        return 1.0 if correct else -1.0

    if expected_action == "ABSTAIN":

        return 1.0 if predicted_action == "ABSTAIN" else -1.0

    raise ValueError(
        f"Unknown expected action: {expected_action}"
    )


def generate_action(task):
    """
    Generate one model response for a task.

    Returns:
        raw_output
        generated_ids
        input_ids
    """

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
    """
    Calculate the log probability of the generated sequence.

    Returns a scalar tensor containing:

        sum(log P(token_i | previous tokens))
    """

    if generated_ids.numel() == 0:
        return torch.tensor(
            0.0,
            device=model.device,
        )

    full_ids = torch.cat(
        [
            input_ids,
            generated_ids.unsqueeze(0),
        ],
        dim=1,
    )

    with torch.no_grad():
        outputs = model(
            input_ids=full_ids,
        )

    logits = outputs.logits

    prompt_length = input_ids.shape[1]

    # Logits predicting each generated token.
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
        sequence_log_probability
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
    valid_format = parsed["valid_format"]

    expected_action = task["expected_action"]
    expected_answer = task["expected_answer"]

    # Semantic correctness.
    correct = calculate_correctness(
        expected_action,
        expected_answer,
        predicted_action,
        predicted_answer,
    )

    # Reward.
    reward = calculate_reward(
        expected_action,
        expected_answer,
        predicted_action,
        predicted_answer,
        correct,
    )

    # Save everything in JSON-safe Python types.
    trajectory = {
        "task": task["task"],

        "expected_action": expected_action,
        "expected_answer": expected_answer,

        "raw_output": raw_output,

        "predicted_action": predicted_action,
        "predicted_answer": predicted_answer,

        "valid_format": valid_format,
        "correct": correct,
        "reward": reward,

        # JSON-safe token IDs.
        "generated_ids": (
            generated_ids
            .detach()
            .cpu()
            .tolist()
        ),

        "input_ids": (
            input_ids
            .detach()
            .cpu()
            .tolist()
        ),

        "token_count": int(
            generated_ids.numel()
        ),

        "sequence_log_probability": float(
            log_probability.item()
        ),
    }

    return (
        trajectory,
        log_probability,
    )


def evaluate(
    output_path="baseline_trajectories.json",
):
    """
    Run the full evaluation set and save trajectories to JSON.

    Args:
        output_path: Path where the trajectory JSON will be saved.

    Returns:
        List of trajectory dictionaries.
    """

    results = []

    # Overall statistics.
    correct = 0
    total_reward = 0.0

    # Protocol statistics.
    valid = 0
    invalid = 0

    # Answer / abstain statistics.
    correct_answers = 0
    correct_abstentions = 0

    wrong_answers = 0
    wrong_abstentions = 0

    for task in ABSTAIN_TASKS:

        trajectory, log_probability = rollout(task)

        results.append(trajectory)

        expected_action = trajectory[
            "expected_action"
        ]

        predicted_action = trajectory[
            "predicted_action"
        ]

        is_correct = trajectory[
            "correct"
        ]

        reward = trajectory[
            "reward"
        ]

        # Overall statistics.
        if is_correct:
            correct += 1

        total_reward += reward

        # Protocol statistics.
        if trajectory["valid_format"]:
            valid += 1
        else:
            invalid += 1

        # Detailed semantic statistics.
        if expected_action == "ANSWER":

            if is_correct:
                correct_answers += 1

            elif predicted_action == "ABSTAIN":
                wrong_abstentions += 1

            else:
                wrong_answers += 1

        elif expected_action == "ABSTAIN":

            if is_correct:
                correct_abstentions += 1

            elif predicted_action == "ANSWER":
                wrong_answers += 1

            elif predicted_action == "INVALID":
                # Already counted in invalid.
                pass

        else:
            raise ValueError(
                f"Unknown expected action: "
                f"{expected_action}"
            )

        # Print trajectory.
        print("#" * 40)
        print("TASK:")
        print(task["task"])

        print(
            "EXPECTED:",
            trajectory["expected_action"],
        )

        print(
            "EXPECTED ANSWER:",
            trajectory["expected_answer"],
        )

        print(
            "MODEL:",
            trajectory["raw_output"],
        )

        print(
            "PARSED:",
            trajectory["predicted_action"],
        )

        print(
            "ANSWER:",
            trajectory["predicted_answer"],
        )

        print(
            "VALID FORMAT:",
            trajectory["valid_format"],
        )

        print(
            "CORRECT:",
            trajectory["correct"],
        )

        print(
            "REWARD:",
            trajectory["reward"],
        )

        print(
            "TOKENS:",
            trajectory["token_count"],
        )

        print(
            "LOG PROB:",
            trajectory[
                "sequence_log_probability"
            ],
        )

    total = len(ABSTAIN_TASKS)

    print("\n" + "=" * 40)
    print("EVALUATION SUMMARY")
    print("=" * 40)

    print(
        f"Overall accuracy: "
        f"{correct}/{total} "
        f"({correct / total:.2%})"
    )

    print(
        f"Protocol validity: "
        f"{valid}/{total} "
        f"({valid / total:.2%})"
    )

    print(
        f"Invalid responses: "
        f"{invalid}/{total}"
    )

    print(
        f"Correct answers: "
        f"{correct_answers}"
    )

    print(
        f"Correct abstentions: "
        f"{correct_abstentions}"
    )

    print(
        f"Wrong answers: "
        f"{wrong_answers}"
    )

    print(
        f"Wrong abstentions: "
        f"{wrong_abstentions}"
    )

    print(
        f"Total reward: "
        f"{total_reward:.2f}"
    )

    print(
        f"Average reward: "
        f"{total_reward / total:.2f}"
    )

    # Save trajectories as JSON.
    output_path = Path(output_path)

    # Create parent directory if needed.
    if output_path.parent != Path("."):
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        f"\nSaved trajectories to: "
        f"{output_path}"
    )

    return results


def normalize_answer(answer):
    answer = answer.strip().lower()

    # Remove trailing punctuation.
    answer = answer.rstrip(" .!?")

    # Normalize common units.
    answer = re.sub(r"\s+years?$", "", answer)

    # Normalize whitespace.
    answer = re.sub(r"\s+", " ", answer)

    return answer


def calculate_correctness(
    expected_action,
    expected_answer,
    predicted_action,
    predicted_answer,
):
    if predicted_action == "INVALID":
        return False

    # -----------------------------------------
    # ANSWER task
    # -----------------------------------------
    if expected_action == "ANSWER":

        if predicted_action != "ANSWER":
            return False

        if expected_answer is None or predicted_answer is None:
            return False

        expected = normalize_answer(expected_answer)
        predicted = normalize_answer(predicted_answer)

        # -----------------------------------------
        # Numeric expected answer
        # -----------------------------------------
        if re.fullmatch(r"-?\d+(?:\.\d+)?", expected):

            numbers = re.findall(
                r"(?<![\d.])-?\d+(?:\.\d+)?(?![\d.])",
                predicted,
            )

            # Compare numerically rather than as strings.
            expected_num = float(expected)

            return any(
                float(number) == expected_num
                for number in numbers
            )

        # -----------------------------------------
        # Text expected answer
        # -----------------------------------------
        expected_lower = expected.lower()
        predicted_lower = predicted.lower()

        # Exact match.
        if predicted_lower == expected_lower:
            return True

        # Expected answer appears in the model's
        # explanation.
        if expected_lower in predicted_lower:
            return True

        return False

    # -----------------------------------------
    # ABSTAIN task
    # -----------------------------------------
    if expected_action == "ABSTAIN":
        return predicted_action == "ABSTAIN"

    raise ValueError(
        f"Unknown expected action: {expected_action}"
    )

def parse_action(raw_output):
    """
    Parse model output.

    Valid:
        ABSTAIN
        ABSTAIN <optional explanation>

        ANSWER: <answer>

    Invalid:
        ANSWER:
        ANSWER: ABSTAIN
        anything else
    """

    action = raw_output.strip()

    # --------------------------------------------------
    # ABSTAIN
    # --------------------------------------------------
    if action.upper() == "ABSTAIN" or action.upper().startswith("ABSTAIN\n"):
        return {
            "action": "ABSTAIN",
            "answer": None,
            "valid_format": True,
        }

    # --------------------------------------------------
    # ANSWER
    # --------------------------------------------------
    if action.upper().startswith("ANSWER:"):
        answer = action[len("ANSWER:"):].strip()

        # Empty answer
        if not answer:
            return {
                "action": "INVALID",
                "answer": None,
                "valid_format": False,
            }

        # ANSWER: ABSTAIN is invalid
        if answer.upper() == "ABSTAIN":
            return {
                "action": "INVALID",
                "answer": None,
                "valid_format": False,
            }

        return {
            "action": "ANSWER",
            "answer": answer,
            "valid_format": True,
        }

    # --------------------------------------------------
    # Anything else
    # --------------------------------------------------
    return {
        "action": "INVALID",
        "answer": None,
        "valid_format": False,
    }