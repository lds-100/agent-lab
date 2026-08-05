import json
from datetime import datetime, timezone

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent import run_episode
from evaluate import (
    MULTISTEP_TASKS,
    evaluate_tool_trajectory,
    judge_answer,
)
from reward import calculate_multistep_efficiency_reward


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
CHECKPOINT_PATH = "/content/agent-lab/experiments/lora_test_checkpoint_-1"

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
RESULTS_PATH = f"experiments/lora_evaluation_{RUN_ID}.json"


def evaluate_checkpoint(model, tokenizer, task_data):
    """
    Run one task through the LoRA checkpoint and calculate
    the same evaluation metrics used during training.
    """

    trajectory = run_episode(
        model,
        tokenizer,
        task_data["task"],
    )

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

    trajectory.update(
        {
            "expected_answer": task_data["expected_answer"],
            "answer_correct": answer_correct,
            "tool_selection_correct": (
                tool_eval["tool_selection_correct"]
            ),
            "argument_correct": (
                tool_eval["argument_correct"]
            ),
            "missing_required_steps": (
                tool_eval["missing_required_steps"]
            ),
            "invalid_calls": (
                tool_eval["invalid_calls"]
            ),
            "unnecessary_calls": (
                tool_eval["unnecessary_calls"]
            ),
            "reward": reward,
        }
    )

    return trajectory


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype="float16",
        device_map="auto",
    )

    model = PeftModel.from_pretrained(
        base_model,
        CHECKPOINT_PATH,
    )

    model.eval()

    results = []

    for task_data in MULTISTEP_TASKS:
        result = evaluate_checkpoint(
            model,
            tokenizer,
            task_data,
        )

        results.append(result)

        print(
            json.dumps(
                result,
                indent=2,
            )
        )

    with open(RESULTS_PATH, "w") as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()