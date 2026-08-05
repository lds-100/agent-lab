import csv
import json
from pathlib import Path


RESULTS_PATH = (
    "experiments/baseline/"
    "baseline_multistep_v2_20260804.jsonl"
)
RESULTS_PATH = (
    "experiments/lora_test_checkpoint_-1/"
    "lora_evaluation_20260805_083721.json"
)

SUMMARY_DIR = Path("experiments/summaries")
MASTER_CSV = Path("experiments/results_summary.csv")


def load_results(path):
    results = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if line:
                results.append(json.loads(line))

    return results

def load_results(path):
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    return [data]


def analyze(results):
    n = len(results)

    answer_correct = sum(
        bool(r.get("answer_correct", False))
        for r in results
    )

    tool_selection_correct = sum(
        bool(r.get("tool_selection_correct", False))
        for r in results
    )

    argument_correct = sum(
        bool(r.get("argument_correct", False))
        for r in results
    )

    invalid_calls = sum(
        r.get("invalid_calls", 0)
        for r in results
    )

    unnecessary_calls = sum(
        r.get("unnecessary_calls", 0)
        for r in results
    )

    missing_required_steps = sum(
        r.get("missing_required_steps", 0)
        for r in results
    )

    rewards = [
        r["reward"]
        for r in results
        if "reward" in r
    ]

    average_reward = (
        sum(rewards) / len(rewards)
        if rewards
        else None
    )

    return {
        "tasks": n,
        "answer_correct": answer_correct,
        "answer_accuracy": answer_correct / n if n else 0,
        "tool_selection_correct": tool_selection_correct,
        "tool_selection_accuracy": (
            tool_selection_correct / n if n else 0
        ),
        "argument_correct": argument_correct,
        "argument_accuracy": (
            argument_correct / n if n else 0
        ),
        "average_reward": average_reward,
        "invalid_calls": invalid_calls,
        "unnecessary_calls": unnecessary_calls,
        "missing_required_steps": missing_required_steps,
    }


def save_json_summary(summary, experiment_name):
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    path = SUMMARY_DIR / f"{experiment_name}.json"

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    return path


def append_to_master_csv(summary, experiment_name):
    MASTER_CSV.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "experiment": experiment_name,
        **summary,
    }

    file_exists = MASTER_CSV.exists()

    with open(
        MASTER_CSV,
        "a",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=row.keys(),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def main():
    path = Path(RESULTS_PATH)

    if not path.exists():
        raise FileNotFoundError(
            f"Results file not found: {path}"
        )

    results = load_results(path)

    if not results:
        raise ValueError("No results found.")

    summary = analyze(results)

    experiment_name = path.stem

    json_path = save_json_summary(
        summary,
        experiment_name,
    )

    append_to_master_csv(
        summary,
        experiment_name,
    )

    print()
    print("=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)

    print(f"Experiment:             {experiment_name}")
    print(f"Tasks:                  {summary['tasks']}")

    print(
        f"Answer accuracy:        "
        f"{summary['answer_accuracy']:.1%}"
    )

    print(
        f"Tool selection:         "
        f"{summary['tool_selection_accuracy']:.1%}"
    )

    print(
        f"Argument accuracy:      "
        f"{summary['argument_accuracy']:.1%}"
    )

    print(
        f"Average reward:         "
        f"{summary['average_reward']:.3f}"
    )

    print(
        f"Invalid calls:          "
        f"{summary['invalid_calls']}"
    )

    print(
        f"Unnecessary calls:      "
        f"{summary['unnecessary_calls']}"
    )

    print(
        f"Missing required steps: "
        f"{summary['missing_required_steps']}"
    )

    print()
    print(f"Saved JSON: {json_path}")
    print(f"Updated CSV: {MASTER_CSV}")
    print("=" * 50)


if __name__ == "__main__":
    main()