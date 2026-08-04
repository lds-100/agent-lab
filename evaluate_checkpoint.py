import json
from datetime import datetime, timezone

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent import run_episode
from evaluate import MULTISTEP_TASKS


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
CHECKPOINT_PATH = "lora_test_checkpoint-1"

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
RESULTS_PATH = f"experiments/lora_evaluation_{RUN_ID}.json"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

    for task in MULTISTEP_TASKS:
        result = run_episode(
            model,
            tokenizer,
            task,
        )

        results.append(result)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()