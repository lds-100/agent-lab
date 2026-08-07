import torch
from model import model, tokenizer

def agent(task):
    """
    Generate one response from Qwen for a single task.

    Returns the raw model output.
    """

    prompt = task["task"]

    messages = [
        {
            "role": "system",
            "content": (
                "You must either answer the question or abstain. "
                "If the answer is supported by the context, output "
                "exactly: ANSWER: <answer>. "
                "If the answer cannot be determined from the context, "
                "output exactly: ABSTAIN."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    generated_ids = outputs[0][
        inputs["input_ids"].shape[-1]:
    ]

    raw_output = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    return raw_output