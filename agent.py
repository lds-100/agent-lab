from env import calculator, lookup
from model import tokenizer, model

SYSTEM_PROMPT_BASELINE = (
    "You have access to tools. Choose exactly one action."
)

SYSTEM_PROMPT_TOOLS = (
    "Choose exactly one action. "
    "For arithmetic, output CALL_CALCULATOR(expression). "
    "For information requests, output CALL_LOOKUP(topic). "
    "Output nothing else."
)
def agent(task):
    messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT_TOOLS
    },
    {"role": "user", "content": task}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
    )

    answer = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True,
    )

    if answer.startswith("CALL_CALCULATOR"):
        expression = answer[len("CALL_CALCULATOR("):-1]
        return calculator(expression)

    if answer.startswith("CALL_LOOKUP"):
        topic = answer[len("CALL_LOOKUP("):-1]
        return lookup(topic)

    return answer