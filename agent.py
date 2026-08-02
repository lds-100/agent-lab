from env import calculator
from model import tokenizer, model

def agent(task):
    messages = [
    {
        "role": "system",
        "content": "You have access to a calculator. When arithmetic is needed, respond with CALL_CALCULATOR(expression)."
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
        result = calculator(expression)
        return result

    return answer

task = "What is 12 * 25?"
result = agent(task)

print(result)