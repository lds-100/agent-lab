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

SYSTEM_PROMPT_FINAL = (
    "Use the tool result to answer the user's original question. "
    "Give only the final answer."
)

def agent(task, max_steps=2):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_TOOLS,
        },
        {
            "role": "user",
            "content": task,
        },
    ]

    steps = []

    for _ in range(max_steps):

        # Ask Qwen for the next action
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
        )

        action = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        # Execute the selected tool
        if action.startswith("CALL_CALCULATOR"):
            expression = action[len("CALL_CALCULATOR("):-1]
            result = calculator(expression)
            tool = "calculator"

        elif action.startswith("CALL_LOOKUP"):
            topic = action[len("CALL_LOOKUP("):-1]
            result = lookup(topic)
            tool = "lookup"

        else:
            return {
                "task": task,
                "steps": steps,
                "answer": action,
            }

        # Record this step
        steps.append({
            "action": action,
            "tool": tool,
            "observation": result,
        })

        # Give the observation back to Qwen
        messages.append({
            "role": "assistant",
            "content": action,
        })

        messages.append({
            "role": "user",
            "content": f"Tool result: {result}",
        })

    # Ask Qwen for final answer
    messages.append({
        "role": "system",
        "content": SYSTEM_PROMPT_FINAL,
    })

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
    )

    final_answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()

    return {
        "task": task,
        "steps": steps,
        "answer": final_answer,
    }