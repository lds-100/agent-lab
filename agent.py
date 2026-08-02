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

def agent(task):

    # 1. Qwen chooses a tool
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

    action = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()

    # 2. Execute the chosen tool
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
            "tool": "none",
            "result": None,
            "answer": action,
        }

    # 3. Qwen receives the tool observation
    final_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_FINAL,
        },
        {
            "role": "user",
            "content": (
                f"Original question: {task}\n"
                f"Tool used: {tool}\n"
                f"Tool result: {result}\n"
                "Now give the final answer."
            ),
        },
    ]

    # 4. Qwen produces the final answer
    prompt = tokenizer.apply_chat_template(
        final_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
    )

    final_answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()

    # 5. Return the full trajectory for evaluation
    return {
        "task": task,
        "action": action,
        "tool": tool,
        "observation": result,
        "answer": final_answer,
    }