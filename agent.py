from env import calculator, lookup
from model import model, tokenizer

SYSTEM_PROMPT_BASELINE = "You have access to tools. Choose exactly one action."

SYSTEM_PROMPT_TOOLS = (
    "Choose exactly one action. "
    "For arithmetic, output CALL_CALCULATOR(expression). "
    "For information requests, output CALL_LOOKUP(topic). "
    "Output nothing else."
)

SYSTEM_PROMPT_TOOLS = (
    "You are an agent that may either call one tool or answer the user's question. "
    "If you still need information, output exactly one tool call and nothing else. "
    "For arithmetic, output CALL_CALCULATOR(expression). "
    "For information about the profile subject, output CALL_LOOKUP(topic). "
    "The lookup tool contains the profile subject's birth year, death year, "
    "birthplace, book, and book publication year. "
    "Use natural-language lookup topics such as: "
    "CALL_LOOKUP(profile subject birth year), "
    "CALL_LOOKUP(profile subject death year), "
    "CALL_LOOKUP(profile subject birthplace), "
    "CALL_LOOKUP(profile subject book), "
    "CALL_LOOKUP(profile subject book publication year). "
    "Retrieve each required fact separately. "
    "Do not guess profile subject facts. "
    "Once you have enough information to answer the question, "
    "do NOT call another tool. "
    "Instead, reply with the final answer in natural language. "
    "Output either one tool call or the final answer."
)

SYSTEM_PROMPT_FINAL = (
    "Use the tool result to answer the user's original question. "
    "Give only the final answer."
    "Be sure to return answers to all parts of the question."
)


def agent(task, max_steps=4):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TOOLS},
        {"role": "user", "content": task},
    ]

    steps = []

    # Send the current conversation to the model and return its response.
    def generate_response():
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=50)
        return tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        ).strip()

    # Let the model take up to `max_steps` tool actions.
    for _ in range(max_steps):
        action = generate_response()

        # Execute the requested tool.
        if action.startswith("CALL_CALCULATOR"):
            expression = action[len("CALL_CALCULATOR(") : -1]
            result = calculator(expression)
            tool = "calculator"
        elif action.startswith("CALL_LOOKUP"):
            topic = action[len("CALL_LOOKUP(") : -1]
            result = lookup(topic)
            tool = "lookup"
        else:
            # The model returned a final answer instead of a tool call.
            return {
                "task": task,
                "steps": steps,
                "answer": action,
            }

        # Record the tool call and feed the result back to the model.
        steps.append(
            {
                "action": action,
                "tool": tool,
                "observation": result,
            }
        )

        messages.extend(
            [
                {"role": "assistant", "content": action},
                {
                    "role": "user",
                    "content": (
                        f"Tool result: {result}\n\n"
                        "If you have enough information, answer the original question. "
                        "Otherwise, make one more tool call."
                    ),
                },
            ]
        )

    # If we hit the step limit, ask the model for a final answer.
    messages.append({"role": "system", "content": SYSTEM_PROMPT_FINAL})

    return {
        "task": task,
        "steps": steps,
        "answer": generate_response(),
    }
