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
    "Choose exactly one action. "
    "For arithmetic, output CALL_CALCULATOR(expression). "
    "For information requests about the profile subject, output CALL_LOOKUP(topic). "
    "The lookup tool contains facts about the profile subject's birth year, "
    "death year, birthplace, book, and book publication year. "
    "When the question asks for one of these facts, you must use the lookup tool. "
    "Use a clear natural-language topic describing the specific fact you need. "
    "For example: "
    "CALL_LOOKUP(profile subject birth year), "
    "CALL_LOOKUP(profile subject death year), "
    "CALL_LOOKUP(profile subject birthplace), "
    "CALL_LOOKUP(profile subject book), "
    "CALL_LOOKUP(profile subject book publication year). "
    "For multi-step questions, retrieve each required fact separately. "
    "Be sure to return answers to all parts of the question."
    "Do not guess profile subject facts. "
    "Output nothing else."
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
                {"role": "user", "content": f"Tool result: {result}"},
            ]
        )

    # If we hit the step limit, ask the model for a final answer.
    messages.append({"role": "system", "content": SYSTEM_PROMPT_FINAL})

    return {
        "task": task,
        "steps": steps,
        "answer": generate_response(),
    }
