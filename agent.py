import torch

from env import calculator, lookup
from model import model, tokenizer

MAX_NEW_TOKENS = 50
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

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            return_dict_in_generate=True,
            output_scores=True,
        )

        generated_ids = outputs.sequences[:, inputs["input_ids"].shape[-1] :]

        response = tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        ).strip()

        log_prob = torch.tensor(
            0.0,
            device=model.device,
        )

        for token_id, score in zip(
            generated_ids[0],
            outputs.scores,
        ):
            log_probs = torch.log_softmax(
                score[0],
                dim=-1,
            )

            log_prob = log_prob + log_probs[token_id]

        return {"response": response, "log_prob": log_prob}

    # Let the model take up to `max_steps` tool actions.
    for _ in range(max_steps):
        generation = generate_response()
        action = generation["response"]
        log_prob = generation["log_prob"]

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
                "log_prob": log_prob,
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

    final_generation = generate_response()

    return {
        "task": task,
        "steps": steps,
        "answer": final_generation["response"],
        "log_prob": final_generation["log_prob"],
    }


def execute_action(action):
    """
    Execute one tool action.

    Returns:
        tool_name
        observation
        is_final_answer
    """

    if action.startswith("CALL_CALCULATOR("):
        expression = action[len("CALL_CALCULATOR(") : -1]

        result = calculator(expression)

        return (
            "calculator",
            result,
            False,
        )

    if action.startswith("CALL_LOOKUP("):
        topic = action[len("CALL_LOOKUP(") : -1]

        result = lookup(topic)

        return (
            "lookup",
            result,
            False,
        )

    return (
        None,
        None,
        True,
    )


def sequence_log_probability(
    input_ids,
    generated_ids,
):
    """
    Re-run the generated sequence with gradients enabled and
    calculate the log probability of the generated tokens.
    """

    if generated_ids.numel() == 0:
        return torch.tensor(
            0.0,
            device=model.device,
            requires_grad=True,
        )

    full_ids = torch.cat(
        [
            input_ids,
            generated_ids.unsqueeze(0),
        ],
        dim=1,
    )

    outputs = model(
        input_ids=full_ids,
    )

    logits = outputs.logits

    prompt_length = input_ids.shape[1]

    generated_logits = logits[
        :,
        prompt_length - 1 : -1,
        :,
    ]

    log_probs = torch.log_softmax(
        generated_logits,
        dim=-1,
    )

    token_log_probs = log_probs.gather(
        2,
        generated_ids.unsqueeze(0).unsqueeze(-1),
    ).squeeze(-1)

    return token_log_probs.sum()


def generate_action(messages):
    """
    Generate one action without tracking gradients.

    We separately calculate the log probability of the generated
    tokens afterward so REINFORCE can backpropagate through them.
    """

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]

    action = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    return action, generated_ids, inputs["input_ids"]
