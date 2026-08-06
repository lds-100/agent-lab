# import torch

from env import calculator, lookup
# from model import model, tokenizer


MAX_NEW_TOKENS = 50

SYSTEM_PROMPT_BASELINE = (
    "You have access to tools. Choose exactly one action."
)

SYSTEM_PROMPT_TOOLS = (
    "You are an agent that may either call exactly one tool or answer the user's question. "
    "If you still need information, output exactly ONE tool call and NOTHING ELSE. "
    "Never output two tool calls in the same response. "
    "Never combine tool calls with 'and', commas, newlines, or other text. "
    "Never write CALL calculator; the tool name is exactly CALL_CALCULATOR. "
    "Never write CALL lookup; the tool name is exactly CALL_LOOKUP. "
    "For arithmetic, output exactly: CALL_CALCULATOR(expression) "
    "where expression is the arithmetic expression to calculate. "
    "For information about the profile subject, output exactly: CALL_LOOKUP(topic) "
    "where topic is a natural-language lookup topic. "
    "The lookup tool contains the profile subject's birth year, death year, "
    "birthplace, book, and book publication year. "
    "Valid lookup topics include exactly: "
    "CALL_LOOKUP(profile subject birth year), "
    "CALL_LOOKUP(profile subject death year), "
    "CALL_LOOKUP(profile subject birthplace), "
    "CALL_LOOKUP(profile subject book), "
    "CALL_LOOKUP(profile subject book publication year). "
    "Retrieve each required fact separately, using one tool call per response. "
    "Do not guess profile subject facts. "
    "After a tool result, decide whether another fact is required. "
    "If another fact is required, output exactly one more tool call. "
    "Once you have enough information to answer the question, "
    "do NOT call another tool. "
    "Instead, reply with the final answer in natural language. "
    "Your response must be either exactly one valid tool call or one final natural-language answer."
)


SYSTEM_PROMPT_FINAL = (
    "Use the tool result to answer the user's original question. "
    "Give only the final answer. "
    "Be sure to return answers to all parts of the question."
)


def run_episode(
    model,
    tokenizer,
    task,
    max_steps=4,
):
    """
    Run one complete agent episode.

    The model may make tool calls until it produces a final
    answer or reaches the maximum number of steps.

    Returns:
        task
        steps
        answer
    """

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
        )

        generated_ids = (
            outputs.sequences[:, inputs["input_ids"].shape[-1] :]
            if hasattr(outputs, "sequences")
            else outputs[:, inputs["input_ids"].shape[-1] :]
        )

        response = tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        ).strip()

        return response

    for _ in range(max_steps):
        action = generate_response()

        if action.startswith("CALL_CALCULATOR("):
            expression = action[
                len("CALL_CALCULATOR(") : -1
            ]

            result = calculator(expression)
            tool = "calculator"

        elif action.startswith("CALL_LOOKUP("):
            topic = action[
                len("CALL_LOOKUP(") : -1
            ]

            result = lookup(topic)
            tool = "lookup"

        else:
            return {
                "task": task,
                "steps": steps,
                "answer": action,
            }

        steps.append(
            {
                "action": action,
                "tool": tool,
                "observation": result,
            }
        )

        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": action,
                },
                {
                    "role": "user",
                    "content": (
                        f"Tool result: {result}\n\n"
                        "If you have enough information, "
                        "answer the original question. "
                        "Otherwise, make one more tool call."
                    ),
                },
            ]
        )

    messages.append(
        {
            "role": "system",
            "content": SYSTEM_PROMPT_FINAL,
        }
    )

    final_answer = generate_response()

    return {
        "task": task,
        "steps": steps,
        "answer": final_answer,
    }


def agent(task, max_steps=4):
    """
    Backward-compatible wrapper for the existing baseline/training code.

    Uses the model and tokenizer imported from model.py.
    """

    return run_episode(
        model,
        tokenizer,
        task,
        max_steps=max_steps,
    )


def execute_action(action):
    """
    Execute exactly one tool action.

    Returns:
        tool_name
        observation
        is_final_answer
    """

    action = action.strip()

    # Reject multiple tool calls in one action.
    tool_call_count = (
        action.count("CALL_LOOKUP(")
        + action.count("CALL_CALCULATOR(")
    )

    if tool_call_count > 1:
        return (
            None,
            "INVALID_ACTION: multiple tool calls in one action",
            False,
        )

    # Lookup tool
    if action.startswith("CALL_LOOKUP(") and action.endswith(")"):
        topic = action[len("CALL_LOOKUP("):-1].strip()

        if not topic:
            return (
                None,
                "INVALID_ACTION: empty lookup topic",
                False,
            )

        result = lookup(topic)

        return (
            "lookup",
            result,
            False,
        )

    # Calculator tool
    if action.startswith("CALL_CALCULATOR(") and action.endswith(")"):
        expression = action[len("CALL_CALCULATOR("):-1].strip()

        if not expression:
            return (
                None,
                "INVALID_ACTION: empty calculator expression",
                False,
            )

        result = calculator(expression)

        return (
            "calculator",
            result,
            False,
        )

    # Anything else is malformed.
    return (
        None,
        "INVALID_ACTION: malformed action",
        False,
    )



def sequence_log_probability(
    input_ids,
    generated_ids,
):
    """
    Re-run the generated sequence with gradients enabled
    and calculate the log probability of the generated tokens.
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

    Training separately calculates the log probability of the
    generated tokens so REINFORCE can backpropagate through them.
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

    generated_ids = (
        outputs[0][inputs["input_ids"].shape[-1] :]
    )

    action = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    return (
        action,
        generated_ids,
        inputs["input_ids"],
    )