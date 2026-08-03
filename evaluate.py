from agent import agent
from model import model, tokenizer
from reward import (
    calculate_multistep_efficiency_reward,
    calculate_reward,
)

EVAL_TASKS = [
    ("What is 23 * 17?", "calculator"),
    ("What is 144 / 12?", "calculator"),
    ("Calculate 87 + 156.", "calculator"),
    ("What is 15 squared?", "calculator"),
    ("What is 1000 - 347?", "calculator"),
    ("Tell me about Paris.", "lookup"),
    ("Who was Albert Einstein?", "lookup"),
    ("What is the capital of Japan?", "lookup"),
    ("Tell me about the Amazon rainforest.", "lookup"),
    ("Who wrote Romeo and Juliet?", "lookup"),
]

MULTISTEP_TASKS_CALCULATOR = [
    {
        "task": "What is 23 * 17, and is the result greater than 400?",
        "expected_answer": "False",
        "expected_tools": ["calculator", "calculator"],
        "reference_actions": [
            "CALL_CALCULATOR(23 * 17)",
            "CALL_CALCULATOR(391 > 400)",
        ],
    },
    {
        "task": "What is 100 - 37, and is the result less than 70? Answer only True or False.",
        "expected_answer": "True",
        "expected_tools": ["calculator", "calculator"],
        "reference_actions": [
            "CALL_CALCULATOR(100 - 37)",
            "CALL_CALCULATOR(63 < 70)",
        ],
    },
    {
        "task": "What is 12 * 12, and is the result equal to 144?",
        "expected_answer": "True",
        "expected_tools": ["calculator", "calculator"],
        "reference_actions": [
            "CALL_CALCULATOR(12 * 12)",
            "CALL_CALCULATOR(144 == 144)",
        ],
    },
    {
        "task": "What is 500 / 10, and is the result greater than 40?",
        "expected_answer": "True",
        "expected_tools": ["calculator", "calculator"],
        "reference_actions": [
            "CALL_CALCULATOR(500 / 10)",
            "CALL_CALCULATOR(50.0 > 40)",
        ],
    },
    {
        "task": "What is 15 + 25, and is the result less than 50?",
        "expected_answer": "True",
        "expected_tools": ["calculator", "calculator"],
        "reference_actions": [
            "CALL_CALCULATOR(15 + 25)",
            "CALL_CALCULATOR(40 < 50)",
        ],
    },
]

MULTISTEP_TASKS = [
    {
        "task": "Where was the profile subject born, and what book did they write?",
        "expected_answer": "Portland, Oregon; The Glass Harbor",
        "expected_tools": ["lookup", "lookup"],
        "reference_actions": [
            "CALL_LOOKUP(profile subject birthplace)",
            "CALL_LOOKUP(profile subject book)",
        ],
    },
    {
        "task": "What book did the profile subject write, and when was it published?",
        "expected_answer": "The Glass Harbor; 2015",
        "expected_tools": ["lookup", "lookup"],
        "reference_actions": [
            "CALL_LOOKUP(profile subject book)",
            "CALL_LOOKUP(profile subject book publication year)",
        ],
    },
    {
        "task": "When was the profile subject born, and when did they die?",
        "expected_answer": "1987; 2021",
        "expected_tools": ["lookup", "lookup"],
        "reference_actions": [
            "CALL_LOOKUP(profile subject birth year)",
            "CALL_LOOKUP(profile subject death year)",
        ],
    },
    {
        "task": "When was the profile subject born, when did they die, and how old were they when they died?",
        "expected_answer": "1987; 2021; 34",
        "expected_tools": ["lookup", "lookup", "calculator"],
        "reference_actions": [
            "CALL_LOOKUP(profile subject birth year)",
            "CALL_LOOKUP(profile subject death year)",
            "CALL_CALCULATOR(2021-1987)",
        ],
    },
    {
        "task": "Where was the profile subject born, what book did they write, and when was it published?",
        "expected_answer": "Portland, Oregon; The Glass Harbor; 2015",
        "expected_tools": ["lookup", "lookup", "lookup"],
        "reference_actions": [
            "CALL_LOOKUP(profile subject birthplace)",
            "CALL_LOOKUP(profile subject book)",
            "CALL_LOOKUP(profile subject book publication year)",
        ],
    },
    {
        "task": "What book did the profile subject write, when was it published, when did they die, and how many years before their death was the book published?",
        "expected_answer": "The Glass Harbor; 2015; 2021; 6",
        "expected_tools": ["lookup", "lookup", "lookup", "calculator"],
        "reference_actions": [
            "CALL_LOOKUP(profile subject book)",
            "CALL_LOOKUP(profile subject book publication year)",
            "CALL_LOOKUP(profile subject death year)",
            "CALL_CALCULATOR(2021-2015)",
        ],
    },
    {
        "task": "When was the profile subject born, where were they born, when did they die, and how old were they when they died?",
        "expected_answer": "1987; Portland, Oregon; 2021; 34",
        "expected_tools": ["lookup", "lookup", "lookup", "calculator"],
        "reference_actions": [
            "CALL_LOOKUP(profile subject birth year)",
            "CALL_LOOKUP(profile subject birthplace)",
            "CALL_LOOKUP(profile subject death year)",
            "CALL_CALCULATOR(2021-1987)",
        ],
    },
    {
        "task": "What book did the profile subject write, when was it published, when did they die, and how many years after publication did they die?",
        "expected_answer": "The Glass Harbor; 2015; 2021; 6",
        "expected_tools": ["lookup", "lookup", "lookup", "calculator"],
        "reference_actions": [
            "CALL_LOOKUP(profile subject book)",
            "CALL_LOOKUP(profile subject book publication year)",
            "CALL_LOOKUP(profile subject death year)",
            "CALL_CALCULATOR(2021-2015)",
        ],
    },
    {
        "task": "What is 500 / 10, and is the result greater than 40?",
        "expected_answer": "True",
        "expected_tools": ["calculator", "calculator"],
        "reference_actions": [
            "CALL_CALCULATOR(500 / 10)",
            "CALL_CALCULATOR(50.0 > 40)",
        ],
    },
]


def judge_answer(task, expected_answer, actual_answer):
    """
    Judge whether the agent's final answer correctly answers the task.

    Performs basic normalization/rejection of obvious tool-call outputs,
    then uses Qwen for semantic evaluation.
    """

    actual = actual_answer.strip().lower()

    # Normalize common variations of tool-call names so that:
    # CALL_LOOKUP
    # Call Lookup
    # call_lookup
    # CALL-LOOKUP
    # are all detected as tool calls.
    normalized = actual.replace("_", " ").replace("-", " ")

    # Reject empty answers.
    if not actual:
        return False

    # Reject answers that are actually tool calls rather than answers.
    if "call lookup" in normalized:
        return False

    if "call calculator" in normalized:
        return False

    # Reject obvious tool/environment failures.
    if "no information found" in actual:
        return False

    # Reject obvious refusal/non-answer responses.
    if "i'm sorry" in actual:
        return False

    if "i cannot" in actual or "i can't" in actual:
        return False

    if "please provide more" in actual:
        return False

    judge_prompt = f"""
    TASK:
    {task}

    EXPECTED ANSWER:
    {expected_answer}

    AGENT FINAL ANSWER:
    {actual_answer}

    Does the AGENT FINAL ANSWER correctly answer the TASK?

    Return TRUE only if:
    - Every required fact is present.
    - No required fact is missing.
    - No fact is incorrect or contradictory.
    - The answer is semantically equivalent to the expected answer.

    Never infer missing facts.

    Return exactly one word:

    TRUE

    or

    FALSE

    Do not provide an explanation.
    """.strip()

    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": (
                    "You are a strict binary answer evaluator. "
                    "Compare the expected answer and the agent's final answer. "
                    "Never infer missing facts. "
                    "Return exactly one word: TRUE or FALSE."
                ),
            },
            {
                "role": "user",
                "content": judge_prompt,
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=3,
        do_sample=False,
    )

    judgment = (
        tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )
        .strip()
        .upper()
    )

    if judgment.startswith("TRUE"):
        return True

    if judgment.startswith("FALSE"):
        return False

    # Fail closed if the judge does not return TRUE/FALSE.
    return False


def evaluate_single_step():
    results = []

    correct = 0
    total_reward = 0

    for task, expected_tool in EVAL_TASKS:
        result = agent(task)

        reward = calculate_reward(
            result["tool"],
            expected_tool,
        )

        if result["tool"] == expected_tool:
            correct += 1

        total_reward += reward

        evaluation = {
            "task": task,
            "expected_tool": expected_tool,
            "chosen_tool": result["tool"],
            "result": result,
            "reward": reward,
        }

        results.append(evaluation)

        print("\nTASK:", task)
        print("EXPECTED:", expected_tool)
        print("CHOSEN:", result["tool"])
        print("REWARD:", reward)

    print(f"\nScore: {correct}/{len(EVAL_TASKS)}")
    print(f"Average reward: {total_reward / len(EVAL_TASKS):.2f}")

    return results


def lookup_intent(action):
    action = action.lower()

    if "publication" in action:
        return "profile subject book publication year"
    if "birthplace" in action or "place of birth" in action:
        return "profile subject birthplace"
    if "birth" in action or "born" in action:
        return "profile subject birth year"
    if "death" in action or "died" in action:
        return "profile subject death year"
    if "book" in action or "wrote" in action:
        return "profile subject book"

    return None


def check_argument_correct(actual, expected):
    actual = "".join(actual.split())
    expected = "".join(expected.split())

    if actual.startswith("CALL_CALCULATOR("):
        return actual == expected

    if actual.startswith("CALL_LOOKUP("):
        return lookup_intent(actual) == lookup_intent(expected)

    return False


def evaluate_tool_trajectory(
    steps,
    expected_tools,
    reference_actions=None,
):
    actual_actions = [step["action"].strip() for step in steps]

    actual_tools = []
    invalid_calls = 0

    for action in actual_actions:
        if action.startswith("CALL_CALCULATOR("):
            actual_tools.append("calculator")
        elif action.startswith("CALL_LOOKUP("):
            actual_tools.append("lookup")
        else:
            invalid_calls += 1

    tool_selection_correct = actual_tools == expected_tools

    argument_correct_per_call = []

    if reference_actions is not None:
        argument_correct_per_call = [
            check_argument_correct(actual, expected)
            for actual, expected in zip(
                actual_actions,
                reference_actions,
            )
        ]

    known_argument_results = [
        result for result in argument_correct_per_call if result is not None
    ]

    argument_correct = all(known_argument_results) if known_argument_results else None

    order_correct = True
    expected_index = 0

    for tool in actual_tools:
        if expected_index < len(expected_tools):
            if tool == expected_tools[expected_index]:
                expected_index += 1
            else:
                order_correct = False

    missing_required_steps = max(
        0,
        len(expected_tools) - expected_index,
    )

    unnecessary_calls = max(
        0,
        len(actual_tools) - len(expected_tools),
    )

    return {
        "tool_selection_correct": tool_selection_correct,
        "argument_correct": argument_correct,
        "argument_correct_per_call": argument_correct_per_call,
        "order_correct": order_correct,
        "missing_required_steps": missing_required_steps,
        "invalid_calls": invalid_calls,
        "unnecessary_calls": unnecessary_calls,
    }


def evaluate_multistep_efficiency():
    results = []

    for task in MULTISTEP_TASKS:
        result = agent(task["task"])

        # Use the LLM judge for final-answer correctness.
        answer_correct = judge_answer(
            task["task"],
            task["expected_answer"],
            result["answer"],
        )

        # Evaluate tool-use efficiency separately.
        tool_eval = evaluate_tool_trajectory(
            result["steps"],
            task["expected_tools"],
            task["reference_actions"],
        )

        reward = calculate_multistep_efficiency_reward(
            answer_correct,
            tool_eval,
        )

        evaluation = {
            "task": task["task"],
            "answer": result["answer"],
            "expected_answer": task["expected_answer"],
            "steps": result["steps"],
            "expected_tools": task["expected_tools"],
            "answer_correct": answer_correct,
            "tool_selection_correct": tool_eval["tool_selection_correct"],
            "argument_correct": tool_eval["argument_correct"],
            "order_correct": tool_eval["order_correct"],
            "missing_required_steps": tool_eval["missing_required_steps"],
            "invalid_calls": tool_eval["invalid_calls"],
            "unnecessary_calls": tool_eval["unnecessary_calls"],
            "reward": reward,
        }

        results.append(evaluation)

        print("\nTASK:", task["task"])
        print("STEPS:")

        for i, step in enumerate(result["steps"], 1):
            print(
                i,
                step["action"],
                "→",
                step["observation"],
            )

        print("ANSWER:", result["answer"])
        print("EXPECTED:", task["expected_answer"])
        print("ANSWER CORRECT:", answer_correct)
        print("TOOL SELECTION:", tool_eval["tool_selection_correct"])
        print("ARGUMENT CORRECT:", tool_eval["argument_correct"])
        print("ORDER CORRECT:", tool_eval["order_correct"])
        print("MISSING STEPS:", tool_eval["missing_required_steps"])
        print("INVALID CALLS:", tool_eval["invalid_calls"])
        print("UNNECESSARY CALLS:", tool_eval["unnecessary_calls"])
        print("REWARD:", reward)

    return results
