from agent import agent
from model import model, tokenizer
from env import calculator, lookup
from reward import (
    calculate_multistep_efficiency_reward,
    calculate_multistep_reward,
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

MULTISTEP_TASKS_ORIGINAL = [
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
    {
        "task": "Who wrote Romeo and Juliet, and when were they born?",
        "expected_answer": "1564",
        "expected_tools": ["lookup", "lookup"],
        "reference_actions": [
            "CALL_LOOKUP(Romeo_and_Juliet_author)",
            "CALL_LOOKUP(William_Shakespeare_birth_year)",
        ],
    },
    {
        "task": "Who was Albert Einstein, and where were they born?",
        "expected_answer": "Ulm",
        "expected_tools": ["lookup", "lookup"],
        "reference_actions": [
            "CALL_LOOKUP(Albert_Einstein)",
            "CALL_LOOKUP(Albert_Einstein_birthplace)",
        ],
    },
    {
        "task": "When was William Shakespeare born, when did he die, and how old was he when he died?",
        "expected_answer": "52",
        "expected_tools": ["lookup", "lookup", "calculator"],
        "reference_actions": [
            "CALL_LOOKUP(William_Shakespeare_birth_date)",
            "CALL_LOOKUP(William_Shakespeare_death_date)",
            "CALL_CALCULATOR(1616 - 1564)",
        ],
    },
]

MULTISTEP_TASKS_GENERAL = [
    {
        "task": "Who wrote Romeo and Juliet, and what year was William Shakespeare born?",
        "expected_answer": "1564",
        "expected_tools": ["lookup", "lookup"],
    },
    {
        "task": "What year was William Shakespeare born, and what year did he die?",
        "expected_answer": "1616",
        "expected_tools": ["lookup", "lookup"],
    },
    {
        "task": "What year was William Shakespeare born, what year did he die, and how old was he when he died?",
        "expected_answer": "52",
        "expected_tools": ["lookup", "lookup", "calculator"],
    },
    {
        "task": "Who wrote Romeo and Juliet, and where was Albert Einstein born?",
        "expected_answer": "Ulm, Germany",
        "expected_tools": ["lookup", "lookup"],
    },
    {
        "task": "Where was Albert Einstein born, and how many years before William Shakespeare died was Einstein born?",
        "expected_answer": "52",
        "expected_tools": ["lookup", "lookup", "calculator"],
    },
    {
        "task": "Who wrote Romeo and Juliet, and how many years after Shakespeare was born did he die?",
        "expected_answer": "52",
        "expected_tools": ["lookup", "lookup", "calculator"],
    },
    {
        "task": "Where was Albert Einstein born, and what year was William Shakespeare born?",
        "expected_answer": "1564",
        "expected_tools": ["lookup", "lookup"],
    },
    {
        "task": "What year did William Shakespeare die, and how many years after his birth was that?",
        "expected_answer": "52",
        "expected_tools": ["lookup", "lookup", "calculator"],
    },
]

MULTISTEP_TASKS = [
    {
        "task": "Where was the profile subject born, and what book did they write?",
        "expected_answer": "Portland, Oregon; The Glass Harbor",
        "expected_tools": ["lookup", "lookup"],
    },
    {
        "task": "What book did the profile subject write, and when was it published?",
        "expected_answer": "The Glass Harbor; 2015",
        "expected_tools": ["lookup", "lookup"],
    },
    {
        "task": "When was the profile subject born, and when did they die?",
        "expected_answer": "1987; 2021",
        "expected_tools": ["lookup", "lookup"],
    },
    {
        "task": "When was the profile subject born, when did they die, and how old were they when they died?",
        "expected_answer": "1987; 2021; 34",
        "expected_tools": ["lookup", "lookup", "calculator"],
    },
    {
        "task": "Where was the profile subject born, what book did they write, and when was it published?",
        "expected_answer": "Portland, Oregon; The Glass Harbor; 2015",
        "expected_tools": ["lookup", "lookup", "lookup"],
    },
    {
        "task": "What book did the profile subject write, when was it published, when did they die, and how many years before their death was the book published?",
        "expected_answer": "The Glass Harbor; 2015; 2021; 6",
        "expected_tools": ["lookup", "lookup", "lookup", "calculator"],
    },
    {
        "task": "When was the profile subject born, where were they born, when did they die, and how old were they when they died?",
        "expected_answer": "1987; Portland, Oregon; 2021; 34",
        "expected_tools": ["lookup", "lookup", "lookup", "calculator"],
    },
    {
        "task": "What book did the profile subject write, when was it published, when did they die, and how many years after publication did they die?",
        "expected_answer": "The Glass Harbor; 2015; 2021; 6",
        "expected_tools": ["lookup", "lookup", "lookup", "calculator"],
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
    normalized = (
        actual
        .replace("_", " ")
        .replace("-", " ")
    )

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
You are a strict binary evaluator of an AI agent's FINAL ANSWER.

TASK:
{task}

EXPECTED ANSWER:
{expected_answer}

AGENT FINAL ANSWER:
{actual_answer}

Determine whether the AGENT FINAL ANSWER correctly answers the TASK.

Rules:

1. Identify every distinct fact required by the TASK.

2. Every required fact must be present in the AGENT FINAL ANSWER.

3. If even ONE required fact is missing, return FALSE.

4. If even ONE required fact is incorrect, return FALSE.

5. Do not infer missing facts from the TASK.

6. Do not infer missing facts from tool calls or tool results.

7. A tool call is NOT an answer.

8. The agent must actually state the requested information.

9. Full sentences are acceptable.

10. Different formatting is acceptable.

11. Different ordering is acceptable.

12. Semantically equivalent wording is acceptable.

13. Extra correct explanation is acceptable.

14. Extra incorrect or contradictory information makes the answer FALSE.

15. If the answer contains only some of the required facts, return FALSE.

Examples:

EXPECTED:
Portland, Oregon; The Glass Harbor

AGENT:
The Glass Harbor

FALSE

EXPECTED:
Portland, Oregon; The Glass Harbor

AGENT:
The profile subject was born in Portland, Oregon and wrote
The Glass Harbor.

TRUE

EXPECTED:
The Glass Harbor; 2015

AGENT:
The book "The Glass Harbor" was published in 2015.

TRUE

EXPECTED:
1987; 2021

AGENT:
The profile subject was born in 1987.

FALSE

EXPECTED:
1987; 2021; 34

AGENT:
The profile subject was born in 1987 and died in 2021.

FALSE

EXPECTED:
1987; 2021; 34

AGENT:
The profile subject was born in 1987, died in 2021, and was
34 years old when they died.

TRUE

EXPECTED:
The Glass Harbor; 2015; 2021; 6

AGENT:
The Glass Harbor was published in 2015. The profile subject died
in 2021, six years after the book was published.

TRUE

EXPECTED:
The Glass Harbor; 2015; 2021; 6

AGENT:
Call Lookup(profile subject book)_Call Lookup(profile subject death year)

FALSE

Now evaluate the AGENT FINAL ANSWER.

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
                    "Never infer missing facts. "
                    "Return only TRUE or FALSE."
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

    judgment = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip().upper()

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


def evaluate_multistep_correctness():
    results = []

    for task in MULTISTEP_TASKS:
        result = agent(task["task"])

        answer_correct = judge_answer(
            task["task"],
            task["expected_answer"],
            result["answer"],
        )

        evaluation = {
            "task": task["task"],
            "answer": result["answer"],
            "expected_answer": task["expected_answer"],
            "steps": result["steps"],
            "expected_tools": task["expected_tools"],
            "answer_correct": answer_correct,
            "reward": 1 if answer_correct else 0,
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
        print("REWARD:", 1 if answer_correct else 0)

    return results

def normalize_action(action):
    return "".join(action.split())

def check_argument_correct(actual, expected):
    actual = normalize_action(actual)
    expected = normalize_action(expected)

    if actual.startswith("CALL_CALCULATOR("):
        return actual == expected

    if actual.startswith("CALL_LOOKUP("):
        return None

    return False

def evaluate_tool_trajectory(
    steps,
    expected_tools,
    reference_actions=None,
):
    actual_actions = [
        step["action"].strip()
        for step in steps
    ]

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
        result
        for result in argument_correct_per_call
        if result is not None
    ]

    argument_correct = (
        all(known_argument_results)
        if known_argument_results
        else None
    )

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

def evaluate_tool_efficiency(actual_steps, reference_actions):
    actual_actions = [step["action"].strip() for step in actual_steps]

    unnecessary_calls = 0
    invalid_calls = 0
    reference_index = 0

    for action in actual_actions:
        if not (
            action.startswith("CALL_CALCULATOR(") or action.startswith("CALL_LOOKUP(")
        ):
            invalid_calls += 1
            continue

        if reference_index < len(reference_actions):
            expected_action = reference_actions[reference_index]

            if action == expected_action:
                reference_index += 1
            else:
                unnecessary_calls += 1
        else:
            unnecessary_calls += 1

    return {
        "unnecessary_calls": unnecessary_calls,
        "invalid_calls": invalid_calls,
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
        )

        # Base reward comes from the LLM's final-answer judgment.
        reward = 1 if answer_correct else 0

        # Penalize inefficient/invalid tool use.
        reward -= 0.01 * tool_eval["unnecessary_calls"]
        reward -= 0.05 * tool_eval["invalid_calls"]

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