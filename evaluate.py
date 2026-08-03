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
    Use Qwen to determine whether the agent's final answer
    correctly answers the task.
    """

    judge_prompt = f"""
You are evaluating the final answer produced by an AI agent.

TASK:
{task}

EXPECTED ANSWER:
{expected_answer}

AGENT ANSWER:
{actual_answer}

Determine whether the AGENT ANSWER correctly answers the TASK.

Rules:
- Judge factual correctness, not exact wording.
- Full sentences are acceptable.
- Different formatting is acceptable.
- The agent does not need to use the same wording as the expected answer.
- All required facts must be present.
- Missing a required fact means FALSE.
- An incorrect fact means FALSE.
- A contradictory fact means FALSE.
- Extra explanation is acceptable if it does not introduce incorrect information.
- Do not judge whether the agent used the correct tools.
- Do not judge the efficiency of the tool calls.

Return exactly one word:
TRUE
or
FALSE
""".strip()

    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": (
                    "You are a strict answer evaluator. "
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
        max_new_tokens=5,
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

    # If the judge fails to follow the format,
    # treat the answer as incorrect.
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


def judge_answer(task, expected_answer, actual_answer):
    """
    Judge whether the agent's final answer contains every required fact.

    Uses simple deterministic rejection for obvious failures, then uses
    the loaded Qwen model for semantic evaluation.
    """

    actual = actual_answer.strip().lower()

    # Obvious failures should not be passed to the LLM judge.
    if not actual:
        return False

    if actual.startswith("call_lookup"):
        return False

    if actual.startswith("call_calculator"):
        return False

    if "no information found" in actual:
        return False

    if "i'm sorry" in actual:
        return False

    if "i cannot" in actual or "i can't" in actual:
        return False

    if "please provide more" in actual:
        return False

    judge_prompt = f"""
You are a strict binary answer verifier.

TASK:
{task}

REQUIRED ANSWER:
{expected_answer}

AGENT ANSWER:
{actual_answer}

Your job is to determine whether the AGENT ANSWER correctly answers the
TASK.

The REQUIRED ANSWER is a checklist of facts that must all be present.

Rules:

1. Identify every distinct required fact in the REQUIRED ANSWER.

2. Every required fact must be explicitly stated in the AGENT ANSWER,
   or expressed using an unambiguous equivalent.

3. If even ONE required fact is missing, return FALSE.

4. If even ONE required fact is incorrect, return FALSE.

5. Do not infer a missing fact from the TASK.

6. Do not infer a missing fact from something the agent says it looked up.

7. A tool call is NOT an answer. The actual tool result must be reflected
   in the AGENT ANSWER.

8. An error message or "no information found" response cannot satisfy a
   required fact.

9. Full sentences are acceptable.

10. Different ordering or formatting is acceptable.

11. Extra correct information is acceptable.

12. Extra incorrect information makes the answer FALSE.

Examples:

REQUIRED ANSWER:
Portland, Oregon; The Glass Harbor

AGENT ANSWER:
The Glass Harbor

RESULT:
FALSE

REQUIRED ANSWER:
Portland, Oregon; The Glass Harbor

AGENT ANSWER:
The profile subject was born in Portland, Oregon and wrote
The Glass Harbor.

RESULT:
TRUE

REQUIRED ANSWER:
1987; 2021

AGENT ANSWER:
The profile subject was born in 1987 and died in 2021.

RESULT:
TRUE

REQUIRED ANSWER:
1987; 2021; 34

AGENT ANSWER:
The profile subject was born in 1987 and died in 2021.

RESULT:
FALSE

REQUIRED ANSWER:
The Glass Harbor; 2015

AGENT ANSWER:
The book "The Glass Harbor" was published in 2015.

RESULT:
TRUE

REQUIRED ANSWER:
1987; 2021; 34

AGENT ANSWER:
CALL_LOOKUP(profile subject death age)

RESULT:
FALSE

Now evaluate the AGENT ANSWER.

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
                    "You are a strict binary verifier. "
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

    return False

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

        tool_eval = evaluate_tool_trajectory(
            result["steps"],
            task["expected_tools"],
        )

        reward = calculate_multistep_efficiency_reward(
            result["answer"],
            task["expected_answer"],
            unnecessary_calls=tool_eval["unnecessary_calls"],
            invalid_calls=tool_eval["invalid_calls"],
        )

        evaluation = {
            "task": task["task"],
            "answer": result["answer"],
            "expected_answer": task["expected_answer"],
            "steps": result["steps"],
            "expected_tools": task["expected_tools"],
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
        print("TOOL SELECTION:", tool_eval["tool_selection_correct"])
        print("ARGUMENT CORRECT:", tool_eval["argument_correct"])
        print("ORDER CORRECT:", tool_eval["order_correct"])
        print("MISSING STEPS:", tool_eval["missing_required_steps"])
        print("INVALID CALLS:", tool_eval["invalid_calls"])
        print("UNNECESSARY CALLS:", tool_eval["unnecessary_calls"])
        print("REWARD:", reward)

    return results
