from agent import agent
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

MULTISTEP_TASKS = [
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

        reward = calculate_multistep_reward(result["answer"], task["expected_answer"])

        evaluation = {
            "task": task["task"],
            "answer": result["answer"],
            "expected_answer": task["expected_answer"],
            "steps": result["steps"],
            "expected_tools": task["expected_tools"],
            "reward": reward,
        }

        results.append(evaluation)

        print("\nTASK:", task["task"])
        print("STEPS:")

        for i, step in enumerate(result["steps"], 1):
            print(i, step["action"], "→", step["observation"])

        print("ANSWER:", result["answer"])
        print("EXPECTED:", task["expected_answer"])
        print("REWARD:", reward)

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
