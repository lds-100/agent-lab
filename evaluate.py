from agent import agent
from reward import calculate_reward, calculate_multistep_reward, calculate_multistep_efficiency_reward

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

MULTISTEP_TASKS = [
    {
        "task": "What is 23 * 17, and is the result greater than 400?",
        "expected_answer": "False",
        "expected_tools": ["calculator", "calculator"],
    },
    {
        "task": "What is 100 - 37, and is the result less than 70? Answer only True or False.",
        "expected_answer": "True",
        "expected_tools": ["calculator", "calculator"],
    },
    {
        "task": "What is 12 * 12, and is the result equal to 144?",
        "expected_answer": "True",
        "expected_tools": ["calculator", "calculator"],
    },
    {
        "task": "What is 500 / 10, and is the result greater than 40?",
        "expected_answer": "True",
        "expected_tools": ["calculator", "calculator"],
    },
    {
        "task": "What is 15 + 25, and is the result less than 50?",
        "expected_answer": "True",
        "expected_tools": ["calculator", "calculator"],
    },
    {
        "task": "Who wrote Romeo and Juliet, and when were they born?",
        "expected_answer": "1564",
        "expected_tools": ["lookup", "lookup"],
    },
    {
        "task": "Who was Albert Einstein, and where were they born?",
        "expected_answer": "Ulm",
        "expected_tools": ["lookup", "lookup"],
    },
    {
    "task": "When was William Shakespeare born, when did he die, and how old was he when he died?",
    "expected_answer": "52",
    "expected_tools": ["lookup", "lookup", "calculator"],
    },
]


def evaluate():
    correct = 0
    total_reward = 0

    for task, expected_tool in EVAL_TASKS:
        result = agent(task)
        reward = calculate_reward(
            result["tool"],
            expected_tool,
        )

        print("Reward:", reward)

        if result["tool"] == expected_tool:
            correct += 1
        
        total_reward += reward
        print(task)
        print("Expected:", expected_tool)
        print("Chosen:", result["tool"])

    print(f"Score: {correct}/{len(EVAL_TASKS)}")
    print(f"Average reward: {total_reward / len(EVAL_TASKS):.2f}")


def evaluate_multistep():
    for task in MULTISTEP_TASKS:
        result = agent(task["task"])

        reward = calculate_multistep_reward(
            result["answer"],
            task["expected_answer"]
        )

        print("\nTASK:", task["task"])
        print("STEPS:")

        for i, step in enumerate(result["steps"], 1):
            print(i, step["action"], "→", step["observation"])

        print("ANSWER:", result["answer"])
        print("EXPECTED:", task["expected_answer"])
        print("REWARD:", reward)


def evaluate_multistep_efficiency():
    for task in MULTISTEP_TASKS:
        result = agent(task["task"])

        reward = calculate_multistep_efficiency_reward(
            result["answer"],
            task["expected_answer"]
        )

        print("\nTASK:", task["task"])
        print("STEPS:")

        for i, step in enumerate(result["steps"], 1):
            print(i, step["action"], "→", step["observation"])

        print("ANSWER:", result["answer"])
        print("EXPECTED:", task["expected_answer"])
        print("EXPECTED TOOLS:", task["expected_tools"])
        print("REWARD:", reward)