from agent import agent
from reward import calculate_reward, calculate_multistep_reward

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
    ("What is 23 * 17, and is the result greater than 400?", "False"),
    ("What is 100 - 37, and is the result less than 70?", "True"),
    ("What is 12 * 12, and is the result equal to 144?", "True"),
    ("What is 500 / 10, and is the result greater than 40?", "True"),
    ("What is 15 + 25, and is the result less than 50?", "True"),
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
    for task, expected_answer in MULTISTEP_TASKS:
        result = agent(task)

        reward = calculate_multistep_reward(
            result["answer"],
            expected_answer
        )

        print("\nTASK:", task)
        print("STEPS:")

        for i, step in enumerate(result["steps"], 1):
            print(i, step["action"], "→", step["observation"])

        print("ANSWER:", result["answer"])
        print("EXPECTED:", expected_answer)
        print("REWARD:", reward)