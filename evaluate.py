from agent import agent
from reward import calculate_reward

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

def evaluate():
    correct = 0

    for task, expected_tool in EVAL_TASKS:
        result = agent(task)
        reward = calculate_reward(
            result["tool"],
            expected_tool,
        )

        print("Reward:", reward)

        if result["tool"] == expected_tool:
            correct += 1

        print(task)
        print("Expected:", expected_tool)
        print("Chosen:", result["tool"])

    print(f"Score: {correct}/{len(EVAL_TASKS)}")