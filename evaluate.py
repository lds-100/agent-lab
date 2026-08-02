from agent import agent

EVAL_TASKS = [
    ("What is 23 * 17?", "calculator"),
    ("What is 144 / 12?", "calculator"),
    ("Tell me about Paris.", "lookup"),
    ("Who was Albert Einstein?", "lookup"),
]

correct = 0

for task, expected_tool in EVAL_TASKS:
    result = agent(task)

    if result["tool"] == expected_tool:
        correct += 1

    print(task)
    print("Expected:", expected_tool)
    print("Chosen:", result["tool"])
    print()

print(f"Score: {correct}/{len(EVAL_TASKS)}")