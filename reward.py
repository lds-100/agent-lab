def calculate_reward(chosen_tool, expected_tool):
    if chosen_tool == expected_tool:
        return 1
    return 0

def calculate_multistep_reward(answer, expected_answer):
    answer = answer.lower()

    if expected_answer.lower() == "true":
        return 1 if ("yes" in answer or "true" in answer) else 0

    if expected_answer.lower() == "false":
        return 1 if ("no" in answer or "false" in answer) else 0

    return 0