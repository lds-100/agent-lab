def calculate_reward(chosen_tool, expected_tool):
    if chosen_tool == expected_tool:
        return 1
    return 0

def calculate_multistep_reward(answer, expected_answer):
    answer = answer.strip().lower()
    expected_answer = expected_answer.strip().lower()

    if expected_answer == "true":
        return 1 if ("yes" in answer or "true" in answer) else 0

    if expected_answer == "false":
        return 1 if ("no" in answer or "false" in answer) else 0

    return 1 if expected_answer in answer else 0

def calculate_multistep_efficiency_reward(
    answer,
    expected_answer,
    unnecessary_calls=0,
    invalid_calls=0,
):
    reward = calculate_multistep_reward(answer, expected_answer)

    reward -= 0.01 * unnecessary_calls
    reward -= 0.05 * invalid_calls

    return reward