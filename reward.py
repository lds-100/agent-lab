def calculate_reward(chosen_tool, expected_tool):
    if chosen_tool == expected_tool:
        return 1
    return 0

def calculate_multistep_reward(answer, expected_answer):
    if answer.strip().lower() == expected_answer.strip().lower():
        return 1
    return 0