def calculate_reward(chosen_tool, expected_tool):
    if chosen_tool == expected_tool:
        return 1
    return 0


def calculate_multistep_reward(answer_correct):
    return 1 if answer_correct else 0


def calculate_multistep_efficiency_reward(
    answer_correct,
    unnecessary_calls=0,
    invalid_calls=0,
):
    reward = calculate_multistep_reward(answer_correct)

    reward -= 0.01 * unnecessary_calls
    reward -= 0.05 * invalid_calls

    return reward
