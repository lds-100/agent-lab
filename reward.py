def calculate_reward(chosen_tool, expected_tool):
    if chosen_tool == expected_tool:
        return 1
    return 0


def calculate_multistep_reward(answer_correct):
    return 1 if answer_correct else 0


def calculate_multistep_efficiency_reward(
    answer_correct,
    tool_eval,
):
    # Reward correct answers most heavily, then encourage efficient tool use.
    reward = 1.0 if answer_correct else 0.0

    # Bonus for selecting the correct sequence of tools.
    if tool_eval["tool_selection_correct"]:
        reward += 0.1

    # Bonus for using the correct tool arguments.
    if tool_eval["argument_correct"]:
        reward += 0.1

    # Penalize unnecessary and invalid tool calls.
    reward -= 0.05 * tool_eval["unnecessary_calls"]
    reward -= 0.1 * tool_eval["invalid_calls"]

    return reward