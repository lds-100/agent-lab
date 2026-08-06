from tasks import ABSTAIN_TASKS

def parse_action(action):
    """
    Convert model output into a canonical action.

    Valid outputs:

        ANSWER: Portland, Oregon
        ABSTAIN

    Anything else is invalid.
    """

    action = action.strip()

    if action == "ABSTAIN":
        return {
            "action": "ABSTAIN",
            "answer": None,
        }

    if action.startswith("ANSWER:"):
        answer = action[len("ANSWER:"):].strip()

        if not answer:
            return {
                "action": "INVALID",
                "answer": None,
            }

        return {
            "action": "ANSWER",
            "answer": answer,
        }

    return {
        "action": "INVALID",
        "answer": None,
    }

def evaluate(results):
    results = []

    correct = 0
    total_reward = 0

    for task in ABSTAIN_TASKS:
        # result = agent(task)

        predicted_answer = task["predicted_answer"]
        expected_answer = task["expected_answer"]

        reward = calculate_reward(
            task,
            predicted_answer,
        )

        expected_action = task["expected_action"]

        if expected_action == "ANSWER":
            is_correct = (
                predicted_answer.strip() == expected_answer
            )

        elif expected_action == "ABSTAIN":
            is_correct = (
                predicted_answer.strip().upper()
                == "I DON'T KNOW"
            )

        else:
            raise ValueError(
                f"Unknown expected action: {expected_action}"
            )

        if is_correct:
            correct += 1

        total_reward += reward

        evaluation = {
            "task": task["task"],
            "expected_action": expected_action,
            "expected_answer": expected_answer,
            "predicted_answer": predicted_answer,
            "correct": is_correct,
            "reward": reward,
        }

        results.append(evaluation)

        print("#" * 20)
        print("\nTASK:")
        print(task["task"])
        print("EXPECTED ACTION:", expected_action)
        print("EXPECTED ANSWER:", expected_answer)
        print("MODEL:", predicted_answer)
        print("CORRECT:", is_correct)
        print("REWARD:", reward)

    print(f"\nScore: {correct}/{len(ABSTAIN_TASKS)}")
    print(
        f"Average reward: "
        f"{total_reward / len(ABSTAIN_TASKS):.2f}"
    )

    return results

def calculate_reward(
    expected_action,
    expected_answer,
    predicted_action,
    predicted_answer,
):
    """
    Calculate the RL reward for one model decision.

    Correct answer:       +1
    Correct abstention:   +1
    Wrong answer:         -1
    Abstain when answerable: -0.5
    Invalid output:       -1
    """

    if predicted_action == "INVALID":
        return -1.0

    if expected_action == "ANSWER":

        if predicted_action == "ABSTAIN":
            return -0.5

        if predicted_answer.strip() == expected_answer:
            return 1.0

        return -1.0

    if expected_action == "ABSTAIN":

        if predicted_action == "ABSTAIN":
            return 1.0

        return -1.0

    raise ValueError(
        f"Unknown expected action: {expected_action}"
    )