from evaluate import evaluate_tool_trajectory
from evaluate import MULTISTEP_TASKS
from evaluate import judge_answer


EXPECTED_TOOLS = ["calculator", "calculator"]


def test_efficient_trajectory():
    steps = [
        {"action": "CALL_CALCULATOR(12 * 12)"},
        {"action": "CALL_CALCULATOR(144 == 144)"},
    ]

    result = evaluate_tool_trajectory(
        steps,
        EXPECTED_TOOLS,
    )

    assert result["tool_selection_correct"] is True
    assert result["order_correct"] is True
    assert result["missing_required_steps"] == 0
    assert result["invalid_calls"] == 0
    assert result["unnecessary_calls"] == 0


def test_missing_step():
    steps = [
        {"action": "CALL_CALCULATOR(12 * 12)"},
    ]

    result = evaluate_tool_trajectory(
        steps,
        EXPECTED_TOOLS,
    )

    assert result["missing_required_steps"] == 1
    assert result["invalid_calls"] == 0


def test_invalid_call():
    steps = [
        {"action": "CALL_CALCULATOR(12 * 12)"},
        {"action": "CALL_BANANA(144)"},
    ]

    result = evaluate_tool_trajectory(
        steps,
        EXPECTED_TOOLS,
    )

    assert result["invalid_calls"] == 1


def test_unnecessary_calls():
    steps = [
        {"action": "CALL_CALCULATOR(12 * 12)"},
        {"action": "CALL_CALCULATOR(144 == 144)"},
        {"action": "CALL_CALCULATOR(999 + 1)"},
    ]

    result = evaluate_tool_trajectory(
        steps,
        EXPECTED_TOOLS,
    )

    assert result["missing_required_steps"] == 0
    assert result["unnecessary_calls"] == 1


def test_multistep_tasks_are_defined():
    assert len(MULTISTEP_TASKS) == 8

    for task in MULTISTEP_TASKS:
        assert "task" in task
        assert "expected_answer" in task
        assert "expected_tools" in task
        assert len(task["expected_tools"]) >= 2


def test_judge_exact_answer():
    assert judge_answer(
        "What year was the profile subject born?",
        "1987",
        "1987",
    ) is True


def test_judge_sentence_answer():
    assert judge_answer(
        "What year was the profile subject born?",
        "1987",
        "The profile subject was born in 1987.",
    ) is True


def test_judge_wrong_answer():
    assert judge_answer(
        "What year was the profile subject born?",
        "1987",
        "The profile subject was born in 1990.",
    ) is False


def test_judge_missing_fact():
    assert judge_answer(
        "When was the profile subject born and when did they die?",
        "1987; 2021",
        "The profile subject was born in 1987.",
    ) is False