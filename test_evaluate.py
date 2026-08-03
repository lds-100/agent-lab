from evaluate import evaluate_tool_trajectory
from evaluate import MULTISTEP_TASKS

REFERENCE = [
    "CALL_CALCULATOR(12 * 12)",
    "CALL_CALCULATOR(144 == 144)",
]

EXPECTED_TOOLS = ["calculator", "calculator"]


def test_efficient_trajectory():
    steps = [
        {"action": "CALL_CALCULATOR(12 * 12)"},
        {"action": "CALL_CALCULATOR(144 == 144)"},
    ]

    result = evaluate_tool_trajectory(
        steps,
        EXPECTED_TOOLS,
        REFERENCE,
    )

    assert result["tool_selection_correct"] is True
    assert result["argument_correct"] is True
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
        REFERENCE,
    )

    assert result["missing_required_steps"] == 1
    assert result["invalid_calls"] == 0


def test_wrong_argument():
    steps = [
        {"action": "CALL_CALCULATOR(12 * 12)"},
        {"action": "CALL_CALCULATOR(144 != 144)"},
    ]

    result = evaluate_tool_trajectory(
        steps,
        EXPECTED_TOOLS,
        REFERENCE,
    )

    assert result["argument_correct"] is False
    assert result["argument_correct_per_call"] == [True, False]


def test_invalid_call():
    steps = [
        {"action": "CALL_CALCULATOR(12 * 12)"},
        {"action": "CALL_BANANA(144)"},
    ]

    result = evaluate_tool_trajectory(
        steps,
        EXPECTED_TOOLS,
        REFERENCE,
    )

    assert result["invalid_calls"] == 1

def test_lookup_argument_is_unknown():
    steps = [
        {"action": "CALL_LOOKUP(Albert_Einstein)"},
    ]

    expected_tools = ["lookup"]
    reference = ["CALL_LOOKUP(Albert_Einstein_birthplace)"]

    result = evaluate_tool_trajectory(
        steps,
        expected_tools,
        reference,
    )

    assert result["argument_correct"] is None
    assert result["argument_correct_per_call"] == [None]




def test_multistep_tasks_are_defined():
    assert len(MULTISTEP_TASKS) == 8

    for task in MULTISTEP_TASKS:
        assert "task" in task
        assert "expected_answer" in task
        assert "expected_tools" in task
        assert len(task["expected_tools"]) >= 2