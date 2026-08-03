from evaluate import MULTISTEP_TASKS
from env import lookup

def test_profile_subject_facts():
    assert lookup("profile subject birth year") == "1987"
    assert lookup("profile subject death year") == "2021"
    assert lookup("profile subject birthplace") == "Portland, Oregon"
    assert lookup("profile subject book") == "The Glass Harbor"
    assert lookup("profile subject book publication year") == "2015"


def test_multistep_task_chains():
    for task in MULTISTEP_TASKS:
        assert len(task["expected_tools"]) >= 2

        if "calculator" in task["expected_tools"]:
            assert task["expected_tools"][-1] == "calculator"