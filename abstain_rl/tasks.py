ABSTAIN_TASKS = [
    {
        "task": """
Context:
Profile subject:
Birthplace: Portland, Oregon
Book: The Glass Harbor

Question:
Where was the profile subject born?
""",
        "expected_action": "ANSWER",
        "expected_answer": "Portland, Oregon",
    },
    {
        "task": """
Context:
Profile subject:
Birthplace: Portland, Oregon
Book: The Glass Harbor

Question:
What book did the profile subject write?
""",
        "expected_action": "ANSWER",
        "expected_answer": "The Glass Harbor",
    },
    {
        "task": """
Context:
Profile subject:
Birth year: 1987
Death year: 2021

Question:
When did the profile subject die?
""",
        "expected_action": "ANSWER",
        "expected_answer": "2021",
    },
    {
        "task": """
Context:
Profile subject:
Birth year: 1987
Death year: 2021

Question:
How old was the profile subject when they died?
""",
        "expected_action": "ANSWER",
        "expected_answer": "34",
    },
    {
        "task": """
Context:
Book:
The Glass Harbor

Publication year:
2015

Question:
When was The Glass Harbor published?
""",
        "expected_action": "ANSWER",
        "expected_answer": "2015",
    },
    {
        "task": """
Context:
Birth year:
1987

Death year:
2021

Question:
Was the profile subject over 30 when they died?
""",
        "expected_action": "ANSWER",
        "expected_answer": "Yes",
    },
    {
        "task": """
Context:
Birth year:
1987

Book publication:
2015

Question:
How old was the profile subject when the book was published?
""",
        "expected_action": "ANSWER",
        "expected_answer": "28",
    },
    {
        "task": """
Context:
Book:
The Glass Harbor

Publication year:
2015

Death year:
2021

Question:
How many years after publication did the profile subject die?
""",
        "expected_action": "ANSWER",
        "expected_answer": "6",
    },
    {
        "task": """
Context:
500 / 10 = 50

Question:
Is the result greater than 40?
""",
        "expected_action": "ANSWER",
        "expected_answer": "Yes",
    },
    {
        "task": """
Context:
Book:
The Glass Harbor

Publication year:
2015

Question:
Was the book published before 2020?
""",
        "expected_action": "ANSWER",
        "expected_answer": "Yes",
    },
    {
        "task": """
Context:
Birthplace:
Portland, Oregon

Question:
What university did the profile subject attend?
""",
        "expected_action": "ABSTAIN",
        "expected_answer": None,
    },
    {
        "task": """
Context:
Book:
The Glass Harbor

Question:
Who illustrated the book cover?
""",
        "expected_action": "ABSTAIN",
        "expected_answer": None,
    },
    {
        "task": """
Context:
Birth year:
1987

Question:
When did the profile subject die?
""",
        "expected_action": "ABSTAIN",
        "expected_answer": None,
    },
    {
        "task": """
Context:
Publication year:
2015

Question:
How many copies were sold worldwide?
""",
        "expected_action": "ABSTAIN",
        "expected_answer": None,
    },
    {
        "task": """
Context:
Birthplace:
Portland, Oregon

Book:
The Glass Harbor

Question:
Did the profile subject grow up in Portland?
""",
        "expected_action": "ABSTAIN",
        "expected_answer": None,
    },
    {
        "task": """
Context:
Book:
The Glass Harbor

Publication year:
2015

Question:
What awards did the book win?
""",
        "expected_action": "ABSTAIN",
        "expected_answer": None,
    },
    {
        "task": """
Context:
Birth year:
1987

Death year:
2021

Question:
What was the cause of the profile subject's death?
""",
        "expected_action": "ABSTAIN",
        "expected_answer": None,
    },
    {
        "task": """
Context:
Birthplace:
Portland, Oregon

Question:
What is the profile subject's phone number?
""",
        "expected_action": "ABSTAIN",
        "expected_answer": None,
    },
    {
        "task": """
Context:
Book:
The Glass Harbor

Question:
How many pages does the book have?
""",
        "expected_action": "ABSTAIN",
        "expected_answer": None,
    },
    {
        "task": """
Context:
Birth year:
1987

Book publication:
2015

Question:
What city was the book written in?
""",
        "expected_action": "ABSTAIN",
        "expected_answer": None,
    },
]

