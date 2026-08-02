def calculator(expression):
    try:
        return eval(expression)
    except Exception as e:
        return f"INVALID_CALCULATOR_EXPRESSION: {expression}"


LOOKUP_DATA = {
    "Romeo_and_Juliet_authors": "William Shakespeare",
    "Albert_Einstein": "Albert Einstein was born in Ulm, Germany.",
    "William_Shakespeare_birthday": "William Shakespeare was born in 1564.",
    "William_Shakespeare_death": "William Shakespeare died in 1616.",
}


def lookup(topic):
    return LOOKUP_DATA.get(
        topic,
        f"No information found for {topic}"
    )