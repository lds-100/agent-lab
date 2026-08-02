def calculator(expression):
    try:
        return eval(expression)
    except Exception as e:
        return f"INVALID_CALCULATOR_EXPRESSION: {expression}"


LOOKUP_DATA = {
    "romeo and juliet": "William Shakespeare",
    "shakespeare birth": "William Shakespeare was born in 1564.",
    "shakespeare death": "William Shakespeare died in 1616.",
    "albert einstein": "Albert Einstein was born in Ulm, Germany.",
}


def lookup(topic):
    topic = topic.lower().strip()

    for key, value in LOOKUP_DATA.items():
        if key in topic:
            return value

    return f"No information found for {topic}"