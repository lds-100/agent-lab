def calculator(expression):
    try:
        return eval(expression)
    except Exception as e:
        return f"INVALID_CALCULATOR_EXPRESSION: {expression}"


def lookup(topic):
    return f"Information about {topic}"