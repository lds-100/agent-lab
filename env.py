def calculator(expression):
    try:
        return eval(expression)
    except Exception:
        return f"INVALID_CALCULATOR_EXPRESSION: {expression}"


LOOKUP_DATA = {
    "romeo and juliet": "William Shakespeare",
    "shakespeare birth": "William Shakespeare was born in 1564.",
    "shakespeare death": "William Shakespeare died in 1616.",
    "albert einstein": "Albert Einstein was born in Ulm, Germany.",
}


def lookup(topic):
    topic = topic.lower().strip()

    if "romeo" in topic and "juliet" in topic:
        if "birth" in topic or "born" in topic:
            return "William Shakespeare was born in 1564."
        return "William Shakespeare wrote Romeo and Juliet."

    if "shakespeare" in topic:
        if "birth" in topic or "born" in topic:
            return "William Shakespeare was born in 1564."
        if "death" in topic or "died" in topic:
            return "William Shakespeare died in 1616."
        return "William Shakespeare."

    if "einstein" in topic:
        return "Albert Einstein was born in Ulm, Germany."

    return f"No information found for {topic}"
