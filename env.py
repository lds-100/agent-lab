def calculator(expression):
    try:
        return eval(expression)
    except Exception:
        return f"INVALID_CALCULATOR_EXPRESSION: {expression}"


LOOKUP_DATA = {
    "romeo and juliet author": "William Shakespeare",
    "shakespeare birth year": "1564",
    "shakespeare death year": "1616",
    "einstein birthplace": "Ulm, Germany",
}


def lookup(topic):
    topic = topic.lower().strip()

    if "romeo" in topic and "juliet" in topic:
        return "William Shakespeare"

    if "shakespeare" in topic:
        if "birth" in topic or "born" in topic:
            return "1564"
        if "death" in topic or "died" in topic:
            return "1616"

    if "einstein" in topic and (
        "birthplace" in topic or "born" in topic
    ):
        return "Ulm, Germany"

    return f"No information found for {topic}"
