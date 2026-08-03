def calculator(expression):
    try:
        return eval(expression)
    except Exception:
        return f"INVALID_CALCULATOR_EXPRESSION: {expression}"


LOOKUP_DATA = {
    "profile subject birth year": "1987",
    "profile subject death year": "2021",
    "profile subject birthplace": "Portland, Oregon",
    "profile subject book": "The Glass Harbor",
    "profile subject book publication year": "2015",
}


def lookup(topic):
    topic = topic.lower().strip()

    if "birth year" in topic or "born" in topic:
        return "1987"

    if "death year" in topic or "died" in topic:
        return "2021"

    if "birthplace" in topic or "where" in topic and "born" in topic:
        return "Portland, Oregon"

    if "book" in topic and "publication" in topic:
        return "2015"

    if "book" in topic or "wrote" in topic or "written" in topic:
        return "The Glass Harbor"

    return f"No information found for {topic}"