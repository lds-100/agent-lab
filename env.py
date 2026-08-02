def calculator(expression):
    return eval(expression)


def agent(task):
    print("Agent received:", task)


task = "What is 12 * 25?"
result = agent(task)
print(result)