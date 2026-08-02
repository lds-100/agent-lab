from env import calculator


def agent(task):
    return calculator("37 * 48")


task = "What is 12 * 25?"
result = agent(task)

print(result)