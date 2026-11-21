template = "Task: {question}\nReturn one integer."
question = "7 + 8"
rendered = template.format(question=question)
plain = "Task: " + question + "\nReturn one integer."
assert rendered == plain
print(rendered)
