def retrieve(question): return "Library closes at 5 pm Saturday."
def format_prompt(question, source): return f"Question: {question}\nSource: {source}"
question = "When does the library close?"
print(format_prompt(question, retrieve(question)))
