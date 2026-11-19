examples = ["a wonderful film", "a dull film"]
labels = ["positive", "negative"]
def classify(text):
    return "positive" if "wonderful" in text else "negative"
print([(text, classify(text)) for text in examples])
