references = ["cat", "dog", "bird"]
predictions = ["cat", "dog", "car"]
accuracy = sum(a == b for a,b in zip(references,predictions)) / len(references)
print(round(accuracy, 3))
