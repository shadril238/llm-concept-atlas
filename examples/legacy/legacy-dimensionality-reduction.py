documents = {"A": (0.9, 0.1, 0.3), "B": (0.1, 0.8, 0.4)}
projection = {name: (vector[0], vector[1]) for name, vector in documents.items()}
print(projection)
