design = [("2 + 3", 5), ("9 - 4", 5)]
for order in [design, list(reversed(design))]:
    print("demonstrations:", order)
    print("evaluation input:", "7 + 8")
assert all(question != "7 + 8" for question, _ in design)
