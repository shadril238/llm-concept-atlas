relevance = {"translations": 0.88, "evaluation": 0.65}
similarity_to_selected_translation = {"translations": 0.98, "evaluation": 0.20}
diversity = 0.3
scores = {
    term: (1 - diversity) * value - diversity * similarity_to_selected_translation[term]
    for term, value in relevance.items()
}
print({term: round(value, 3) for term, value in scores.items()})
assert max(scores, key=scores.get) == "evaluation"
