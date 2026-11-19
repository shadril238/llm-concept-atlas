labels = {"sports": {"goal", "team"}, "food": {"meal", "recipe"}}
text = "the team scored a goal"
words = set(text.split())
scores = {label: len(words & cues) for label, cues in labels.items()}
print(max(scores, key=scores.get), scores)
