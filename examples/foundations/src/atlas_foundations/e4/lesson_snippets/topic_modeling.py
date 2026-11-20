groups = {
    0: ["translation model", "translation errors"],
    1: ["speech model", "speech recognition"],
}
for topic, documents in groups.items():
    tokens = " ".join(documents).split()
    print("class document", topic, ":", tokens)
print(
    "Next stage: class-normalized term counts and corpus weighting, not just common-word counts."
)
