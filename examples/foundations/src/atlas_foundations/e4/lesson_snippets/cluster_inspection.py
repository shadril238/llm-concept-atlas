inspected = {
    "summary-1": "summarization",
    "summary-2": "summarization",
    "garden": "irrigation",
}
proposed_label = "summarization"
unsupported = [doc for doc, theme in inspected.items() if theme != proposed_label]
print("inspect minority documents:", unsupported)
assert unsupported == ["garden"]
