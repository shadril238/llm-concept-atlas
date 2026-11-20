assignments_before = [0, 0, 1, -1]
assignments_after = [0, 0, 1, -1]
representatives = [
    "translation of code switched text",
    "constrained translation decoding",
    "parallel corpus filtering",
]
proposed_name = "Filtering"
assert assignments_before == assignments_after
print("unchanged assignments; review name:", proposed_name)
for document in representatives:
    print("evidence:", document)
