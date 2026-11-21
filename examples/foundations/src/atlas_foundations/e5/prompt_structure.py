parts = {
    "instruction": "Add the two counts.",
    "data": "7 books and 8 books",
    "format": "Return one integer only.",
}
for role, text in [
    ("system", parts["format"]),
    ("user", parts["instruction"] + " Data: " + parts["data"]),
]:
    print(role + ":", text)
assert "7" not in parts["format"]  # task data is separate from the contract
