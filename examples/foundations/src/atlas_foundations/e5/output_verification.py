import json

raw = '{"item":"pen","quantity":8,"priority":"urgent"}'
value = json.loads(raw)
valid = (
    set(value) == {"item", "quantity", "priority"} and type(value["quantity"]) is int
)
expected = {"item": "pen", "quantity": 3, "priority": "urgent"}
print("basic shape:", valid, "content correct:", value == expected)
# The connected lab adds enums, range, duplicate-key rejection, and full parsing.
