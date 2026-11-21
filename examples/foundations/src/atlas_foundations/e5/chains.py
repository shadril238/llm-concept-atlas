def compute(order):
    prices = {"pen": 2, "book": 8}
    return {**order, "total": prices[order["item"]] * order["quantity"]}


first = {"item": "pen", "quantity": 8}
second_input = compute(first)
print(second_input)
assert second_input["total"] == 16
