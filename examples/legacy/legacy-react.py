tools = {"lookup": lambda item: {"hours": "5 pm"}.get(item, "unknown")}
plan = [("lookup", "hours")]
for tool, argument in plan:
    print(tool, argument, "->", tools[tool](argument))
