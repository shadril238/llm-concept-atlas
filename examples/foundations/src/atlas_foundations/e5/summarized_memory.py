reference = {"code": "LARK-27", "day": "Monday", "units": "metric"}
notes = "Meeting Friday. Please provide a new project code."
for field, value in reference.items():
    print(field, "literal present:", value.casefold() in notes.casefold())
print("stale Friday present:", "friday" in notes.casefold())
