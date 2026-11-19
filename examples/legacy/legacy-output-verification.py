allowed = {"positive", "negative"}
candidate = "positive"
print(candidate in allowed)
print("retry or flag" if candidate not in allowed else "accept shape")
