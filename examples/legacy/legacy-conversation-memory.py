history = [("user", "My name is Sam"), ("assistant", "Hello Sam")]
history.append(("user", "What is my name?"))
print("\n".join(f"{role}: {text}" for role, text in history[-3:]))
