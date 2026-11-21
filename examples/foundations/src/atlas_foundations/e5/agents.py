allowed = {"sum": lambda: 7 + 8, "difference": lambda: 7 - 8}
choice = "sum"
observation = allowed[choice]()
trace = [{"action": choice, "observation": observation}, {"final": str(observation)}]
print(trace)
assert len(trace) <= 3
