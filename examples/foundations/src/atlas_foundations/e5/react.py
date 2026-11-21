observations = []
fixture = {"library_closes": "17:30"}
observations.append({"tool": "lookup", "result": fixture["library_closes"]})
print("actual observation:", observations[-1])
print("next decision input:", observations)
assert observations[-1]["result"] == "17:30"
