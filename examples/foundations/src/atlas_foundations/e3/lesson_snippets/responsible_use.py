source = {"Saturday": "10:00–16:00"}
for day in ["Saturday", "Sunday"]:
    print(day, source.get(day, "No verified source; ask or abstain"))
