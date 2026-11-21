messages = [
    "u1: code LARK-27",
    "a1: noted",
    "u2: Friday, metric",
    "a2: noted",
    "u3: correction Monday",
    "a3: noted",
    "u4: blue paint",
    "a4: noted",
]
print("buffer:", messages)
print("two recent turn pairs:", messages[-4:])
assert "LARK-27" not in " ".join(messages[-4:])
