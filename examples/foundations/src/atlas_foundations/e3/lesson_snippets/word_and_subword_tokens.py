text = "cats 👋"
print("whitespace:", text.split())
print("characters:", list(text))
print("UTF-8 bytes:", list(text.encode("utf8")))
assert bytes(text.encode("utf8")).decode("utf8") == text
