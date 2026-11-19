instruction = "Return one label: positive or negative."
example = "Text: I loved it. Label: positive"
query = "Text: The ending was dull. Label:"
prompt = "\n".join([instruction, example, query])
print(prompt)
