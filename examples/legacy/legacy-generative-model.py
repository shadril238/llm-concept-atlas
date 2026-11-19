choices = {'table': 0.7, 'floor': 0.2, 'moon': 0.1}
context = 'The cup is on the'
next_token = max(choices, key=choices.get)
print(context + ' ' + next_token)
