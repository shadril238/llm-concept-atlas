cache = {'keys': [], 'values': []}
for token in ['cats', 'sleep']:
    key, value = len(token), token.upper()  # invented stand-ins
    cache['keys'].append(key)
    cache['values'].append(value)
    print(token, 'can use', len(cache['keys']), 'stored positions')
