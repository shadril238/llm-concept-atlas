prefix = ['A', 'cat', 'sat', 'on', 'the']
for position, token in enumerate(prefix):
    visible = prefix[:position + 1]
    print(token, '<-', ' '.join(visible))
