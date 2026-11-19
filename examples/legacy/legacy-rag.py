question = 'When does the library close on Saturday?'
passages = {'hours': 'The library closes at 5 p.m. on Saturday.',
            'cafe': 'The café serves coffee and tea.'}
selected = ['hours']
answer = passages['hours'] + ' [hours]' if 'hours' in selected else 'No supported answer.'
print(answer)
