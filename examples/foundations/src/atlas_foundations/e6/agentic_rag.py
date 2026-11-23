import json
from pathlib import Path
data = json.loads(Path('src/data/retrieval-results.json').read_text())
row = next(r for r in data['grounded']['rows'] if r['id']=='eval-hours' and r['variant']=='agentic')
first = row['plan']['calls'][0]
assert 'finish' not in first['choices']
assert first['raw']==first['choices'][max(range(len(first['choices'])),key=lambda i:first['mean_log_probabilities'][i])]
print('actual finite decision:',first['raw'])
print('searches:',row['search_calls'],'stop:',row['plan']['stop_reason'])
print('scored tokens:',row['scored_tokens'],'generated tokens:',row['output_tokens'])
