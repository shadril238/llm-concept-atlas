import json
from pathlib import Path
data = json.loads(Path('src/data/retrieval-results.json').read_text())
row = next(r for r in data['grounded']['rows'] if r['id']=='eval-hours' and r['variant']=='baseline')
for source in row['context']:
    assert source['text'] in row['answer_call']['prompt']
print('actual context IDs:',[s['id'] for s in row['context']])
print('raw draft:',row['answer_call']['raw'])
print('model citations:',row['model_citations'])
print('application display:',row['display_answer'])
