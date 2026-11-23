import json
from pathlib import Path
data = json.loads(Path('src/data/retrieval-results.json').read_text())
row = next(r for r in data['grounded']['rows'] if r['id']=='eval-hop' and r['variant']=='baseline')
print('raw answer:',row['answer_call']['raw'])
for source in row['context']:
    if source['document_id'] in ['wren','birch','aspen']:
        print(source['id'],source['text'])
print('Reference answer is evaluation-only:',row['reference_answer'])
