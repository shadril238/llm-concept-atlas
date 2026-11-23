import json
from pathlib import Path
data = json.loads(Path('src/data/retrieval-results.json').read_text())
rows = data['chat']['rows']
for i in [0,2]:
    assert rows[i]['raw'] in rows[i+1]['rendered_prompt']
    print(rows[i]['condition'],'pixel shape:',rows[i]['input_shapes']['pixel_values'])
    print('price answer:',rows[i+1]['raw'].strip())
print('The second-turn history also changes with the first generated answer.')
