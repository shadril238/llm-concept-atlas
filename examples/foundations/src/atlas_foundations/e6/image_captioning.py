import json
from pathlib import Path
data = json.loads(Path('src/data/retrieval-results.json').read_text())
for row in data['caption']['rows']:
    print('image:',row['image_id'],'prompt:',repr(row['prompt']))
    print('raw generation:',row['raw'])
    print('output cap:',row['max_new_tokens'])
print('Inspect the actual image before accepting any descriptive detail.')
