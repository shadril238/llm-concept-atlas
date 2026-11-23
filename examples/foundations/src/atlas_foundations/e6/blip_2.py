import json
from pathlib import Path
data = json.loads(Path('src/data/retrieval-results.json').read_text())
v = data['caption']
print('genuine checkpoint:',next(iter(v['models'])))
for row in v['rows']:print(row['image_id'],row['trace'])
print('Actual CPU profile checked:',v['cpu_profile_verified'])
print('No BLIP-2 training was performed.')
