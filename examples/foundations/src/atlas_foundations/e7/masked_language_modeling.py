import json
from pathlib import Path
data = json.loads(Path('src/data/training-results.json').read_text())['mlm']
trace = data['mask_trace']
for row in trace:
    if row['special_or_padding']:
        assert row['label'] == -100
    if row['label'] != -100:
        assert row['label'] == row['original_id']
print('selected actual targets:', sum(r['label'] != -100 for r in trace))
print('same initial downstream head/pooler:', data['downstream'][0]['head_pooler_initial_sha256'] == data['downstream'][1]['head_pooler_initial_sha256'])
print('two-step smoke / evaluated steps:', data['smoke']['steps'], data['evaluated']['steps'])
