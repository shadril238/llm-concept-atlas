import json
from pathlib import Path
data = json.loads(Path('src/data/retrieval-results.json').read_text())
row = next(r for r in data['grounded']['rows'] if r['id']=='eval-hours' and r['variant']=='routing')
route = row['plan']['trace'][0]
print('actual decision:',route['selected'],'valid:',route['valid'])
print('executed collection:',row['plan']['searches'][0]['collection'])
if not route['valid']:assert row['plan']['searches'][0]['collection']=='all'
print('A successful fallback is not evidence of a correct route.')
