import json
from pathlib import Path
data = json.loads(Path('src/data/retrieval-results.json').read_text())
cfg = data['caption']['bridge_config']
q,d,h = cfg['query_count'],cfg['query_hidden'],cfg['language_hidden']
print('learned query values:',q*d)
print('biased projection parameters:',d*h+h)
print('actual query output:',data['caption']['rows'][0]['trace']['query_tokens'])
print('actual projected output:',data['caption']['rows'][0]['trace']['projected_queries'])
