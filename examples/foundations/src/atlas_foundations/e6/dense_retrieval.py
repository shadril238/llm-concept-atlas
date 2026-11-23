import json
from pathlib import Path
data = json.loads(Path('src/data/retrieval-results.json').read_text())
import math
example = data['retrieval']['numerical_example']
query = example['query_vector']
assert abs(sum(x*x for x in query)-1)<1e-5
scores = [(row['id'],sum(a*b for a,b in zip(query,row['vector']))) for row in example['passages']]
print('actual query width:',len(query))
for id,score in sorted(scores,key=lambda row:(-row[1],row[0])):
    print(id,round(score,4))
print('scope:',example['scope'])
