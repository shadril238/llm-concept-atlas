import sys
sys.path.insert(0, 'examples/foundations/src')
import json
from pathlib import Path
from atlas_foundations.e6.retrieval import citation_metrics
review = json.loads(Path('examples/data/fixtures/rag-author-review.json').read_text())
row = next(r for r in review['rows'] if r['id']=='eval-hours' and r['variant']=='baseline')
print('Author answer judgment:',row['answer_correct'])
print('Citation metrics:',citation_metrics(row['claims']))
print('Independent review:',review['independent_review'])
