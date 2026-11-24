import sys
sys.path.insert(0, 'examples/foundations/src')
from atlas_foundations.ner_contract import entity_scores
gold = [['B-PER', 'I-PER', 'O', 'B-ORG', 'I-ORG', 'O']]
predicted = [['B-PER', 'I-PER', 'O', 'B-ORG', 'O', 'O']]
result = entity_scores(gold, predicted)
print('token accuracy:', round(sum(a == b for a, b in zip(gold[0], predicted[0])) / 6, 4))
print('whole typed entities:', result)
assert result['tp'] == 1 and result['f1'] == 0.5
