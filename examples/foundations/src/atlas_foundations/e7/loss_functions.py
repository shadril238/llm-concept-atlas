import sys
sys.path.insert(0, 'examples/foundations/src')
from atlas_foundations.e7.contracts import cross_entropy, mnr_loss
score = 0.8
print('regression toward one:', round((score - 1) ** 2, 4))
print('regression toward zero:', round(score ** 2, 4))
print('pair classification CE:', round(cross_entropy([2.0, 1.0], 0), 4))
matrix = [[0.8, 0.4], [0.3, 0.9]]
print('directional MNR mean:', round(mnr_loss(matrix, scale=5)['loss'], 4))
print('Different labels and denominators define different objectives.')
