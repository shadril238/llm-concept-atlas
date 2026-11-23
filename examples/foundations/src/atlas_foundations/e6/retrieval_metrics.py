import sys
sys.path.insert(0, 'examples/foundations/src')
from atlas_foundations.e6.retrieval import metrics
score = metrics(['A','B','C','D'],{'A':2,'B':0,'C':1,'D':2},3)
assert abs(score['ap']-5/9)<1e-12
print('AP terms:',score['ap_terms'],'all relevant:',score['relevant_documents'])
print('P@3:',round(score['precision'],4),'AP@3:',round(score['ap'],4),'nDCG@3:',round(score['ndcg'],4))
print('Absent-query AP:',metrics(['A'],{},3)['ap'])
