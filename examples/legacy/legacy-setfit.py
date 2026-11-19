from math import exp, log
positive_similarity = 0.8
negative_similarities = [0.2, 0.4]
all_scores = [positive_similarity, *negative_similarities]
loss = -log(exp(all_scores[0]) / sum(exp(x) for x in all_scores))
print(round(loss, 3))
