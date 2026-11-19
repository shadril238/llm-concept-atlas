from math import sqrt
image = (0.8, 0.6)
captions = {"a dog": (0.7, 0.7), "a car": (-0.6, 0.8)}
score = lambda a,b: sum(x*y for x,y in zip(a,b)) / (sqrt(sum(x*x for x in a))*sqrt(sum(y*y for y in b)))
print(max(captions, key=lambda text: score(image, captions[text])))
