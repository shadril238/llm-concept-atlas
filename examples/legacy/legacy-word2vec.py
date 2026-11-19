sentence = 'the cat chased the mouse'.split()
window = 1
pairs = [(sentence[i], sentence[j])
         for i in range(len(sentence))
         for j in range(max(0,i-window), min(len(sentence),i+window+1))
         if i != j]
print(pairs[:4])
