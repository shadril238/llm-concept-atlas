observed = {"song-a", "song-b"}
catalog = ["song-a", "song-b", "song-c", "song-d"]
candidates = [song for song in catalog if song not in observed]
rank = 2
print(
    "eligible:", candidates, "recall@3:", int(rank <= 3), "reciprocal rank:", 1 / rank
)
