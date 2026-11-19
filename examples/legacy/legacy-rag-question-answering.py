from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

passages = [
    ("hours", "The library closes at 5 p.m. on Saturday."),
    ("loans", "Members may borrow laptops for three days."),
    ("cafe", "The café serves coffee and tea."),
]
question = "When does the library close on Saturday?"

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
document_vectors = embedder.encode([text for _, text in passages], convert_to_tensor=True)
query_vector = embedder.encode(question, convert_to_tensor=True)
scores = util.cos_sim(query_vector, document_vectors)[0]
best_index = int(scores.argmax())
source_id, evidence = passages[best_index]
print("Retrieved:", source_id, evidence)

prompt = (
    "Answer using only the source. If it does not answer the question, say "
    "'I cannot answer from this source.'\n"
    f"Source [{source_id}]: {evidence}\nQuestion: {question}\nAnswer:"
)
generator = pipeline("text2text-generation", model="google/flan-t5-small")
answer = generator(prompt, max_new_tokens=60)[0]["generated_text"]
print("Draft answer:", answer)
print("Source to inspect:", source_id)
