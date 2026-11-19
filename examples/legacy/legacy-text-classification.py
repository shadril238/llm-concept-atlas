from transformers import pipeline

classifier = pipeline(
    task="text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)
reviews = [
    "The setup was simple and the result was excellent.",
    "The app crashes every time I open it.",
]
for review, prediction in zip(reviews, classifier(reviews)):
    print(review, "=>", prediction["label"], round(prediction["score"], 3))
