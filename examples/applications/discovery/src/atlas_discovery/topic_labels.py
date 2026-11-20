"""Optional local TextGeneration representation using the fitted L08 artifact."""

import json
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_discovery.common import ROOT, T5, T5_REV, save_result, setup
from atlas_discovery.topics import PinnedBackend

import numpy as np
from bertopic import BERTopic
from bertopic.representation import TextGeneration
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


def main():
    setup()
    source = ROOT / "examples/artifacts/e4-topics"
    if not (source / "model").exists():
        raise ValueError(
            "Run modular-bertopic first; the safe fitted artifact is required"
        )
    recorded = json.loads((source / "results.json").read_text())
    rows = recorded["documents"]
    documents = [r["text"] for r in rows]
    assignments = [r["topic"] for r in rows]
    model = BERTopic.load(str(source / "model"), embedding_model=PinnedBackend())
    tokenizer = AutoTokenizer.from_pretrained(T5, revision=T5_REV)
    generator = AutoModelForSeq2SeqLM.from_pretrained(T5, revision=T5_REV).eval()
    generate = pipeline(
        "text2text-generation", model=generator, tokenizer=tokenizer, device=-1
    )
    prompt = "Give a short research topic name supported by these excerpts. Do not invent a finding.\nExcerpts:\n[DOCUMENTS]\nCandidate terms: [KEYWORDS]\nTopic name:"
    representation = TextGeneration(
        generate,
        prompt=prompt,
        nr_docs=2,
        doc_length=35,
        tokenizer="whitespace",
        random_state=42,
        pipeline_kwargs={"max_new_tokens": 24, "do_sample": False, "truncation": True},
    )
    model.update_topics(
        documents,
        topics=assignments,
        vectorizer_model=CountVectorizer(stop_words="english"),
        ctfidf_model=ClassTfidfTransformer(),
        representation_model=representation,
        top_n_words=5,
    )
    assert np.array_equal(model.topics_, assignments)
    topic_ids = sorted(model.get_topics())
    generated = [
        {
            "topic": topic,
            "label": model.get_topic(topic)[0][0],
            "prompt": actual_prompt,
            "review_status": "author-review-required",
        }
        for topic, actual_prompt in zip(topic_ids, representation.prompts_)
    ]
    assert len(generated) == len(topic_ids)
    result = {
        "parent_artifact_sha256": hashlib.sha256(
            (source / "results.json").read_bytes()
        ).hexdigest(),
        "model": T5,
        "revision": T5_REV,
        "prompt_template": prompt,
        "labels": generated,
        "assignments_unchanged": True,
        "status": "executed-local-model; labels need human support review",
        "limitations": [
            "No new clustering or model-weight fine-tuning occurs.",
            "A short fluent name can ignore minority documents or invent an unsupported focus.",
            "The -1 group contains noise, not one coherent semantic topic; its generated name is not accepted as a topic.",
            "Prompt excerpts are capped at 35 whitespace words per representative document; full source documents remain visible.",
        ],
    }
    save_result("e4-topic-labels", result)
    print(
        json.dumps(
            {
                "assignments_unchanged": True,
                "generated_count": len(generated),
                "labels": [
                    {"topic": r["topic"], "label": r["label"]} for r in generated
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
