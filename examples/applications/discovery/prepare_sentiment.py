"""Rebuild the attributed UCI subset; no evaluation or model selection occurs here."""

import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[3]
URL = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"
ARCHIVE_SHA = "afc26626d710899948693e1a61405dce197f57ffa719fa1130d346b4cc095343"
TEXT_SHA = "aef2e49e3da25714d61175e3a6e68eeef74a20a2f914318dc3be9947ea86512d"


def canonical(text):
    return re.sub(r"\W+", " ", text.casefold()).strip()


def build(raw):
    assert hashlib.sha256(raw).hexdigest() == ARCHIVE_SHA
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        source = archive.read("sentiment labelled sentences/imdb_labelled.txt")
    assert hashlib.sha256(source).hexdigest() == TEXT_SHA
    unique = {}
    pending = []
    first_line = None
    for line_number, line in enumerate(source.decode("utf8").splitlines(), 1):
        if first_line is None:
            first_line = line_number
        if "\t" not in line:
            pending.append(line)
            continue
        text, label = line.rsplit("\t", 1)
        text = " ".join(pending + [text])
        pending = []
        source_line = first_line
        first_line = None
        text = text.strip()
        key = canonical(text)
        if key in unique:
            if unique[key]["label"] != int(label):
                raise ValueError("Conflicting duplicate labels")
            continue
        unique[key] = {
            "id": f"imdb-{source_line:04d}",
            "text": text,
            "label": int(label),
        }
    if pending:
        raise ValueError("Unlabelled trailing source text")
    rows = []
    for label in [0, 1]:
        ordered = sorted(
            [r for r in unique.values() if r["label"] == label],
            key=lambda r: hashlib.sha256(
                ("atlas-e4-v1:" + r["id"]).encode()
            ).hexdigest(),
        )
        for split, start, stop in [
            ("train", 0, 100),
            ("validation", 100, 120),
            ("test", 120, 160),
        ]:
            rows.extend(
                dict(r, split=split, source="uci-imdb") for r in ordered[start:stop]
            )
    # Original transfer probes; binary gold is an author judgment, not a benchmark truth.
    probes = [
        ("negation", "The film is not boring; its quiet scenes kept me fascinated.", 1),
        ("negation", "I cannot recommend this confusing and joyless movie.", 0),
        (
            "mixed",
            "The actors are wonderful, but the story is so incoherent that I regret watching it.",
            0,
        ),
        (
            "mixed",
            "The opening drags, yet the ending and performances make the whole film worthwhile.",
            1,
        ),
        (
            "domain-shift",
            "The database migration finished perfectly and every record was preserved.",
            1,
        ),
        (
            "domain-shift",
            "The API lost all my saved work and support never replied.",
            0,
        ),
        (
            "domain-shift",
            "This router is remarkably reliable during long video calls.",
            1,
        ),
        (
            "domain-shift",
            "The backup tool reports success but restores only broken files.",
            0,
        ),
        ("sarcasm", "Wonderful, another two hours I will never get back.", 0),
        ("ambiguous", "The film is beautifully photographed and emotionally empty.", 0),
        ("ambiguous", "An uneven film, but I am glad I saw it.", 1),
        ("negation", "It is not a masterpiece, but it is good entertainment.", 1),
    ]
    rows.extend(
        {
            "id": f"probe-{i:02d}",
            "text": text,
            "label": label,
            "split": "challenge",
            "source": "atlas-original",
            "group": group,
        }
        for i, (group, text, label) in enumerate(probes, 1)
    )
    return {
        "id": "uci-imdb-atlas-v1",
        "labels": {"0": "negative", "1": "positive"},
        "rows": rows,
    }


def main():
    raw = urllib.request.urlopen(URL, timeout=60).read()
    fixture = ROOT / "examples/data/fixtures/sentiment.json"
    fixture.write_text(json.dumps(build(raw), indent=2, ensure_ascii=False) + "\n")
    print("Wrote", fixture.relative_to(ROOT))


if __name__ == "__main__":
    main()
