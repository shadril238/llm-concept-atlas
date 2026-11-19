"""Optional local Last.fm-1K TSV importer; no download or profile processing."""

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path


def convert(source, limit=200000):
    events = defaultdict(list)
    titles = {}
    with source.open(encoding="utf8") as stream:
        for i, line in enumerate(stream):
            if i >= limit:
                break
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 6:
                raise ValueError(
                    "Expected the six-column listening-event TSV, not profiles"
                )
            user, timestamp, artist_id, artist, track_id, track = fields
            song = (
                track_id
                or hashlib.sha256((artist + "\0" + track).encode()).hexdigest()[:20]
            )
            titles[song] = track + " — " + artist
            events[user].append(
                (datetime.fromisoformat(timestamp.replace("Z", "+00:00")), song)
            )
    playlists = []
    for user, rows in events.items():
        # User identity is used for splitting only; it is not included in output.
        bucket = int(hashlib.sha256(user.encode()).hexdigest()[:8], 16) % 10
        split = "train" if bucket < 8 else "validation" if bucket == 8 else "test"
        session = []
        previous = None
        for stamp, song in sorted(rows):
            if previous is not None and (stamp - previous).total_seconds() > 1800:
                if len(set(session)) >= 4:
                    playlists.append(
                        {
                            "id": f"session-{len(playlists)}",
                            "split": split,
                            "songs": list(dict.fromkeys(session)),
                        }
                    )
                session = []
            session.append(song)
            previous = stamp
        if len(set(session)) >= 4:
            playlists.append(
                {
                    "id": f"session-{len(playlists)}",
                    "split": split,
                    "songs": list(dict.fromkeys(session)),
                }
            )
    return {
        "items": [
            {"id": song, "title": title} for song, title in sorted(titles.items())
        ],
        "playlists": playlists,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tsv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-events", type=int, default=200000)
    args = parser.parse_args()
    if args.max_events <= 0:
        parser.error("max-events must be positive")
    data = convert(args.tsv, args.max_events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(
        json.dumps(
            {
                "sessions": len(data["playlists"]),
                "items": len(data["items"]),
                "source": "https://zenodo.org/records/6090214",
                "license": "Non-commercial use; see source terms. Not covered by repository MIT license.",
            }
        )
    )


if __name__ == "__main__":
    main()
