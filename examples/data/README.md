# Fixture provenance and splits

`fixtures/retrieval.json` contains five original atlas-authored sentences under the repository's MIT license. `manifests/retrieval.json` records the content hash, revision, provenance, and split policy.

It is demonstration-only: no training, fitting, tuning, or held-out quality claim. Three document sentences and two authored queries test the embedding pipeline. Later realistic datasets must add license/source/revision, stable IDs, deduplication/grouping policy, and train/validation/test membership before training. Do not tune on a final test set.
