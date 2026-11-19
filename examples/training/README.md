# Training environment roadmap

E3 adds [CPU foundation labs](./reference/E3.md) for generation, pretrained representations, learned playlist embeddings, a transparent transformer, and cache equivalence using the same locked reference environment. Their model downloads run only in the opt-in `e3` workflow job; ordinary CI runs the dependency-light checks.

E1 verifies a standard-library foundation environment and one CPU application slice. E2 now provides [CPU LoRA and NER reference training slices](./reference/README.md), with pinned tokenizers/models, original split fixtures, actual evaluation, and execution records. Later phases may split embedding, representation, and generation environments when their dependencies conflict. GPU quantization, CUDA, hosted APIs, and training configurations have no verified environment here yet. Each added family needs a resolved lock, data provenance/splits, real execution record, and documented hardware contract.
