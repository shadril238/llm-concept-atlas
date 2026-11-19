# Training environment roadmap

E1 verifies a standard-library foundation environment and one CPU application slice. E2 introduces supported LoRA and NER training slices. Later phases may split embedding, representation, and generation environments when their dependencies conflict. GPU quantization, CUDA, hosted APIs, and training configurations have no verified environment here yet. Each added family needs a resolved lock, data provenance/splits, real execution record, and documented hardware contract.
