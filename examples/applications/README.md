# CPU application environment

The supported E1 slice is pretrained sentence embedding inference with PyTorch 2.8.0, Transformers 4.57.1, and NumPy 2.2.6 on Python 3.11. All transitive versions and distribution hashes are in `uv.lock`.

```bash
uv sync --locked --project examples/applications
python3 scripts/run-examples.py --id embedding-retrieval --allow-network
python3 scripts/run-examples.py --id embedding-retrieval --offline
```

Run from repository root. The first run downloads the public Apache-2.0 `sentence-transformers/paraphrase-MiniLM-L3-v2` checkpoint at `4ca70771034acceecb2e72475f72050fcdde4ddc`. It uses about 70 MB of model weights. The package installation is larger and platform-dependent. No authentication is required.

The model's [official card](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2) describes mask-aware mean pooling. This example shows that operation directly, followed by L2 normalization and cosine ranking. It is an inference smoke exercise, with five original sentences and no training. Later phases add full evaluated workflows.

Source: `src/atlas_applications/retrieval.py`. Notebook: `notebooks/embedding-retrieval.ipynb`; use this environment's Python kernel. Records and measured resources: `../verification/records/embedding-retrieval.json`. See the [shared guide](../README.md) for failure handling, artifacts, and reproducibility limits.
