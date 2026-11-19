# Foundation environment

Python 3.11, standard library only. From repository root:

```bash
uv sync --locked --project examples/foundations
python3 scripts/run-examples.py --id attention-softmax
python3 scripts/run-examples.py
```

The default suite runs all registered offline calculations in this interpreter. Legacy exit-success checks certify execution only. The new attention example also checks translation invariance and its numerical answer. See the [shared guide](../README.md) for records and synchronization. Open `notebooks/attention-softmax.ipynb` with this environment's Python kernel, then run all cells.
