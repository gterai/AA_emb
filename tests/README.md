# Optional verification

These checks are for validating the repository; you do not need to run them
before simply plotting the figures. Run from the repository root:

```bash
pip install -r figures/requirements.txt
python tests/check_data.py
python -m unittest discover -s tests -v
```

Checks cover distributed data hashes, the 9,926-transcript cohort, all 150 fixed
partitions, cluster/function leakage, command construction for primary/protein/
function experiments, Holm correction, and Figure 5/6 input validation.

To check a locally prepared input (requires NumPy):

```bash
python tests/check_data.py --input data/processed/input.pkl.gz
```

`--strict-hashes` additionally checks exact feature bytes. Generated embeddings
can differ across environments, so this is stricter than schema compatibility.

These checks validate the distributed data and analysis code. They do not rerun
model training or generate protein embeddings.
