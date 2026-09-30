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

To compare all 12 reference fields exactly:

```bash
python tests/check_data.py --input data/processed/input.pkl.gz --strict-hashes
```

This checks `oht`, `TE`, `x_mask`, `y_mask`, `emb_T5b`, `emb_T5u`, `emb_ank`,
`emb_ank3`, `emb_esm2`, `emb_esm2L`, `emb_aacom` and `emb_dipep` against the
hashes in `prepare_input/input_reference.json`. All 12 fields must be present;
array shapes, data types and value bytes must match. Additional fields are not
compared. Generated embeddings can differ across environments, so this is
stricter than schema compatibility.

These checks validate the distributed data and analysis code. They do not rerun
model training or generate protein embeddings.
