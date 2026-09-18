"""Recalculate FigPTR statistics from user-supplied Table EV3 and reference predictions."""
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = ('c50', 'c70', 'c90')


def read_tsv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def load_table(path):
    path = Path(path)
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as archive:
            members = [n for n in archive.namelist() if Path(n).name == 'Table_EV3.tsv' and not n.startswith('__MACOSX/')]
            if len(members) != 1:
                raise ValueError('Archive must contain exactly one Table_EV3.tsv')
            data = archive.read(members[0])
    else:
        data = path.read_bytes()
    reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig')), delimiter='\t')
    fields = reader.fieldnames or []
    tissues = [c[:-4] for c in fields if c.endswith('_PTR')]
    if 'EnsemblTranscriptID' not in fields or len(tissues) != 29:
        raise ValueError('Expected Table EV3 transcript IDs and 29 *_PTR columns')
    values = {}
    for row in reader:
        sid = row['EnsemblTranscriptID'].split('.')[0]
        if not sid or sid in values:
            raise ValueError(f'Duplicate or empty source transcript ID: {sid}')
        parsed = [float(row[t + '_PTR']) if row[t + '_PTR'].strip().upper() not in ('', 'NA') else np.nan for t in tissues]
        if np.isinf(parsed).any():
            raise ValueError(f'Infinite PTR in {sid}')
        values[sid] = np.asarray(parsed, dtype=float)
    return values, tissues, hashlib.sha256(data).hexdigest()


def load_inputs(table_path, predictions):
    ptr, tissues, checksum = load_table(table_path)
    ref = json.loads((ROOT / 'figure_data/figure6/metadata.json').read_text())
    if tissues != ref['ptr_tissues'] or len(ptr) != ref['table_ev3_rows']:
        raise ValueError('Table EV3 tissue order or row count differs from the reference dataset')
    if checksum != ref['table_ev3_sha256']:
        print('NOTE: Table EV3 checksum differs (e.g. line endings); validating IDs and recomputing values.', flush=True)
    cohort = {r['transcript_id_base'] for r in read_tsv(ROOT / 'partitions/metadata/cohort.tsv')}
    matched = int(sum(sid in cohort and np.isfinite(v).any() for sid, v in ptr.items()))
    if matched != ref['matched_cohort']:
        raise ValueError(f'Expected {ref["matched_cohort"]} matched transcripts, found {matched}')
    data = {}
    for threshold in THRESHOLDS:
        rows = read_tsv(Path(predictions) / f'{threshold}.tsv')
        ids = [r['transcript_id'] for r in rows]
        if len(ids) != ref[threshold] or len(set(ids)) != len(ids):
            raise ValueError(f'Unexpected prediction cohort for {threshold}')
        if any(s.split('.')[0] not in cohort or s.split('.')[0] not in ptr for s in ids):
            raise ValueError('Prediction IDs must belong to both the model cohort and Table EV3')
        obs = np.stack([ptr[s.split('.')[0]] for s in ids])
        x0 = np.asarray([float(r['mrna_only_median_te']) for r in rows])
        x1 = np.asarray([float(r['mrna_t5u_median_te']) for r in rows])
        if not np.isfinite(obs).any(axis=1).all() or not np.isfinite([x0, x1]).all():
            raise ValueError('Non-finite predictions or a transcript without observed PTR')
        data[threshold] = (ids, obs, x0, x1)
    return data, tissues, {'table_ev3_sha256': checksum, 'n_source_rows': len(ptr), 'n_matched': matched}


def correlations(y, x):
    keep = np.isfinite(y) & np.isfinite(x)
    if keep.sum() < 2:
        raise ValueError('Fewer than two observations for correlation')
    return int(keep.sum()), float(pearsonr(y[keep], x[keep]).statistic), float(spearmanr(y[keep], x[keep]).statistic)


def write_tsv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def calculate(data, tissues, output, bootstrap=5000, seed=20260826):
    if bootstrap < 2:
        raise ValueError('At least two bootstrap replicates are required')
    membership = defaultdict(dict)
    for r in read_tsv(ROOT / 'partitions/metadata/protein_cluster_membership.tsv'):
        membership[r['threshold']][r['transcript_id']] = r['cluster_id']
    rng = np.random.default_rng(seed)
    primary, tissue_rows, bootrows = [], [], []
    for threshold in THRESHOLDS:
        ids, obs, x0, x1 = data[threshold]
        y = np.nanmedian(obs, axis=1)
        n, r0, rho0 = correlations(y, x0); _, r1, rho1 = correlations(y, x1)
        groups = defaultdict(list)
        for i, sid in enumerate(ids):
            groups[membership[threshold][sid.split('.')[0]]].append(i)
        groups = [np.asarray(v, dtype=int) for v in groups.values()]
        vals = np.empty((bootstrap, 2))
        for b in range(bootstrap):
            ix = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
            vals[b] = [pearsonr(y[ix], x1[ix]).statistic - pearsonr(y[ix], x0[ix]).statistic,
                       spearmanr(y[ix], x1[ix]).statistic - spearmanr(y[ix], x0[ix]).statistic]
            if (b + 1) % 1000 == 0:
                print(f'{threshold}: bootstrap {b+1}/{bootstrap}', flush=True)
        row = {'threshold': threshold, 'n': n, 'mrna_only_pearson': r0, 'mrna_t5u_pearson': r1}
        for j, metric, difference in [(0, 'pearson', r1-r0), (1, 'spearman', rho1-rho0)]:
            lo, hi = np.quantile(vals[:, j], [.025, .975])
            row.update({f'{metric}_difference': difference, f'{metric}_ci95_low': lo, f'{metric}_ci95_high': hi})
            bootrows.append({'threshold': threshold, 'metric': metric, 'n_transcripts': n,
                             'n_clusters': len(groups), 'difference': difference, 'ci95_low': lo,
                             'ci95_high': hi, 'bootstrap_replicates': bootstrap})
        row.update(mrna_only_spearman=rho0, mrna_t5u_spearman=rho1); primary.append(row)
        for j, tissue in enumerate(tissues):
            n, r0, rho0 = correlations(obs[:, j], x0); _, r1, rho1 = correlations(obs[:, j], x1)
            tissue_rows.append({'threshold': threshold, 'tissue': tissue, 'n': n,
                               'mrna_only_pearson': r0, 'mrna_t5u_pearson': r1, 'pearson_difference': r1-r0,
                               'mrna_only_spearman': rho0, 'mrna_t5u_spearman': rho1, 'spearman_difference': rho1-rho0})
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    for name, rows in [('figure_primary_correlations.tsv',primary), ('figure_tissue_differences.tsv',tissue_rows), ('primary_cluster_bootstrap.tsv',bootrows)]:
        write_tsv(output/name, rows)
    return primary, tissue_rows
