#!/usr/bin/env python3
"""Plot run-wise Pearson and Spearman performance for the two public analyses."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def panel(ax, rows, group_key, groups, conditions, metric, labels):
    colors = ['#0072B2', '#D55E00', '#009E73']
    names = {'mrna_only': 'mRNA-only', 'mrna_t5u': 'mRNA + T5u', 't5u_only': 'T5u-only'}
    for i, condition in enumerate(conditions):
        for j, group in enumerate(groups):
            subset = [r for r in rows if r[group_key] == group and r['condition'] == condition]
            if len(subset) != 10 or {int(r['seed']) for r in subset} != set(range(10)):
                raise ValueError(f'Expected seeds 0..9 for {condition}/{group}')
            values = np.array([float(r[f'mean_{metric}']) for r in subset])
            if not np.isfinite(values).all():
                raise ValueError('Non-finite run mean')
            x = j + (i - (len(conditions)-1)/2) * .22
            ax.scatter(x + np.linspace(-.045, .045, len(values)), values, s=12,
                       facecolors='white', edgecolors=colors[i], linewidths=.7)
            ax.errorbar(x, values.mean(), yerr=values.std(ddof=1), fmt='o', color=colors[i],
                        capsize=3, label=names[condition] if j == 0 else None)
    ax.set_xticks(range(len(groups)), labels, rotation=25, ha='right')
    ax.set_ylabel(f'Mean tissue {metric.capitalize()} correlation')
    ax.grid(axis='y', alpha=.2)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False, fontsize=8)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True, help='Directory containing both analysis summaries')
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    protein=read(a.results/'protein_cluster_baseline/per_seed_summary.tsv')
    function=read(a.results/'go_slim_holdout/per_seed_summary_final.tsv')
    groups=['GO:0005840','GO:0005886','GO:0005576','GO:0005739','GO:0002376','GO:0003723']
    names=['Ribosome','Plasma membrane','Extracellular region','Mitochondrion','Immune process','RNA binding']
    for metric in ['pearson','spearman']:
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),gridspec_kw={'width_ratios':[1,1.8]},layout='constrained')
        panel(axes[0],protein,'threshold',['c50','c70','c90'],['mrna_only','mrna_t5u','t5u_only'],metric,['c50','c70','c90'])
        panel(axes[1],function,'go_slim_id',groups,['mrna_only','mrna_t5u'],metric,names)
        axes[0].set_title('Protein-cluster partitions')
        axes[1].set_title('Leave-one-function-out with c50 exclusion')
        for ext in ['png','pdf']:
            fig.savefig(a.output/f'benchmarks_{metric}.{ext}',dpi=180)
        plt.close(fig)


if __name__=='__main__':
    main()
