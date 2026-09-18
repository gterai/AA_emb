# The paper's data partitions

A partition specifies which transcripts are used for training, validation and
testing. The files here contain IDs and labels, not sequences or embeddings.
Each `seed_0` to `seed_9` directory contains a lowercase `class.txt` file.

```text
transcript_id train
transcript_id validation
transcript_id test
```

The actual IDs differ on each line. There is no header. Keep the supplied row
order as well as membership when comparing with reference runs.

| Directory | Partition |
| --- | --- |
| `mrna_c80/` | 80% mRNA sequence clustering; primary model comparison |
| `protein_c50/` | 50% protein sequence clustering |
| `protein_c70/` | 70% protein sequence clustering |
| `protein_c90/` | 90% protein sequence clustering |
| `function_holdout/` | One GO functional category reserved for testing |

The cohort contains 9,926 transcripts. The mRNA partition has 9,872 clusters;
protein c50/c70/c90 have 8,624/9,502/9,843 clusters. Complete clusters are assigned
to train/validation/test with target proportions 0.6/0.2/0.2. CD-HIT identity
thresholds do not guarantee a maximum pairwise similarity between partitions.
Protein clustering used CD-HIT 4.8.1, `-G 1 -d 0`, `-n 3` for c50 and `-n 5`
for c70/c90, without an additional minimum alignment-coverage threshold.

For the function partitions, each category is the fixed test set. Other
transcripts in the same c50 clusters as test transcripts are excluded. Remaining
clusters are divided into training and validation with target proportions
0.75/0.25. The categories are ribosome (GO:0005840), plasma membrane
(GO:0005886), extracellular region (GO:0005576), mitochondrion (GO:0005739),
immune system process (GO:0002376) and RNA binding (GO:0003723).

`metadata/` records cohort IDs, cluster membership and selected GO membership.
GO resources use the 2026-06-19 release; Ensembl BioMart annotations were
retrieved on 2026-08-26. GO mapping propagated `is_a`, `part_of`, `regulates`,
`positively_regulates` and `negatively_regulates`. Category membership can overlap.
Upstream sources: [GO](https://geneontology.org/) and
[Ensembl BioMart](https://www.ensembl.org/biomart/martview).

You do not need to regenerate these files. For advanced use, the function
partitions can be reconstructed with:

```bash
python partitions/make_function_partitions.py partitions/metadata/cohort.tsv partitions/metadata/protein_cluster_membership.tsv partitions/metadata/gene_go_slim_membership.tsv outputs/function_partitions
```
