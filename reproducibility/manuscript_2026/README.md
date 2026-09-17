# Manuscript Reproducibility Package

This directory contains the exact compact inputs, reference outputs,
provenance, and analysis scripts used for the reported GuacaMol, SATURN, and
frequency-mode experiments.

## Scientific Invariants

- The evolved genotype is theta.
- GuacaMol phenotype proposals and BRICS genotype operations are disabled.
- SATURN uses theta-only NSGA-II and jointly evaluates docking, QED, and
  synthetic accessibility.
- Parent-lineage fields are observability metadata, not evolved variables.

## Reproduction Commands

Run from the repository root:

```bash
python run.py --config configs/guacamol_manuscript.toml
python run.py --config configs/saturn_table8.toml
python run.py --config configs/frequency_ablation.toml
```

Print bundled headline tables or validate every packaged checksum:

```bash
python run.py --config configs/results.toml
python run.py --config configs/verify.toml
```

The complete GuacaMol and SATURN runs are computationally expensive. External
docking and comparison dependencies are documented in the repository README.

## Reported Results

### GuacaMol, 10 matched seeds (7-16)

| Method | Aggregate mean +/- SD | Seed wins |
|---|---:|---:|
| SpectralMol | 15.230486 +/- 0.176493 | 10/10 |
| GraphGA | 14.574242 +/- 0.256197 | 0/10 |
| Paired difference | +0.656244 +/- 0.295991 | 10/10 positive |

Paired sign test: `p=9.8e-4`.

### SATURN Table 8, 1,000 oracle calls, 10 seeds

| Docking threshold | Replicates | Modes | Yield | QED | SA | MolWt |
|---|---:|---:|---:|---:|---:|---:|
| < -9 kcal/mol | 10/10 | 94.9 +/- 11.5 | 316.2 +/- 38.1 | 0.85 +/- 0.01 | 2.66 +/- 0.04 | 312.4 +/- 5.2 |
| < -10 kcal/mol | 10/10 | 8.1 +/- 3.0 | 12.9 +/- 6.9 | 0.84 +/- 0.02 | 2.62 +/- 0.12 | 309.6 +/- 7.1 |

### Frequency-Mode Ablation

| Condition | Mean best score | Delta vs full | Speedup |
|---|---:|---:|---:|
| low-only | 0.730258 | +0.019037 | 1.046x |
| random-matrix | 0.725250 | +0.014029 | 1.036x |
| full-spectrum | 0.711221 | 0 | 1.000x |
| high-only | 0.664649 | -0.046572 | 1.154x |

The parent-lineage outputs support a cautious interpretation: frequency
restrictions bias parent-child molecular changes, but no frequency band maps
universally to one chemical operation.

## Contents

- `inputs/guacamol`: shared GuacaMol population and compressed ablation source.
- `inputs/saturn`: ten initial populations and docking structures.
- `results/guacamol`: aggregate, per-seed, per-task, trajectory, and top-200 tables.
- `results/saturn`: Table 8, molecule, Pareto, crowding, and figure source tables.
- `results/frequency_ablation`: score, runtime, feature, and lineage analyses.
- `scripts`: manuscript table and figure generators.
- `provenance`: package versions, historical run records, and checksums.
