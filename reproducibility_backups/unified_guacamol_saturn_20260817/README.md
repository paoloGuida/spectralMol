# Unified GuacaMol and SATURN Snapshot

This directory retains compact verification outputs from the final unified
software snapshot. Active launch settings were migrated to the public Python
configuration interface:

- `configs/guacamol_manuscript.toml`
- `configs/guacamol_task_profiles.json`
- `configs/saturn_table8.toml`
- `configs/saturn_table8_v119_environment.json`

Run or inspect the profiles from the repository root:

```bash
python run.py --config configs/guacamol_manuscript.toml
python run.py --config configs/saturn_table8.toml
python run.py --config configs/results.toml
python run.py --config configs/verify.toml
```

The `verification/` directory is historical reference data. The publication
tables are under `reproducibility/manuscript_2026/results/`.
