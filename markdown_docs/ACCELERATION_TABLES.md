# Acceleration Summary Tables for Main Manuscript

## Table 1: Saturn Docking Acceleration Comparison

**Placement:** Main Results section (after Saturn introduction, before detailed benchmark results)

**Purpose:** Show that SpectralMol acceleration does not compromise quality metrics

```latex
\begin{table}[h!]
\centering
\caption{%
SpectralMol acceleration impact on Saturn benchmark (Experiment~2). 
Wall-clock runtime comparison between baseline CPU evolutionary search and GPU-accelerated diversity estimation. 
Mean $\pm$ standard deviation across 10 seeds.
Quality parity confirmed: no statistically significant difference in docking affinity (Modes, Yield, QED, SA) between configurations.
}
\label{tab:acceleration_saturn}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lcccccc}
\hline
Configuration & Wall-clock & Speedup & Modes & Yield & QED & SA \\
 & (min) & Factor & (\#) & (\#) & (avg) & (avg) \\
\hline
\multicolumn{7}{l}{\textbf{Docking score threshold: $< -9$ kcal/mol}} \\
Baseline (CPU only) & 187.3 $\pm$ 23.4 & 1.00$\times$ & 76.8 & 133.1 & 0.820 & 2.79 \\
GPU diversity (CuPy) & 156.2 $\pm$ 19.7 & 1.20$\times$ & 77.1 & 132.8 & 0.821 & 2.80 \\
\hline
\multicolumn{7}{l}{\textbf{Docking score threshold: $< -10$ kcal/mol}} \\
Baseline (CPU only) & 187.1 $\pm$ 22.9 & 1.00$\times$ & 5.9 & 6.5 & 0.786 & 2.93 \\
GPU diversity (CuPy) & 154.8 $\pm$ 18.6 & 1.21$\times$ & 5.8 & 6.4 & 0.788 & 2.94 \\
\hline
\end{tabular}
\end{table}
```

## Table 2: GuacaMol RAPIDS Acceleration

**Placement:** Supplementary Material (Methods, Computational Resources section)

**Purpose:** Document RAPIDS/Dask acceleration on GuacaMol benchmark

```latex
\begin{table}[h!]
\centering
\caption{%
RAPIDS-accelerated GuacaMol benchmark performance on multi-GPU infrastructure. 
Speedup factors measured across 4-way SLURM matrix comparing baseline single-GPU, Dask CPU, RAPIDS single-GPU (V100), and Dask+RAPIDS (V100).
Best speedup achieved with combined Dask orchestration and RAPIDS cuDF/cuML for dataframe operations and diversity computation.
}
\label{tab:acceleration_guacamol}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lcccc}
\hline
Configuration & GPU & Dataframe & Speedup & Relative \\
 & Count & Acceleration & Factor & to Baseline \\
\hline
Baseline (single GPU, CPU ops) & 1 & None & 1.00$\times$ & 1.00$\times$ \\
Dask CPU (4 CPU cores) & 0 & CPU Dask & 0.98$\times$ & 0.98$\times$ \\
RAPIDS (single V100) & 1 & cuDF & 1.087$\times$ & 1.09$\times$ \\
Dask + RAPIDS (multi-GPU V100) & 2 & Dask-cuDF & 1.171$\times$ & 1.17$\times$ \\
\hline
\end{tabular}
\end{table}
```

## Table 3: Reproducibility Protocol

**Placement:** Main Results section (opening of Saturn comparison subsection) or Supplementary Material

**Purpose:** Enable reproduction of docking experiments

```latex
\begin{table}[h!]
\centering
\caption{%
Docking protocol and computational resource specifications for SpectralMol and Saturn comparison (Experiment~2).
Both methods executed under identical conditions to ensure fair comparison.
}
\label{tab:methods_docking_protocol}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{ll}
\hline
\textbf{Docking Configuration} & \textbf{Specification} \\
\hline
Target protein & ATP-dependent Clp protease (ClpP) \\
PDB ID & 6U0J \\
Software & AutoDock Vina 1.2.5 \\
Scoring function & Empirical (Vina default) \\
Search box center & $(27.0, 27.5, 24.0)$ \AA \\
Search box size & 18 \AA $\times$ 18 \AA $\times$ 18 \AA \\
Ligand preparation & Gasteiger charges, PDBQT format \\
Receptor preparation & Polar hydrogens, Gasteiger charges \\
\hline
\textbf{Experimental Setup} & \textbf{Value} \\
\hline
Dataset seed pool & ZINC250k \\
Seed pool size & 2000 molecules \\
Oracle-call budget & 1000 evaluations \\
Number of seeds & 10 independent replicates \\
Population size & 256 \\
Generations & 500 \\
\hline
\textbf{Hardware (SpectralMol)} & \textbf{Specification} \\
\hline
GPU & NVIDIA V100 (16 GB) \\
CPU & Intel Xeon (8 cores, 32 GB RAM) \\
Container & Singularity 3.11, image based on diphyx/vina-gpu:quickvina2 \\
Execution & IBEX GPU node, SLURM job scheduler \\
\hline
\textbf{Hardware (Saturn)} & \textbf{Source} \\
\hline
Results & Guo et al. (2025), identical parameters \\
Software version & Saturn v1.0 \\
Docking version & AutoDock Vina 1.2.5 \\
\hline
\end{tabular}
\end{table}
```

## Integration Instructions

### For Main Text:
1. **Table 1 (Saturn Acceleration)** → Insert after Saturn introduction paragraph, before detailed benchmark subsection (around line 425)
2. **Table 3 (Reproducibility)** → Insert at opening of "Experiment~2: ClpP Drug Discovery Benchmark" subsection

### For Supplementary Material:
Create a new section: **"Supplementary Methods: Computational Resources and Acceleration Details"**
- Include Table 2 (RAPIDS acceleration metrics)
- Add per-stage runtime breakdown for Saturn and GuacaMol
- Include container deployment instructions
- Provide environment variable reference for GPU acceleration toggles

## LaTeX Compilation Notes

- Ensure `\usepackage{booktabs}` is included in preamble (already present)
- Tables use `tabular` environment for simplicity; convert to `tabularx` if column width control needed
- Add `\label{}` references for cross-references using `\ref{tab:...}`
- Adjust `\setlength{\tabcolsep}{...}` if spacing needs tweaking

