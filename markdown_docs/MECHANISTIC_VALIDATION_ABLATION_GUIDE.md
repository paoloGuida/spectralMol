# Mechanistic Validation vs. Performance Check

## Overview

This document explains the distinction between **mechanistic validation** and **performance checks** in the context of SpectralMol, and why mechanistic validation is critical for scientific credibility and publication acceptance.

---

## Quick Summary

| Aspect | Performance Check | Mechanistic Validation |
|--------|------------------|----------------------|
| **Question** | "Does the method work?" | "Why does it work? Which component causes the improvement?" |
| **Example** | 2× more scaffolds than Saturn | Low-freq mutations drive scaffold changes; high-freq enable refinement |
| **Evidence Type** | Benchmark scores, hits, diversity metrics | Ablation studies isolating each mechanism |
| **Publication Impact** | Borderline; raises "why?" questions | Strong; claims are scientifically grounded |

---

## Current State: Performance Check ✅

SpectralMol currently demonstrates:
- ✅ Generates 2× more unique scaffolds than Saturn at −9 kcal/mol
- ✅ Pareto front maintains balance (86 non-dominated molecules, 3.36% of population)
- ✅ Comparable overall benchmark scores to GraphGA (14.7 vs 14.8 on GuacaMol)
- ✅ Competitive QED and drug-likeness metrics

**What this proves:** The method produces good empirical results.

**What it doesn't prove:** *Why* the Fourier parameterization is responsible for those results.

---

## Missing: Mechanistic Validation ❌

### The Core Claim

> "Low-frequency Fourier coefficients induce coherent changes across the sequence (scaffold hopping), while high-frequency coefficients produce localized variations (lead optimization). This frequency-dependent behavior is the key innovation enabling effective multi-objective exploration."

### Why Validation is Needed

Without ablation studies, readers cannot distinguish between:

1. **Hypothesis A (Fourier is key):** 
   - The Fourier structure creates systematic benefits
   - Low-frequency mutations naturally drive scaffold changes
   - High-frequency mutations naturally enable refinement
   - *Removing this structure degrades performance*

2. **Hypothesis B (Fourier is incidental):**
   - NSGA-II + deterministic embeddings would work equally well
   - The improvements come from the evolutionary algorithm, not the representation
   - Fourier structure just adds mathematical complexity without benefit

**Reviewers will assume Hypothesis B until you prove Hypothesis A.**

---

## Mechanistic Ablation Design

### Ablation Conditions

Run 10 seeds for each condition under matched oracle-call budgets (1000 evaluations).

#### **Condition 1: Low-Frequency Only** 
Evolve only the low-frequency Fourier coefficients (constant mode + lowest K/2 harmonics):
- $\Theta \in \mathbb{R}^{(1+K/2) \times D}$ instead of $(1+2K) \times D$
- Prevents high-frequency perturbations
- **Expected outcome:** Fewer but more cohesive scaffolds; slower refinement


#### **Condition 2: High-Frequency Only**
Evolve only the high-frequency Fourier coefficients (highest K/2 harmonics):
- $\Theta \in \mathbb{R}^{(2K - K/2) \times D}$ instead of $(1+2K) \times D$
- No low-frequency modes to drive large-scale changes
- **Expected outcome:** Stuck in local regions; good within-scaffold refinement but poor scaffold exploration


#### **Condition 3: Full-Spectrum** (Baseline)
Current implementation:
- $\Theta \in \mathbb{R}^{33 \times D}$ (1 + 16 cosines + 16 sines)
- **Expected outcome:** Balanced scaffold exploration + refinement

#### **Condition 4: Random Matrix** (Control)
Evolve an unstructured real-valued matrix without Fourier structure:
- $\Theta_{random} \in \mathbb{R}^{33 \times D}$ (same dimensionality)
- No basis projection; direct mutation on latent vectors
- **Expected outcome:** Should underperform full-spectrum if Fourier is beneficial

---

## Metrics to Compare

### Primary Metrics (Per-Seed, 10 seeds each)

| Metric | Significance |
|--------|---|
| **Unique Scaffolds (Modes)** | Low-freq validates scaffold diversity; high-freq tests within-scaffold refinement |
| **Docking Hits at −9 kcal/mol** | Overall optimization effectiveness |
| **Docking Hits at −10 kcal/mol** | Stringent threshold tests mechanism under pressure |
| **QED (avg)** | Drug-likeness maintained; should not degrade |
| **SA (avg)** | Synthetic accessibility; should not degrade |
| **Wall-clock Time (sec)** | Computational efficiency per generation |

### Secondary Metrics (Final Population Analysis)

| Metric | What it reveals |
|--------|---|
| **Pareto Front Size ($F_1$)** | Non-dominated molecules; tests multi-objective capability |
| **Crowding Distance (median)** | Diversity within the Pareto front |
| **Coefficient of Variation (Modes)** | Consistency across seeds; robustness indicator |

---

## Expected Results & Interpretation

### If Hypothesis A is True (Fourier is Key)

| Condition | Expected Profile | Interpretation |
|-----------|-----------------|---|
| Low-freq only | 10–20 modes, excellent docking, good QED | Low frequencies drive scaffold changes effectively ✅ |
| High-freq only | 5–8 modes, poor docking, stuck locally | High frequencies alone cannot explore broadly ✅ |
| Full-spectrum | 70–80 modes, excellent docking, balanced QED | Both frequency bands are synergistic ✅ |
| Random matrix | 30–40 modes, mediocre docking | Unstructured representation underperforms ✅ |

**Conclusion:** Fourier structure is the mechanism. Claim validated. 🎯

---

### If Hypothesis B is True (Fourier is Incidental)

| Condition | Observed Profile |
|-----------|---|
| Low-freq only | 70–80 modes (comparable to full), excellent docking |
| High-freq only | 70–80 modes (comparable to full), excellent docking |
| Full-spectrum | 70–80 modes, excellent docking |
| Random matrix | 70–80 modes, excellent docking |

**Conclusion:** Fourier structure doesn't matter; NSGA-II does the work. Claim invalidated. ❌

---

## Implementation Checklist

### Code Changes Needed

- [ ] Add `--fourier-mode` argument to SpectralMol:
  - `low-only`: K_max = K/2
  - `high-only`: K_min = K/2 + 1
  - `full-spectrum`: K_min = 1, K_max = K (default)
  - `random-matrix`: No Fourier basis; direct latent evolution

- [ ] Modify basis construction in your genotype-to-phenotype mapping

- [ ] Log condition in all outputs for traceability

### Experiment Protocol

- [ ] Run each condition × 10 seeds
- [ ] Use same initialization: ZINC250k seed pool, same 256 molecules
- [ ] Matched budgets: 1000 oracle calls per run
- [ ] ClpP benchmark (Experiment 2 from Guo et al.)
- [ ] Record per-generation best score, scaffold count, hit count
- [ ] Save final population for Pareto analysis

### Output & Visualization

- [ ] **Table:** Summary statistics (mean ± std) across conditions
- [ ] **Figure 1:** Best-score trajectories (mean + 95% CI bands)
- [ ] **Figure 2:** Scaffold count evolution
- [ ] **Figure 3:** Box plots: modes, docking hits, QED, SA across conditions
- [ ] **Supplementary:** Per-seed scatter plots and success rates

---

## Statistical Analysis

### Pairwise Comparisons

For each metric:
- **Parametric:** ANOVA (if normally distributed) + post-hoc Tukey HSD
- **Non-parametric:** Kruskal-Wallis + post-hoc Dunn test

### Effect Sizes

- **Cohen's d:** (full-spectrum − low-freq) / pooled_std
- **Report:** d > 0.8 = large effect, validates mechanism

### Per-Seed Reporting

Always report:
- Mean ± std (across 10 seeds)
- Median (Q50) and IQR (Q25–Q75)
- Min, max (to show range)
- Seed-level success rates

---

## Timeline & Effort Estimate

| Phase | Duration | Effort |
|-------|----------|--------|
| Code modifications | 2–3 hours | Low |
| Ablation runs (40 total) | 3–5 days | Computational |
| Analysis & visualization | 1 day | Low |
| Writing supplementary section | 0.5–1 day | Low |
| **Total** | ~5–7 days | Moderate |

---

## Publication Impact

### Without Ablation
- **Outcome:** Desk reject or major revision (top venues); acceptance (workshops)

### With Ablation
- **Outcome:** Acceptance likely (journals, ML conferences); strong visibility (top venues)

