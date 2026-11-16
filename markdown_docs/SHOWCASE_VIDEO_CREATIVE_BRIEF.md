# SpectralMol Wall-Loop — "Frequency Becomes Molecule"

Silent, looping scientific-visualization installation. Companion to `Draft_EvolutionaryAlgorithm.pdf`.
Prepared 2026-11-15. Written for: the author (ablation + acceleration contributor) and whoever builds the PowerPoint/video.

---

## 0. Read this first — five facts that constrain the visuals

I read the LaTeX source (`overleaf/templatePRIME.tex`, `templateArxiv.tex`) and the code, not the PDF itself (no PDF parser on this node), so figure numbers may differ from your PDF. I could not open the GitHub repo; I used the local `core/`, which holds the same primitives.

1. **The Θ→Z→token→molecule pipeline is real and drawable.** `core/fourier_theta.py` (`build_fourier_basis`, `theta_to_Z`, `mutate_Theta`), `core/embedding.py` (`build_structured_E`), `core/decoder.py` (`compute_dist2`, `dist2_to_ids_sample`, `ids_to_selfies`, `decode_theta_best_of_k`), plus `visualizePipeline.py`, which already does Θ→Z→distance→token for aspirin. This is the hero content.
2. **The GuacaMol "frequency-mode ablation" is NOT a Fourier-space ablation.** `benchmarks/Guacamol/evolve_vs_molscore_benchmark.py:915-935` says so in its own comment: *"The active evolution path is SMILES-operator-based, not latent-Fourier-native."* `low-only`, `high-only` and `random-matrix` are presets of immigrant fraction, parent-pool fraction, elite fraction and mutation step count. The video must not show measured wave spectra for those runs. Scenes about the ablation should be labelled **"conceptual"** in a small caption, or you should first run a real Θ-space ablation (§5, item R1).
3. **Acceleration is modest (1.05×–1.17×), with identical quality** (0.7522 mean best score in all four configs; Dask+RAPIDS V100 = −14.6 % wall-clock). Visuals must show **same destination, slightly earlier arrival, wider front** — never a dramatic 10× race. Exaggerating it would be visible to any scientist who knows the number. The honest hook is *width of exploration per unit time*, which is true by construction of Dask/GPU parallelism.

4. **The real embedding table E is a small constellation, not ~100 stars** *(found while building G3)*. With the real vocabulary (178 tokens incl. macros) and `build_structured_E`, only **29 rows are unique** (18 distinct positions in a 2D PCA). Charged variants (`[C]`/`[C+1]`) and all `[Branch*]` tokens have *identical* embeddings, so pure nearest-token cannot separate them; the decoder resolves ties through validity masks/state. Scenes 2 and F4/F5/G3 should therefore show a **29-star constellation** (not ~100), of which aspirin lights only 6. PCA collapses many of these onto each other, so G3 uses a 3D metric-MDS layout (all 29 separated, min spacing 0.52) plus a slow camera sway. The "fog masks" metaphor is not decoration: it is what disambiguates tied tokens. A Θ→aspirin genotype reconstructs Z exactly (error ~1e-15), so all 19 slots lock once ties are respected.

5. **Acceleration = the same search, shorter generation steps, not a wider front** *(found while building G10)*. For a given seed the best-so-far trajectory is bit-identical across baseline CPU / Dask CPU / RAPIDS V100 / Dask+RAPIDS V100, and per-task speedups are 1.12–1.23× (median 1.17×). Concepts B2 (swarm), B3 (expanding fronts) and B4 (fan-out) in §3B, and the "many simultaneous exploration paths" phrasing, imply extra parallel paths that the code does not run; treat them as **metaphor only, never as data**. G10 (real wall-clock race on one shared lineage) is the faithful version and replaces G11.

Also note for the ablation story: `high-only` scored **+0.005** over full-spectrum, `low-only` **−0.012**, `random-matrix` **−0.102** (6 seeds × 20 tasks). Only random-matrix is a clear effect. The "low = scaffold, high = refinement" metaphor is the paper's hypothesis (`MECHANISTIC_VALIDATION_ABLATION_GUIDE.md` itself calls it unvalidated). Use it as a *design intuition*, not as a result. The one claim the data supports: **structure matters; unstructured randomness collapses.**

---

## 1. Creative Direction

**Title (never displayed as text):** *Frequency Becomes Molecule.*

**Look:** a night-ocean planetarium. Near-black deep-teal background (#05101A), light as the only material. Waves are luminous ribbons, never axes. Molecules are drawn as thin glowing line-art (RDKit SVG, white-cyan strokes, atom colours only for N/O/S/halogens). One accent per idea:

| Idea | Colour |
|---|---|
| Low frequency (broad) | deep blue → indigo |
| High frequency (fine) | amber → gold |
| Selected / elite / Pareto | white-hot |
| Random / noise | desaturated grey static |
| GPU/parallel | cyan |
| CPU/serial | warm grey |

**Motion language:** everything breathes in periodic motion (sines), because the subject is Fourier. All wave motion uses integer harmonics of the loop length, so the loop is mathematically seamless, and the video *is* a demonstration of the idea. Cuts are replaced by **continuous transformations** (a ribbon becomes a path becomes a bond). No text longer than 4 words; no numbers except possibly a single "1.17×" ghosted in scene 8 (optional).

**Standalone rule:** every scene opens with a 1.5 s "legend glance": a fixed 3-icon strip at the bottom (three drawn glyphs: wave → constellation → molecule) with the current stage lit. A visitor arriving mid-loop sees where they are in the chain.

**Metaphors that read in 3 seconds (no notation):**

| Symbol | Visual metaphor | Why it reads |
|---|---|---|
| Genotype Θ | A **mixing desk**: a bank of vertical sliders, one per frequency, glowing amber (high) → blue (low), each slider is a stack of a few thin channels | Anyone knows "turn the knobs → different sound" |
| Fourier basis Φ | A **fixed set of glass rulers/tuning forks** — a row of standing sine waves that never move or change | Constant = "the instrument"; Θ = "the performance" |
| Phenotype Z | The **ribbon field**: the sum of all waves laid out along a line of slots (positions) | Θ played through Φ gives one flowing curve per channel |
| Embedding E | A **constellation chart** of ~100 token-stars grouped by chemical family (aromatic atoms, ring closers, branches) in clusters | Neighbourhoods = chemical similarity |
| Distance field d(i,v) | A **gravity well / glow radius**: each slot's point drifts and the nearest star ignites; neighbours get a dimmer halo = probability | "Closest star wins" |
| Constraint masks | **Fog**: invalid stars are dimmed/greyed at that moment | Shows "can't pick that now" without words |
| NSGA-II | A **glowing frontier curve** sweeping outward; individuals on it turn white-hot, dominated ones fade; crowding = the spacing of lights on the frontier | The frontier is the story, not the sorting |
| Crossover | Two ribbon bundles **braid** and split into two children | Universal |
| Mutation | A single slider **nudges**; the ribbon ripples downstream | Cause-and-effect locality |

---

## 2. Storyboard — 75 s loop (30 fps, 1920×1080)

Seamless: scene 9 resolves into the frame of scene 0. Total = 75 s.

| # | Time | Scene | Visual objective | Motion language | Transition out |
|---|---|---|---|---|---|
| 0 | 0–6 | **Spectrum** | 8 coloured sine ribbons (cos/sin modes, k=1…4) drift, their sum forms a rolling landscape | Slow integer-harmonic phase drift | Landscape's crests rise and thin into a row of sliders |
| 1 | 6–14 | **The Desk** | Mixing desk (Θ) × fixed glass rulers (Φ) → ribbon field (Z). One slider moves → whole ribbon reshapes smoothly | Sliders ease in/out; ribbon follows with 0.2 s lag | Ribbon is sliced into ~20 slots; each slot's tip becomes a drifting point |
| 2 | 14–24 | **Nearest Star** | Each slot's point glides through the token constellation; nearest star ignites; probability halo; fog masks drop in/out | Points drift on curved paths, lock with a soft "settle" bounce | Ignited stars unspool into a bead chain |
| 3 | 24–32 | **Crystallise** | Beads (SELFIES) slide together; branch beads fork; ring-closure bead snaps a ring shut; chain morphs into the 2D molecule | Elastic snap on ring closure only (one accent motion) | Molecule shrinks to a thumbnail "card" bottom-centre |
| 4 | 32–44 | **Frequency Brush** | Two brushes over a canvas. Wide blue brush (low-k) moves the whole scaffold: ring appears/vanishes. Fine amber brush (high-k) swaps one substituent. Θ desk in a corner mirrors which sliders move | Wide = slow sweeping arcs; fine = quick short flicks | Brush trails dissolve into many small ribbon bundles = a population |
| 5 | 44–54 | **Evolution** | ~40 ribbon bundles braid (crossover), ripple (mutation); each decodes to a faint molecule; a white frontier sweeps; survivors brighten; the best molecule's drawing sharpens each generation | Generation = 1.2 s heartbeat; frontier advances on the beat | Frontier curve straightens into a ring → panel grid |
| 6 | 54–62 | **Which frequencies?** *(conceptual caption)* | 2×2 wave-fields: Full (rich, balanced), High-only (only fine ripples, tidy), Low-only (smooth, bulky), Random (grey static, nothing converges). Each feeds the same decoder; molecule complexity "blooms" under Full/High, stalls under Random | All four share one clock; Random jitters, others flow | Panels merge into one landscape → zoom out |
| 7 | 62–70 | **Parallel Exploration** | Split screen over one chemical-space galaxy: left, one warm-grey comet probes; right, a cyan swarm of many probes fans out. Both reach the same bright cluster (target family). Cyan front arrives a bit earlier and wider | Comet = single smooth line; swarm = synchronized pulses | Both streams converge into a single star |
| 8 | 70–75 | **Galaxy** | Zoom out: millions of stars, the target family glowing, a faint Pareto arc of molecule line-art | Slow dolly-out | The galaxy's spiral arms flatten into the sine ribbons of scene 0 |

Optional last frame hold (0.5 s): a small ghosted caption "same destination · sooner" — only if you want one line of words.

---

## 3. Visualisation alternatives (as requested)

### 3A. Frequency-mode ablation without plots (≥5; all conceptual unless R1 is run)

| # | Concept | How it shows Full / High-only / Low-only / Random |
|---|---|---|
| A1 | **Four ocean panels** | Same sea, different wave content: full = swells + ripples; high = glassy surface with fine chop; low = big slow swells, no detail; random = chaotic uncorrelated peaks. A boat (best molecule so far) navigates; only structured seas let it reach the lighthouse |
| A2 | **Harmonic decomposition peel** | A rich waveform is split into its sine components; frequencies are removed (low or high) and the reconstructed curve and the decoded molecule update live. Shows *what* each condition keeps |
| A3 | **Molecular complexity bloom** | A single molecule per condition grows atom-by-atom over generations (real lineage from run CSVs). Random never forms rings; Full and High form fused/ring structures |
| A4 | **Spectral energy aurora** | Θ's per-frequency energy as a curtain of light (x = frequency, brightness = energy, y = generation). Full: broad glow; High: only the amber edge; Low: only the blue edge; Random: flat white noise. Needs R1 data |
| A5 | **Search-footprint constellations** | Each condition's explored molecules as dots in a 2D chemical embedding (t-SNE/UMAP of Morgan fingerprints). Full: wide structured cloud with bright top; Random: wide but dim, no bright region |
| A6 | **Decoder radio dial** | Four radios tuned to the same station: full = clean, low = muffled, high = tinny, random = static; the signal is the decoded molecule sharpening or not |
| A7 | **Orchestra** | Four ensembles playing the same score: full orchestra, only piccolos, only basses, random notes. Waveform of each; *the tune (molecule) is only recognisable in the first three* |

Recommended trio for the video: **A1 + A3 + A2** (A1 as the scene 6 frame, A3 as the progress, A2 as a standalone GIF).

### 3B. Acceleration without charts (≥5; "same destination, faster exploration")

| # | Concept | Mechanism |
|---|---|---|
| B1 | **Branching path growth** | Four root-to-canopy trees grown on a shared map, all ending at the same glowing region. CPU = a single trunk with few branches; Dask = a few more; RAPIDS = a dense bush; Dask+RAPIDS = widest canopy. Growth speed identical per branch — *more branches at the same time* |
| B2 | **Synchronized particle swarms** | N particles per config diffusing toward an attractor (the target family). Same attractor, different particle counts. All reach it; the wider swarm gets there marginally earlier |
| B3 | **Expanding search fronts** | Radial wavefronts from one start point in chemical space; each config = one ring front with a different thickness; same covered area |
| B4 | **Parallel exploration trees** | CPU = a depth-first single path; GPU = a breadth-first fan-out. Both converge into one Pareto-family cluster |
| B5 | **Lanes on a road** | Four roads from A to B with 1, 2, 4, 8 lanes of luminous cars (batches); all cars the same speed; wider road → the whole column arrives earlier |
| B6 | **Many-hands assembly** | One worker places beads, many workers place beads, all produce the same molecule, which assembles in ¼ of the time-slice — echoing scene 3 |
| B7 | **Orchestrated fireflies** | Dask orchestrates: fireflies flash in staggered bursts (scheduler), GPUs flash as dense synchronized pulses (batched distance computation) |

Use **B1 (hero) + B2 (GIF) + B5 (fallback for non-experts)**. Anchor honesty: measured gains are only 1.05–1.17×, so keep the width differences between configs small and qualitative, and do not label lanes or branches with numbers.

---

## 4. New Figure Proposals (12, none in the paper)

| ID | Figure | Purpose | Data source | Generation method | Style |
|---|---|---|---|---|---|
| F1 | **Frequency Landscape** | Show Θ as a terrain built from modes; evolution reshapes it | `build_fourier_basis(L,K)`, any evolved Θ | numpy surface z = Σ θ_m φ_m(i) over (position × channel) → matplotlib `plot_surface` or Blender displacement mesh | Dark ocean, blue→gold height colouring |
| F2 | **Fourier Mode Orchestra** | Show each basis function as a sine/cosine ribbon plus their sum | `build_fourier_basis` (index 0 = DC, 1..K = cos, K+1..2K = sin) | Stack ribbons with transparency; sum overlay in white | Stained-glass ribbons |
| F3 | **Theta Desk** | The mixing-desk metaphor for Θ | Θ matrix of one individual (M×D) | Draw M sliders, amplitude = row norm, hue by frequency | Studio-board, flat glow |
| F4 | **Token Starfield (E)** | Show the embedding space as chemically structured constellations | `build_structured_E` + `vocab` | PCA/UMAP of E to 2D, colour by group (atom_aliph, atom_arom, bond_double, branch, ring, macro) | Constellation, labelled by glyph not text |
| F5 | **Gravity Wells (distance field)** | Show d²(i,v) as a landscape per slot | `compute_dist2(Z, E)` | For one position i, project E to 2D, colour a Voronoi/KDE field by softmax(−d²/T); mark Z_i | Topographic glow |
| F6 | **Probability Landscape over Sequence** | Token-probability heatmap along the sequence | softmax(−d²) before masking, masked after | Render L × V matrix as luminous heat band, mask fog overlay; top token spark | Waterfall of light |
| F7 | **Z-Trajectory in Embedding Space** | How Z_i drifts across generations and lands on tokens | Z over generations (needs R3) | 2D projection of Z rows through E's PCA; trail per slot | Neon trails on starfield |
| F8 | **Molecule Morph Ribbon** | Lineage of one best molecule through generations | `molecule_score_rows` (generation, smiles, score) in runs | Parent→child by Tanimoto nearest in prior generation; RDKit MCS-based coordinate interpolation | White line-art strip |
| F9 | **Frequency Brush canvas** | Low-k vs high-k mutations affect scaffold vs substituents | Mutate Θ with mask-low vs mask-high (R1) from same parent, decode both | Tanimoto/scaffold-change statistic per mask; draw as two brush strokes with the resulting child molecules | Painterly strokes |
| F10 | **Pareto Frontier Sweep** | NSGA-II as a moving frontier | **Built from GuacaMol ablation run data, not Saturn** (the Saturn outputs in the repo are 24-molecule smoke runs without docking or NSGA-2): `runs/<model>/seed_<n>/case_*/benchmark_runs/*/*_Median_molecules_1/{scores.csv,molecule_scores_by_generation.tsv}`. These runs optimised the geometric mean of two similarities (camphor, menthol) with the SMILES-operator evolver, not NSGA-II | Per generation compute non-dominated set; animate frontier curve and fading dominated points | Starfield with white-hot arc |
| F11 | **Chemical Space Galaxy** | Search coverage of runs as a galaxy | `molecules_all.csv`, Morgan-FP → UMAP | Render 10⁵–10⁶ points with datashader / Blender point cloud; target family glowing | Deep-space |
| F12 | **Parallel Evolution Tree** | Serial vs parallel exploration | Per-seed timings (`model_runtime.tsv`), run lineage | Draw tree depths/widths per config; node birth time = wall-clock; same final cluster | Luminous branching |
| F13 | **Spectral Energy Aurora** | Θ energy by frequency over generations | Per-generation Θ (R3) | Heatmap (freq × generation) with additive blending | Aurora curtain |
| F14 | **Ring-Closure Snap** | Show the validity-constraint magic | Decoder trace (R2) | Bead chain + ring token, animate bond formation | Elastic bead chain |

---

## 5. GIF concepts (12 seamless loops, 3–8 s each)

| ID | GIF | Loop mechanism | Duration | Rendering |
|---|---|---|---|---|
| G1 | **Spectrum breathing** — F2 ribbons with phase drift | Time t ∈ [0,T); all modes use integer cycles per T so frame T = frame 0 exactly | 6 s, 30 fps | matplotlib → PNG sequence → ffmpeg palettegen |
| G2 | **Theta → Z** — one slider cycles ±, ribbon follows | Slider follows A·sin(2πt/T) | 5 s | matplotlib / Manim |
| G3 | **Nearest star** — F5 point orbits and ignites the nearest star | Closed Lissajous path for Z_i | 8 s | Matplotlib + KDE field; precompute frames |
| G4 | **Crystallise** — beads → molecule → beads | Ping-pong (forward then reverse, eased) | 6 s | RDKit SVG + cairosvg |
| G5 | **Frequency Brush** — broad and fine strokes alternating | Stroke path closes on itself; canvas cross-fades through its start state | 8 s | Skia/Cairo or Manim |
| G6 | **Evolution heartbeat** — population braids, frontier sweeps | Generation k → k+1 repeated, last generation morphs back via crossfade to the first | 8 s | Matplotlib scatter + RDKit thumbnails |
| G7 | **Four oceans** — A1 panels | Same periodic clock; Random panel uses a periodic noise (seeded noise tiled on t) | 6 s | Shader-like numpy render |
| G8 | **Harmonic peel** — A2 sine components removed and restored | Remove low → restore → remove high → restore (cycle) | 8 s | Matplotlib |
| G9 | **Complexity bloom** — A3 molecule grows atom by atom | Grows then fades to the first state with a 1 s crossfade | 8 s | RDKit + generation lineage |
| G10 | **Branching growth** — B1 trees | Growth → hold → fade → regrow; start/end both empty | 7 s | Matplotlib / Blender |
| G11 | **Swarm convergence** — B2 particles toward attractor | Particles wrap: after arrival, they dissolve into the starting cloud with a mirrored path | 6 s | numpy + matplotlib or datashader |
| G12 | **Galaxy dolly** — F11 slow zoom | Zoom-in-zoom-out, dolly with matched ends; spirals rotate by 2π/T | 8 s | Blender or datashader frames |

GIF practicals: ≤ 600 px wide, ≤ 12 MB each, 30 fps; for slides prefer MP4 with the "Loop until stopped" setting (§7) because GIFs blow up the PPTX size.

---

## 6. Repository Enhancement Plan

### 6.0 What exists vs what is missing

| Need | Exists? | Where |
|---|---|---|
| Φ basis, Θ→Z, Θ mutation | ✅ | `core/fourier_theta.py:5-53` |
| Embedding E (structured) | ✅ | `core/embedding.py` |
| dist², token sampling, SELFIES string | ✅ | `core/decoder.py:155-260, 500-` |
| Θ→Z→dist→token figure for aspirin | ✅ (static PNG) | `visualizePipeline.py` (`theta_to_molecule.png`) |
| Real **Θ-space** evolution loop that logs Θ per generation | ❌ in this checkout | `mutate_Theta`/`make_initial_population` are only referenced by `tests/unit/test_module_purpose_core.py`. The GuacaMol runner is SMILES-based. **Verify in the GitHub repo** whether a Θ-native NSGA loop lives there; if so, hook there instead. |
| Per-generation molecules and scores | ✅ (SMILES loop) | `molecule_score_rows` in `evolve_on_task` (columns `generation, smiles, score`) — confirm these are written to disk in `runs/*/`; `best_score_rows` per generation |
| Per-run timing | ✅ | `model_runtime.tsv`, `model_runtime_summary.tsv` |
| Real "timeline" of evolution | ⚠️ | `evolve_molecules_timeline.py` just adds a methyl each round. **Do not present it as evolution footage.** |

### 6.1 Required modifications (in priority order)

**R1 — Real frequency masks in Θ space** (makes the ablation scientifically honest and gives true "low vs high" visuals)
- File: `core/fourier_theta.py`
- Add `mode_frequency(K)` → array of length M=1+2K mapping index→k (0→0, 1..K→1..K, K+1..2K→1..K).
- Add `make_mode_mask(K, kind, cutoff)` for `"all" | "low" | "high" | "random"`.
- Extend `mutate_Theta(Theta, gen, generations, rng, mode_mask=None)`: multiply the noise mask and `row_reset` by `mode_mask[:, None]`.
- Add `project_Theta(Theta, mode_mask)` to apply masks to Φ for the `high-only`/`low-only` variants (zero rows), and a `random_basis` variant for the random-matrix control.
- Outputs: none directly; enables F9, A2, A4, A5 with real data.

**R2 — Decode trace** (powers F5, F6, G3, G4, F14)
- File: `core/decoder.py`
- New `trace_decode(Theta, Phi, E, id2tok, pad_id, ...)` returning a dict: `Z (L×D)`, `dist2 (L×V)`, `probs_unmasked (L×V)`, `valid_mask (L×V)` per step, `chosen_ids (L)`, `selfies`. It composes `compute_dist2` + the masking inside `dist2_to_ids_sample`. Export `.npz`.
- Check `dist2_to_ids_sample` (line 255) for where the validity mask is computed; return it instead of discarding.

**R3 — Generation logger** (powers F7, F13, F8, G6, G9)
- Where the Θ-native loop lives (see 6.0): after each generation, `np.savez_compressed("viz_trace/gen_{g:04d}.npz", Theta=…, smiles=…, scores=…, parent_ids=…, front_rank=…)`. Add `--viz-trace-dir` CLI flag. Keep it off by default (no overhead for benchmark runs).
- For the existing SMILES runner: add `parent_smiles`, `operator` (the `mutate_*` function name) and `front_rank` columns to `molecule_score_rows` so lineage and mutation type can be drawn.

**R4 — Asset exporters** (new folder `viz/`)

| Script | Output | Format |
|---|---|---|
| `viz/export_basis.py` | Φ ribbons, mode sums | SVG + PNG sequence (G1, F2) |
| `viz/export_pipeline_trace.py` | Z, dist², probs for a chosen molecule | NPZ + PNG sequence (F5/F6/G3) |
| `viz/export_embedding.py` | E projected 2D, group colours | JSON + SVG (F4) |
| `viz/export_lineage.py` | Molecule lineage strips from run CSVs | SVG per node + MP4 (F8, G9) |
| `viz/export_galaxy.py` | All `molecules_all.csv` → UMAP points | Parquet + datashader PNG sequence (F11, G12) |
| `viz/export_timing_tree.py` | Per-config lineage with wall-clock birth times (from `model_runtime.tsv`) | SVG/MP4 (F12, B1) |
| `viz/render_molecule_svgs.py` | Line-art molecules using `rdMolDraw2D.MolDraw2DSVG`, transparent background | SVG (all scenes) |

**R5 — Make the GPU story true to data.** Add, to run logs, timestamps per evaluated batch (`time.perf_counter()` at batch completion with run id and config). Without it, B1/B4/F12 can only be *illustrative*; with it, "same destination, earlier" can be rendered from actual arrival times of each config's best-so-far score. If time-stamps exist already in `scores.csv`/`logs/`, reuse them.

**Formats to generate:** SVG for molecules and static figures (vector, recolourable); PNG sequences at 1920×1080 (transparent where compositing is needed); MP4 (H.264, yuv420p) for loops; GIF only for the 12 GIFs above; NPZ for numerical traces.

**Caveat on R1:** after R1, the full/high/low/random comparison on real Θ will not necessarily reproduce the operator-proxy numbers in the paper. Run it small (a few tasks) before putting "Low = scaffold" in a video; if low/high don't behave as hypothesised, the Frequency-Brush scene (4) should be reframed as *"coarse vs fine edits"* using Tanimoto-distance of parent→child as the measured quantity.

---

## 7. PowerPoint Video Production Guide

**Philosophy:** render each heavy scene as a 1920×1080 MP4 (matplotlib/Manim/Blender via PNG sequences + ffmpeg), use PowerPoint only as sequencer and for light overlays. PowerPoint's own animation engine is poor at hundreds of moving ribbons.

**Slide order (one slide per scene, 16:9, 13.333×7.5 in, black background):**

| Slide | Content | Duration |
|---|---|---|
| 1 | Scene 0 Spectrum (MP4 from G1) | 6 s |
| 2 | Scene 1 Desk (G2) | 8 s |
| 3 | Scene 2 Nearest star (G3) | 10 s |
| 4 | Scene 3 Crystallise (G4) | 8 s |
| 5 | Scene 4 Brush (G5) | 12 s |
| 6 | Scene 5 Evolution (G6) | 10 s |
| 7 | Scene 6 Frequencies (G7+G8+G9 composited; "conceptual" caption) | 8 s |
| 8 | Scene 7 Parallel (G10+G11) | 8 s |
| 9 | Scene 8 Galaxy (G12) | 5 s |

**Animation settings**
- Video on each slide: Insert → Video → "Start: Automatically", "Play Full Screen" off, "Hide while not playing" off, "Loop until stopped" off (the slide loop handles repetition), "Rewind after playing" off, no volume.
- Transition between slides: **Fade, 0.5 s** — or **Morph** only where two slides share identical shapes. Morph does not animate videos/GIFs, so avoid it between video slides.
- Slide timing: Transitions → "After" = scene duration (+ transition) and uncheck "On Mouse Click".
- Legend strip (wave → constellation → molecule glyphs): place on the slide master; the highlighted stage changes per slide (use three layouts or on-slide highlight).
- Do not use PowerPoint's animated entrances for text; if you add the optional caption, use "Fade" 0.5 s.

**GIF integration**
- PowerPoint plays GIFs only in Slide Show / exported video. Prefer converting the GIFs to MP4 (`ffmpeg -i g.gif -movflags faststart -pix_fmt yuv420p g.mp4`) for 8× smaller files; use real GIFs only for small inline accents (< 3 MB).
- Never Morph-transition into a GIF slide.

**Export settings**
- File → Export → Create a Video → 1920×1080 (Full HD), **Use Recorded Timings and Narrations** (timings set above), no narration.
- Render clips at 30 fps (not 24) so PowerPoint's 30 fps export needs no frame duplication; the storyboard timings are unchanged.
- Final polish with ffmpeg:
  `ffmpeg -i showcase.mp4 -c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p -movflags +faststart showcase_final.mp4`

**Loop settings**
- PowerPoint video export itself does not loop. Choose one:
  1. Wall screen: play `showcase_final.mp4` in VLC (`vlc --loop --fullscreen --no-video-title-show`) or a media player with loop (most reliable).
  2. In PowerPoint: Slide Show → Set Up Slide Show → **"Browsed at a kiosk (full screen)"**; this loops continuously until Esc with the saved timings.
- To make the *video file* itself seamless, make the last 12 frames cross-dissolve into the first 12 (scene 8 → scene 0 are designed to match: both are sine ribbons on the dark background).
- Test the seam: `ffmpeg -stream_loop 2 -i showcase_final.mp4 -c copy test_loop.mp4` and scrub the join.

---

## 8. Priority Ranking

Scores 1–5. Priority = (Scientific × 1.5 + Visual × 1.5 + Ease × 1.0) / 4. "Scientific" = how faithfully it reflects the actual method and data.

| Rank | Item | Scientific | Visual | Ease | Priority | Needs |
|---|---|---|---|---|---|---|
| 1 | **G4 Crystallise (beads → molecule)** | 5 | 5 | 4 | 4.75 | RDKit + decoder output |
| 2 | **F2/G1 Fourier Mode Orchestra** | 5 | 4 | 5 | 4.62 | `build_fourier_basis` only |
| 3 | **F5/G3 Gravity wells / Nearest star** | 5 | 5 | 3 | 4.50 | R2 (or recompute with `compute_dist2`) |
| 4 | **F3/G2 Θ Desk → Z ribbon** | 5 | 4 | 4 | 4.38 | existing code |
| 5 | **F4 Token Starfield** | 5 | 4 | 4 | 4.38 | existing E |
| 6 | **F13 Spectral Aurora** | 5 | 5 | 2 | 4.25 | R1 + R3 |
| 7 | **F8/G9 Molecule Morph / Complexity bloom** | 4 | 5 | 3 | 4.12 | run lineage; R3 columns |
| 8 | **F10/G6 Pareto Frontier Sweep** | 5 | 4 | 3 | 4.12 | Saturn per-gen data |
| 9 | **A1/G7 Four oceans (ablation) — *conceptual until R1*** | 3 | 5 | 4 | 4.00 | illustrative; R1 makes it measured |
| 10 | **A2/G8 Harmonic peel** | 4 | 4 | 4 | 4.00 | R1 (nice-to-have) |
| 11 | **B1/G10 Branching growth (acceleration)** | 3 | 5 | 4 | 4.00 | illustrative; better with R5 |
| 12 | **F7 Z-trajectory** | 5 | 4 | 2 | 3.88 | R3 |
| 13 | **F11/G12 Chemical Space Galaxy** | 3 | 5 | 3 | 3.75 | `molecules_all.csv` + UMAP |
| 14 | **F14 Ring-closure snap** | 4 | 4 | 3 | 3.75 | R2 |
| 15 | **F9/G5 Frequency Brush (scientific 3 → 5 after R1)** | 3 | 5 | 2 | 3.50 | R1 + decode |
| 16 | **B2/G11 Swarm convergence** | 2 | 4 | 5 | 3.50 | illustrative |
| 17 | **F12 Parallel evolution tree** | 4 | 4 | 2 | 3.50 | R5 |

**Recommended build order (a 1-week pass):**
1. G1, G2 (no new code; ½ day).
2. G3 + G4 from `compute_dist2` and `ids_to_selfies` with the aspirin example from `visualizePipeline.py` (1 day).
3. F4 starfield + F10 Pareto sweep (1 day).
4. A1/B1/B2 as illustrative loops; tag **conceptual** (1 day).
5. R1 + R3 only if you want measured spectra/brush scenes (≈2 days code + a small test run on the cluster).
6. Assemble in PPT/ffmpeg and loop-test (½ day).

**Tests of honesty before the wall goes up:** (a) no scene implies a measured spectrum for the SMILES-operator ablation; (b) no visual shows a speedup greater than ~1.2×; (c) the "low = scaffold, high = refinement" story is shown only as a design intuition unless R1 verifies it.
