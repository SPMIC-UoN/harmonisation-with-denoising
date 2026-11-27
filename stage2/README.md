# Stage 2 analysis code

Code for the Stage 2 registered report *Does Denoising Reduce Cross-Vendor Variability in Diffusion MRI? A Travelling
Heads Evaluation*. It takes the ON-Harmony Phase B data (OpenNeuro ds004712) from download to every table and figure
in the manuscript.

```
pipeline/   download, preprocessing, model inputs, registered model fits, results tables
results/    one notebook per hypothesis: the tables and figures of the manuscript
tests/      unit tests of the input tables and of the model-fitting layer
processed-data/   model input tables, model results and figure data (zipped; see its README)
```

To refit the models or redraw most of the figures without running the image processing, see `processed-data/README.md`.

## From notebook to manuscript

Each notebook reads the pipeline outputs, shows the results table, and saves its figures (PNG and PDF, with the
caption as a `.md` file) to `<derivatives>/analysis/figures/stage2/`. All plotting code is in `results/figs.py`.

| Notebook | Tables | Figures |
|---|---|---|
| `H1.ipynb` | S11, S15 | 2, 6 |
| `H2.ipynb` | S12 | 3, 7 |
| `H3.ipynb` | S13 | 4, 8 |
| `H4.ipynb` | S14 | 5 |
| `H4_variogram_null.ipynb` | – | 9, S18 (and the ROI-wise autocorrelation check) |
| `Sensitivity.ipynb` | S7, S16a, S16b, S17, S19, S20 | S8 |
| `Supplementary.ipynb` | S6 | S9, S10 |

Each notebook has a "Registered results" part and, where there is one, an "Exploratory results" part.

To run a notebook, set two environment variables (or edit the first code cell):

```
export RR_DERIV=/path/to/derivatives
export RR_MANIFEST=/path/to/ds004712_meta/sessions_manifest.csv
```

Run `H1.ipynb` and `H2.ipynb` before the others: they write files that the H2 and H4 notebooks read. `FSLDIR` must also be set: the brain maps are drawn on the MNI152 templates shipped with FSL.

`H4_variogram_null.ipynb` and `Supplementary.ipynb` fit models (surrogate fields; the effect of denoising on TOPUP and
EDDY). They store what they compute under `<derivatives>/analysis/`, so only the first run is slow (about an hour
each on the machine used here). Delete the cached files named in each notebook to
recompute.

## Running the pipeline

`pipeline/run_pipeline.sh` lists every step in order. Set the paths at the top of the script, then run it, or run the
steps one at a time; each script has a header describing what it does and `--help` for its options. Finished jobs are
skipped when a step is re-run.

| Step | Script | What it does |
|---|---|---|
| 1 | `00_download.py`, `01_prepare_inputs.py` | Download Phase B, check the metadata, prepare the per-session inputs |
| 2 | `02_denoise.py` | MPPCA denoising (DESIGNER) for the denoised arm |
| 3 | `03_brc.py`, `04_eddy_qc.py` | BRC structural and diffusion preprocessing (TOPUP, EDDY) for both arms; EDDY QC |
| 4 | `05_templates.py` – `08_mmorf.py` | Subject templates, FA/MD/MK in template space, white-matter masks and ROIs, registration to MNI (H3a) |
| 5 | `build_inputs.py`, `check_inputs.py`, `run_models.py`, `summarise_results.py` | Model input tables, the registered models, results and convergence tables |
| 6 | `run_models.py --h4-col` | Exploratory: H4 with z-scored \|off-resonance\| |
| 7 | `run_models.py --draws 8000` | Exploratory: re-run of the fits that did not meet the convergence criteria, with four times as many draws |
| 8 | `build_inputs.py --checkerboard parity`, `run_models.py` | Exploratory: the voxel-wise models on the dense checkerboard (Table S20) |
| 9 | `h2b_integrated.py` | Exploratory: the ROI-wise H2 model with the ROI effects integrated out (Table S16a) |
| 10 | `h3a_mk_persubject.py`, `run_models.py` | Exploratory: voxel-wise H3 of MK with validity required only within each subject (Table S17) |
| 11 | `mk_exclusion.py` | Why the voxel-wise H3 analysis of MK keeps 26.5% of the voxels (Supplementary Section 5, Figure S8) |
| 12 | `descriptives.py`, `h1_maps.py`, `h2_maps.py`, `h4_maps.py` | Native-unit summaries and group maps used by the figures |

### Software

- FSL 6.0.6.4 (and FSL 6.0.7.18 for MMORF), BRC pipelines 1.6.3, ANTs 2.4.4, Singularity
- DESIGNER 2.0.11 (Singularity image)
- The Stage 1 simulation container (built from `simulations/env.def` in this repository; PyMC 5.10.4). The registered
  models are fitted inside it.
- Python environment for the pipeline scripts and the H1–H4 notebooks: `environment.yml` (`h2b_integrated.py` fits its model in this environment, with the PyMC version of the Stage 1 container)
- `H4_variogram_null.ipynb` and `Supplementary.ipynb` also need `brainsmash` (0.11.0), `bambi` (0.14.0), `pymc`,
  `arviz`, `seaborn` and `scikit-learn`

### What you will need to adapt

The scripts ran on one workstation. Three things are specific to it:

- `08_mmorf.py` has the FSL installation paths at the top of the file.
- `03_brc.py` and `08_mmorf.py` call a local GPU-allocation wrapper (`migtools`) before EDDY and MMORF. Replace it with
  whatever selects a GPU on your system.
- `run_models.py` expects the Python interpreter at the path it has inside the Stage 1 container.

## How the registered models are fitted

`pipeline/models.py` imports the model functions from the Stage 1 `simulations/` folder and calls them unchanged. Each
file is checked against its SHA-256 before use, and the fit stops if a file differs. The `simulations/` folder is found
automatically when `stage2/` sits next to it, as in this repository; otherwise set `RR_SIM_DIR`.

Two points of implementation:

- **Bayes factors** are computed as in the Stage 1 simulation code: a Savage–Dickey density ratio with a normal
  approximation to the posterior.
- **H1a and H3** have up to about a million rows. Their likelihood is evaluated on sufficient
  statistics per cell (subject × condition for H1a; subject × scanner for H3) instead of one term per row. The model,
  priors and sampler settings are those of the simulation code, and the log-density is the same; `tests/test_models.py`
  checks this.

## Sessions that needed special handling (`pipeline/session_exceptions.csv`)

Two sessions needed special handling (the second in two steps). `01_prepare_inputs.py` and `03_brc.py` read what to do
with them from `pipeline/session_exceptions.csv`:

- **sub-16745, GE Premier 21ch:** acquired with 2.4 mm slices; included unchanged.
- **sub-16793, GE Premier 21ch:** the reverse-phase b0 was acquired left–right instead of anterior–posterior. TOPUP is
  run with custom acquisition parameters. The sign of the left–right readout direction cannot be read from the
  metadata, so the `d14test` stage of `03_brc.py` runs TOPUP with both signs and reports which one fits; the chosen sign
  is the fourth row of the file (`D14`), and the diffusion preprocessing of this session waits for it.

## Voxel sampling

The voxel-wise models (H1a–H4a) use one white-matter voxel in eight (i, j, k all even on the 2 mm grid; `lattice` in the
file names). The dense checkerboard (i + j + k even, about half of the voxels; `parity`) is used for Table S20 only.

Two short names recur in the code and in the output files:

| Name | Meaning |
|---|---|
| `d41` | FA or MD is missing in both arms where it is ≤ 0 (or non-finite) in either arm |
| `d56` | MK is missing in both arms where it is ≤ 0 or > 3 in either arm |

## Off-resonance in H4

The H4 fits use `off_hz`: the absolute TOPUP field in Hz, averaged over the repeats and the two arms. β0 is therefore the
denoising effect at zero off-resonance and β1 its change per Hz (Figure 5 shows β1 per SD of the pair's |off-resonance|).
The input tables also hold `off_z` (`off_hz` z-scored within pair) for the exploratory fits.

## Tests

```
python -m pytest tests
```

`tests/test_inputs.py` checks the functions that build the model inputs on small arrays with known answers.
`tests/test_models.py` checks the model-fitting layer, and needs PyMC and the Stage 1 `simulations/` folder.
