# Processed data

The tables that the registered models were fitted to, the model results, and the summaries and group maps behind the
figures. With these files, all registered models can be refitted, and the figures redrawn, without running the
image processing.

The raw data are ON-Harmony Phase B ([OpenNeuro ds004712](https://doi.org/10.18112/openneuro.ds004712.v2.0.1), v2.0.1,
CC0). Please cite the dataset (Warrington et al., 2025, *Scientific Data*) if you use these files.

## Unpacking

```
python unpack.py --out /path/to/derivatives
```

Download the whole folder first. The script (Python 3 standard library only) extracts every zip, joins the two H3 tables
that were split to stay below GitHub's file-size limit, and checks each file against `checksums.sha256`. The folder it
creates has the layout used by the pipeline, so it can be used directly as `RR_DERIV`.

## Contents

| Files | Unpacks to | What it holds |
|---|---|---|
| `model_inputs_H1.zip` … `model_inputs_H4.zip` | `analysis/inputs/` | Input tables of the registered models, and `counts.json` (numbers of voxels and ROIs kept and excluded) |
| `model_results*.zip` | `analysis/results*/` | Model results (see below) |
| `figure_data*.zip` | `analysis/descriptives/`, `analysis/exploratory/`, `analysis/supplementary/`, `qc/`, `atlas/` | Summaries in native units, results of the exploratory analyses, TOPUP/EDDY and EDDY QC values, and the ROI atlas |
| `group_maps*.zip` | `analysis/descriptives/`, `analysis/exploratory/` | The brain maps shown in the figures (NIfTI, MNI152, 1 or 2 mm) |

### Input tables

One file per hypothesis, level (`a` voxel-wise, `b` ROI-wise), metric (FA, MD, MK) and, for H2 and H4, subject.
Voxel-wise tables hold one white-matter voxel in eight. `unit` is the voxel (flat index of its grid) or the ROI
(label of the 1,298-label atlas), `den` is 0 for non-denoised and 1 for denoised.

| File | Columns | Modelled variable |
|---|---|---|
| `h1{a,b}_{metric}` | `subject`, `unit`, `den`, `cov`, `y` | `y`: cross-scanner CoV (`cov`), z-scored over all subjects and both conditions |
| `h2{a,b}_{subject}_{metric}` | `unit`, `delta_bias`, `y` | `y`: ΔBias divided by its SD, not centred |
| `h3{a,b}_{metric}` | `subject`, `scanner`, `unit`, `r`, `den`, `y_raw`, `y` | `y`: metric value (`y_raw`) minus the unit's mean within its condition, divided by one SD shared by both conditions. `r` is the unit index |
| `h4{a,b}_{subject}_{metric}` | `unit`, `cov_nonden`, `cov_den`, `delta`, `off_hz`, `off_z` | `delta`: ΔCoV, z-scored with the two conditions pooled. Covariate: `off_hz` (mean off-resonance magnitude, Hz); `off_z` is the z-scored version used in Table S17 |

CoVs are fractions, not percentages. MD is in mm²/s.

### Model results

| Folder | What it holds |
|---|---|
| `results/` | Registered models (Tables S10–S13) |
| `results_extended_x4/` | Fits that did not meet the convergence criteria, re-run with four times as many draws (Table S15) |
| `results_h4_abs_z/` | H4 with the z-scored off-resonance magnitude (Table S17) |
| `results_dense/` | Voxel-wise models on the dense checkerboard (Table S18); its ROI-wise results are not used |

In each, `<model>/summary.csv` has the posterior mean, SD, 95% interval and convergence diagnostics of each parameter,
and `<model>/result.json` the Bayes factor, sampler settings and convergence (the name and checksum of the input
table are also listed; in `results_extended_x4/` and `results_dense/` the checksums refer to other copies of the
tables, so they differ from the files here). H3 folders also have the ICC draws
(`icc_draws.npz`). Except for `results_h4_abs_z/`, `results.csv` collects all tests with their verdicts, and
`convergence_all.csv` and `convergence_by_var.csv` the diagnostics of all parameters.

## Refitting the models

```
python ../pipeline/run_models.py --inputs <derivatives>/analysis/inputs --out <new folder> --container env.sif --bind /path/to
python ../pipeline/run_models.py --out <new folder> --combine
python ../pipeline/summarise_results.py --results <new folder>
```

`env.sif` is the container built from `simulations/env.def`.

## Redrawing the figures

Set `RR_DERIV` to the unpacked folder (`RR_MANIFEST` can be left unset) and run the notebooks in `../results`. `FSLDIR`
must be set, as described in `../README.md`.

| Notebook | What it makes from these files |
|---|---|
| `H1.ipynb` | Table S10, Figures 2 and 6. Under "Exploratory results", run only the last cell (Figure 6); the others read the images of the full pipeline. Table S14 holds the subject means of `analysis/exploratory/H1_spread/decomposition.csv` |
| `H2.ipynb` | All its tables and figures |
| `H3.ipynb` | Table S12, Figures 4 and 8. Under "Exploratory results", run only the last cell (Figure 8); the others read the images of the full pipeline |
| `H4.ipynb` | All its tables and figures |
| `H4_variogram_null.ipynb` | All its figures, from the stored surrogate results |
| `Sensitivity.ipynb` | All its tables |
| `Supplementary.ipynb` | All its tables and figures, from the stored TOPUP/EDDY values (it refits the models, about an hour) |

## Not included

These are large and are regenerated by the pipeline from the raw data: preprocessed diffusion images, the metric maps
and off-resonance maps of each session, subject templates and transforms, the posterior samples, and the input tables of
the dense checkerboard (`build_inputs.py --checkerboard parity`). Regenerating images from the raw data may not give
bit-identical values on different hardware.
