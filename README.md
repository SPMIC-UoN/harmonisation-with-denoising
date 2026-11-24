# Does Denoising Reduce Cross-Vendor Variability in Diffusion MRI? A Travelling Heads Evaluation

Code for the PCI Registered Report by F. D'Antonio, J.-P. Manzano-Patron, O. Mougin, P. S. Morgan, S. N. Sotiropoulos
and S. Warrington. The data are ON-Harmony Phase B, openly available from OpenNeuro
([ds004712](https://openneuro.org/datasets/ds004712)).

## Contents

| Folder | Stage | What it holds |
|---|---|---|
| `simulations/` | 1 | Data-adequacy simulations for H1–H4 and the container definition (`env.def`) used to run them. The Stage 2 models are fitted with these functions, unchanged |
| `atlas-ROI/` | 1 | Sub-parcellation of the XTRACT white-matter atlas |
| `supplementary-materials/` | 1 | Pilot analysis of the effect of denoising on TOPUP and EDDY (Stage 1 supplement, Section 3) |
| `stage2/` | 2 | The full Stage 2 analysis: preprocessing pipeline, registered models, and the notebooks that produce every table and figure of the manuscript. Start with `stage2/README.md` |

The Stage 2 analysis used the 1,298-label parcellation, which is included as `stage2/pipeline/xtract_sub_1298.nii.gz`.

## Licence

MIT (see `LICENSE`).
