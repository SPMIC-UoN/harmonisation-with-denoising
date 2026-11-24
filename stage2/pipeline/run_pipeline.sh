#!/bin/bash
# run_pipeline.sh - the Stage 2 pipeline in order: download -> preprocessing (both arms) -> template-space maps ->
# input tables -> pre-registered model fits -> results tables, descriptives and maps read by ../results/*.ipynb.
#
# Requires FSL 6.0.6.4 ($FSLDIR), BRC pipelines 1.6.3, ANTs, MMORF (FSL 6.0.7.18), Singularity, the DESIGNER 2.0.11
# image and the Stage 1 simulation image (built from simulations/env.def). Set the paths below, then run.
set -euo pipefail

BIDS=/path/to/ds004712                 # raw BIDS (ON-Harmony, written by step 00)
META=/path/to/ds004712_meta            # manifest, QC report
DERIV=/path/to/derivatives             # all outputs
DESIGNER_SIF=/path/to/designer2_v2.0.11.sif
ENV_SIF=/path/to/env.sif               # Stage 1 simulation container (simulations/env.def)
BIND=/path/to                          # folder bound into the container (contains $DERIV)
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}

# 1. Data
$PY "$HERE/00_download.py" --bids-dir "$BIDS" --out-dir "$META"
$PY "$HERE/01_prepare_inputs.py" --bids-dir "$BIDS" --meta-dir "$META" --deriv-dir "$DERIV"

# 2. Denoised arm: MPPCA (DESIGNER)
$PY "$HERE/02_denoise.py" --deriv-dir "$DERIV" --sif "$DESIGNER_SIF"

# 3. BRC: structural, TOPUP sign test for the left-right reverse b0 (sub-16793), dMRI (TOPUP + EDDY) in both arms
$PY "$HERE/03_brc.py" --deriv-dir "$DERIV" --bids-dir "$BIDS" --meta-dir "$META" \
    --stages struc d14test dmri --arms nondenoised denoised
for arm in nondenoised denoised; do $PY "$HERE/04_eddy_qc.py" --deriv-dir "$DERIV" --arm $arm; done

# 4. Subject templates, template-space DTI/DKI fits, WM masks and ROIs, MNI registration (H3a)
mkdir -p "$DERIV/atlas" && cp "$HERE/xtract_sub_1298.nii.gz" "$DERIV/atlas/"
$PY "$HERE/05_templates.py" --deriv-dir "$DERIV" --meta-dir "$META"
$PY "$HERE/06_fit.py" --deriv-dir "$DERIV" --meta-dir "$META"
$PY "$HERE/07_masks_rois.py" --templates-dir "$DERIV/templates" --atlas "$DERIV/atlas/xtract_sub_1298.nii.gz"
$PY "$HERE/08_mmorf.py" --deriv-dir "$DERIV" --meta-dir "$META"
$PY "$HERE/08_mmorf.py" --deriv-dir "$DERIV" --meta-dir "$META" --scalars-only
$PY "$HERE/08_mmorf.py" --deriv-dir "$DERIV" --meta-dir "$META" --mni-nonpos

# 5. Input tables and pre-registered models (4 chains x (2,000 warm-up + 2,000 draws), in the Stage 1 container)
A="$DERIV/analysis"
$PY "$HERE/build_inputs.py" --deriv-dir "$DERIV" --manifest "$META/sessions_manifest.csv" --out "$A/inputs"
$PY "$HERE/check_inputs.py" --inputs "$A/inputs"
$PY "$HERE/run_models.py" --inputs "$A/inputs" --out "$A/results" --container "$ENV_SIF" --bind "$BIND"
$PY "$HERE/run_models.py" --out "$A/results" --combine
$PY "$HERE/summarise_results.py" --results "$A/results"

# 6. Exploratory: H4 with z-scored |off-resonance| (Table S17)
H4=$(cd "$A/inputs" && ls h4?_*.csv.gz | sed 's/^h/H/; s/\.csv\.gz$//')
$PY "$HERE/run_models.py" --inputs "$A/inputs" --out "$A/results_h4_abs_z" --h4-col off_z --only $H4 \
    --container "$ENV_SIF" --bind "$BIND"

# 7. Exploratory: fits that fail R-hat < 1.01 / bulk ESS > 1000 are re-run with 4x draws (H3: both arms of the pair)
EXT=$(cut -d: -f1 "$A/results/ess_extend.txt" | while read -r j; do
    echo "$j"; case $j in H3*_den0) echo "${j%0}1";; H3*_den1) echo "${j%1}0";; esac; done | sort -u)
if [ -n "$EXT" ]; then
    $PY "$HERE/run_models.py" --inputs "$A/inputs" --out "$A/results_extended_x4" --only $EXT --draws 8000 \
        --container "$ENV_SIF" --bind "$BIND"
    $PY "$HERE/run_models.py" --out "$A/results_extended_x4" --combine
    $PY "$HERE/summarise_results.py" --results "$A/results_extended_x4"
fi

# 8. Voxel-wise models on the dense checkerboard (1 in 2 voxels; Table S18)
$PY "$HERE/build_inputs.py" --deriv-dir "$DERIV" --manifest "$META/sessions_manifest.csv" --out "$A/inputs_dense" \
    --checkerboard parity --hyp H1 H2 H3a H4
DENSE=$(cd "$A/inputs_dense" && ls h?a_*.csv.gz | sed 's/^h/H/; s/\.csv\.gz$//' | while read -r j; do
    case $j in H3*) echo "${j}_den0"; echo "${j}_den1";; *) echo "$j";; esac; done)
$PY "$HERE/run_models.py" --inputs "$A/inputs_dense" --out "$A/results_dense" --only $DENSE \
    --container "$ENV_SIF" --bind "$BIND"
$PY "$HERE/run_models.py" --out "$A/results_dense" --combine

# 9. Native-unit descriptives and group maps used by the figures
$PY "$HERE/descriptives.py" --deriv-dir "$DERIV" --manifest "$META/sessions_manifest.csv" --out "$A/descriptives"
for f in --only-h2-per-arm --only-h1-value; do
    $PY "$HERE/descriptives.py" --deriv-dir "$DERIV" --manifest "$META/sessions_manifest.csv" --out "$A/descriptives" $f
done
for h in h1 h2 h4; do $PY "$HERE/${h}_maps.py" --deriv-dir "$DERIV" --manifest "$META/sessions_manifest.csv"; done
