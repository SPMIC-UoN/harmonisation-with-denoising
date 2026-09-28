# Note on the parcellated XTRACT atlas (added 2026-09-28)

Two versions of the parcellated XTRACT atlas exist:

| File | Labels | sha256 | Status |
|---|---|---|---|
| `xtract_sub_1298.nii.gz` (+ `.map.json`, `total_parcels: 1298`) | **1,298** | `d3adee781d5e3f399fec4d7a0c0a63de3f1f112c8c32e70c2ef57f97b271412c` | **Used in the study.** Matches the 1,298 ROIs reported in the Stage 1 manuscript (Table 2). |
| `xtract_sub.nii.gz` | 1,300 | `467b922cdc39ed00696b55a6ce81e108bf41e1a682cc83749d2216241e1947a4` | Committed on 2026-01-14 (`ea39140`) and present at the Stage 1 submission. Superseded; kept for provenance. |

Both come from the same subparcellation of the XTRACT tract atlas (`subparcellate.py`; target size 400 voxels,
minimum component 50, minimum final size 200) on the FSL MNI152 1 mm grid. They have identical labelled voxels
(521,310) and differ only in tracts #19 (37 vs 38 parcels) and #40 (24 vs 25 parcels), where the 1,300-label version
has one extra split each, consistent with a re-run of the PCA-median splitting.
This discrepancy is recorded as a deviation (decision D34) in the study's preprocessing decision log.
