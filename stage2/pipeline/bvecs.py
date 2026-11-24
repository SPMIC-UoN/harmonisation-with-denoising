"""
bvecs.py - Rotate FSL bvecs by the rotation part of a (concatenated) FLIRT affine.

FLIRT matrices map FSL scaled-voxel coordinates of the input to those of the reference, and FSL bvecs
are expressed in the same scaled-voxel (radiological) convention. The bvecs of an image resampled with
a FLIRT matrix A are therefore R @ v, where R is the rotation part of A's linear block:
    R = A (A^T A)^(-1/2)      (polar decomposition; computed as U V^T from A = U S V^T)
This is the same convention FSL uses when it rotates bvecs itself. b0 columns (all zero) stay zero;
other columns are renormalised to unit length after rotation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def read_flirt(path: Path) -> np.ndarray:
    m = np.loadtxt(path)
    if m.shape != (4, 4):
        raise ValueError(f"{path}: not a 4x4 FLIRT matrix")
    return m


def rotation_part(affine: np.ndarray) -> np.ndarray:
    """Closest rotation to the 3x3 linear block (polar decomposition). Refuses reflections."""
    a = np.asarray(affine, dtype=float)[:3, :3]
    u, _, vt = np.linalg.svd(a)
    r = u @ vt
    if np.linalg.det(r) < 0:
        raise ValueError("affine contains a reflection; refusing to rotate bvecs")
    return r


def rotate_bvecs(bvecs: np.ndarray, rotation: np.ndarray, zero_tol: float = 1e-6) -> np.ndarray:
    v = np.asarray(bvecs, dtype=float)
    if v.ndim != 2 or v.shape[0] != 3:
        raise ValueError(f"bvecs must be 3 x N, got {v.shape}")
    out = rotation @ v
    norms = np.linalg.norm(out, axis=0)
    nonzero = np.linalg.norm(v, axis=0) > zero_tol
    out[:, nonzero] /= norms[nonzero]
    out[:, ~nonzero] = 0.0
    return out


def rotate_bvec_file(src: Path, dst: Path, matrix: Path) -> np.ndarray:
    r = rotation_part(read_flirt(matrix))
    v = np.loadtxt(src, ndmin=2)
    out = rotate_bvecs(v, r)
    np.savetxt(dst, out, fmt="%.6f")
    return r
