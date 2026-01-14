import numpy as np
import nibabel as nib
from sklearn.decomposition import PCA
from scipy.ndimage import label as cc_label
import argparse
import json

# Utility functions
def load_nifti(path):
    img = nib.load(path)
    # keep original numeric values (do not cast to int) so non-integer labels are preserved
    data = img.get_fdata()
    return img, data

def save_nifti(data, ref_img, out_path):
    # Create a new image with an explicit integer dtype for labels to avoid
    # re-using the old header which might indicate an 8-bit dtype (max 255).
    arr = data.astype(np.int32)
    new_img = nib.Nifti1Image(arr, ref_img.affine)
    new_img.header.set_data_dtype(np.int32)
    nib.save(new_img, out_path)


# Split a set of voxel coordinates using PCA median split
def pca_median_split(coords):
    """
    coords: (N, 3) array of voxel coordinates
    Returns: boolean mask splitting into two roughly equal sets
    """
    if coords.shape[0] < 2:
        return np.zeros(coords.shape[0], dtype=bool)
    pca = PCA(n_components=1)
    proj = pca.fit_transform(coords)
    median_val = np.median(proj)
    return proj[:, 0] <= median_val

# Recursive splitting
def recursive_split(coords, k):
    """
    Recursively split voxel coordinates into k parts using PCA median splits.
    Returns a list of arrays of voxel indices.
    """
    if k == 1:
        return [coords]
    # first split
    mask = pca_median_split(coords)
    left = coords[mask]
    right = coords[~mask]

    # distribute requested subdivisions
    k_left = max(1, round(k * len(left) / (len(left) + len(right))))
    k_right = k - k_left

    parts = []
    parts.extend(recursive_split(left, k_left))
    parts.extend(recursive_split(right, k_right))
    return parts

# Connected-component fix
def enforce_contiguity(subcoords, atlas_shape, min_component=50):
    """
    Break subparcel into connected components. Merge small ones.
    Returns a list of contiguous voxel coordinate arrays.
    """
    # handle empty input
    if subcoords is None or len(subcoords) == 0:
        return []

    vol = np.zeros(atlas_shape, dtype=int)
    for (x, y, z) in subcoords:
        vol[x, y, z] = 1

    labeled, ncc = cc_label(vol)
    components = []

    for cc_id in range(1, ncc+1):
        xs, ys, zs = np.where(labeled == cc_id)
        comp = np.vstack([xs, ys, zs]).T
        components.append(comp)

    if len(components) == 0:
        return []

    # separate large and small components
    large_components = [c for c in components if len(c) >= min_component]
    small_components = [c for c in components if len(c) < min_component]

    if len(large_components) == 0:
        # everything small; merge all
        all_coords = np.vstack(components)
        return [all_coords]

    if small_components:
        # Merge each small component into the nearest large component (by centroid distance)
        # Precompute centroids for large components
        large_centroids = [c.mean(axis=0) for c in large_components]

        for sc in small_components:
            sc_centroid = sc.mean(axis=0)
            # find nearest large component
            dists = [np.linalg.norm(sc_centroid - lc) for lc in large_centroids]
            nearest_idx = int(np.argmin(dists))
            # append small comp to the selected large component
            large_components[nearest_idx] = np.vstack([large_components[nearest_idx], sc])
            # update centroid for that large component
            large_centroids[nearest_idx] = large_components[nearest_idx].mean(axis=0)

        return large_components
    else:
        return large_components

# Main subparcellation routine
def subparcellate(in_path, out_path, target_size=400, min_component=50, min_final_size=200, label_tolerance=0.0):
    img, data = load_nifti(in_path)
    atlas_shape = data.shape

    # identify non-background voxels using tolerance (0.0 means exact != 0)
    if label_tolerance == 0.0:
        mask_nonzero = data != 0
    else:
        mask_nonzero = np.abs(data) > label_tolerance

    labels = np.unique(data[mask_nonzero])

    # compute sizes using tolerance-aware matching
    if label_tolerance == 0.0:
        sizes = {lab: int(np.sum(data == lab)) for lab in labels}
    else:
        sizes = {lab: int(np.sum(np.isclose(data, lab, atol=label_tolerance, rtol=0))) for lab in labels}

    # initial desired splits
    desired = {lab: max(1, round(sizes[lab] / target_size)) for lab in labels}

    new_label_data = np.zeros_like(data, dtype=int)
    next_label_id = 1
    # mapping: original_label (string) -> list of {new_label: int, size: int}
    mapping = {}

    for lab in labels:
        # fetch coordinates for this label (tolerance-aware)
        if label_tolerance == 0.0:
            coords = np.vstack(np.where(data == lab)).T  # Nx3
        else:
            coords = np.vstack(np.where(np.isclose(data, lab, atol=label_tolerance, rtol=0))).T
        k = desired[lab]

        # split into k parts
        parts = recursive_split(coords, k)

        # enforce contiguity within each part
        final_parts = []
        for p in parts:
            cc_parts = enforce_contiguity(p, atlas_shape, min_component=min_component)
            final_parts.extend(cc_parts)

        # enforce minimum parcel size by merging tiny parcels
        # sort by size
        final_parts_sorted = sorted(final_parts, key=lambda x: len(x), reverse=True)
        merged_parts = []

        for fp in final_parts_sorted:
            if len(fp) < min_final_size:
                # merge into largest
                if len(merged_parts) == 0:
                    merged_parts.append(fp)
                else:
                    merged_parts[0] = np.vstack([merged_parts[0], fp])
            else:
                merged_parts.append(fp)

        # assign new labels (vectorized) and record mapping info
        lab_key = str(lab)
        mapping.setdefault(lab_key, [])
        for part in merged_parts:
            if part is None or len(part) == 0:
                continue
            # vectorized assignment
            try:
                idx = tuple(part.T)
                new_label_data[idx] = next_label_id
            except Exception:
                # fallback to safe loop if something unexpected occurs
                for (x, y, z) in part:
                    new_label_data[x, y, z] = next_label_id

            mapping[lab_key].append({"new_label": int(next_label_id), "size": int(len(part))})
            next_label_id += 1

    save_nifti(new_label_data, img, out_path)
    print(f"Saved subparcellated atlas to {out_path}. Total parcels: {next_label_id-1}")

    # write mapping file (JSON) next to output
    map_json_path = out_path + ".map.json"

    result = {
        "input_file": in_path,
        "output_file": out_path,
        "total_parcels": int(next_label_id - 1),
        "parcels": mapping,
        "settings": {
            "target_size": int(target_size),
            "min_component": int(min_component),
            "min_final_size": int(min_final_size),
            "label_tolerance": float(label_tolerance),
        },
    }

    try:
        with open(map_json_path, 'w') as jf:
            json.dump(result, jf, indent=2)

        print(f"Wrote mapping file: {map_json_path}")
    except Exception as e:
        print(f"Warning: failed to write mapping files: {e}")


def _build_argparser():
    p = argparse.ArgumentParser(
        description="Subparcellate a labelled NIfTI atlas into smaller contiguous parcels.")
    p.add_argument("input", help="Input labelled NIfTI file (atlas)")
    p.add_argument("output", help="Output NIfTI path for subparcellated atlas")
    p.add_argument("--target-size", type=int, default=400,
                   help="Approximate target parcel size (voxels). Default: 400")
    p.add_argument("--min-component", type=int, default=50,
                   help="Minimum connected-component size during contiguity enforcement. Default: 50")
    p.add_argument("--min-final-size", type=int, default=200,
                   help="Minimum final parcel size; smaller parcels will be merged. Default: 200")
    p.add_argument("--label-tolerance", type=float, default=0.0,
                   help="Tolerance for matching label values in the atlas (useful for non-integer labels). Default: 0.0 (exact match)")
    return p


def main(argv=None):
    parser = _build_argparser()
    args = parser.parse_args(argv)

    subparcellate(args.input, args.output,
                      target_size=args.target_size,
                      min_component=args.min_component,
                      min_final_size=args.min_final_size,
                      label_tolerance=args.label_tolerance)


if __name__ == "__main__":
    main()
