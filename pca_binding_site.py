#!/usr/bin/env python3
"""
PCA of binding-site conformations across an ensemble of aligned structures.

Given a set of pre-aligned protein structures (e.g. crystal structures of the
same target with different ligands, or frames of an MD trajectory saved as PDBs)
and a mapping from each structure to its ligand, this script:

  1. Detects the binding-site residues around the ligand in each structure.
  2. Keeps the residues shared by a configurable fraction of the ensemble.
  3. Extracts the coordinates of those residues (CA or all heavy atoms).
  4. Runs PCA on the stacked coordinates and clusters the projection
     (Ward hierarchical clustering with a distance cut).
  5. Writes PC scatter plots (PC1-PC2, PC1-PC3, PC2-PC3, 3D), a coloured
     dendrogram, a projections CSV and a text summary.

The structures must already be superposed onto a common frame - this tool does
not align them for you; it only aligns the *selected binding-site atoms* before
PCA so that the projection reflects local conformational change, not rigid-body
drift.

Nothing about the target is hard-coded: the structure list, ligands and the
optional category labels all come from plain-text input files (see --help).
"""

import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import mdtraj as md

# Marker shapes cycled through when an optional category file is supplied.
CATEGORY_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


# --------------------------------------------------------------------------- #
# Input parsing
# --------------------------------------------------------------------------- #
def read_ligand_mapping(path):
    """Parse a TSV/whitespace mapping file into {structure_filename: selection}.

    Each non-comment line is:  <structure_file>  <ligand_resname[,resname2,...]>
    Multiple comma-separated residue names are OR-ed into one MDTraj selection.
    """
    mapping = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                print(f"WARNING: skipping malformed mapping line: {line!r}")
                continue
            structure, ligand = parts[0], parts[1]
            resnames = [r.strip() for r in ligand.split(",") if r.strip()]
            mapping[structure] = " or ".join(f"resname {r}" for r in resnames)
    if not mapping:
        sys.exit(f"ERROR: no usable entries found in mapping file {path}")
    return mapping


def read_categories(path):
    """Parse an optional category file into {label: category_name}.

    Each non-comment line is:  <label>  <category name with spaces allowed>
    'label' must match the short label derived from the structure filename
    (see label_of / --label-regex).
    """
    categories = {}
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                print(f"WARNING: skipping malformed category line: {line!r}")
                continue
            categories[parts[0]] = parts[1].strip()
    return categories


def make_label_fn(label_regex):
    """Return a function mapping a structure filename to a short display label.

    By default the label is the filename stem (no directory, no extension).
    With --label-regex the first capturing group of the regex is used, which
    lets you pull e.g. the PDB code out of 'aligned_4NM3_prep.pdb'.
    """
    pattern = re.compile(label_regex) if label_regex else None

    def label_of(name):
        base = os.path.splitext(os.path.basename(name))[0]
        if pattern is None:
            return base
        m = pattern.search(os.path.basename(name))
        if m and m.groups():
            return m.group(1)
        return base

    return label_of


# --------------------------------------------------------------------------- #
# Binding-site detection
# --------------------------------------------------------------------------- #
def protein_atom_indices(structure, atoms):
    """Indices of the protein atoms used to represent the binding site."""
    if atoms == "ca":
        return structure.topology.select("name CA and protein")
    # heavy: all non-hydrogen protein atoms
    return np.array(
        [a.index for a in structure.topology.atoms
         if a.residue.is_protein and not a.name.startswith("H")]
    )


def find_binding_site_residues(structure, ligand_selection, cutoff_nm, atoms,
                               max_pairs=5_000_000):
    """Residue indices whose representative atoms lie within cutoff of the ligand.

    Distances are computed between the chosen protein atoms (CA or heavy) and
    every ligand atom; a residue is in the binding site if any of its atoms is
    within `cutoff_nm`. Large pair lists are processed in chunks to bound memory.
    """
    ligand_indices = structure.topology.select(ligand_selection)
    if len(ligand_indices) == 0:
        raise ValueError(f"no atoms matched ligand selection {ligand_selection!r}")

    prot_indices = protein_atom_indices(structure, atoms)
    if len(prot_indices) == 0:
        raise ValueError("no protein atoms found for binding-site detection")

    close_atoms = set()
    pairs = np.array([[p, l] for p in prot_indices for l in ligand_indices])
    for start in range(0, len(pairs), max_pairs):
        chunk = pairs[start:start + max_pairs]
        dists = md.compute_distances(structure, chunk)[0]
        for offset in np.where(dists < cutoff_nm)[0]:
            close_atoms.add(int(chunk[offset][0]))

    residues = {structure.topology.atom(idx).residue.index for idx in close_atoms}
    return residues


def find_common_residues(structures, ligand_selections, cutoff_nm, atoms,
                         presence_threshold):
    """Residue indices present in >= presence_threshold fraction of structures.

    Residues are compared across structures by (chain index, residue number) so
    that equivalent positions match even when absolute atom indices differ.
    Returns a per-structure list of residue-index sets restricted to the shared
    positions.
    """
    per_structure_positions = []
    for i, (structure, sel) in enumerate(zip(structures, ligand_selections)):
        try:
            residues = find_binding_site_residues(structure, sel, cutoff_nm, atoms)
        except Exception as exc:  # keep going; a bad structure shouldn't abort all
            print(f"WARNING: structure {i}: {exc}")
            residues = set()
        positions = set()
        for res_idx in residues:
            res = structure.topology.residue(res_idx)
            positions.add((res.chain.index, res.resSeq))
        per_structure_positions.append(positions)
        print(f"  structure {i}: {len(positions)} binding-site residues")

    counts = defaultdict(int)
    for positions in per_structure_positions:
        for pos in positions:
            counts[pos] += 1

    threshold = max(1, int(round(len(structures) * presence_threshold)))
    common = {pos for pos, n in counts.items() if n >= threshold}
    print(f"Residue positions present in >= {threshold}/{len(structures)} "
          f"structures ({presence_threshold:.0%}): {len(common)}")
    if not common:
        sys.exit("ERROR: no common binding-site residues; lower --presence "
                 "or increase --cutoff")

    common_indices = []
    for structure in structures:
        idx = {res.index for res in structure.topology.residues
               if (res.chain.index, res.resSeq) in common}
        common_indices.append(idx)
    return common_indices


# --------------------------------------------------------------------------- #
# Coordinate extraction
# --------------------------------------------------------------------------- #
def residue_atom_indices(structure, residue_indices, atoms):
    """Atom indices (CA or heavy) for the given residues, sorted for consistency."""
    selected = []
    for res_idx in sorted(residue_indices):
        res = structure.topology.residue(res_idx)
        if atoms == "ca":
            selected.extend(a.index for a in res.atoms if a.name == "CA")
        else:
            selected.extend(a.index for a in res.atoms if not a.name.startswith("H"))
    return selected


def extract_coordinates(structures, common_residues, atoms):
    """Stack binding-site coordinates into (n_structures, n_atoms * 3).

    Requires the same atom count in every structure - satisfied when the input
    ensemble shares the same residue numbering (as after a common alignment).
    """
    per_structure_atoms = [residue_atom_indices(s, r, atoms)
                           for s, r in zip(structures, common_residues)]
    counts = {len(a) for a in per_structure_atoms}
    if len(counts) != 1 or 0 in counts:
        sys.exit(f"ERROR: inconsistent binding-site atom counts across structures "
                 f"({sorted(counts)}). Ensure all structures share the same "
                 f"residue numbering and are aligned.")

    coords = np.array([s.atom_slice(a).xyz[0]
                       for s, a in zip(structures, per_structure_atoms)])
    n_structures, n_atoms, _ = coords.shape
    return coords.reshape(n_structures, n_atoms * 3), n_atoms


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def scatter_pcs(projected, x, y, labels, colours, marker_of, cluster_legend,
                pca, outfile, title):
    """One PCi-vs-PCj scatter, coloured by cluster and shaped by category."""
    plt.figure(figsize=(12, 10))
    for marker in sorted(set(marker_of.values())):
        idx = [i for i, lab in enumerate(labels) if marker_of[lab] == marker]
        if idx:
            plt.scatter(projected[idx, x], projected[idx, y], s=100,
                        c=[colours[i] for i in idx], marker=marker)
    for i, lab in enumerate(labels):
        plt.annotate(lab, (projected[i, x], projected[i, y]),
                     fontsize=8, ha="center", va="center")
    plt.xlabel(f"PC{x + 1} ({pca.explained_variance_ratio_[x]:.2%})")
    plt.ylabel(f"PC{y + 1} ({pca.explained_variance_ratio_[y]:.2%})")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(handles=cluster_legend, title="Clusters", loc="upper right")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"  wrote {outfile}")


def plot_dendrogram(Z, labels, clusters, n_clusters, outfile):
    plt.figure(figsize=(12, 8))
    colour_of = {c: f"C{c - 1}" for c in range(1, n_clusters + 1)}

    def link_colour(k):
        return colour_of[clusters[k]] if k < len(clusters) else "gray"

    dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=8,
               link_color_func=link_colour)
    ax = plt.gca()
    for i, lbl in enumerate(ax.get_xmajorticklabels()):
        lbl.set_color(colour_of[clusters[i]])
    plt.title("Hierarchical clustering of binding-site PCA projection")
    plt.xlabel("Structure")
    plt.ylabel("Distance")
    legend = [plt.Line2D([0], [0], marker="o", color="w",
                         markerfacecolor=colour_of[c], markersize=10,
                         label=f"Cluster {c}") for c in range(1, n_clusters + 1)]
    ax.legend(handles=legend, title="Clusters", loc="upper right")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"  wrote {outfile}")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(args):
    label_of = make_label_fn(args.label_regex)
    mapping = read_ligand_mapping(args.mapping)
    categories = read_categories(args.categories) if args.categories else {}
    os.makedirs(args.outdir, exist_ok=True)

    # Load structures in the order given by the mapping file.
    structures, labels, selections = [], [], []
    for name, selection in mapping.items():
        path = os.path.join(args.pdb_dir, name)
        if not os.path.exists(path):
            print(f"WARNING: file not found, skipping: {path}")
            continue
        try:
            structures.append(md.load(path))
        except Exception as exc:
            print(f"WARNING: could not load {path}: {exc}")
            continue
        labels.append(label_of(name))
        selections.append(selection)
    if len(structures) < 2:
        sys.exit("ERROR: need at least 2 loadable structures for PCA")
    print(f"Loaded {len(structures)} structures")

    cutoff_nm = args.cutoff / 10.0  # Angstrom -> nm for MDTraj
    print("Detecting binding-site residues...")
    common_residues = find_common_residues(
        structures, selections, cutoff_nm, args.atoms, args.presence)

    coords, n_atoms = extract_coordinates(structures, common_residues, args.atoms)
    n_structures = coords.shape[0]
    print(f"PCA on {n_structures} structures x {n_atoms} {args.atoms} atoms")

    n_components = min(3, n_structures - 1)
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(coords)

    Z = linkage(projected, "ward")
    clusters = fcluster(Z, args.cluster_distance, criterion="distance")
    n_clusters = len(np.unique(clusters))
    print(f"Hierarchical clustering -> {n_clusters} clusters "
          f"(Ward distance cut {args.cluster_distance})")

    cluster_colours = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 1)))
    colours = [cluster_colours[c - 1] for c in clusters]

    # Category -> marker shape (only if a category file was supplied).
    category_of = {lab: categories.get(lab, "Other") for lab in labels}
    unique_categories = sorted(set(category_of.values()))
    marker_by_category = {cat: CATEGORY_MARKERS[i % len(CATEGORY_MARKERS)]
                          for i, cat in enumerate(unique_categories)}
    marker_of = {lab: marker_by_category[category_of[lab]] for lab in labels}

    cluster_legend = [plt.Line2D([0], [0], marker="o", color="w",
                                 markerfacecolor=cluster_colours[i], markersize=10,
                                 label=f"Cluster {i + 1}") for i in range(n_clusters)]

    print("Writing plots...")
    scatter_pcs(projected, 0, 1, labels, colours, marker_of, cluster_legend, pca,
                os.path.join(args.outdir, "pca_pc1_pc2.png"),
                "Binding-site conformations (PC1 vs PC2)")
    if n_components >= 3:
        scatter_pcs(projected, 0, 2, labels, colours, marker_of, cluster_legend, pca,
                    os.path.join(args.outdir, "pca_pc1_pc3.png"),
                    "Binding-site conformations (PC1 vs PC3)")
        scatter_pcs(projected, 1, 2, labels, colours, marker_of, cluster_legend, pca,
                    os.path.join(args.outdir, "pca_pc2_pc3.png"),
                    "Binding-site conformations (PC2 vs PC3)")
        _plot_3d(projected, labels, colours, marker_of, cluster_legend, pca,
                 os.path.join(args.outdir, "pca_3d.png"))

    plot_dendrogram(Z, labels, clusters, n_clusters,
                    os.path.join(args.outdir, "dendrogram.png"))

    write_outputs(args, labels, projected, clusters, category_of, n_components,
                  n_clusters, n_structures, n_atoms, pca)


def _plot_3d(projected, labels, colours, marker_of, cluster_legend, pca, outfile):
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection="3d")
    for marker in sorted(set(marker_of.values())):
        idx = [i for i, lab in enumerate(labels) if marker_of[lab] == marker]
        if idx:
            ax.scatter(projected[idx, 0], projected[idx, 1], projected[idx, 2],
                       s=100, c=[colours[i] for i in idx], marker=marker)
    for i, lab in enumerate(labels):
        ax.text(projected[i, 0], projected[i, 1], projected[i, 2], lab, fontsize=8)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.2%})")
    ax.set_title("Binding-site conformations (3D PCA)")
    ax.legend(handles=cluster_legend, title="Clusters", loc="upper right")
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"  wrote {outfile}")


def write_outputs(args, labels, projected, clusters, category_of, n_components,
                  n_clusters, n_structures, n_atoms, pca):
    has_categories = bool(args.categories)

    rows = []
    for i, lab in enumerate(labels):
        row = {"label": lab}
        for j in range(n_components):
            row[f"PC{j + 1}"] = projected[i, j]
        row["cluster"] = int(clusters[i])
        if has_categories:
            row["category"] = category_of[lab]
        rows.append(row)
    csv_path = os.path.join(args.outdir, "pca_projections.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}")

    txt_path = os.path.join(args.outdir, "pca_summary.txt")
    with open(txt_path, "w") as f:
        f.write("PCA of binding-site conformations\n")
        f.write("=================================\n\n")
        f.write(f"Structures analysed : {n_structures}\n")
        f.write(f"Atoms per structure : {n_atoms} ({args.atoms})\n\n")
        f.write("Explained variance:\n")
        for i, ratio in enumerate(pca.explained_variance_ratio_):
            f.write(f"  PC{i + 1}: {ratio:.4f} ({ratio:.2%})\n")
        f.write(f"  cumulative: {np.sum(pca.explained_variance_ratio_):.2%}\n\n")
        f.write(f"Clusters ({n_clusters}):\n")
        for c in range(1, n_clusters + 1):
            members = [labels[i] for i in range(len(clusters)) if clusters[i] == c]
            f.write(f"  Cluster {c} ({len(members)}): {', '.join(members)}\n")
        if has_categories:
            f.write("\nCategories:\n")
            by_cat = defaultdict(list)
            for lab in labels:
                by_cat[category_of[lab]].append(lab)
            for cat in sorted(by_cat):
                f.write(f"  {cat}: {', '.join(by_cat[cat])}\n")
    print(f"  wrote {txt_path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mapping", required=True,
                   help="TSV: '<structure_file> <ligand_resname[,resname2,...]>' "
                        "per line; '#' comments allowed. Defines which structures "
                        "to load and their ligand(s).")
    p.add_argument("--pdb-dir", default=".",
                   help="Directory containing the structure files (default: .)")
    p.add_argument("--atoms", choices=["ca", "heavy"], default="ca",
                   help="Atoms used to represent the binding site and run PCA "
                        "(default: ca)")
    p.add_argument("--cutoff", type=float, default=6.0,
                   help="Binding-site distance cutoff from the ligand, in Angstrom "
                        "(default: 6.0)")
    p.add_argument("--presence", type=float, default=0.5,
                   help="Fraction of structures a residue must appear in to be kept "
                        "as a shared binding-site residue (default: 0.5)")
    p.add_argument("--cluster-distance", type=float, default=0.75,
                   help="Ward-linkage distance cut for hierarchical clustering of "
                        "the PCA projection (default: 0.75)")
    p.add_argument("--categories", default=None,
                   help="Optional TSV: '<label> <category name>' to mark points by "
                        "shape and add a category column. Labels must match those "
                        "derived from filenames (see --label-regex).")
    p.add_argument("--label-regex", default=None,
                   help="Regex with one capture group to extract a short label from "
                        "each filename, e.g. 'aligned_(.+?)_prep' -> '4NM3'. "
                        "Default: filename without directory/extension.")
    p.add_argument("--outdir", default="pca_out",
                   help="Output directory for plots and tables (default: pca_out)")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
