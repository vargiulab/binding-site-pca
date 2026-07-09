# Binding-site PCA of a structural ensemble

**Author:** Han Kurt

Principal-component analysis of the **binding-site conformations** across an
ensemble of pre-aligned protein structures. The tool finds the residues that
line the ligand in each structure, keeps the ones shared by the ensemble,
extracts their coordinates (Cα or all heavy atoms), and projects the ensemble
onto its principal components. The projection is clustered (Ward hierarchical
clustering with a distance cut) and rendered as PC scatter plots, a 3D plot and
a coloured dendrogram, plus a projections table and a text summary.

Use it to answer *"how many distinct binding-site conformations does my target
adopt, and which structures share one?"* — for a panel of co-crystal structures,
docked poses, or PDB snapshots of an MD trajectory.

The target is **not** hard-coded: the structure list, the ligand of each
structure, and the optional grouping labels all come from small plain-text
files, so the same script works for any system.

---

## What it does

1. **Detect the binding site.** For each structure, protein atoms (Cα or heavy)
   within `--cutoff` Å of the mapped ligand define the binding-site residues.
2. **Keep shared residues.** Residues are matched across structures by
   *(chain, residue number)*; those present in at least `--presence` of the
   ensemble are retained, so a residue missing from a few structures doesn't
   drop the whole position.
3. **Extract coordinates.** The selected atoms are stacked into one matrix
   (one row per structure). All structures must share the same residue
   numbering and be superposed beforehand (see the note below).
4. **PCA + clustering.** PCA (up to 3 components) followed by Ward clustering of
   the projection with a distance cut (`--cluster-distance`).
5. **Report.** PC1–PC2 / PC1–PC3 / PC2–PC3 scatters, a 3D scatter, a coloured
   dendrogram, `pca_projections.csv` and `pca_summary.txt`.

> **Alignment.** This script aligns only the *selected binding-site atoms* before
> PCA, so the projection reflects local conformational change rather than
> rigid-body motion. It assumes your structures are **already superposed** onto a
> common reference (e.g. `align` in PyMOL/VMD, or your MD alignment step). It
> does not do the whole-structure superposition for you.

---

## Requirements

| Package | Used for |
|---|---|
| **[MDTraj](https://www.mdtraj.org/)** | reading structures, distances, atom slicing |
| **[scikit-learn](https://scikit-learn.org/)** | PCA |
| **[SciPy](https://scipy.org/)** | hierarchical clustering |
| **[pandas](https://pandas.pydata.org/)** | projections table |
| **[matplotlib](https://matplotlib.org/)** | plots |

```bash
conda create -n bspca python=3.11 mdtraj scikit-learn scipy pandas matplotlib
conda activate bspca
```

---

## Input files

**Mapping file** (`--mapping`, required) — one line per structure:

```
# structure_file        ligand_resname[,resname2,...]
aligned_1J1B_prep.pdb   ANP
aligned_4NM3_prep.pdb   ADP
aligned_5KPK_prep.pdb   6VK
```

The structure files are read from `--pdb-dir`. Multiple comma-separated residue
names are combined (e.g. `ADP,MG`). See [`example_mapping.tsv`](example_mapping.tsv).

**Category file** (`--categories`, optional) — assigns a group to each label so
points are drawn with distinct marker shapes and a `category` column is added:

```
# label   category name
4NM3      ADP/ANP
5KPK      Type-I inhibitors
```

Labels must match the short label derived from each filename. See
[`example_categories.tsv`](example_categories.tsv).

---

## Usage

```bash
# Simplest: Cα PCA, labels = filename stem
python pca_binding_site.py --pdb-dir ./structures --mapping example_mapping.tsv

# Pull a clean PDB code out of 'aligned_4NM3_prep.pdb' and group by category
python pca_binding_site.py \
    --pdb-dir ./structures \
    --mapping example_mapping.tsv \
    --label-regex 'aligned_(.+?)_prep' \
    --categories example_categories.tsv \
    --atoms heavy \
    --outdir pca_out
```

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--mapping` | *(required)* | structure → ligand mapping file |
| `--pdb-dir` | `.` | directory holding the structure files |
| `--atoms` | `ca` | `ca` or `heavy` atoms for the binding site + PCA |
| `--cutoff` | `6.0` | binding-site distance from ligand, in Å |
| `--presence` | `0.5` | fraction of structures a residue must appear in to be kept |
| `--cluster-distance` | `0.75` | Ward distance cut for clustering the projection |
| `--categories` | *(none)* | optional label → group file for marker shapes |
| `--label-regex` | *(none)* | regex (one capture group) to shorten filenames into labels |
| `--outdir` | `pca_out` | output directory |

---

## Outputs (in `--outdir`)

| File | Contents |
|---|---|
| `pca_pc1_pc2.png`, `pca_pc1_pc3.png`, `pca_pc2_pc3.png` | PC scatter plots, coloured by cluster, shaped by category |
| `pca_3d.png` | 3D PCA scatter |
| `dendrogram.png` | Ward dendrogram coloured by cluster |
| `pca_projections.csv` | per-structure PC scores, cluster and category |
| `pca_summary.txt` | variance explained, cluster membership, category listing |

---

## Notes & tips

- **No common residues?** Lower `--presence` or raise `--cutoff`.
- **"Inconsistent atom counts" error** means your structures don't share the same
  residue numbering for the binding site — renumber/curate them first.
- The number of clusters is set by the `--cluster-distance` cut, not fixed in
  advance; lower it for more, finer clusters.
- The bundled `example_mapping.tsv` / `example_categories.tsv` come from a
  GSK-3β co-crystal panel and are provided purely as a format reference — the
  PDB files themselves are not included.

---

## License

MIT — see [LICENSE](LICENSE).
