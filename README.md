# TF-LMedSAM code and supporting records

This repository brings the current code, Figure 1–9 assets and source CSVs, Tables 1–5, and linked evaluation records into one portable directory. Start with the material map below. The supplied evaluation records are a generated reference package for checking calculations and workflows; they are not a substitute for trained model checkpoints or measured inference logs. Keep [the source notice](材料来源与适用范围.txt) and [the unified notice](统一说明.txt) with the package.

## Git staging scope

The bulk folders `records/unified/data/` and `records/unified/conditions/*/predictions/` are retained locally and excluded from Git at the author's request. This Git revision supplies code, figures, CSV/JSON evaluation records and documentation; it does not include those bulk image/mask files. The full local inventory and pixel-level verification commands require those folders to be restored separately before use in a fresh checkout.

## Directory map

```text
TF-LMedSAM/
├── README.md
├── project_paths.py
├── train.py, evaluate.py, benchmark.py
├── unified_records.py, render_unified.py, export_figure_data.py
├── verify_repository.py, audit_timing.py
├── configs/                    Model, prompts and resource configurations
├── tflmedsam/                  Model, metrics, statistics, I/O and data modules
├── tests/                      Model contracts and record/statistics checks
├── data/                       Input manifest template
├── figures/                    All current images, draw.io sources and figure CSVs
├── tables/                     Current XLSX tables and frozen table targets
├── records/
│   ├── unified/                Active images, masks, partitions, metrics and statistics
│   ├── build_inputs/           Manifest and initial reference-score ordering
├── docs/                       Reviewer map, licenses and prior workspace notes
```

## Find a material

| Material | Path |
|---|---|
| All figures and their CSV data | [figures/figure_index.csv](figures/figure_index.csv), [figure guide](figures/README.md) |
| Editable method diagrams | [Figure 1.drawio](figures/Figure%201.drawio), [Figure 2.drawio](figures/Figure%202.drawio) |
| Current workbooks | [tables/](tables/README.md); Tables 2–4 include Statistics worksheets |
| Shared images and reference masks | `records/unified/data/`; [manifest](records/unified/manifest.json) |
| Partition IDs and nested training subsets | `records/unified/splits/` |
| Condition settings and packed four-class outputs | `records/unified/conditions/<result_hash>/config.json` and `predictions/*.npz` |
| Image-level confusion counts and lesion metrics | [per_image_class.json](records/unified/per_image_class.json) |
| Image-level macro scores | [per_image.json](records/unified/per_image.json) |
| Estimates, intervals and paired comparisons | [summary.json](records/unified/summary.json), [paired_comparisons.json](records/unified/paired_comparisons.json) |
| Statistical protocol and empty-reference checks | `records/unified/comment3/` |
| Hardware inventory and CPU benchmark specification | `records/unified/comment4/` |
| Figure source inventory and license links | `records/unified/comment5/figure_provenance.json`, [data licenses](docs/DATA_AND_LICENSES.md) |
| One-to-one map for the five reviewer comments | [docs/REVIEWER_MATERIALS.md](docs/REVIEWER_MATERIALS.md) |
| Model configuration | [configs/protocol.json](configs/protocol.json), `configs/prompts.json`, `configs/resources.json` |
| Frozen table targets and reference FPS values | `tables/source_data/`; `figures/figure7_report_values.csv`, `figure9_report_values.csv` |
| File hashes and packaging validation | [repository_manifest.json](repository_manifest.json), [packaging_validation.json](packaging_validation.json) |

The manuscript directory label `unified_records` corresponds to `records/unified/` in this Git layout. See [the path migration table](docs/PATH_MIGRATION.csv) for previous workspace names.

## Verify supplied records

Run from the repository root with Python 3.12 and NumPy, SciPy and Pillow installed:

```powershell
python verify_repository.py
python unified_records.py verify
python -m unittest discover -s tests -p "test_*records.py" -v
python -m unittest discover -s tests -p "test_statistics_eligibility.py" -v
```

The full neural contract suite additionally needs PyTorch. The dependency files retain the project version pins; the recorded environments are in `records/unified/runtime.json` and `comment3/`. Installed versions should be checked before running the model workflows.

## Figures and CSV export

```powershell
python export_figure_data.py
python render_unified.py
python build_drawio_figures.py
```

The first command exports CSVs for all nine figures, their source-file links and the Figure 6 spatial fields. The second redraws Figures 3, 4, 5, 7, 8 and 9 from the active records into `figures/`. The third regenerates the editable diagrams and their original graphic assets. To export the native diagrams with the installed draw.io application, use `export_drawio_figures.ps1 -DrawioExecutable <path-to-draw.io.exe>`. Exporting or editing figures changes file hashes; `python verify_repository.py --refresh` intentionally refreshes the inventory after review.

Figure 9 FPS coordinates remain reference values. The files under `records/unified/comment4/reference_timing/` are explicitly generated timing examples and are rejected by the measured-timing audit. They do not describe measured speed on this CPU.

## Recompute or rebuild the reference package

```powershell
python unified_records.py derive
python unified_records.py build --output records/new_build
python unified_records.py verify --output records/new_build
```

Use an empty new directory for a rebuild. The builder reuses the supplied images/reference masks, the manifest in `records/build_inputs/`, the initial reference-score ordering and frozen report targets. It does not train the neural methods. After changing summaries, run `export_figure_data.py`, review figure/table consistency, and refresh the file inventory.

## Neural model entry points

`train.py`, `evaluate.py`, `prepare_manifest.py`, `make_subsets.py` and `export_text.py` provide the model workflow. Install model dependencies from `requirements-models.txt` and supply actual dataset files, the pretrained backbone, text resources and matching trained checkpoints. These resources are not supplied by the generated-record package.

```powershell
python benchmark.py --checkpoint <trained.pt> --backbone-checkpoint <backbone.pth> --text-features <text.npz> --device cpu --threads 4 --interop-threads 1 --repeats 5 --method-label TF-LMedSAM --output records/runs/timing.json
```

This benchmark loads the trained state strictly and checks backbone/text hashes and the architecture label. U-Net, TransUNet and fully fine-tuned MedSAM require their own implementations and checkpoints. Relabelling a TF-LMedSAM variant does not benchmark a different architecture.

## Publishing this directory

The existing `.git` history and remote configuration are preserved. `.gitignore` excludes caches, `node_modules`, local runs and local backups. Stage the current code, figure assets/CSVs, current tables, unified evaluation records, reviewer documentation and both source notices. Local archives, duplicate snapshots, manuscript editing logs and workbook-export intermediates are excluded by .gitignore and the publication inventory. The old workspace folders outside this repository are retained as local originals and are not required by the commands above. Historical code snapshots and manuscript-editing logs remain local and are excluded from the review publication subset. This operation does not push to GitHub, publish a release, modify the Word manuscript or update the external DOI record.
