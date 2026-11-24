# TF-LMedSAM code and supporting records

This repository brings the current code, Figure 1–9 assets and source CSVs, Tables 1–5, and linked evaluation records into one portable directory. Start with the material map below.


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

The first command exports CSVs for all nine figures, their source-file links and the Figure 6 spatial fields. The second redraws Figures 3, 4, 5, 7, 8 and 9 from the active records into `figures/`. The third regenerates the editable diagrams and their original graphic assets