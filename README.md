# TF-LMedSAM Directory Guide

## Root Scripts

| File | Purpose |
| --- | --- |
| `README.md` | File index and purpose descriptions for this directory. |
| `train.py` | Training entry point that calls `tflmedsam/engine.py` for manifest-based training or pipeline checks. |
| `evaluate.py` | Loads a trained checkpoint, evaluates a selected dataset partition, and exports predictions, per-image metrics, and evaluation context. |
| `benchmark.py` | Measures inference speed and parameter counts and exports profiling results. |
| `prepare_manifest.py` | Scans dataset images and lesion annotations, matches file paths, and generates image identifiers and training, validation, and test partitions. |
| `audit_data.py` | Audits images and masks listed in a manifest and summarizes lesion pixels and data distributions. |
| `make_subsets.py` | Generates nested training subsets at 10%, 20%, 40%, 60%, 80%, and 100%, together with identifier lists. |
| `export_text.py` | Encodes lesion text prompts and exports text features for the model. |
| `generate_figures.py` | Reads configurations and numerical records to generate Figures 1–9 and export plotting values and arrays; some images and masks are constructed programmatically. |
| `generate_curves.py` | Generates deterministic simulated training curves and epoch records for plotting and pipeline exercises. |
| `build_reference.py` | Constructs reference masks, prediction masks, and per-image metrics from the aggregate values in Tables 2–4 for metric and pipeline checks. |
| `verify_masks.py` | Checks saved mask hashes, recomputes confusion counts, Dice, and IoU, and compares them with recorded values. |
| `verify_run.py` | Loads a pipeline-check checkpoint and verifies reproduced prediction masks and pixel-level metrics. |
| `run_all.ps1` | Runs contract tests, constructed-mask verification, simulated curve generation, figure generation, and pipeline checks for four model variants. |

## Configuration Files: `configs/`

| File | Purpose |
| --- | --- |
| `configs/protocol.json` | Defines the random seed, lesion classes, partition counts, preprocessing, model architecture, training parameters, loss settings, evaluation rules, and ablation configurations. |
| `configs/prompts.json` | Stores English text prompts for the MA, HE, EX, and SE lesion classes. |
| `configs/environment.json` | Records Python and dependency versions, execution environment details, and references to dependency files. |
| `configs/resources.json` | Records external resource links and version information for IDRiD, DDR, MedSAM, PubMedCLIP, and Segment Anything. |

## Data Manifest Template: `data/`

| File | Purpose |
| --- | --- |
| `data/manifest_template.csv` | Defines manifest fields for dataset names, image identifiers, partitions, patient and eye identifiers, image paths, and lesion mask paths. |

## Table and Figure Values: `source_data/`

| File | Corresponding Content and Purpose |
| --- | --- |
| `source_data/table1.csv` | Table 1: dataset image counts, partition counts, and lesion pixel fractions. |
| `source_data/table2.csv` | Table 2: aggregate Dice and IoU values for methods evaluated on IDRiD and DDR. |
| `source_data/table3.csv` | Table 3: Dice, IoU, and metric gains for ablation configurations. |
| `source_data/table4.csv` | Table 4: aggregate Dice values for cross-dataset evaluation from IDRiD to DDR and from DDR to IDRiD. |
| `source_data/table5.json` | Table 5: comparison entries containing method names, years, segmentation designs, guidance types, adaptation strategies, and DOIs. |
| `source_data/figure7.csv` | Figure 7: training sample fractions, image counts, and Dice values for TF-LMedSAM and TransUNet. |
| `source_data/figure9.csv` | Figure 9: FPS and IDRiD Dice values for each method. |

## Model and Data Modules: `tflmedsam/`

| File | Purpose |
| --- | --- |
| `tflmedsam/__init__.py` | Defines package-level lesion class constants. |
| `tflmedsam/model.py` | Implements adapters, image–text interaction, segmentation models, loss functions, and simplified encoders and segmenters for pipeline checks. |
| `tflmedsam/engine.py` | Implements model construction, text feature loading, training, validation, checkpoint saving, and prediction export. |
| `tflmedsam/data.py` | Implements manifest handling, image and mask loading, data auditing, and check datasets. |
| `tflmedsam/metrics.py` | Computes pixel confusion counts, Dice, IoU, and aggregate metrics. |
| `tflmedsam/io.py` | Provides CSV and JSON input/output, file hashing, and identifier handling utilities. |

## Tests and Dependencies

| File | Purpose |
| --- | --- |
| `tests/test_contract.py` | Tests implementation contracts for model structure, parameters, outputs, loss functions, and evaluation rules. |
| `requirements.txt` | Core runtime dependencies. |
| `requirements-models.txt` | Additional dependencies for model resource loading and text feature export. |
| `requirements-plotting.txt` | Plotting dependencies. |
