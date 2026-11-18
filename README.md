# TF-LMedSAM experiment workflow

Use Python 3.12.14 and install requirements.txt. For pretrained model preparation and text export, install requirements-models.txt. Use requirements-plotting.txt in a separate plotting environment. The dependency versions and environment roles are recorded in configs/environment.json. The configuration follows manuscript Sections 2 and 3: 12 ViT-B layers, rank 64 adapters, taps 3/6/9/12, 256 fusion channels, 8 heads, 2 decoder layers, AdamW, 100 epochs and no random augmentation.

## Files and commands

Run `python generate_figures.py` to redraw Figures 1–9 from source_data and records. This command does not train a network. The two tabular data directories retain report inputs and executed model outputs separately.

Run `python train.py --mode smoke --device cpu --output ../数据记录/pipeline_check` for a 64×64 pipeline check. This uses a small randomly initialized 12-layer encoder and random fixed text features, with two epochs. It writes actual optimization checkpoints and per-image masks from that run.

Download official dataset archives and populate data/manifest_template.csv using actual filenames. `prepare_manifest.py` enumerates files and rejects unresolved masks unless an explicit missing-mask option is selected after annotation review. Dataset image IDs remain those supplied by the release. No patient or eye identifiers are inferred.

For image–text model training, install the official package at https://github.com/facebookresearch/segment-anything and obtain a MedSAM ViT-B checkpoint from https://github.com/bowang-lab/MedSAM . Use PubMedCLIP_ViT32.pth from the sarahESL/PubMedCLIP release at revision c5885a5c0ba8ac8d86341e57c8466e49783874c1. This source checkpoint requires conversion to a CLIPTextModel-compatible local text resource before export; the repository of original .pth files is not a ready-to-load Transformers model directory. Use the CLIP BPE tokenizer assets from openai/clip-vit-base-patch32 at revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268, with a 77-token context and a 49,408-token vocabulary. Selected identifiers and access URLs are in configs/resources.json. Record local file checksums after preparing the resources.

Export text features:
`python export_text.py --model-dir assets/text_encoder --model-id sarahESL/PubMedCLIP:PubMedCLIP_ViT32.pth@c5885a5c0ba8ac8d86341e57c8466e49783874c1 --source-url https://huggingface.co/sarahESL/PubMedCLIP --max-length 77 --output assets/text_features.npz`

Train:
`python train.py --mode train --dataset IDRiD --manifest data/idrid.csv --backbone-checkpoint assets/medsam_vit_b.pth --text-features assets/text_features.npz --variant full --output ../数据记录/idrid_full`

Variants: baseline and adapter use the same linear head, with adapters disabled/enabled. text adds text fusion and the bidirectional decoder. full adds uncertainty weighting. All final logits are bilinearly resized to the input dimensions before sigmoid and thresholding. Loss and evaluation retain independent lesion classes and positive-reference eligibility.

Audit pixels: `python audit_data.py --manifest data/idrid.csv --output ../数据记录/idrid_audit`

Create nested training subsets: `python make_subsets.py --manifest data/ddr.csv --dataset DDR --output ../数据记录/ddr_subsets`

Evaluate: `python evaluate.py --checkpoint ../数据记录/idrid_full/best_model.pth --backbone-checkpoint assets/medsam_vit_b.pth --text-features assets/text_features.npz --manifest data/ddr.csv --dataset DDR --output ../数据记录/idrid_to_ddr`

Profile: `python benchmark.py --checkpoint ../数据记录/idrid_full/best_model.pth --backbone-checkpoint assets/medsam_vit_b.pth --text-features assets/text_features.npz --device cuda --output ../数据记录/profile.json`

Measured throughput is written by benchmark.py; source_data/figure9.csv is a separate report input. Third-party comparison model implementations are not included. Table 5 is a methodological comparison. Source datasets, author confirmation, and archive registration remain external requirements. Read ../统一说明.txt for the provenance of this material package.

## Report-linked mask-response records

`report_linked_records.py` builds a procedural response simulator and reloads
its saved states to generate binary outputs. Its configuration uses the
reported aggregate Dice and IoU as calibration targets. The response generator
requires reference masks; it does not perform image-based neural segmentation.
Saved `.npz` files contain simulator parameters. Calibration logs record the
actual parameter fitting process; no model-training epochs are represented.

Build in a new empty directory:
`python report_linked_records.py build --output ../数据记录/report_linked_demo_new`

Replay and verify existing records:
`python report_linked_records.py verify --output ../数据记录/report_linked_demo`

The output contains `split.csv`, separate train/validation/test lists,
1024 by 1024 generated images and four reference masks per case, saved response
states, calibration histories and logs, per-image-class outputs and metrics,
aggregate metrics, and verification records. Identifiers begin with DEMO_.
Patient and eye identifiers are empty. See the external provenance notice for
the scope and interpretation of these records.
