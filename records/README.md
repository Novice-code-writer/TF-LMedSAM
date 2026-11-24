# Evaluation records

`unified/` is the active record family. Its manifest, shared references, packed condition outputs, image-level counts, summaries and statistical tests form one linked package. `build_inputs/` stores the original manifest and the historical reference-score ordering required by the generated-record builder; image files are reused from `unified/data/`. The local `provenance_archive/` retains earlier environment/index/source maps and workbook snapshots and is excluded from the review publication subset. Original hashes in archived material describe the earlier workspace, not the relocated repository.

Use `python unified_records.py verify` to verify the active records. Use `python verify_repository.py` for the portable file inventory and figure CSV checks. These checks establish arithmetic and file consistency, not a real neural experiment. Model dependencies, trained weights and genuine dataset inputs are separate requirements.
