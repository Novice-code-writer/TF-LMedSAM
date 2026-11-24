# Figures and source data

All current Figure 1–9 exports, editable diagrams and corresponding CSV files are stored here. `figure_index.csv` maps each figure to its image, CSV and supporting records. Run `python export_figure_data.py` from the repository root to regenerate the CSV tables. Raster image pixels and masks remain in the linked PNG/NPZ records; source CSV files identify those files rather than duplicating millions of pixels as text.

`Figure 1.drawio` and `Figure 2.drawio` contain editable text, modules and orthogonal connectors, with embedded original graphics. `figure1_elements.csv` and `figure2_elements.csv` list diagram objects. `drawio_assets/` can be regenerated with `build_drawio_figures.py`; embedded assets are sufficient to open the retained diagrams.

`figure6_arrays.npz` and `figure6_pixels.csv` contain the original schematic background and spatial fields. Figure 9 vertical intervals describe Dice, while its horizontal FPS coordinates remain reference values. Generated timing examples are kept separately under `records/unified/comment4/reference_timing/`.

Original images or annotations from IDRiD/DDR are not redistributed as such in these author-created visual panels. See `../docs/DATA_AND_LICENSES.md` and the external source notices.
