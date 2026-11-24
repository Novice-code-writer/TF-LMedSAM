"""Portable paths anchored to this repository, independent of the working directory."""
from pathlib import Path
REPO = Path(__file__).resolve().parent
RECORDS = REPO / "records"
UNIFIED = RECORDS / "unified"
FIGURES = REPO / "figures"
TABLES = REPO / "tables"
DOCS = REPO / "docs"
def source_file(name):
    if name in ("figure7.csv", "figure9.csv"):
        return FIGURES / (Path(name).stem + "_report_values.csv")
    return TABLES / "source_data" / name
