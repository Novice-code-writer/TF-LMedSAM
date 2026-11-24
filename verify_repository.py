"""Validate portable material paths, CSV/figure joins and the repository inventory."""
import argparse,csv,json,hashlib,os
from pathlib import Path
from project_paths import REPO,UNIFIED,FIGURES

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def inventory():
    for folder,dirs,names in os.walk(REPO,followlinks=False):
        dirs[:]=[d for d in dirs if d not in ['.git','node_modules','__pycache__','.venv','.pytest_cache','local_backups','new_build','runs','examples','code_snapshot','legacy','provenance_archive']
                 and not (Path(folder)/d).is_symlink() and not (Path(folder)/d).is_junction()]
        for name in sorted(names):
            p=Path(folder)/name
            if name in ['repository_manifest.json','packaging_validation.json','manuscript_changes.json','workbook_data.json','workbook_export.json','run_all.ps1'] or name.endswith(('.pyc','.inspect.ndjson')) or name.startswith(('~$','previous_')):continue
            yield p
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--refresh',action='store_true');a=ap.parse_args()
    checked=0
    def exists(path):
        nonlocal checked
        if not path.is_file():raise ValueError(f'Missing material: {path}')
        checked+=1
    for name in ['README.md','材料来源与适用范围.txt','统一说明.txt','docs/REVIEWER_MATERIALS.md','docs/DATA_AND_LICENSES.md']:
        exists(REPO/name)
    entries=rows(FIGURES/'figure_index.csv');assert len(entries)==9
    for row in entries:
        exists(FIGURES/row['image_file']);exists(FIGURES/row['csv_file'])
        if row['editable_source']:exists(FIGURES/row['editable_source'])
    for name in ['figure3_panels.csv','figure5_regions.csv','figure8_panels.csv']:
        for row in rows(FIGURES/name):
            for field in ['source_path','image_path','reference_path','prediction_path']:
                if row.get(field):exists(REPO/row[field])
    intervals=rows(FIGURES/'figure9_statistics.csv')
    assert [float(r['reported_fps']) for r in intervals]==[45.,20.,15.,28.]
    assert all(r['timing_measurement_supplied'].lower()=='false' for r in intervals)
    manifest=REPO/'repository_manifest.json'
    if a.refresh:
        files=[dict(path=str(p.relative_to(REPO)).replace('\\','/'),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(inventory())]
        manifest.write_text(json.dumps(dict(path_base='repository root',files=files),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    expected=json.loads(manifest.read_text(encoding='utf-8'))['files']
    for row in expected:
        p=REPO/row['path'];exists(p)
        if p.stat().st_size!=row['bytes'] or digest(p)!=row['sha256']:raise ValueError(f'Changed file: {row["path"]}')
    actual={str(p.relative_to(REPO)).replace('\\','/') for p in inventory()}
    assert actual=={r['path'] for r in expected}, 'Inventory is missing current files; review and use --refresh'
    result=dict(passed=True,figure_count=9,files_hashed=len(expected),material_path_checks=checked,
        total_bytes=sum(r['bytes'] for r in expected),maximum_file_bytes=max(r['bytes'] for r in expected),
        internal_paths_verified=True,git_history_preserved=(REPO/'.git').is_dir(),
        source_notices_preserved=True,fps_reference_status_preserved=True,neural_benchmark_executed=False)
    (REPO/'packaging_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
