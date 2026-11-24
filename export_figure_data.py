"""Export the CSV source tables accompanying all current manuscript figures."""
import csv,json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from project_paths import REPO,UNIFIED,FIGURES

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(name,rows):
    rows=list(rows);fields=list(dict.fromkeys(k for r in rows for k in r))
    with (FIGURES/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})

def main():
    FIGURES.mkdir(exist_ok=True)
    manifest={r['image_id']:r for r in read(UNIFIED/'manifest.json')['rows']}
    source=read(UNIFIED/'figure_sources.json')['figures']
    metrics=read(UNIFIED/'per_image_class.json')['rows']
    lookup={(r['result_id'],r['image_id'],r['lesion']):r for r in metrics}
    def record_path(p):return 'records/unified/'+p
    for n in [1,2]:
        tree=ET.parse(FIGURES/f'Figure {n}.drawio');rows=[]
        for cell in tree.iter('mxCell'):
            if not (cell.get('vertex')=='1' or cell.get('edge')=='1'):continue
            g=cell.find('mxGeometry');row=dict(object_id=cell.get('id'),object_type='edge' if cell.get('edge')=='1' else 'vertex',
                text=cell.get('value',''),style=cell.get('style',''))
            # Embedded PNG payloads stay in draw.io; CSV contains geometry/text only.
            if 'image=data:' in row['style']:row['style']=row['style'].split('image=data:')[0]+'image=embedded PNG'
            if g is not None:
                row.update({k:g.get(k,'') for k in ['x','y','width','height']})
                row['waypoints']=[dict(point.attrib) for point in g.iter('mxPoint')]
            rows.append(row)
        write(f'figure{n}_elements.csv',rows)
    rows=[]
    for i,image_id in enumerate(source['Figure 3']['image_ids'],1):
        r=manifest[image_id]
        for field in ['image_path','MA_path','HE_path','EX_path','SE_path']:
            rows.append(dict(panel_row=i,image_id=image_id,dataset_context=r['dataset'],panel=field,
                source_path=record_path(r[field]),material_type='author-created graphic or mask'))
    write('figure3_panels.csv',rows)
    with (UNIFIED/'comment3/figure4_image_scores.csv').open(encoding='utf-8-sig',newline='') as f:write('figure4_image_scores.csv',csv.DictReader(f))
    rows=[]
    for j,region in enumerate(source['Figure 5']['regions'],1):
        for method in ['U-Net','TransUNet','MedSAM (Full)','TF-LMedSAM']:
            r=lookup[(f'comparison/IDRiD/{method}',region['image_id'],region['lesion'])]
            rows.append(dict(panel_row=j,method=method,**region,image_path=record_path(r['image_path']),
                reference_path=record_path(r['reference_path']),prediction_path=record_path(r['prediction_path']),
                prediction_channel=r['prediction_channel'],dice=r['dice'],iou=r['iou']))
    write('figure5_regions.csv',rows)
    with np.load(FIGURES/'figure6_arrays.npz',allow_pickle=False) as arrays:
        image=arrays['input'];a=arrays['without_text'];b=arrays['with_text']
        with (FIGURES/'figure6_pixels.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f);writer.writerow(['row','column','input_intensity','without_text_field','with_text_field'])
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):writer.writerow([y,x,float(image[y,x]),float(a[y,x]),float(b[y,x])])
    with (UNIFIED/'comment3/figure7_statistics.csv').open(encoding='utf-8-sig',newline='') as f:write('figure7_statistics.csv',csv.DictReader(f))
    rows=[]
    for j,image_id in enumerate(source['Figure 8']['image_ids'],1):
        for method in ['TransUNet','TF-LMedSAM']:
            for lesion in ['MA','HE','EX','SE']:
                r=lookup[(f'cross_DDR_to_IDRiD/IDRiD/{method}',image_id,lesion)]
                rows.append(dict(panel_row=j,image_id=image_id,method=method,lesion=lesion,
                    evaluation_direction='DDR-to-IDRiD',image_path=record_path(r['image_path']),
                    reference_path=record_path(r['reference_path']),prediction_path=record_path(r['prediction_path']),
                    prediction_channel=r['prediction_channel'],dice=r['dice'],iou=r['iou']))
    write('figure8_panels.csv',rows)
    with (UNIFIED/'comment3/figure9_statistics.csv').open(encoding='utf-8-sig',newline='') as f:write('figure9_statistics.csv',csv.DictReader(f))
    csvs={1:'figure1_elements.csv',2:'figure2_elements.csv',3:'figure3_panels.csv',4:'figure4_image_scores.csv',
          5:'figure5_regions.csv',6:'figure6_pixels.csv',7:'figure7_statistics.csv',8:'figure8_panels.csv',9:'figure9_statistics.csv'}
    descriptions={1:'Editable diagram objects',2:'Editable diagram objects',3:'Panel source paths',4:'Image-level macro Dice scores',
        5:'Crop coordinates and linked masks',6:'Schematic image and spatial fields',7:'Dice and confidence intervals',
        8:'Cross-domain panel paths and class metrics',9:'Dice intervals and reference FPS'}
    write('figure_index.csv',[dict(figure=n,image_file=f'Figure {n}.png',csv_file=csvs[n],
        editable_source=f'Figure {n}.drawio' if n in [1,2] else '',description=descriptions[n],
        record_source='../records/unified/figure_sources.json' if n not in [1,2,6] else
            'drawio_validation.json' if n in [1,2] else 'figure6_arrays.npz') for n in range(1,10)])
    print('Exported CSV source tables and the figure index for all nine figures.')

if __name__=='__main__':main()
