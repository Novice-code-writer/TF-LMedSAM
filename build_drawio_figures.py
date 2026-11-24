"""Editable draw.io method diagrams with embedded text-free PNG assets."""
from pathlib import Path
import base64, json, math, hashlib
import xml.etree.ElementTree as E
from PIL import Image, ImageDraw

from project_paths import REPO as ROOT, FIGURES as OUT
ASSETS=OUT/'drawio_assets'
ASSETS.mkdir(exist_ok=True)
INK='#233441';BLUE='#DFEBFA';GREEN='#E2EFDC';PURPLE='#E9E0F0'

def assets():
    # Original schematic icons, with alpha backgrounds and no text.
    s=768
    im=Image.new('RGBA',(s,s),(0,0,0,0));px=im.load()
    for y in range(s):
        for x in range(s):
            r=math.hypot((x-384)/340,(y-384)/340)
            if r<=1:
                light=max(0,1-r*r)
                px[x,y]=(int(93+70*light),int(40+38*light),int(23+19*light),255)
    d=ImageDraw.Draw(im)
    d.ellipse((128,305,241,418),fill='#EDCE83')
    vessels=[[(190,363),(300,250),(430,180),(605,165)],[(190,363),(320,320),(490,290),(657,325)],
             [(190,363),(320,420),(480,510),(610,590)],[(190,363),(260,500),(350,610)],
             [(300,250),(310,150)],[(490,290),(550,235)],[(480,510),(490,620)]]
    for line in vessels:d.line(line,fill='#743B2C',width=9,joint='curve')
    d.ellipse((460,375,481,396),fill='#6E2523');d.ellipse((515,450,548,483),fill='#E9D895')
    im.save(ASSETS/'fundus_schematic.png',dpi=(600,600))
    im=Image.new('RGBA',(s,s),(0,0,0,0));d=ImageDraw.Draw(im)
    colors=['#ABC8DF','#C5DAB7','#E8C69B','#C7BDDA']
    for row in range(4):
        for col in range(4):
            x=70+col*160;y=70+row*160
            d.rounded_rectangle((x,y,x+135,y+135),radius=15,fill=colors[(row+col)%4],outline=INK,width=6)
    im.save(ASSETS/'token_grid.png',dpi=(600,600))
    im=Image.new('RGBA',(s,s),(0,0,0,0));d=ImageDraw.Draw(im)
    for k,c in [(2,'#BBD3E8'),(1,'#C4DEC1'),(0,'#D7C8E5')]:
        x=95+k*85;y=170-k*60
        d.rounded_rectangle((x,y,x+400,y+440),radius=20,fill=c,outline=INK,width=8)
        for j in range(1,4):
            d.line((x+j*100,y,x+j*100,y+440),fill='#FFFFFF',width=5)
            d.line((x,y+j*110,x+400,y+j*110),fill='#FFFFFF',width=5)
    im.save(ASSETS/'feature_maps.png',dpi=(600,600))
    im=Image.new('RGBA',(s,s),(0,0,0,0));d=ImageDraw.Draw(im)
    for j,c in enumerate(['#AFC8DF','#BBD5B6','#E3C89F','#CABCDD']):
        x=40+(j%2)*365;y=40+(j//2)*365
        d.rounded_rectangle((x,y,x+320,y+320),radius=22,fill='#F9FBFD',outline=INK,width=8)
        d.ellipse((x+60,y+70,x+140,y+150),fill=c)
        d.ellipse((x+180,y+180,x+238,y+238),fill=c)
    im.save(ASSETS/'lesion_masks.png',dpi=(600,600))

class Diagram:
    def __init__(self,n,w,h):
        self.n=n;self.w=w;self.h=h;self.items=[];self.labels=[];self.modules=[];self.edges=[]
        self.mx=E.Element('mxfile',host='app.diagrams.net',agent='draw.io',version='26.0.0',type='device')
        diagram=E.SubElement(self.mx,'diagram',id=f'figure-{n}',name=f'Figure {n}')
        model=E.SubElement(diagram,'mxGraphModel',dx=str(w),dy=str(h),grid='1',gridSize='10',guides='1',
          tooltips='1',connect='1',arrows='1',fold='1',page='1',pageScale='1',pageWidth=str(w),pageHeight=str(h),
          math='0',shadow='0',background='#FFFFFF')
        self.root=E.SubElement(model,'root');E.SubElement(self.root,'mxCell',id='0');E.SubElement(self.root,'mxCell',id='1',parent='0')
    def cell(self,id,value,style,x,y,w,h):
        c=E.SubElement(self.root,'mxCell',id=id,value=value,style=style,vertex='1',parent='1')
        E.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'})
        return c
    def panel(self,id,x,y,w,h,title,color):
        self.cell(id,'',f'rounded=1;arcSize=4;whiteSpace=wrap;html=0;fillColor={color};strokeColor=#A1AFB7;strokeWidth=2;connectable=0;',x,y,w,h)
        self.items.append(('rect',x,y,w,h,color,True))
        self.text(id+'-title',title,x+25,y+15,min(w-50,800),40,28,True,align='left')
    def text(self,id,value,x,y,w,h,size=20,bold=False,align='center'):
        self.cell(id,value,f'text;html=0;whiteSpace=wrap;overflow=hidden;fillColor=none;strokeColor=none;fontFamily=Arial;fontSize={size};fontColor={INK};fontStyle={1 if bold else 0};align={align};verticalAlign=middle;spacing=0;connectable=0;',x,y,w,h)
        self.labels.append((id,x,y,w,h));self.items.append(('text',x,y,w,h,value,size,bold,align))
    def module(self,id,x,y,w,h,title,lines,color='#FFFFFF',body_size=20,asset=None):
        self.cell(id,'',f'rounded=1;arcSize=7;whiteSpace=wrap;html=0;fillColor={color};strokeColor={INK};strokeWidth=2;',x,y,w,h)
        self.modules.append((id,x,y,w,h));self.items.append(('rect',x,y,w,h,color,False))
        self.text(id+'-title',title,x+12,y+12,w-24,36,22,True)
        if asset:
            self.asset(id+'-asset',asset,x+20,y+58,84,84)
            self.text(id+'-body','\n'.join(lines),x+116,y+56,w-130,h-65,body_size)
        else:self.text(id+'-body','\n'.join(lines),x+15,y+58,w-30,h-70,body_size)
    def asset(self,id,name,x,y,w,h):
        data=base64.b64encode((ASSETS/name).read_bytes()).decode()
        self.cell(id,'',f'shape=image;imageAspect=1;aspect=fixed;verticalLabelPosition=bottom;verticalAlign=top;image=data:image/png,{data};',x,y,w,h)
        self.items.append(('image',x,y,w,h,name))
    def edge(self,id,pts):
        assert all(a[0]==b[0] or a[1]==b[1] for a,b in zip(pts,pts[1:])),id
        c=E.SubElement(self.root,'mxCell',id=id,value='',style=f'edgeStyle=none;rounded=0;curved=0;html=0;endArrow=block;endFill=1;strokeWidth=2;strokeColor={INK};',edge='1',parent='1')
        g=E.SubElement(c,'mxGeometry',relative='1',**{'as':'geometry'})
        E.SubElement(g,'mxPoint',x=str(pts[0][0]),y=str(pts[0][1]),**{'as':'sourcePoint'})
        E.SubElement(g,'mxPoint',x=str(pts[-1][0]),y=str(pts[-1][1]),**{'as':'targetPoint'})
        if len(pts)>2:
            arr=E.SubElement(g,'Array',**{'as':'points'})
            for x,y in pts[1:-1]:E.SubElement(arr,'mxPoint',x=str(x),y=str(y))
        self.edges.append((id,pts));self.items.append(('edge',pts))
    def verify(self):
        # Lines may touch box boundaries but must not intersect label/module interiors.
        for edge,pts in self.edges:
            for a,b in zip(pts,pts[1:]):
                for name,x,y,w,h in self.labels+self.modules:
                    if a[0]==b[0]:hit=x+.5<a[0]<x+w-.5 and max(min(a[1],b[1]),y+.5)<min(max(a[1],b[1]),y+h-.5)
                    else:hit=y+.5<a[1]<y+h-.5 and max(min(a[0],b[0]),x+.5)<min(max(a[0],b[0]),x+w-.5)
                    assert not hit,(edge,'crosses',name)
        for c in self.root.findall('mxCell'):
            assert not any('\u4e00'<=ch<='\u9fff' for ch in c.get('value',''))
        return {'editable_text_labels':len(self.labels),'modules':len(self.modules),'orthogonal_arrows':len(self.edges),'line_text_intersections':0}
    def save(self):
        check=self.verify()
        E.indent(self.mx)
        E.ElementTree(self.mx).write(OUT/f'Figure {self.n}.drawio',encoding='utf-8',xml_declaration=True)
        return check

def figure1():
    f=Diagram(1,1640,830)
    f.panel('encoding',20,20,1600,350,'A  Image-to-feature encoding',BLUE)
    f.panel('layer',20,450,1600,350,'B  Pre-normalized Transformer layer','#F2F3F4')
    f.module('input',60,125,280,170,'Fundus image',['RGB input','1024 × 1024 pixels'],asset='fundus_schematic.png',body_size=18)
    f.module('embedding',410,125,340,170,'Token embedding',['16 × 16 patch projection','Positional embeddings'],asset='token_grid.png',body_size=18)
    f.module('encoder',820,125,370,170,'MedSAM ViT-B encoder',['12 Transformer layers','Embedding dimension: 768'],'#CADCF3')
    f.module('features',1260,125,320,170,'Visual representations',['Spatial token features','For lesion segmentation'],asset='feature_maps.png',body_size=18)
    for j,(a,b) in enumerate([(340,410),(750,820),(1190,1260)]):f.edge(f'encoding-flow-{j}',[(a,210),(b,210)])
    f.edge('layer-detail',[(1005,295),(1005,410),(810,410),(810,450)])
    f.text('detail-label','Encoder layer detail',1090,385,300,35,18)
    f.module('layer-input',60,565,240,140,'Layer input',['Token sequence'])
    f.module('attention',370,540,380,190,'Attention sublayer',['Layer normalization (LN)','Multi-head self-attention (MSA)','Residual addition'],GREEN,19)
    f.module('ffn',820,540,400,190,'Feed-forward sublayer',['Layer normalization (LN)','Multilayer perceptron (MLP)','Residual addition'],'#F9DFC2',19)
    f.module('layer-output',1290,565,290,140,'Layer output',['Updated tokens'])
    for j,(a,b) in enumerate([(300,370),(750,820),(1220,1290)]):f.edge(f'layer-flow-{j}',[(a,635),(b,635)])
    f.text('repeat-note','The same layer structure is repeated across the 12-layer image encoder.',240,750,1160,32,18)
    return f

def figure2():
    f=Diagram(2,1700,1430)
    f.panel('visual',20,20,600,1380,'A  Visual stream',BLUE)
    f.panel('textual',650,20,1030,460,'B  Textual stream',GREEN)
    f.panel('fusion',650,510,1030,890,'C  Fusion and decoding stream',PURPLE)
    f.module('fundus',140,110,420,160,'Fundus image input',['RGB input','1024 × 1024 pixels'],asset='fundus_schematic.png')
    f.module('adapted',140,340,420,300,'Adapted MedSAM ViT-B',[
      'Frozen pretrained encoder · 12 layers','Trainable adapter in every FFN','Down-projection · GELU · up-projection',
      'Bottleneck width: 64','Depth groups: 1–3 / 4–6 / 7–9 / 10–12','Feature taps: layers 3, 6, 9 and 12'],'#CADCF3',19)
    f.module('alignment',140,770,420,200,'Depth-feature alignment',[
      '1 × 1 projection to 256 channels','Pooling factors: 1 / 2 / 4 / 8','Resize to the native grid · mean fusion'],body_size=19)
    f.edge('fundus-to-encoder',[(350,270),(350,340)])
    f.edge('encoder-to-alignment',[(350,640),(350,770)])
    f.text('depth-label','Four depth features',365,690,220,35,18,align='left')
    f.text('aligned-label','Aligned visual features',160,985,380,35,18)
    f.text('training-note','Pretrained image weights: frozen\nAdapters and added modules: trainable',160,1040,400,65,18)
    # Short textual-stream boxes use their own title and body labels.
    for id,y,h,title,body in [
      ('descriptions',100,100,'Fixed clinical lesion descriptions','MA · HE · EX · SE'),
      ('text-encoder',250,95,'Frozen PubMedCLIP text encoder','ViT-B/32 text tower · 512-dimensional outputs')]:
        f.cell(id,'',f'rounded=1;arcSize=7;html=0;fillColor=#FFFFFF;strokeColor={INK};strokeWidth=2;',760,y,780,h)
        f.modules.append((id,760,y,780,h));f.items.append(('rect',760,y,780,h,'#FFFFFF',False))
        f.text(id+'-title',title,775,y+10,750,32,22,True)
        f.text(id+'-body',body,775,y+52,750,30,20)
    f.cell('embeddings','',f'rounded=1;arcSize=7;html=0;fillColor=#FFFFFF;strokeColor={INK};strokeWidth=2;',760,395,780,60)
    f.modules.append(('embeddings',760,395,780,60));f.items.append(('rect',760,395,780,60,'#FFFFFF',False))
    f.text('embeddings-label','Token states + validity mask   |   Global EOS embeddings',775,405,750,40,21,True)
    f.edge('descriptions-to-text-encoder',[(1150,200),(1150,250)])
    f.edge('text-encoder-to-embeddings',[(1150,345),(1150,395)])
    f.module('tpfm',760,710,730,190,'Text Prompt Fusion Module (TPFM)',[
      'Visual queries · text token keys and values','256 dimensions · 8 attention heads','Padding mask · visual residual addition'],'#FFF0CA',21)
    f.edge('visual-to-tpfm',[(560,870),(690,870),(690,805),(760,805)])
    f.edge('tokens-to-tpfm',[(1010,455),(1010,495),(635,495),(635,670),(1060,670),(1060,710)])
    f.text('tokens-label','Projected text tokens\n256 dimensions',825,590,340,60,20)
    f.module('decoder',760,980,730,140,'Prompt-guided lightweight decoder',[
      '2 bidirectional interaction layers','256 dimensions · 8 heads · coarse lesion logits'],'#F7DCC8',21)
    f.edge('fusion-to-decoder',[(1125,900),(1125,980)])
    f.text('fusion-label','Fused features',1190,925,235,30,19)
    f.edge('global-to-decoder',[(1450,455),(1620,455),(1620,1050),(1490,1050)])
    f.text('global-label','Global text\nprompt queries',1495,605,120,60,18)
    f.module('output',760,1190,730,165,'Mask refinement and output',[
      'Upsample + projected first-layer features · 1 × 1 convolution',
      'Resize to 1024 × 1024 · sigmoid · threshold 0.5',
      'Four independent masks: MA / HE / EX / SE'],body_size=19)
    f.edge('decoder-to-output',[(1125,1120),(1125,1190)])
    f.edge('refinement-skip',[(140,565),(75,565),(75,1270),(760,1270)])
    f.text('skip-label','First-layer features\nChannel projection · refinement skip',170,1175,405,65,19)
    f.text('abbreviations','MA: microaneurysms · HE: hemorrhages · EX: hard exudates · SE: soft exudates',740,1360,850,30,16)
    return f

if __name__=='__main__':
    assets()
    figures=[figure1(),figure2()]
    report={f'Figure {f.n}':f.save() for f in figures}
    report['assets']={p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'mode':Image.open(p).mode,
      'origin':'Original programmatically drawn schematic; no clinical dataset pixels; no text'} for p in ASSETS.glob('*.png')}
    (OUT/'drawio_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (OUT/'drawio_layout.json').write_text(json.dumps({f'Figure {f.n}':{'width':f.w,'height':f.h,'items':f.items} for f in figures},ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2))
