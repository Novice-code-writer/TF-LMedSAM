"""Export deterministic epoch profiles for plotting and pipeline exercises."""
from pathlib import Path
import csv, json
import numpy as np

def main():
    out=Path(__file__).resolve().parent/'records/epoch_profiles'
    for dataset,top in [('IDRiD',.80),('DDR',.74)]:
        rng=np.random.default_rng(20261112+(dataset=='DDR'));rows=[]
        for epoch in range(1,101):
            loss=.35+1.8*np.exp(-epoch/24)+float(rng.normal(0,.008))
            dice=top-(top-.30)*np.exp(-epoch/23)+float(rng.normal(0,.003))
            valid=epoch%5==0
            rows.append(dict(epoch=epoch,train_loss=round(loss,6),learning_rate=1e-6+(1e-4-1e-6)*(1+np.cos(np.pi*(epoch-1)/100))/2,
                val_dice=round(dice,6) if valid else '',val_iou=round(dice/(2-dice),6) if valid else ''))
        folder=out/dataset;folder.mkdir(parents=True,exist_ok=True)
        with (folder/'history.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        (folder/'profile.json').write_text(json.dumps(dict(record_mode='reference_profile',seed=20261112,
            epochs=100,validation_interval=5,checkpoint=None,optimization_executed=False),indent=2),encoding='utf-8')
    print(out)
if __name__=='__main__':main()
