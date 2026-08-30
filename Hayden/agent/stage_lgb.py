"""Stage 1 — LightGBM only. NO torch import (they segfault together via libomp)."""
import csv, pathlib, sys, time
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; KIT = HERE.parent/"kuairand-starter-kit"
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(KIT))
import lightgbm as lgb
from data import load, encode
from evaluate import evaluate

NUM=["long_time_play_cnt","valid_play_cnt","complete_play_cnt","play_cnt",
     "short_time_play_cnt","like_cnt","comment_cnt","share_cnt","collect_cnt","follow_cnt"]
stat={}
with open(KIT/"KuaiRand-Pure"/"data"/"video_features_statistic_pure.csv") as fh:
    for r in csv.DictReader(fh): stat[r["video_id"]]=r
def block(vids):
    out=np.zeros((len(vids),len(NUM)+2),dtype=np.float32)
    for i,v in enumerate(vids):
        s=stat.get(v)
        if not s: continue
        show=max(float(s.get("show_cnt",1) or 1),1.0)
        for j,c in enumerate(NUM):
            try: out[i,j]=float(s.get(c,0) or 0)/show
            except: pass
        try: out[i,len(NUM)]=float(s.get("play_progress",0) or 0)
        except: pass
        out[i,len(NUM)+1]=np.log1p(show)
    return out

splits=load(str(KIT/"KuaiRand-Pure"/"data")); enc,dim=encode(splits)
Xtr,ytr,utr=enc["train"]
Ftr=np.hstack([Xtr, block([r[2] for r in splits["train"]])]).astype(np.float32)
u2i={u:i for i,u in enumerate(sorted(set(utr)))}
ui=np.fromiter((u2i[u] for u in utr),dtype=np.int64,count=len(utr))
o=np.argsort(ui,kind="stable"); _,c=np.unique(ui[o],return_counts=True)
ds=lgb.Dataset(Ftr[o],label=ytr[o],group=c,categorical_feature=[0,1,2,3,4],free_raw_data=False)
t0=time.time()
b=lgb.train(dict(objective="lambdarank",metric="ndcg",ndcg_eval_at=[5],learning_rate=0.05,
                 num_leaves=63,min_data_in_leaf=50,feature_fraction=0.9,bagging_fraction=0.9,
                 bagging_freq=1,verbose=-1,num_threads=8), ds, num_boost_round=300)
print(f"trained {time.time()-t0:.0f}s", flush=True)
for split in ("valid","test"):
    X,y,u=enc[split]
    F=np.hstack([X, block([r[2] for r in splits[split]])]).astype(np.float32)
    p=b.predict(F); np.save(HERE/"cache"/f"lgb_{split}.npy", p)
    print(f"  {split}: primary {evaluate(u,y,p)['primary']:.4f} -> cache/lgb_{split}.npy", flush=True)
