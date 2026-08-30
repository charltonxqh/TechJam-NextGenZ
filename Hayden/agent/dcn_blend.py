"""Does DCN earn a place in the blend? Valid + test, 5 seeds, GAUC-selected."""
import itertools, pathlib, sys, time
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path("agent").resolve()))
sys.path.insert(0, str(pathlib.Path("kuairand-starter-kit").resolve()))
from evaluate import evaluate
from features_v2 import load_video_meta, read_log, encode, FM, DATA, LOGS, SPLITS
from architectures import DCN, train as atrain
from groupce_listce import listce
import torch.nn as nn

HERE = pathlib.Path("agent").resolve()
FIELDS = ["user","video","author","tab","dur","hour","tag","age"]
SEEDS = (0,100,200,300,400)

meta = load_video_meta(); rows=[]
for f in LOGS: rows += read_log(DATA/f, meta)
tr=[x for x in rows if SPLITS["train"][0]<=x[0]<=SPLITS["train"][1]]
va=[x for x in rows if SPLITS["valid"][0]<=x[0]<=SPLITS["valid"][1]]
te=[x for x in rows if SPLITS["test"][0]<=x[0]<=SPLITS["test"][1]]
enc,dim = encode(tr, {"valid":va,"test":te}, FIELDS)
Xtr,ytr,_ = enc["train"]
print(f"dim {dim:,}", flush=True)

def vrf(users):
    u2i={u:i for i,u in enumerate(sorted(set(users)))}
    ui=np.fromiter((u2i[u] for u in users),dtype=np.int64,count=len(users))
    def f(x):
        o=np.lexsort((x,ui)); su=ui[o]; n=len(x)
        new=np.r_[True,su[1:]!=su[:-1]]
        gs=np.maximum.accumulate(np.where(new,np.arange(n),0))
        _,c=np.unique(su,return_counts=True); sz=np.repeat(c,c)
        out=np.empty(n); out[o]=(np.arange(n)-gs)/np.maximum(sz-1,1); return out
    return f

def fm_member(kind, Xp, seed):
    Xt=torch.from_numpy(Xtr).long(); yt=torch.from_numpy(ytr).float()
    Xq=torch.from_numpy(Xp).long(); uid=torch.from_numpy(Xtr[:,0].astype(np.int64))
    m=FM(dim,seed=seed)
    opt=torch.optim.Adam([m.V,m.W],lr=1e-3,weight_decay=1e-6); ob=torch.optim.SGD([m.b],lr=1e-3)
    rng=np.random.default_rng(seed)
    for _ in range(6):
        idx=rng.permutation(len(ytr))
        for i in range(0,len(idx),8192):
            j=torch.from_numpy(idx[i:i+8192]).long()
            opt.zero_grad(set_to_none=True); ob.zero_grad(set_to_none=True)
            z=m(Xt[j]); yb=yt[j]
            L=nn.functional.binary_cross_entropy_with_logits(z,yb)
            if kind=="listce": L=L+listce(z,yb,uid[j])
            L.backward(); opt.step(); ob.step()
    return m.predict(Xq)

P={}
for split in ("valid","test"):
    X,y,u = enc[split]; vr=vrf(u); P[split]={}
    for kind in ("fm","listce"):
        t0=time.time()
        P[split][kind]=np.mean([vr(np.asarray(fm_member(kind,X,s),np.float64)) for s in SEEDS],axis=0)
        print(f"  {split}/{kind:7s} GAUC {evaluate(u,y,P[split][kind])['GAUC']:.4f} [{time.time()-t0:.0f}s]",flush=True)
    t0=time.time()
    P[split]["dcn"]=np.mean([vr(np.asarray(atrain(DCN,Xtr,ytr,X,dim,s),np.float64)) for s in SEEDS],axis=0)
    print(f"  {split}/dcn     GAUC {evaluate(u,y,P[split]['dcn'])['GAUC']:.4f} [{time.time()-t0:.0f}s]",flush=True)
    lp=HERE/"cache"/f"lgb_{split}.npy"
    if lp.exists():
        P[split]["lgb"]=vr(np.load(lp).astype(np.float64))
        print(f"  {split}/lgb     GAUC {evaluate(u,y,P[split]['lgb'])['GAUC']:.4f}")

print("\ncorrelations (valid):")
ks=list(P["valid"])
for a,b in itertools.combinations(ks,2):
    print(f"  {a:7s} x {b:7s} {np.corrcoef(P['valid'][a],P['valid'][b])[0,1]:.3f}")

Xv,yv,uv=enc["valid"]; Xt_,yt_,ut_=enc["test"]
best=(None,-1)
for w in itertools.product(np.arange(0,1.01,0.1),repeat=len(ks)-1):
    if sum(w)>1+1e-9: continue
    wv=list(w)+[1-sum(w)]
    g=evaluate(uv,yv,sum(wv[i]*P["valid"][ks[i]] for i in range(len(ks))))["GAUC"]
    if g>best[1]: best=(wv,g)
wv=best[0]
print("\nbest weights: "+", ".join(f"{k}={x:.1f}" for k,x in zip(ks,wv)))
print(f"  valid GAUC {best[1]:.4f}   (v2 was 0.6730)")
bt=sum(wv[i]*P["test"][ks[i]] for i in range(len(ks)))
r=evaluate(ut_,yt_,bt)
d=((r["GAUC"]-0.6610)+(r["nDCG@5"]-0.5282))/2
print(f"TEST  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
print(f"      delta {d:+.4f}   (v2: GAUC 0.6669, delta +0.0047)")
if r["GAUC"]>0.6669:
    from submit import write_submission, read_submission
    out=HERE/"state"/"submission_v4.csv"
    write_submission(str(out),[(x[0],x[1],x[2]) for x in te],bt)
    read_submission(str(out),[(x[0],x[1],x[2]) for x in te])
    print(f"wrote {out} (beats v2)")
else:
    print("does not beat v2 — no submission written")
