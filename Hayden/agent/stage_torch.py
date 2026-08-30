"""Stage 2 — torch members + blend. NO lightgbm import (segfaults with torch)."""
import pathlib, sys, time
import numpy as np, torch, torch.nn as nn
HERE=pathlib.Path(__file__).resolve().parent; KIT=HERE.parent/"kuairand-starter-kit"
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(KIT))
from data import load, encode
from evaluate import evaluate
from submit import write_submission, read_submission
from groupce_listce import FM, listce

SEEDS=(0,100,200,300,400)
splits=load(str(KIT/"KuaiRand-Pure"/"data")); enc,dim=encode(splits)
Xtr,ytr,utr=enc["train"]; Xtr=Xtr.astype(np.int64)

def vrank_factory(users):
    u2i={u:i for i,u in enumerate(sorted(set(users)))}
    ui=np.fromiter((u2i[u] for u in users),dtype=np.int64,count=len(users))
    def f(x):
        o=np.lexsort((x,ui)); su=ui[o]; n=len(x)
        new=np.r_[True,su[1:]!=su[:-1]]
        gs=np.maximum.accumulate(np.where(new,np.arange(n),0))
        _,c=np.unique(su,return_counts=True); sz=np.repeat(c,c)
        out=np.empty(n); out[o]=(np.arange(n)-gs)/np.maximum(sz-1,1); return out
    return f

def member(kind, Xp, seed):
    Xt=torch.from_numpy(Xtr).long(); yt=torch.from_numpy(ytr).float()
    Xq=torch.from_numpy(Xp.astype(np.int64)).long()
    uid=torch.from_numpy(Xtr[:,0].astype(np.int64))
    m=FM(dim,seed=seed)
    opt=torch.optim.Adam([m.V,m.W],lr=1e-3,weight_decay=1e-6)
    ob=torch.optim.SGD([m.b],lr=1e-3); rng=np.random.default_rng(seed)
    for _ in range(7):
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
    X,y,u=enc[split]; vr=vrank_factory(u); P[split]={}
    for kind in ("fm","listce"):
        t0=time.time()
        pr=[vr(np.asarray(member(kind,X,s),dtype=np.float64)) for s in SEEDS]
        P[split][kind]=np.mean(pr,axis=0)
        print(f"  {split}/{kind:7s} {evaluate(u,y,P[split][kind])['primary']:.4f} [{time.time()-t0:.0f}s]",flush=True)
    P[split]["lgb"]=vr(np.load(HERE/"cache"/f"lgb_{split}.npy").astype(np.float64))
    print(f"  {split}/lgb     {evaluate(u,y,P[split]['lgb'])['primary']:.4f}",flush=True)

# tune weights on VALIDATION only
Xv,yv,uv=enc["valid"]; best=(None,-1)
for wf in np.arange(0,0.7,0.1):
    for wl in np.arange(0,1.01-wf,0.1):
        wg=1-wf-wl
        if wg<-1e-9: continue
        b=wf*P["valid"]["fm"]+wl*P["valid"]["listce"]+wg*P["valid"]["lgb"]
        p=evaluate(uv,yv,b)["primary"]
        if p>best[1]: best=((round(wf,2),round(wl,2),round(wg,2)),p)
wf,wl,wg=best[0]
print(f"\nbest weights (tuned on valid): fm={wf} listce={wl} lgb={wg} -> {best[1]:.4f}")

Xt_,yt_,ut_=enc["test"]
bt=wf*P["test"]["fm"]+wl*P["test"]["listce"]+wg*P["test"]["lgb"]
r=evaluate(ut_,yt_,bt)
d=((r["GAUC"]-0.6610)+(r["nDCG@5"]-0.5282))/2
print(f"TEST  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
print(f"      vs baseline 0.5946: {r['primary']-0.5946:+.4f}   SCORED DELTA {d:+.4f}")
out=HERE/"state"/"submission_blend.csv"
write_submission(str(out), splits["test"], bt); read_submission(str(out), splits["test"])
print(f"wrote {out} — passes official validator")
