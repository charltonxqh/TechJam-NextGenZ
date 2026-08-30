"""Blend tuned for GAUC (not primary). No lightgbm import — torch only."""
import pathlib, sys, time, itertools
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

def member(kind,Xp,seed):
    Xt=torch.from_numpy(Xtr).long(); yt=torch.from_numpy(ytr).float()
    Xq=torch.from_numpy(Xp.astype(np.int64)).long()
    uid=torch.from_numpy(Xtr[:,0].astype(np.int64))
    m=FM(dim,seed=seed)
    opt=torch.optim.Adam([m.V,m.W],lr=1e-3,weight_decay=1e-6); ob=torch.optim.SGD([m.b],lr=1e-3)
    rng=np.random.default_rng(seed)
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
    X,y,u=enc[split]; vr=vrf(u); P[split]={}
    for kind in ("fm","listce"):
        pr=[vr(np.asarray(member(kind,X,s),dtype=np.float64)) for s in SEEDS]
        P[split][kind]=np.mean(pr,axis=0)
        r=evaluate(u,y,P[split][kind])
        print(f"  {split}/{kind:7s} GAUC {r['GAUC']:.4f} primary {r['primary']:.4f}",flush=True)
    for tag,f in (("lgb","lgb"),("lgbfull","lgbfull")):
        p=np.load(HERE/"cache"/f"{f}_{split}.npy").astype(np.float64)
        P[split][tag]=vr(p)
        r=evaluate(u,y,P[split][tag])
        print(f"  {split}/{tag:7s} GAUC {r['GAUC']:.4f} primary {r['primary']:.4f}",flush=True)

keys=["fm","listce","lgb","lgbfull"]
Xv,yv,uv=enc["valid"]; Xt_,yt_,ut_=enc["test"]
best=(None,-1)
grid=np.arange(0,1.01,0.1)
for w in itertools.product(grid,repeat=3):
    if sum(w)>1.0+1e-9: continue
    wv=list(w)+[1-sum(w)]
    b=sum(wv[i]*P["valid"][keys[i]] for i in range(4))
    g=evaluate(uv,yv,b)["GAUC"]
    if g>best[1]: best=(wv,g)
wv=best[0]
print(f"\nbest weights for GAUC (valid): " + ", ".join(f"{k}={w:.1f}" for k,w in zip(keys,wv)))
print(f"  valid GAUC {best[1]:.4f}")
bt=sum(wv[i]*P["test"][keys[i]] for i in range(4))
r=evaluate(ut_,yt_,bt)
print(f"TEST  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
print(f"      GAUC vs baseline 0.6610: {r['GAUC']-0.6610:+.4f}")
d=((r["GAUC"]-0.6610)+(r["nDCG@5"]-0.5282))/2
print(f"      scored delta {d:+.4f}")
out=HERE/"state"/"submission_gauc.csv"
write_submission(str(out), splits["test"], bt); read_submission(str(out), splits["test"])
print(f"wrote {out}")
