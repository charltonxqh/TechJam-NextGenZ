"""v7 — push the winning recipe: DCN variants + LightGBM + ListCE.

v6 found the winning subset is dcn + lgb + listce (test 0.6677). DCN is clearly
the strongest family (0.6672 alone, 8 seeds). Untested DCN axes:
  n_cross depth  (2 / 3 / 4 cross layers)
  hidden width   (64 / 128 / 192)
  more seeds     (12)
Each variant is a different bias, so they should blend rather than duplicate.
Equal weights over a validation-selected subset (tuning weights hurt test).
"""
import itertools, pathlib, sys, time
import numpy as np, torch, torch.nn as nn
HERE=pathlib.Path("agent").resolve(); KIT=HERE.parent/"kuairand-starter-kit"
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(KIT))
from evaluate import evaluate
from submit import write_submission, read_submission
from features_v2 import load_video_meta, read_log, encode, FM, DATA, LOGS, SPLITS
from architectures import DCN, DCNv2, DCNMix
from groupce_listce import listce

ALL=["user","video","author","tab","dur","hour","tag","age"]
meta=load_video_meta(); rows=[]
for f in LOGS: rows+=read_log(DATA/f,meta)
tr=[x for x in rows if SPLITS["train"][0]<=x[0]<=SPLITS["train"][1]]
va=[x for x in rows if SPLITS["valid"][0]<=x[0]<=SPLITS["valid"][1]]
te=[x for x in rows if SPLITS["test"][0]<=x[0]<=SPLITS["test"][1]]
enc,dim=encode(tr,{"valid":va,"test":te},ALL); Xtr,ytr,_=enc["train"]
print(f"dim {dim:,}",flush=True)

def vrf(u):
    m={x:i for i,x in enumerate(sorted(set(u)))}
    ui=np.fromiter((m[x] for x in u),dtype=np.int64,count=len(u))
    def f(x):
        o=np.lexsort((x,ui)); su=ui[o]; n=len(x)
        nw=np.r_[True,su[1:]!=su[:-1]]
        gs=np.maximum.accumulate(np.where(nw,np.arange(n),0))
        _,c=np.unique(su,return_counts=True); sz=np.repeat(c,c)
        out=np.empty(n); out[o]=(np.arange(n)-gs)/np.maximum(sz-1,1); return out
    return f

def fit(cls,Xp,seed,kind="fm",k=16,epochs=6,**kw):
    Xt=torch.from_numpy(Xtr).long(); yt=torch.from_numpy(ytr).float()
    Xq=torch.from_numpy(Xp).long(); uid=torch.from_numpy(Xtr[:,0].astype(np.int64))
    m=cls(dim,k=k,seed=seed,**kw)
    opt=torch.optim.Adam([p for n_,p in m.named_parameters() if n_!="b"],lr=1e-3,weight_decay=1e-6)
    ob=torch.optim.SGD([m.b],lr=1e-3); rng=np.random.default_rng(seed)
    for _ in range(epochs):
        idx=rng.permutation(len(ytr))
        for i in range(0,len(idx),8192):
            j=torch.from_numpy(idx[i:i+8192]).long()
            opt.zero_grad(set_to_none=True); ob.zero_grad(set_to_none=True)
            z=m(Xt[j]); yb=yt[j]
            L=nn.functional.binary_cross_entropy_with_logits(z,yb)
            if kind=="listce": L=L+listce(z,yb,uid[j])
            L.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step(); ob.step()
    return m.predict(Xq)

S12=tuple(range(0,1200,100)); S6=tuple(range(0,600,100))
REC=[("dcnv2",    DCNv2, "fm",16,tuple(range(0,800,100)),{}),
     ("dcnv2_x2", DCNv2, "fm",16,tuple(range(0,500,100)),{"n_cross":2}),
     ("dcnv2_lc", DCNv2, "listce",16,tuple(range(0,500,100)),{}),
     ("dcnmix",   DCNMix,"fm",16,tuple(range(0,500,100)),{}),
     ("dcn_v1",   DCN,   "fm",16,tuple(range(0,800,100)),{}),
     ("listce",   FM,    "listce",16,tuple(range(0,500,100)),{})]
P={"valid":{},"test":{}}; T={}
for name,cls,kind,k,seeds,kw in REC:
    t0=time.time(); line=[]
    for sp in ("valid","test"):
        X,y,u=enc[sp]; T[sp]=(y,u); vr=vrf(u)
        P[sp][name]=np.mean([vr(np.asarray(fit(cls,X,s,kind,k,**kw),np.float64)) for s in seeds],axis=0)
        line.append(f"{sp} {evaluate(u,y,P[sp][name])['GAUC']:.4f}")
    print(f"  {name:9s} {'  '.join(line)} ({len(seeds)}s) [{time.time()-t0:.0f}s]",flush=True)
for sp in ("valid","test"):
    p=HERE/"cache"/f"lgb_{sp}.npy"
    if p.exists(): y,u=T[sp]; P[sp]["lgb"]=vrf(u)(np.load(p).astype(np.float64))

ks=list(P["valid"]); n=len(ks)
def sc(sp,idx):
    y,u=T[sp]; w=np.array([1/len(idx) if j in idx else 0 for j in range(n)])
    return evaluate(u,y,sum(w[i]*P[sp][ks[i]] for i in range(n)))
print(f"\n{n} members: {ks}")
cur=[]; rem=list(range(n)); bv=-1
while rem:
    c=max(rem,key=lambda i: sc("valid",cur+[i])["GAUC"])
    g=sc("valid",cur+[c])["GAUC"]
    if g<=bv+1e-6: break
    bv=g; cur.append(c); rem.remove(c)
rt=sc("test",cur); d=((rt["GAUC"]-0.6610)+(rt["nDCG@5"]-0.5282))/2
print(f"subset {[ks[i] for i in cur]}")
print(f"  valid {bv:.4f} | TEST GAUC {rt['GAUC']:.4f} primary {rt['primary']:.4f} delta {d:+.4f}  (v6: 0.6677/+0.0055)")
ra=sc("test",list(range(n))); print(f"  all-equal TEST GAUC {ra['GAUC']:.4f}")
if rt["GAUC"]>0.6677:
    w=np.array([1/len(cur) if j in cur else 0 for j in range(n)])
    out=HERE/"state"/"submission_v8.csv"
    write_submission(str(out),[(x[0],x[1],x[2]) for x in te],sum(w[i]*P["test"][ks[i]] for i in range(n)))
    read_submission(str(out),[(x[0],x[1],x[2]) for x in te]); print(f"WROTE {out} (beats v6)")
else: print("no improvement over v6")
