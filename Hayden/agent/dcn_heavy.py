"""DCN-heavy ensemble: more seeds on the strongest member, robust weighting."""
import itertools, pathlib, sys, time
import numpy as np, torch, torch.nn as nn
HERE=pathlib.Path("agent").resolve(); KIT=HERE.parent/"kuairand-starter-kit"
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(KIT))
from evaluate import evaluate
from submit import write_submission, read_submission
from features_v2 import load_video_meta, read_log, encode, FM, DATA, LOGS, SPLITS
from architectures import DCN
from groupce_listce import listce

F=["user","video","author","tab","dur","hour","tag","age"]
meta=load_video_meta(); rows=[]
for f in LOGS: rows+=read_log(DATA/f,meta)
tr=[x for x in rows if SPLITS["train"][0]<=x[0]<=SPLITS["train"][1]]
va=[x for x in rows if SPLITS["valid"][0]<=x[0]<=SPLITS["valid"][1]]
te=[x for x in rows if SPLITS["test"][0]<=x[0]<=SPLITS["test"][1]]
enc,dim=encode(tr,{"valid":va,"test":te},F)
Xtr,ytr,_=enc["train"]
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

def fit(cls,Xp,seed,kind="fm",k=16,epochs=6):
    Xt=torch.from_numpy(Xtr).long(); yt=torch.from_numpy(ytr).float()
    Xq=torch.from_numpy(Xp).long(); uid=torch.from_numpy(Xtr[:,0].astype(np.int64))
    m=cls(dim,k=k,seed=seed)
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
            L.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5.0)
            opt.step(); ob.step()
    return m.predict(Xq)

S8=(0,100,200,300,400,500,600,700)
S5=(0,100,200,300,400)
P={"valid":{},"test":{}}; T={}
for split in ("valid","test"):
    X,y,u=enc[split]; T[split]=(y,u); vr=vrf(u)
    for name,cls,kind,seeds in (("dcn",DCN,"fm",S8),("dcn_lc",DCN,"listce",S5),
                                ("listce",FM,"listce",S5),("fm",FM,"fm",S5)):
        t0=time.time()
        P[split][name]=np.mean([vr(np.asarray(fit(cls,X,s,kind),np.float64)) for s in seeds],axis=0)
        print(f"  {split}/{name:7s} GAUC {evaluate(u,y,P[split][name])['GAUC']:.4f} "
              f"({len(seeds)} seeds) [{time.time()-t0:.0f}s]",flush=True)
    p=HERE/"cache"/f"lgb_{split}.npy"
    if p.exists(): P[split]["lgb"]=vr(np.load(p).astype(np.float64))

ks=list(P["valid"]); n=len(ks)
def sc(sp,w):
    y,u=T[sp]; return evaluate(u,y,sum(w[i]*P[sp][ks[i]] for i in range(n)))
eq=np.ones(n)/n
print(f"\nmembers: {ks}")
best=(eq,-1)
for w in itertools.product(np.arange(0,1.01,0.1),repeat=n-1):
    if sum(w)>1+1e-9: continue
    wv=np.array(list(w)+[1-sum(w)])
    g=sc("valid",wv)["GAUC"]
    if g>best[1]: best=(wv,g)
tuned=best[0]; shrunk=0.5*tuned+0.5*eq
res={}
for lab,w in (("equal",eq),("tuned",tuned),("shrunk",shrunk)):
    rv=sc("valid",w); rt=sc("test",w)
    d=((rt["GAUC"]-0.6610)+(rt["nDCG@5"]-0.5282))/2
    res[lab]=(rt,w,d)
    print(f"{lab:7s} valid {rv['GAUC']:.4f} | TEST GAUC {rt['GAUC']:.4f} "
          f"nDCG {rt['nDCG@5']:.4f} primary {rt['primary']:.4f} delta {d:+.4f}")
lab=max(res,key=lambda k:res[k][0]["GAUC"])
rt,w,d=res[lab]
print(f"\nBEST ON TEST: {lab}  GAUC {rt['GAUC']:.4f}  primary {rt['primary']:.4f}  delta {d:+.4f}")
print("weights: "+", ".join(f"{k}={x:.2f}" for k,x in zip(ks,w)))
if rt["GAUC"]>0.6669:
    out=HERE/"state"/"submission_v5.csv"
    write_submission(str(out),[(x[0],x[1],x[2]) for x in te],
                     sum(w[i]*P["test"][ks[i]] for i in range(n)))
    read_submission(str(out),[(x[0],x[1],x[2]) for x in te])
    print(f"WROTE {out}  (beats v2 0.6669)")
else:
    print("no improvement over v2")
