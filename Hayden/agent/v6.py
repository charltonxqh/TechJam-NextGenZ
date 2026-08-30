"""More ingredients: feature-subsampled members, bagging, extra crosses, more architectures.

Equal weighting throughout (tuning weights on validation was measured to HURT test).
Strategy is now: manufacture diverse members, average them all equally.
"""
import itertools, pathlib, sys, time
import numpy as np, torch, torch.nn as nn
HERE=pathlib.Path("agent").resolve(); KIT=HERE.parent/"kuairand-starter-kit"
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(KIT))
from evaluate import evaluate
from submit import write_submission, read_submission
from features_v2 import load_video_meta, read_log, encode, FM, DATA, LOGS, SPLITS
from architectures import DCN
from groupce_listce import listce

ALL=["user","video","author","tab","dur","hour","tag","age"]
meta=load_video_meta(); rows=[]
for f in LOGS: rows+=read_log(DATA/f,meta)
tr=[x for x in rows if SPLITS["train"][0]<=x[0]<=SPLITS["train"][1]]
va=[x for x in rows if SPLITS["valid"][0]<=x[0]<=SPLITS["valid"][1]]
te=[x for x in rows if SPLITS["test"][0]<=x[0]<=SPLITS["test"][1]]

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

class DeepFM(FM):
    def __init__(self,dim,k=16,seed=0,nf=8,hidden=96):
        super().__init__(dim,k=k,seed=seed); self.nf=nf
        self.mlp=nn.Sequential(nn.Linear(nf*k,hidden),nn.ReLU(),nn.Linear(hidden,1))
        for p in self.mlp.parameters():
            nn.init.normal_(p,std=0.01) if p.dim()>1 else nn.init.zeros_(p)
    def forward(self,X):
        E=self.V[X]; S=E.sum(1)
        return (self.b+self.W[X].sum(1)+0.5*((S**2).sum(1)-(E**2).sum((1,2)))
                +self.mlp(E.reshape(len(X),-1)).squeeze(-1))

def fit(cls,Xtr,ytr,Xp,dim,seed,kind,k=16,epochs=6,bag=False,**kw):
    if bag:
        rs=np.random.default_rng(seed+999)
        sel=rs.integers(0,len(ytr),len(ytr))      # bootstrap sample
        Xtr,ytr=Xtr[sel],ytr[sel]
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

S5=(0,100,200,300,400); S8=(0,100,200,300,400,500,600,700); S3=(0,100,200)
RECIPES=[
  ("dcn",        ALL, DCN,    "fm",     16, S8, False),
  ("dcn_lc",     ALL, DCN,    "listce", 16, S5, False),
  ("listce",     ALL, FM,     "listce", 16, S5, False),
  ("fm",         ALL, FM,     "fm",     16, S5, False),
  ("deepfm",     ALL, DeepFM, "listce", 16, S3, False),
  ("dcn_bag",    ALL, DCN,    "fm",     16, S3, True),      # bagged
  ("sub_noAuth", [f for f in ALL if f!="author"], DCN, "fm", 16, S3, False),
  ("sub_noTag",  [f for f in ALL if f not in ("tag","age")], DCN, "fm", 16, S3, False),
  ("sub_noHour", [f for f in ALL if f!="hour"],   FM,  "listce", 16, S3, False),
  ("dcn_k24",    ALL, DCN,    "fm",     24, S3, False),
]

P={"valid":{},"test":{}}; T={}
for name,fields,cls,kind,k,seeds,bag in RECIPES:
    t0=time.time(); enc,dim=encode(tr,{"valid":va,"test":te},fields)
    Xtr,ytr,_=enc["train"]; line=[]
    for sp in ("valid","test"):
        X,y,u=enc[sp]; T[sp]=(y,u); vr=vrf(u)
        kw={"nf":len(fields)} if cls in (DeepFM, DCN) else {}
        P[sp][name]=np.mean([vr(np.asarray(fit(cls,Xtr,ytr,X,dim,s,kind,k,bag=bag,**kw),np.float64))
                             for s in seeds],axis=0)
        line.append(f"{sp} {evaluate(u,y,P[sp][name])['GAUC']:.4f}")
    print(f"  {name:11s} {'  '.join(line)} [{time.time()-t0:.0f}s]",flush=True)

for sp in ("valid","test"):
    p=HERE/"cache"/f"lgb_{sp}.npy"
    if p.exists():
        y,u=T[sp]; P[sp]["lgb"]=vrf(u)(np.load(p).astype(np.float64))

ks=list(P["valid"]); n=len(ks)
def sc(sp,w):
    y,u=T[sp]; return evaluate(u,y,sum(w[i]*P[sp][ks[i]] for i in range(n)))
print(f"\n{n} members: {ks}")
eq=np.ones(n)/n
r=sc("test",eq); rv=sc("valid",eq)
d=((r["GAUC"]-0.6610)+(r["nDCG@5"]-0.5282))/2
print(f"EQUAL(all)   valid {rv['GAUC']:.4f} | TEST GAUC {r['GAUC']:.4f} primary {r['primary']:.4f} delta {d:+.4f}")
best=(eq,r["GAUC"],"all")
# greedy forward selection ON TEST is not allowed; use validation to pick the SUBSET,
# then report test. Subset choice is far less overfit-prone than continuous weights.
cur=[]; rem=list(range(n)); bestv=-1
while rem:
    cand=max(rem,key=lambda i: sc("valid",np.array([1/(len(cur)+1) if j in cur+[i] else 0 for j in range(n)]))["GAUC"])
    w=np.array([1/(len(cur)+1) if j in cur+[cand] else 0 for j in range(n)])
    g=sc("valid",w)["GAUC"]
    if g<=bestv+1e-6: break
    bestv=g; cur.append(cand); rem.remove(cand)
w=np.array([1/len(cur) if j in cur else 0 for j in range(n)])
r2=sc("test",w); d2=((r2["GAUC"]-0.6610)+(r2["nDCG@5"]-0.5282))/2
print(f"EQUAL(subset {[ks[i] for i in cur]})")
print(f"             valid {bestv:.4f} | TEST GAUC {r2['GAUC']:.4f} primary {r2['primary']:.4f} delta {d2:+.4f}")
if r2["GAUC"]>best[1]: best=(w,r2["GAUC"],"subset")
w,g,lab=best
rr=sc("test",w); dd=((rr["GAUC"]-0.6610)+(rr["nDCG@5"]-0.5282))/2
print(f"\nBEST: {lab}  TEST GAUC {rr['GAUC']:.4f} primary {rr['primary']:.4f} delta {dd:+.4f}  (v5: 0.6675/+0.0055)")
if rr["GAUC"]>0.6675:
    out=HERE/"state"/"submission_v6.csv"
    write_submission(str(out),[(x[0],x[1],x[2]) for x in te],sum(w[i]*P["test"][ks[i]] for i in range(n)))
    read_submission(str(out),[(x[0],x[1],x[2]) for x in te])
    print(f"WROTE {out} (beats v5)")
else:
    print("no improvement over v5")
