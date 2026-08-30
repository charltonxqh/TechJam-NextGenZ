import sys, os, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kuairand-starter-kit'))
from evaluate import evaluate

class FM(nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V = nn.Parameter(torch.from_numpy(rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
        self.W = nn.Parameter(torch.zeros(dim))
        self.b = nn.Parameter(torch.zeros(()))

    def forward(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter

    @torch.no_grad()
    def predict(self, X, bs=200_000):
        self.eval()
        out = [self(X[i:i + bs]).cpu().numpy() for i in range(0, len(X), bs)]
        self.train()
        return np.concatenate(out)

def run(D, seed=0):
    k, lr, l2, bs, epochs = 16, 0.001, 1e-6, 4096, 20
    Xtr = torch.from_numpy(D['Xtr']).long(); ytr = torch.from_numpy(D['ytr']).float()
    Xva = torch.from_numpy(D['Xva']).long(); uva, yva = D['uva'], D['yva']
    
    m = FM(D['dim'], k=k, seed=seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=l2)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0))
    
    best, best_state, bad, history = -1.0, None, 0, []
    n_batches = len(ytr) // bs

    for ep in range(1, epochs + 1):
        losses = []
        idx = torch.randperm(len(ytr))
        for i in range(n_batches):
            batch_idx = idx[i*bs:(i+1)*bs]
            opt.zero_grad()
            preds = m(Xtr[batch_idx])
            loss = criterion(preds, ytr[batch_idx])
            loss.backward(); opt.step(); losses.append(loss.item())
        
        p = evaluate(uva, yva, m.predict(Xva))['primary']
        history.append({'epoch': ep, 'train_loss': float(np.mean(losses)), 'valid_primary': float(p)})
        if p > best + 1e-5: best, bad = p, 0; best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else: bad += 1
        if bad >= 3: break
    m.load_state_dict(best_state)
    return m.predict(Xva), history