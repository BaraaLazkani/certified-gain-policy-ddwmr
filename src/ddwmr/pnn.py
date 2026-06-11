"""PNN (anchor Table 2) and the gain-policy MLP used by WP1/WP3.

Anchor architecture: input (x_ref, y_ref) -> 128 ReLU -> 64 ReLU -> 64 tanh
-> 64 ReLU -> 2 linear; ADAM lr=1e-3 betas=(0.9, 0.99); batch 128; 360 epochs.

Two output heads are supported:
  - 'linear'  : raw gains (anchor's supervised PNN; trained on labels)
  - 'sigmoid' : gains squashed into the corrected box (BPTT policy; guarantees
                Kp,Kth > 0 so the frozen-gain Lyapunov condition Eq. 35 holds
                by construction)
"""
import numpy as np
import torch
import torch.nn as nn

from .params import KP_MIN, KP_MAX, KTH_MIN, KTH_MAX


class GainMLP(nn.Module):
    def __init__(self, head="linear"):
        super().__init__()
        self.l1 = nn.Linear(2, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 64)
        self.l4 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 2)
        self.head = head

    def forward(self, xy):
        h = torch.relu(self.l1(xy))
        h = torch.relu(self.l2(h))
        h = torch.tanh(self.l3(h))
        h = torch.relu(self.l4(h))
        o = self.out(h)
        if self.head == "sigmoid":
            kp = KP_MIN + (KP_MAX - KP_MIN) * torch.sigmoid(o[..., 0])
            kth = KTH_MIN + (KTH_MAX - KTH_MIN) * torch.sigmoid(o[..., 1])
            return torch.stack([kp, kth], dim=-1)
        return o


def train_supervised(targets, labels, *, seed=0, epochs=360, batch=128,
                     lr=1e-3, betas=(0.9, 0.99), head="linear", val_frac=0.15):
    """Anchor-style supervised training on brute-force/BO labels."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(targets)
    idx = rng.permutation(n)
    nval = max(1, int(val_frac * n))
    vi, ti = idx[:nval], idx[nval:]
    X = torch.tensor(targets, dtype=torch.float32)
    Y = torch.tensor(labels, dtype=torch.float32)
    net = GainMLP(head=head)
    opt = torch.optim.Adam(net.parameters(), lr=lr, betas=betas)
    lossf = nn.MSELoss()
    hist = []
    for ep in range(epochs):
        perm = rng.permutation(len(ti))
        ep_loss = 0.0; nb = 0
        for s in range(0, len(ti), batch):
            b = ti[perm[s:s + batch]]
            opt.zero_grad()
            loss = lossf(net(X[b]), Y[b])
            loss.backward()
            opt.step()
            ep_loss += loss.item(); nb += 1
        with torch.no_grad():
            vloss = lossf(net(X[vi]), Y[vi]).item()
        hist.append((ep, ep_loss / max(nb, 1), vloss))
    return net, hist


@torch.no_grad()
def predict_gains(net, targets, clip_box=True):
    xy = torch.tensor(np.asarray(targets, float).reshape(-1, 2), dtype=torch.float32)
    g = net(xy).numpy().astype(float)
    if clip_box:
        g[:, 0] = np.clip(g[:, 0], KP_MIN, KP_MAX)
        g[:, 1] = np.clip(g[:, 1], KTH_MIN, KTH_MAX)
    return g
