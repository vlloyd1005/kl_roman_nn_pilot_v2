#!/usr/bin/env python3
"""shearnet: heteroscedastic shear regression from Roman imaging + grism stamps.

Predicts (g+, gx) and a per-galaxy uncertainty for each, trained with
  * a fresh noise realisation every time a galaxy is seen (clean + sqrt(var) N(0,1)),
  * physical auxiliary targets (cos i, intrinsic PA as spin-2, v sin i, sky g1/g2, h/r),
  * EMA weights and checkpoint selection on validation shear MSE.
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common as C


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.sc = None
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        h = F.gelu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.gelu(h + (x if self.sc is None else self.sc(x)))


class ShearNet(nn.Module):
    def __init__(self, cin, n_scal, n_aux, w=32, drop=0.2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(cin, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.GELU())
        self.body = nn.Sequential(
            Block(w, w), Block(w, w),
            Block(w, 2 * w, 2), Block(2 * w, 2 * w),
            Block(2 * w, 4 * w, 2), Block(4 * w, 4 * w),
        )
        feat = 4 * w
        # keep coarse spatial layout (4x4 pool) instead of pure global pooling:
        # shear lives in the relative geometry of features, not just their presence
        self.pool = nn.AdaptiveAvgPool2d(2)
        self.scal = nn.Sequential(nn.Linear(n_scal, 32), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(feat * 4 + 32, 256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, 128), nn.GELU(),
        )
        self.mu = nn.Linear(128, 2)
        self.logsig = nn.Linear(128, 2)
        self.aux = nn.Linear(128, n_aux) if n_aux else None

    def forward(self, x, s):
        h = self.pool(self.body(self.stem(x))).flatten(1)
        h = self.head(torch.cat([h, self.scal(s)], 1))
        aux = self.aux(h) if self.aux is not None else None
        return self.mu(h), self.logsig(h), aux


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------

def batches(idx, bs, shuffle, rng):
    idx = rng.permutation(idx) if shuffle else idx
    for i in range(0, len(idx), bs):
        yield idx[i:i + bs]


@torch.no_grad()
def predict(model, pre, X, S, idx, bs=512):
    model.eval()
    mus, sigs = [], []
    for b in batches(idx, bs, False, None):
        mu, ls, _ = model(pre(torch.from_numpy(X[b])), torch.from_numpy(S[b]))
        mus.append(mu.numpy())
        sigs.append(torch.exp(ls).numpy())
    return np.concatenate(mus) * C.G_SCALE, np.concatenate(sigs) * C.G_SCALE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/user/data/v3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--crop", type=int, default=24)
    ap.add_argument("--aux_w", type=float, default=0.3)
    ap.add_argument("--nll_w", type=float, default=0.2)
    ap.add_argument("--input", choices=["noisy", "clean"], default="noisy",
                    help="clean = noiseless-ceiling experiment (train and test on clean stamps)")
    ap.add_argument("--resample", type=float, default=1.0,
                    help="probability a training example gets fresh noise from clean+var")
    ap.add_argument("--no_align", action="store_true")
    ap.add_argument("--no_scalars", action="store_true")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max_train", type=int, default=0)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    t, noisy, clean, var, sig, rolls = C.load(a.data)
    S = C.scalars(t)
    tr, va, te = C.group_split(t, seed=0, fold=a.fold)
    if a.max_train:
        tr = tr[:a.max_train]
    S = (S - S[tr].mean(0)) / S[tr].std(0)
    if a.no_scalars:
        S[:] = 0
    S = S.astype(np.float32)
    Y = (t[C.TARGETS].to_numpy() / C.G_SCALE).astype(np.float32)
    AUX = C.aux_targets(t)
    std = np.sqrt(var).astype(np.float32)
    Xeval = clean if a.input == "clean" else noisy
    print(f"train {len(tr)} val {len(va)} test {len(te)} rows; input={a.input}", flush=True)

    pre = Preproc = C.Preproc(sig, rolls, crop=a.crop, align=not a.no_align)
    model = ShearNet(pre.n_ch, S.shape[1], AUX.shape[1] if a.aux_w > 0 else 0, w=a.width)
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    print("params", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * math.ceil(len(tr) / a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.15)

    best, best_state, hist = 1e9, None, []
    for ep in range(a.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for b in batches(tr, a.bs, True, rng):
            b = np.sort(b)
            if a.input == "clean":
                x = torch.from_numpy(clean[b])
            else:
                x = torch.from_numpy(noisy[b])
                if a.resample > 0:
                    fresh = torch.from_numpy(clean[b]) + torch.from_numpy(std[b]) * torch.randn(len(b), 6, 32, 32)
                    use = torch.rand(len(b)) < a.resample
                    x = torch.where(use.view(-1, 1, 1, 1), fresh, x)
            y = torch.from_numpy(Y[b])
            mu, ls, aux = model(pre(x), torch.from_numpy(S[b]))
            mse = ((mu - y) ** 2).mean()
            # NLL for the uncertainty head on a detached mean, so it cannot distort the estimate
            ls = ls.clamp(-4, 3)
            nll = (0.5 * ((mu.detach() - y) ** 2) * torch.exp(-2 * ls) + ls).mean()
            loss = mse + a.nll_w * nll
            if aux is not None:
                loss = loss + a.aux_w * F.smooth_l1_loss(aux, torch.from_numpy(AUX[b]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(a.ema).add_(pm.detach(), alpha=1 - a.ema)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            tot += mse.item() * len(b)
        pv, sv = predict(ema, pre, Xeval, S, va)
        yv = Y[va] * C.G_SCALE
        vmse = float(np.mean((pv - yv) ** 2))
        r = [np.corrcoef(pv[:, k], yv[:, k])[0, 1] for k in range(2)]
        hist.append(dict(epoch=ep, train_mse=tot / len(tr), val_mse=vmse, r_gp=r[0], r_gx=r[1]))
        flag = ""
        if vmse < best:
            best, best_state, flag = vmse, copy.deepcopy(ema.state_dict()), " *"
        print(f"ep {ep:3d} train {tot / len(tr):.4f} val_mse {vmse / C.G_SCALE**2:.4f} "
              f"r_g+ {r[0]:.3f} r_gx {r[1]:.3f} {time.time() - t0:.0f}s{flag}", flush=True)

    ema.load_state_dict(best_state)
    torch.save(dict(state=best_state, args=vars(a), sig=sig, rolls=rolls,
                    s_mean=C.scalars(t)[tr].mean(0), s_std=C.scalars(t)[tr].std(0)), out / "model.pt")
    res = {}
    for name, idx in [("val", va), ("test", te)]:
        p, s = predict(ema, pre, Xeval, S, idx)
        yt = Y[idx] * C.G_SCALE
        np.savez(out / f"pred_{name}.npz", idx=idx, pred=p, sigma=s, true=yt)
        for k, comp in enumerate(["g_plus", "g_cross"]):
            m = C.calib_metrics(p[:, k], yt[:, k])
            m["sig_eff_weighted"] = C.weighted_sig_eff(p[:, k], yt[:, k], s[:, k])
            res[f"{name}_{comp}"] = m
    json.dump(dict(results=res, history=hist, args=vars(a)), open(out / "results.json", "w"), indent=1)
    for k, v in res.items():
        print(k, {kk: round(vv, 4) for kk, vv in v.items()})


if __name__ == "__main__":
    main()
