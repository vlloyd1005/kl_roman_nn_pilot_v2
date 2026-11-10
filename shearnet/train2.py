#!/usr/bin/env python3
"""shearnet v2: canonicalised CNN for galaxy-frame shear, optionally pretrained
on forward-model simulations and fine-tuned on the real training rows.

Every example gets a fresh noise draw each time it is used:
  x = clean + sqrt(var) * N(0, 1),  var = sigma_bkg^2 + max(clean, 0) / gain
(real rows use their stored var map, simulations compute it from the base row's gains).
Validation / test always use the PROVIDED noisy stamps of held-out galaxies.
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import common as C
from canon import CanonPreproc
from train import Block


class ShearNet2(nn.Module):
    def __init__(self, cin, n_scal, n_aux, w=32, drop=0.2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(cin, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.GELU())
        self.body = nn.Sequential(
            Block(w, w), Block(w, w),
            Block(w, 2 * w, 2), Block(2 * w, 2 * w),
            Block(2 * w, 4 * w, 2), Block(4 * w, 4 * w),
        )
        self.pool = nn.AdaptiveAvgPool2d(3)
        self.scal = nn.Sequential(nn.Linear(n_scal, 64), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(4 * w * 9 + 64, 384), nn.GELU(), nn.Dropout(drop),
            nn.Linear(384, 128), nn.GELU(),
        )
        self.mu = nn.Linear(128, 2)
        self.logsig = nn.Linear(128, 2)
        self.aux = nn.Linear(128, n_aux)

    def forward(self, x, s):
        h = self.pool(self.body(self.stem(x))).flatten(1)
        h = self.head(torch.cat([h, self.scal(s)], 1))
        return self.mu(h), self.logsig(h), self.aux(h)


class Source:
    """A set of rows that can be drawn with fresh noise."""

    def __init__(self, clean, table, sig, var=None):
        self.clean = clean
        self.t = table
        self.Y = (table[C.TARGETS].to_numpy() / C.G_SCALE).astype(np.float32)
        self.AUX = C.aux_targets(table)
        self.S = C.scalars(table)
        self.var = var
        if var is None:
            self.gain = np.stack([table[f"gain.{c}"].to_numpy() for c in C.CHANNELS], 1).astype(np.float32)
            self.sig2 = (np.asarray(sig, np.float32) ** 2)

    def noisy(self, b, gen):
        c = torch.from_numpy(self.clean[b])
        if self.var is not None:
            v = torch.from_numpy(self.var[b])
        else:
            v = torch.from_numpy(self.sig2).view(1, 6, 1, 1) + c.clamp_min(0) / torch.from_numpy(self.gain[b]).view(-1, 6, 1, 1)
        return c + v.sqrt() * torch.randn(c.shape, generator=gen)


def run_eval(model, pre, X, S, idx, bs=512):
    model.eval()
    mus, sigs = [], []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            xin, ang = pre(torch.from_numpy(X[b]))
            mu, ls, _ = model(xin, torch.cat([torch.from_numpy(S[b]), ang], 1))
            mus.append(mu.numpy())
            sigs.append(torch.exp(ls.clamp(-4, 3)).numpy())
    return np.concatenate(mus) * C.G_SCALE, np.concatenate(sigs) * C.G_SCALE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/user/data/v3")
    ap.add_argument("--sims", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps_per_epoch", type=int, default=0, help="0 = one pass over the pool")
    ap.add_argument("--real_frac", type=float, default=1.0,
                    help="fraction of each batch drawn from real training rows (rest from sims)")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--crop", type=int, default=24)
    ap.add_argument("--aux_w", type=float, default=0.3)
    ap.add_argument("--nll_w", type=float, default=0.2)
    ap.add_argument("--init", default=None, help="checkpoint to fine-tune from")
    ap.add_argument("--no_canon", action="store_true")
    ap.add_argument("--kin", action="store_true", help="kinematic channels + moment-kinematics scalars")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    t, noisy, clean, var, sig, rolls = C.load(a.data)
    tr, va, te = C.group_split(t, seed=0, fold=a.fold)
    real = Source(clean, t, sig, var=var)
    s_mean, s_std = real.S[tr].mean(0), real.S[tr].std(0)
    norm = lambda S: ((S - s_mean) / s_std).astype(np.float32)
    real.S = norm(real.S)
    sims = []
    for d in a.sims:
        st = pd.read_parquet(Path(d) / "table.parquet")
        sc = np.load(Path(d) / "clean.npy", mmap_mode="r")
        n = min(len(st), len(sc))
        src = Source(sc, st.iloc[:n].reset_index(drop=True), sig)
        src.S = norm(src.S)
        sims.append(src)
    n_sim = sum(len(s.Y) for s in sims)
    print(f"real train {len(tr)} val {len(va)} test {len(te)}; sims {n_sim}", flush=True)

    pre = CanonPreproc(sig, rolls, crop=a.crop, canon=not a.no_canon, kin=a.kin)
    n_scal = real.S.shape[1] + pre.n_ang
    model = ShearNet2(pre.n_ch, n_scal, real.AUX.shape[1], w=a.width)
    if a.init:
        model.load_state_dict(torch.load(a.init, weights_only=False)["state"])
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    print("params", sum(p.numel() for p in model.parameters()), flush=True)

    pool = len(tr) + (n_sim if a.real_frac < 1 else 0)
    spe = a.steps_per_epoch or math.ceil(pool / a.bs)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * spe,
                                                pct_start=0.1 if a.init is None else 0.02)

    def draw(nb):
        """Return (x, S, Y, AUX) for a batch mixing real and simulated rows."""
        n_real = nb if not sims else int(round(nb * a.real_frac))
        parts = []
        if n_real:
            b = np.sort(rng.choice(tr, n_real, replace=False))
            parts.append((real.noisy(b, gen), real.S[b], real.Y[b], real.AUX[b]))
        n_s = nb - n_real
        if n_s:
            k = rng.integers(len(sims))
            s = sims[k]
            b = np.sort(rng.choice(len(s.Y), n_s, replace=False))
            parts.append((s.noisy(b, gen), s.S[b], s.Y[b], s.AUX[b]))
        x = torch.cat([p[0] for p in parts])
        return x, *(torch.from_numpy(np.concatenate([p[i] for p in parts])) for i in (1, 2, 3))

    best, best_state, hist = 1e9, None, []
    for ep in range(a.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for _ in range(spe):
            x, S, y, aux_t = draw(a.bs)
            xin, ang = pre(x)
            mu, ls, aux = model(xin, torch.cat([S, ang], 1))
            mse = ((mu - y) ** 2).mean()
            ls = ls.clamp(-4, 3)
            nll = (0.5 * (mu.detach() - y) ** 2 * torch.exp(-2 * ls) + ls).mean()
            loss = mse + a.nll_w * nll + a.aux_w * F.smooth_l1_loss(aux, aux_t)
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
            tot += mse.item()
        pv, _ = run_eval(ema, pre, noisy, real.S, va)
        yv = real.Y[va] * C.G_SCALE
        vmse = float(np.mean((pv - yv) ** 2))
        r = [np.corrcoef(pv[:, k], yv[:, k])[0, 1] for k in range(2)]
        hist.append(dict(epoch=ep, train_mse=tot / spe, val_mse=vmse / C.G_SCALE**2, r_gp=r[0], r_gx=r[1]))
        flag = ""
        if vmse < best:
            best, best_state, flag = vmse, copy.deepcopy(ema.state_dict()), " *"
        print(f"ep {ep:3d} train {tot / spe:.4f} val_mse {vmse / C.G_SCALE**2:.4f} "
              f"r_g+ {r[0]:.3f} r_gx {r[1]:.3f} {time.time() - t0:.0f}s{flag}", flush=True)
        torch.save(dict(state=best_state, args=vars(a), s_mean=s_mean, s_std=s_std,
                        sig=sig, rolls=rolls, n_scal=n_scal, n_aux=real.AUX.shape[1]), out / "model.pt")

    ema.load_state_dict(best_state)
    res = {}
    for name, idx in [("val", va), ("test", te)]:
        p, s = run_eval(ema, pre, noisy, real.S, idx)
        yt = real.Y[idx] * C.G_SCALE
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
