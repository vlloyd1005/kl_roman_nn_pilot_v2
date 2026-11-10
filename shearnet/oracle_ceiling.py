#!/usr/bin/env python3
"""Information ceiling from PERFECT (noiseless, infinitely resolved) observables.

From the truths we compute what an ideal instrument would measure:
  * the full velocity-field geometry of a sheared thin disk (the 2x2 map
    sky <- disk plane, up to overall scale and the unknown position angle),
  * the projected rotation amplitude V_obs = vcirc * sin(i),
  * the second-moment ellipse of the (thick) imaging disk,
  * the Tully-Fisher prior mean (pop.prior_vcirc_mu_kms, observable).
A gradient-boosted regressor (grouped 5-fold CV on pop.id) then estimates the
posterior mean of g+ and gx from these.  No image noise or PSF is involved, so
the resulting correlation is an UPPER bound on what any image-based estimator
can reach on this data set; what remains is the irreducible degeneracy
(Tully-Fisher scatter for g+, unknown disk thickness h/r).
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

t = pd.read_parquet(sys.argv[1] if len(sys.argv) > 1 else "/home/user/data/v3/table.parquet")
gp, gx = t["truth.g_plus"].to_numpy(), t["truth.g_cross"].to_numpy()
c = t["truth.cosi"].to_numpy()
s = np.sqrt(1 - c**2)
hr = t["truth.h_over_r"].to_numpy()

# Galaxy-frame map disk-plane -> sky (up to the unknown left rotation by theta_int)
#   A = (I + G) diag(1, c),  G = [[g+, gx], [gx, -g+]]
A11, A12, A21, A22 = 1 + gp, gx * c, gx, (1 - gp) * c
# rotation invariants of A (A^T A), normalised by the unknown overall scale
M11 = A11**2 + A21**2
M12 = A11 * A12 + A21 * A22
M22 = A12**2 + A22**2
kin_q2 = M22 / M11
kin_skew = M12 / M11
# kinematic major axis in sky (galaxy frame): A e1
kin_ang = np.arctan2(A21, A11)

# thick-disk image moments: diag(1, q_int^2) sheared
qi2 = c**2 + hr**2 * s**2
Q11 = (1 + gp) ** 2 + gx**2 * qi2
Q22 = gx**2 + (1 - gp) ** 2 * qi2
Q12 = (1 + gp) * gx + gx * (1 - gp) * qi2
img_ang = 0.5 * np.arctan2(2 * Q12, Q11 - Q22)
tr, det = Q11 + Q22, Q11 * Q22 - Q12**2
lam1 = tr / 2 + np.sqrt(tr**2 / 4 - det)
lam2 = tr / 2 - np.sqrt(tr**2 / 4 - det)
img_q = np.sqrt(lam2 / lam1)

V_obs = t["truth.vel.vcirc"].to_numpy() * s
mu_tf = t["pop.prior_vcirc_mu_kms"].to_numpy()

feats = {
    "kin_q2": kin_q2, "kin_skew": kin_skew, "img_q": img_q,
    "img_minus_kin_ang": img_ang - kin_ang,
    "log_vobs_over_tf": np.log10(V_obs / mu_tf),
}
groups = t["pop.id"].to_numpy()


def cv(X, y, label):
    pred = np.zeros_like(y)
    for tr_i, te_i in GroupKFold(5).split(X, y, groups):
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31,
                              min_child_samples=30, verbose=-1)
        m.fit(X[tr_i], y[tr_i])
        pred[te_i] = m.predict(X[te_i])
    r = np.corrcoef(pred, y)[0, 1]
    bins = [0.05, 0.3, 0.6, 0.95]
    per = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        k = (c >= lo) & (c < hi)
        per.append(f"cosi[{lo},{hi}) r={np.corrcoef(pred[k], y[k])[0, 1]:.2f}")
    print(f"{label:48s} r={r:.3f}  rmse={np.sqrt(np.mean((pred - y) ** 2)):.4f}  | " + "  ".join(per))


X_all = np.column_stack(list(feats.values()))
print("Oracle ceilings (noiseless, perfect geometry), grouped 5-fold CV")
cv(X_all, gp, "g+  all observables")
cv(X_all, gx, "gx  all observables")
X_noTF = np.column_stack([feats[k] for k in ["kin_q2", "kin_skew", "img_q", "img_minus_kin_ang"]])
cv(X_noTF, gp, "g+  without rotation amplitude / TF prior")
X_kin = np.column_stack([feats[k] for k in ["kin_q2", "kin_skew", "log_vobs_over_tf"]])
cv(X_kin, gp, "g+  kinematics + TF only (no imaging)")
X_img = np.column_stack([feats[k] for k in ["img_q"]])
cv(X_img, gp, "g+  imaging axis ratio only")
