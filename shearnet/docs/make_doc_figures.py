"""Figures for the KL documentation PDF."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

out = "docs/figs/"

# ---------------------------------------------------------------- 1. KL schematic
def vfield(cosi, gp, gx, theta=0.0, n=201, L=3.0, rv=0.6):
    ax = np.linspace(-L, L, n); X, Y = np.meshgrid(ax, ax)
    G = np.array([[1 + gp, gx], [gx, 1 - gp]])
    A = G @ np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) @ np.diag([1, cosi])
    Ai = np.linalg.inv(A)
    u = Ai[0, 0] * X + Ai[0, 1] * Y; v = Ai[1, 0] * X + Ai[1, 1] * Y
    R = np.hypot(u, v) + 1e-9
    I = np.exp(-R / 0.8)
    V = np.arctan(R / rv) * u / R
    return X, Y, I, V, A

fig, axs = plt.subplots(1, 3, figsize=(13, 4.4))
cases = [(0.0, 0.0, "no shear"), (0.0, 0.25, r"$g_\times$ (exaggerated, 0.25)"), (0.25, 0.0, r"$g_+$ (exaggerated, 0.25)")]
cosi = 0.55
for a, (gp, gx, ttl) in zip(axs, cases):
    X, Y, I, V, A = vfield(cosi, gp, gx)
    a.imshow(np.where(I > 0.04, V, np.nan), extent=[-3, 3, -3, 3], origin="lower", cmap="RdBu_r", vmin=-1.2, vmax=1.2)
    a.contour(X, Y, I, levels=[0.08, 0.25, 0.6], colors="k", linewidths=0.8)
    # photometric major axis from image moments
    w = I / I.sum(); qxx = (w * X * X).sum(); qyy = (w * Y * Y).sum(); qxy = (w * X * Y).sum()
    pa = 0.5 * np.arctan2(2 * qxy, qxx - qyy)
    a.plot([-2.8 * np.cos(pa), 2.8 * np.cos(pa)], [-2.8 * np.sin(pa), 2.8 * np.sin(pa)], "k-", lw=2, label="photometric major axis")
    k = A @ np.array([1, 0]); kpa = np.arctan2(k[1], k[0])
    a.plot([-2.8 * np.cos(kpa), 2.8 * np.cos(kpa)], [-2.8 * np.sin(kpa), 2.8 * np.sin(kpa)], "--", color="#2b8a3e", lw=2, label="kinematic major axis")
    m = A @ np.array([0, 1]); mpa = np.arctan2(m[1], m[0])
    a.plot([-2.2 * np.cos(mpa), 2.2 * np.cos(mpa)], [-2.2 * np.sin(mpa), 2.2 * np.sin(mpa)], ":", color="#2b8a3e", lw=2, label="kinematic minor axis (v = v$_0$)")
    a.set_title(ttl); a.set_xticks([]); a.set_yticks([]); a.set_xlim(-3, 3); a.set_ylim(-3, 3)
axs[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
fig.suptitle("Line-of-sight velocity (colour) and light (contours) of an inclined disk, cos i = 0.55", y=0.99)
plt.tight_layout(); plt.savefig(out + "kl_schematic.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 2. TF-limited g+ precision
c = np.linspace(0.05, 0.95, 200)
s_lnv = 0.049 * np.log(10)
sig = 0.5 * (1 - c**2) / c**2 * s_lnv
fig, a = plt.subplots(figsize=(6.2, 3.8))
a.semilogy(c, sig, lw=2, label=r"$\sigma(g_+) \approx \frac{1}{2}\tan^2 i\;\sigma_{\ln v}$ (TF scatter 0.049 dex)")
a.axhline(0.0577, color="k", ls="--", lw=1, label="prior width of $g_+$ (uniform ±0.1)")
a.set_xlabel("cos i (0 = edge-on, 1 = face-on)"); a.set_ylabel(r"$\sigma(g_+)$ per galaxy, perfect data")
a.set_ylim(5e-3, 20); a.legend(fontsize=8); a.grid(alpha=0.3)
a.set_title("Thin-disk Tully-Fisher limit on $g_+$")
plt.tight_layout(); plt.savefig(out + "tf_limit.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 3. progression
labels = ["v3 cnn_feat\n(run01/02)", "v3 imgfit_f0_v2\n(run04, prev. best)", "E1\nreal only", "E3\n60k sims\n(old model)",
          "E4\n60k sims\n(fixed model)", "E5\n120k sims", "E5+E6\n240k sims\n(released)"]
gp = [0.300, 0.372, 0.19, 0.385, 0.405, 0.416, 0.420]
gx = [0.457, 0.491, 0.0, 0.510, 0.556, 0.580, 0.590]
x = np.arange(len(labels))
fig, a = plt.subplots(figsize=(10, 4))
a.bar(x - 0.2, gp, 0.4, label="r(g+)", color="#1f77b4"); a.bar(x + 0.2, gx, 0.4, label="r(g×)", color="#ff7f0e")
for xi, v in zip(list(x - 0.2) + list(x + 0.2), gp + gx):
    a.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
a.axhline(0.56, color="#1f77b4", ls=":", lw=1.2); a.text(2.2, 0.57, "g+ ceiling with perfect, noiseless data = 0.56", fontsize=8, color="#1f77b4", va="bottom", ha="center")
a.axvline(1.5, color="0.6", lw=1)
a.text(0.5, 0.68, "previous work", ha="center", fontsize=9, color="0.3"); a.text(4, 0.68, "this work", ha="center", fontsize=9, color="0.3")
a.set_xticks(x); a.set_xticklabels(labels, fontsize=8); a.set_ylim(0, 0.75); a.set_ylabel("test correlation r")
a.legend(loc="upper left", fontsize=8); a.set_title("Held-out correlation with true shear (v3, line SNR ×10)")
plt.tight_layout(); plt.savefig(out + "progression.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 4. forward-model fixes
steps = ["initial\nthin disk", "+ velocity field\ncentred on origin", "+ continuum\n±20 px"]
med = [359, 42, 30]; mean = [1246, 246, 75]
fig, a = plt.subplots(figsize=(6.2, 3.6))
xx = np.arange(3)
a.bar(xx - 0.2, med, 0.4, label="median"); a.bar(xx + 0.2, mean, 0.4, label="mean")
a.axhline(4096, color="k", ls="--", lw=1); a.text(2.45, 4096 * 1.1, "pure noise (4 stamps)", ha="right", fontsize=8)
a.set_yscale("log"); a.set_xticks(xx); a.set_xticklabels(steps, fontsize=8)
a.set_ylabel(r"grism $\chi^2$ per galaxy (4 rolls)"); a.legend(fontsize=8)
a.set_title("Forward-model mismatch vs noiseless mocks (40 random rows)")
plt.tight_layout(); plt.savefig(out + "forward_fixes.png", dpi=150); plt.close()
print("ok")
