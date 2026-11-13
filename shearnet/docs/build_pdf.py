#!/usr/bin/env python3
"""Build the KL shear-inference documentation PDF (docs/KL_shear_inference_report.pdf)."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = HERE / "figs"
FIGS2 = ROOT / "figs"
OUT = HERE / "KL_shear_inference_report.pdf"

import matplotlib, os
D = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf") + "/"
pdfmetrics.registerFont(TTFont("DV", D + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DV-B", D + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DV-I", D + "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFont(TTFont("DV-BI", D + "DejaVuSans-BoldOblique.ttf"))
pdfmetrics.registerFont(TTFont("DVM", D + "DejaVuSansMono.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

ss = getSampleStyleSheet()
INK = colors.HexColor("#1b1f24")
ACC = colors.HexColor("#1f4e79")
body = ParagraphStyle("body", parent=ss["Normal"], fontName="DV", fontSize=9.6, leading=13.6, textColor=INK,
                      spaceAfter=6)
small = ParagraphStyle("small", parent=body, fontSize=8.2, leading=11)
cap = ParagraphStyle("cap", parent=body, fontSize=8.2, leading=11, textColor=colors.HexColor("#444"),
                     spaceBefore=2, spaceAfter=10)
h1 = ParagraphStyle("h1", parent=body, fontName="DV-B", fontSize=15, leading=19, textColor=ACC,
                    spaceBefore=6, spaceAfter=8)
h2 = ParagraphStyle("h2", parent=body, fontName="DV-B", fontSize=11.2, leading=15, textColor=ACC,
                    spaceBefore=8, spaceAfter=4)
eq = ParagraphStyle("eq", parent=body, alignment=TA_CENTER, fontName="DV", fontSize=9.8, spaceBefore=2, spaceAfter=8)
code = ParagraphStyle("code", parent=body, fontName="DVM", fontSize=7.8, leading=10.2,
                      backColor=colors.HexColor("#f3f4f6"), borderPadding=5, spaceBefore=4, spaceAfter=10)
title = ParagraphStyle("title", parent=body, fontName="DV-B", fontSize=22, leading=28, textColor=ACC, alignment=TA_CENTER)
sub = ParagraphStyle("sub", parent=body, fontSize=11.5, leading=16, alignment=TA_CENTER, textColor=colors.HexColor("#333"))
cell = ParagraphStyle("cell", parent=body, fontSize=8.2, leading=10.4, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="DV-B")

story = []
P = lambda t, s=body: story.append(Paragraph(t, s))
H1 = lambda t: story.append(Paragraph(t, h1))
H2 = lambda t: story.append(Paragraph(t, h2))
EQ = lambda t: story.append(Paragraph(t, eq))


def bullets(items, style=body):
    story.append(ListFlowable([ListItem(Paragraph(i, style), leftIndent=12, value="•") for i in items],
                              bulletType="bullet", start="•", leftIndent=12, bulletFontName="DV",
                              spaceAfter=6))


def fig(path, width, caption):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(path)).getSize()
    img = Image(str(path), width=width, height=width * ih / iw)
    story.append(KeepTogether([img, Paragraph(caption, cap)]))


def table(rows, widths, header=True, bold_rows=()):
    data = [[Paragraph(str(c), cellb if (header and i == 0) or i in bold_rows else cell) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8ccd2")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")))
    for r in bold_rows:
        st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fff4e0")))
    t.setStyle(TableStyle(st))
    story.append(t)
    story.append(Spacer(1, 8))


W = 6.9 * inch

# =====================================================================================
# Title page
# =====================================================================================
story.append(Spacer(1, 1.3 * inch))
P("Kinematic-lensing shear inference from Roman imaging and slitless grism", title)
story.append(Spacer(1, 14))
P("Physics, previous approaches, physical limits, a simulation-trained network (shearnet), results and next steps", sub)
story.append(Spacer(1, 26))
P("Data set: <b>kl_roman_nn_v3_lsnr10x</b> (13,481 mock galaxies, Hα line depth ×10)<br/>"
  "Repository: kl_roman_nn_pilot_v2, branch <font name='DVM'>shearnet-sim-trained</font>, folder <font name='DVM'>shearnet/</font><br/>"
  "November 2026", sub)
story.append(Spacer(1, 40))
P("<b>Contents</b>", h2)
toc = ["1. Summary", "2. The physics of kinematic lensing", "3. The v3 mock data set",
       "4. Physical limits: what any estimator can reach", "5. Previous approaches (train_kl_grism*.py)",
       "6. This work: from diagnosis to a simulation-trained model", "7. Main results",
       "8. Using the model in a shear pipeline", "9. Caveats and limitations", "10. Next steps",
       "Appendix A. Files and reproduction", "Appendix B. Metric definitions"]
for t_ in toc:
    P(t_, ParagraphStyle("toc", parent=body, leftIndent=30, spaceAfter=2))
story.append(PageBreak())

# =====================================================================================
# 1 Summary
# =====================================================================================
H1("1. Summary")
P("The goal is a fast estimator of the galaxy-frame shear (g<sub>+</sub>, g<sub>×</sub>) of a single disk "
  "galaxy from six Roman stamps: two imaging bands (F129, F158) and four slitless-grism exposures at roll "
  "angles 0°, 10°, 167° and 177°. It should be accurate enough to use in the Roman kinematic-lensing (KL) "
  "pipeline. Earlier networks (<font name='DVM'>train_kl_grism*.py</font>) reached a held-out correlation of "
  "r ≈ 0.37 (g<sub>+</sub>) and 0.49 (g<sub>×</sub>) on the v3 ×10 set. This document records why they stalled, "
  "what the data can support in principle, and a new model that improves on them.")
table([
    ["", "r(g<sub>+</sub>)", "r(g<sub>×</sub>)", "σ<sub>eff</sub>(g<sub>+</sub>)", "σ<sub>eff</sub>(g<sub>×</sub>)", "per-galaxy σ", "bias after calibration"],
    ["Previous best (run04 imgfit_f0_v2)", "0.372", "0.491", "≈ 0.145", "≈ 0.10", "no", "not measured"],
    ["<b>shearnet E5+E6 ensemble (released)</b>", "<b>0.420</b>", "<b>0.590</b>", "<b>0.097</b>", "<b>0.044</b>",
     "yes, calibrated", "m = +0.014 ± 0.036 (g<sub>+</sub>), +0.025 ± 0.016 (g<sub>×</sub>); |c| < 0.002"],
    ["Ceiling with perfect, noiseless data", "≈ 0.56", "1.0", "", "", "", ""],
], [1.75 * inch, 0.55 * inch, 0.55 * inch, 0.7 * inch, 0.7 * inch, 0.8 * inch, 1.85 * inch], bold_rows=(2,))
P("σ<sub>eff</sub> is the effective per-galaxy shear noise of the bias-corrected, inverse-variance-weighted "
  "estimator: N galaxies measure a common shear to σ<sub>eff</sub>/√N. Conventional shape-based weak lensing has "
  "about 0.26 per component. For a fixed shear precision the new model needs about <b>5×</b> fewer galaxies "
  "than the previous best for g<sub>×</sub> and about <b>2×</b> fewer for g<sub>+</sub>.")
P("<b>Key findings</b>")
bullets([
    "<b>g<sub>+</sub> is physically limited.</b> For a thin disk, a shear along the major axis is exactly "
    "degenerate with inclination in both the light and the velocity field. Only the Tully-Fisher (TF) prior on "
    "v<sub>circ</sub> (0.049 dex scatter) and the disk thickness break the degeneracy. Even with perfect data "
    "r(g<sub>+</sub>) ≤ 0.56. Edge-on disks carry almost no g<sub>+</sub> information.",
    "<b>g<sub>×</sub> is limited by noise and resolution, not physics</b> (the perfect-data ceiling is r = 1). Its "
    "signal is a small misalignment between the kinematic and photometric axes.",
    "<b>The earlier networks were data-starved.</b> 13,481 rows contain only 3,014 distinct galaxies. A CNN "
    "trained on them memorises the galaxies (training loss falls while validation rises, even with fresh noise "
    "every epoch) and learns no g<sub>×</sub> without hand-built kinematic inputs.",
    "<b>The fix was simulation.</b> The mock renderer was rebuilt (<font name='DVM'>forward.py</font>) and "
    "validated pixel by pixel against the noiseless stamps (median χ² ≈ 5 per stamp against 1024 for pure noise). "
    "The network was then trained on 240,000 simulated galaxies plus the real training rows. Matching the renderer "
    "exactly mattered: one convention error (where the velocity field is centred) cost about 0.1 in r(g<sub>×</sub>).",
    "<b>The network returns a calibrated per-galaxy uncertainty</b> (pull width 0.91-1.04). Inverse-variance "
    "weighting is responsible for much of the σ<sub>eff</sub> gain, because the information is concentrated in "
    "face-on and mid-inclination disks.",
])
fig(FIG / "progression.png", W, "Figure 1. Held-out correlation between predicted and true shear for the earlier "
    "v3 runs and the runs of this work. Test splits differ between the two groups (±0.02 split-to-split). The "
    "dotted line is the g<sub>+</sub> ceiling with perfect, noiseless observables (Section 4).")
story.append(PageBreak())

# =====================================================================================
# 2 Physics
# =====================================================================================
H1("2. The physics of kinematic lensing")
H2("2.1 Weak lensing and shape noise")
P("Weak gravitational lensing by foreground mass applies a small linear distortion to a background galaxy image, "
  "described by the reduced shear g = g<sub>1</sub> + i g<sub>2</sub> (|g| ≲ 0.05 in practice). Conventional "
  "weak lensing measures the observed ellipticity e<sub>obs</sub> ≈ e<sub>int</sub> + g and averages over many "
  "galaxies, assuming the intrinsic ellipticities e<sub>int</sub> are randomly oriented. The intrinsic scatter, "
  "about 0.26 per component (the <i>shape noise</i>), dominates the error budget. Measuring a shear to 0.01 needs "
  "about 700 galaxies.")
H2("2.2 The kinematic-lensing idea")
P("A rotating disk carries a second, independent record of its orientation: its velocity field. For a thin, "
  "circular disk at inclination i and intrinsic position angle θ<sub>int</sub>, the line-of-sight velocity is")
EQ("v<sub>los</sub>(R, φ) = v<sub>0</sub> + v<sub>rot</sub>(R) · sin i · cos φ,&nbsp;&nbsp;&nbsp;"
   "v<sub>rot</sub>(R) = v<sub>circ</sub> · (2/π) · arctan(R / r<sub>v</sub>)")
P("where (R, φ) are polar coordinates in the disk plane. Without lensing, the photometric major axis and the "
  "kinematic major axis (the line of extreme velocities) coincide, and the kinematic minor axis (v = v<sub>0</sub>) "
  "is perpendicular to it. Lensing shears the light and the velocity map with the <i>same</i> transformation, "
  "but the two respond differently. That difference lets the shear be measured from one galaxy, with no "
  "shape-noise averaging. Following the mocks, the shear is expressed in the galaxy frame:")
EQ("g<sub>+</sub> + i g<sub>×</sub> = (g<sub>1</sub> + i g<sub>2</sub>) · e<sup>−2iθ<sub>int</sub></sup>")
P("g<sub>+</sub> stretches the galaxy along its major axis and g<sub>×</sub> stretches it at 45° to it.")
fig(FIG / "kl_schematic.png", W, "Figure 2. Thin-disk velocity field (colour) and light (contours), cos i = 0.55, with "
    "exaggerated shears. <b>Middle:</b> g<sub>×</sub> rotates the photometric axis more than the kinematic axis, and "
    "the kinematic axes are no longer perpendicular. This misalignment is the g<sub>×</sub> signal. <b>Right:</b> "
    "g<sub>+</sub> only changes the axis ratio, so the sheared disk looks exactly like an unsheared disk at a "
    "different inclination.")
H2("2.3 Where the information on each component comes from")
P("<b>g<sub>×</sub>: axis misalignment.</b> To first order in shear, for a thin disk with c = cos i, the kinematic "
  "major axis rotates by g<sub>×</sub> and the photometric major axis by g<sub>×</sub>(1 + c²)/(1 − c²). The "
  "observable misalignment is therefore")
EQ("Δφ = φ<sub>kin</sub> − φ<sub>phot</sub> ≈ −2 g<sub>×</sub> c² / (1 − c²)")
P("For g<sub>×</sub> = 0.05 this is about 0.2° for an edge-on disk (c = 0.2), 2° at c = 0.5 and 10° at c = 0.8. On "
  "galaxies only 3-6 pixels across (scale radius ≈ 0.18″ = 1.6 px, PSF FWHM ≈ 2.5 px), measuring axes to a "
  "degree is the core difficulty, and it gets easier toward face-on disks. A sheared thin disk's velocity field "
  "also has a non-perpendicular kinematic minor axis. With perfect data this determines g<sub>×</sub> exactly at "
  "any inclination, but at Roman resolution it is hard to exploit for edge-on systems.")
P("<b>g<sub>+</sub>: inclination plus Tully-Fisher.</b> A shear along the major axis multiplies the axis ratio by "
  "(1 − g<sub>+</sub>)/(1 + g<sub>+</sub>) ≈ 1 − 2g<sub>+</sub>. It does the same to the velocity-field geometry, "
  "which is identical to a disk with cos i′ = cos i (1 − 2g<sub>+</sub>). Shape alone therefore cannot separate "
  "g<sub>+</sub> from inclination. The degeneracy is broken by the rotation amplitude v<sub>circ</sub> sin i, "
  "provided v<sub>circ</sub> is known independently from the Tully-Fisher relation (v<sub>circ</sub> from "
  "stellar mass). Then sin i, hence cos i, hence the intrinsic axis ratio, and finally g<sub>+</sub> follow. A finite "
  "disk thickness h/r also helps: the light has axis ratio q<sub>int</sub>² = cos² i + (h/r)² sin² i while the gas "
  "kinematics trace the thin-disk geometry.")
H2("2.4 What Roman observes")
P("Roman's High Latitude Wide Area Survey provides imaging and slitless spectroscopy (grism, 1.1 nm/pixel, "
  "0.11″ pixels). In a grism exposure every sky pixel's spectrum is dispersed along one detector axis. The Hα line "
  "image is shifted along the dispersion direction by Δx = λ<sub>obs</sub> v<sub>los</sub> / (c · 1.1 nm), about "
  "0.004 px per km s<sup>−1</sup> at z ≈ 1.2, or ≈ 0.7 px for 150 km s<sup>−1</sup>. The velocity field "
  "therefore appears as a sub-pixel distortion of the line image along the dispersion axis. The continuum smears "
  "into a band along the same axis. Two rolls about 180° apart (0°/177°, 10°/167°) reverse the sky relative to the "
  "dispersion direction, so the velocity distortion changes sign between them while the light distribution does "
  "not. This is what makes the rotation measurable.")
story.append(PageBreak())

# =====================================================================================
# 3 Data
# =====================================================================================
H1("3. The v3 mock data set")
P("<b>kl_roman_nn_v3_lsnr10x</b> holds 13,481 rows built by <font name='DVM'>kl_roman_pipe</font>. Each row is a "
  "COSMOS2025 galaxy with a painted Hα line, given a random orientation, inclination, shear and noise. The grism "
  "noise is reduced ×10 relative to the survey (line SNR ×10, about 100× exposure); the imaging is at survey "
  "depth. Selection: realized total line SNR ≥ 100 (median 150), z = 0.55-1.9. The data are summarised below.")
table([
    ["Quantity", "Value / distribution"],
    ["Rows / distinct galaxies", "13,481 / 3,014 (about 4.5 appearances each, with fresh orientation, shear, v<sub>circ</sub> and noise)"],
    ["Inputs", "6 × 32×32 stamps, 0.11″ pixels: F129, F158 (μJy/px) and grism rolls 0°, 10°, 167°, 177°; variance maps; noiseless versions"],
    ["Targets", "g<sub>+</sub>, g<sub>×</sub> independently uniform on [−0.1, 0.1] (prior std 0.0577)"],
    ["Inclination", "cos i uniform on [0.05, 0.95] before selection"],
    ["Tully-Fisher", "v<sub>circ</sub> = μ<sub>TF</sub> · 10<sup>N(0, 0.049 dex)</sup>; μ<sub>TF</sub> (pop.prior_vcirc_mu_kms) is observable"],
    ["Disk thickness", "h/r lognormal, median 0.25, 0.32 in ln (unknown to the observer)"],
    ["Sizes", "imaging scale radius median 0.18″ (1.6 px); Hα scale radius median 0.19″; v<sub>rscale</sub> median 0.09″"],
    ["Centroids", "independent per component (F129, F158, Hα, continuum), σ = 0.11″ (about 1 px)"],
    ["Noise", "Gaussian: var = σ<sub>bkg</sub>² + max(clean, 0)/gain"],
], [1.6 * inch, 5.3 * inch])
fig(FIGS2 / "examples.png", 3.9 * inch, "Figure 3. Four example rows: noiseless (upper) and noisy (lower) stamps for "
    "F129, F158 and the four grism rolls. The horizontal bands in the grism stamps are the dispersed continuum.")
P("<b>Split protocol.</b> Every result in this document uses one split by catalogue galaxy (pop.id, seed 0): "
  "70% train, 15% validation (2,056 rows), 15% test (2,073 rows). Simulated training galaxies are drawn only from "
  "training-split parameters. All decisions (checkpoints, ensembles, calibration) use validation; test was "
  "evaluated once per final model. The earlier runs used their own random splits, so comparisons across the two "
  "groups carry about ±0.02 in r.")
story.append(PageBreak())

# =====================================================================================
# 4 Limits
# =====================================================================================
H1("4. Physical limits: what any estimator can reach")
H2("4.1 Tully-Fisher-limited g<sub>+</sub>")
P("Combining Section 2.3 with the TF scatter σ<sub>ln v</sub> = 0.049 · ln 10 = 0.113 gives, for a thin disk "
  "with perfect data,")
EQ("σ(g<sub>+</sub>) ≈ ½ · tan² i · σ<sub>ln v</sub>")
P("This is 0.013 at cos i = 0.9, 0.17 at cos i = 0.5 and 1.4 at cos i = 0.2. Since the prior width of "
  "g<sub>+</sub> is 0.058, only galaxies with cos i ≳ 0.6 can constrain g<sub>+</sub> individually.")
fig(FIG / "tf_limit.png", 4.3 * inch, "Figure 4. Per-galaxy g<sub>+</sub> precision allowed by the TF scatter for a "
    "perfectly observed thin disk. Above the dashed line a galaxy is less informative than the prior.")
H2("4.2 Oracle experiment")
P("To include thickness and the selection-shaped inclination distribution, <font name='DVM'>oracle_ceiling.py</font> "
  "computes, from the truth table, what an ideal instrument would observe: the full sheared thin-disk velocity "
  "geometry (up to scale and the unknown PA), the rotation amplitude v<sub>circ</sub> sin i, the thick-disk image "
  "ellipse and μ<sub>TF</sub>. It then regresses the shear on these noiseless observables (gradient-boosted trees, "
  "grouped 5-fold cross-validation). This is an upper bound for any image-based method on this data set.")
table([
    ["Observables (perfect, noiseless)", "r(g<sub>+</sub>)", "cos i < 0.3", "0.3-0.6", "> 0.6", "r(g<sub>×</sub>)"],
    ["All (kinematic geometry, amplitude + TF, image shape)", "0.56", "0.34", "0.45", "0.85", "1.00"],
    ["Without amplitude / TF", "0.46", "0.35", "0.38", "0.67", ""],
    ["Kinematics + TF, no imaging", "0.45", "0.12", "0.26", "0.82", ""],
    ["Image axis ratio only", "0.28", "0.32", "0.09", "0.55", ""],
], [2.9 * inch, 0.7 * inch, 0.8 * inch, 0.7 * inch, 0.7 * inch, 0.8 * inch])
P("<b>Consequences.</b> (i) No estimator can exceed r(g<sub>+</sub>) ≈ 0.56 on this set. Per-galaxy R² for "
  "g<sub>+</sub> is at most about 0.3, so models should be judged on calibration and σ<sub>eff</sub>, not R². "
  "(ii) g<sub>×</sub> has no physical ceiling, so its remaining shortfall is due to noise, resolution and the "
  "estimator. (iii) The v3 README's likelihood-fit expectation of σ(g) ≈ 0.04-0.12 per galaxy, depending on "
  "inclination, is consistent with this picture.")
H2("4.3 Data volume")
P("The second limit is practical. The set has 3,014 distinct galaxies. A network has to learn to measure "
  "degree-level axis misalignments and sub-pixel velocity shifts from these alone, while the targets (shear) are "
  "only weakly encoded. A model with about 10<sup>6</sup> parameters can instead memorise the 3,014 clean "
  "galaxies, and that is what happens (Section 6.2).")
story.append(PageBreak())

# =====================================================================================
# 5 Previous work
# =====================================================================================
H1("5. Previous approaches (train_kl_grism*.py)")
P("The repository records a sequence of increasingly physics-informed attempts:")
table([
    ["Stage", "Data", "Approach", "Held-out result"],
    ["v2 pilot (kl_grism_runs, fkl_grism_runs)", "pilot_v2 (5,870 rows, shear N(0, 0.03), catalogue SNR cut)",
     "Pretrained EfficientNet, small CNN, moment features + GBT; auxiliary targets (cos i, v<sub>circ</sub>, v sin i), noise ramps",
     "r ≲ 0.13 (g<sub>+</sub>), ≲ 0.10 (g<sub>×</sub>); several runs near zero skill"],
    ["v2 ×10 (kl_grism_runs_lsnr10x)", "pilot v2 with line depth ×10",
     "5-fold small CNN, feature GBT, stacking, continuum-subtraction tests",
     "best k-fold r ≈ 0.11 (g<sub>+</sub>), 0.14 (g<sub>×</sub>)"],
    ["v3 run01/02 (cnn_feat_cosi017)", "v3 ×10, cos i > 0.17", "Two-stream CNN + feature scalars (below), 5-fold",
     "r = 0.30 (g<sub>+</sub>), 0.46 (g<sub>×</sub>)"],
    ["v3 run02 (feat_k5_tf)", "v3 ×10", "Feature GBT with TF features", "r ≈ 0.26 / 0.36"],
    ["v3 run03 (cnn_feat_k10, ablate_no_tf)", "v3 ×10", "CNN + observable catalogue inputs (TF prior, fluxes, mass)",
     "r = 0.37 / 0.48 (0.34 / 0.49 without TF)"],
    ["<b>v3 run04 (imgfit_f0_v2)</b>", "v3 ×10", "+ PSF-corrected parametric imaging fit (q, size, fluxes, χ²)",
     "<b>r = 0.372 / 0.491</b>, m<sub>raw</sub> ≈ −0.86 / −0.75"],
], [1.45 * inch, 1.45 * inch, 2.45 * inch, 1.55 * inch], bold_rows=(6,))
P("The v3 script (<font name='DVM'>train_kl_grism_v3.py</font>) combined substantial physics-motivated "
  "preprocessing: roll alignment of the grism stamps, a canonical frame rotated to the measured photometric "
  "PA, parity symmetry, velocity maps from roll-pair differences (D·J/(J² + ε), dividing the pair difference by "
  "the image gradient along the dispersion direction), moment-based velocity gradients, kinematic-minus-photometric "
  "PA features, a kinematics calibration, continuum subtraction, recentring, TF features, a PSF-corrected imaging "
  "fit, EMA weights, per-component checkpoint selection and a shrinkage calibration.")
P("<b>Assessment.</b> This preprocessing was effectively substituting for training data. The hand-built "
  "kinematic estimators (especially the PA misalignment) gave the network the g<sub>×</sub> observable directly, "
  "which a CNN cannot discover from 3,014 galaxies (Section 6). Gains from successive refinements were small "
  "(0.30 → 0.37 in g<sub>+</sub>) because the model was approaching what the real training set alone can teach. "
  "g<sub>+</sub> was also closer to its physical ceiling than was apparent. The pilot-v2 sets were harder still: "
  "the shear prior was narrower (σ = 0.03, comparable to or below the per-galaxy noise) and the selection used a "
  "crude catalogue SNR that scatters ×0.35-2.4 around the realized value. Their near-zero skill was largely a "
  "property of the data, not of the methods.")
story.append(PageBreak())

# =====================================================================================
# 6 This work
# =====================================================================================
H1("6. This work: from diagnosis to a simulation-trained model")
H2("6.1 Plain CNN baselines")
P("A ResNet on the six stamps (no canonicalisation) did not learn at all in 11 epochs, even when trained and "
  "tested on <i>noiseless</i> stamps (r ≈ 0). The shear signal is too subtle and too orientation-dependent to "
  "find from scratch on this data volume.")
H2("6.2 Canonical CNN on real data (E1): memorisation")
P("Rotating all stamps to the measured photometric PA (grism stamp j by −(PA − roll<sub>j</sub>)) let the network "
  "start learning g<sub>+</sub> (peak r = 0.19), but g<sub>×</sub> stayed at zero. After 30 epochs the training MSE "
  "had fallen to 0.60 while the validation MSE rose to 1.07, although every example received a fresh noise "
  "realisation each epoch. The network memorised the 3,014 clean galaxies. This confirmed that data volume, "
  "not architecture, was the binding constraint.")
H2("6.3 Rebuilding the forward model")
P("Unlimited training data require a simulator that reproduces the mocks. <font name='DVM'>forward.py</font> was "
  "reverse-engineered from the metadata and validated by χ² against the noiseless stamps at the true parameters "
  "(χ² summed over a 32×32 stamp with the stored variance; pure noise gives about 1024).")
table([
    ["Component", "Convention established", "How it was found"],
    ["Imaging", "GalSim InclinedExponential (sech² thickness h/r), rotate θ<sub>int</sub>, shear (g<sub>1</sub>, g<sub>2</sub>), shift; Roman WFI PSF SCA 10, pupil_bin 4",
     "Matched on the first attempt: χ² median 0.65"],
    ["Grism geometry", "stamp j = sky rotated by <b>−roll<sub>j</sub></b>; dispersion +x, 1.1 nm/px, centred on Hα (656.461 nm) at the catalogue z",
     "Grid search over 8 sign conventions (χ² 50-170 vs 300-3000)"],
    ["Kinematics", "thin-disk arctan rotation curve, sin i projection; <b>velocity field centred on the stamp origin</b>, not on the Hα centroid",
     "Per-row fits showed roll-independent line shifts (a net Doppler shift): median χ² 359 → 42"],
    ["Continuum", "flat f<sub>λ</sub> over <b>±20 px</b> (±22 nm), amplitude ×0.95", "Edge profiles of bright-continuum rows: mean χ² 246 → 75"],
    ["Line", "Gaussian profile with the true dispersion, amplitude ×0.93", "Face-on rows (no velocity signal)"],
    ["Ruled out", "3-D line-of-sight integration through the thick disk; other rotation-curve radii; extra PSF blur; lower oversampling", "χ² did not improve"],
], [1.05 * inch, 3.35 * inch, 2.5 * inch])
fig(FIG / "forward_fixes.png", 4.6 * inch, "Figure 5. Grism mismatch at the true parameters (40 random rows) as the "
    "conventions were fixed. The final model sits about two orders of magnitude below the noise level.")
fig(FIGS2 / "forward_validation.png", W, "Figure 6. Final forward model on 200 random rows: χ² per stamp for imaging "
    "(median 0.65) and grism (median 4.8, mean 14.6, 95th percentile 59), with example residual maps in units of "
    "the noise σ (colour range ±1).")
H2("6.4 Simulation generator")
P("<font name='DVM'>gen_sims.py</font> takes the physical parameters of a random <i>training-split</i> row (sizes, "
  "fluxes, inclination, redshift, dispersion, thickness, gains) and redraws everything that is independent of the "
  "selection: θ<sub>int</sub> ~ U(0, 2π), (g<sub>+</sub>, g<sub>×</sub>) ~ U(−0.1, 0.1)², v<sub>circ</sub> = "
  "μ<sub>TF</sub>·10<sup>N(0, 0.049)</sup>, per-component centroids ~ N(0, 0.11″) and the sign of v<sub>0</sub>. "
  "Noise is added during training. Each galaxy is thus seen in thousands of configurations and its shear cannot "
  "be memorised. Throughput is about 45 galaxies/s on 4 CPU cores, using a precomputed table of Roman PSF kernels "
  "(5 nm steps).")
H2("6.5 Network (train2.py)")
bullets([
    "<b>Canonical frame:</b> the photometric PA comes from windowed moments of F129 + F158. All stamps are rotated "
    "so the major axis lies along +x, and the dispersion direction in that frame is an input.",
    "<b>12 input channels:</b> canonical F129/F158, four canonical continuum-subtracted grism stamps, four raw grism "
    "stamps, and two opposite-roll difference maps (0° − 177°, 10° − 167°). In the difference maps the continuum "
    "cancels and the rotation dipole remains.",
    "<b>Scalars:</b> z, μ<sub>TF</sub>, observed mass, line and band fluxes (observable catalogue values only), plus "
    "10 moment-kinematic statistics (<font name='DVM'>kinfeat.py</font>). These come from solving how each roll's "
    "second moments shear along its dispersion direction, which gives the velocity-gradient vector and its angle "
    "to the photometric axis.",
    "<b>Architecture:</b> 6-block ResNet (32-64-128 channels), 3×3 spatial pooling, MLP (1.2 M parameters). Heads "
    "give the mean (g<sub>+</sub>, g<sub>×</sub>), log σ, and 7 auxiliary truths (cos i, spin-2 PA, v sin i, sky "
    "g<sub>1</sub>/g<sub>2</sub>, h/r).",
    "<b>Loss:</b> MSE on the mean (so the mean is the posterior mean under the training prior); Gaussian NLL for σ on "
    "a detached mean, so the uncertainty head cannot bias the estimate; auxiliary smooth-L1 with weight 0.3. "
    "AdamW, one-cycle learning rate, EMA weights, best checkpoint by validation MSE.",
    "<b>Data mix:</b> 80-85% simulations and 15-20% real training rows per batch, each with a fresh noise draw.",
])
H2("6.6 Experiment log")
table([
    ["Run", "Training data / change", "Val r (g<sub>+</sub> / g<sub>×</sub>)", "Test r", "Lesson"],
    ["E1", "real rows only, canonical", "0.19 / 0.0", "", "memorises; g<sub>×</sub> not learned"],
    ["E2", "+ 60k sims (initial forward model)", "0.24 / 0.0 (stopped)", "", "sims help g<sub>+</sub>; g<sub>×</sub> needs kinematic inputs"],
    ["GBT", "kinematic features only", "", "0.21 / 0.27", "features carry g<sub>×</sub> signal"],
    ["E3", "E2 + kinematic channels and scalars", "0.35 / 0.49", "0.385 / 0.510", "on held-out <i>sims</i> r(g<sub>×</sub>) = 0.60: sim-to-real gap"],
    ["fix", "forward model corrected (Section 6.3)", "", "", "held-out sims and real now agree"],
    ["E4", "60k corrected sims, init E3", "0.373 / 0.544", "0.405 / 0.556", "gap closed; small g<sub>+</sub> offset"],
    ["E5", "120k corrected sims, 20% real", "0.376 / 0.560", "0.416 / 0.580", "offset gone"],
    ["E6", "240k corrected sims, init E5", "0.381 / 0.561", "", "gains flattening (val MSE −0.4%)"],
    ["<b>E5+E6</b>", "<b>ensemble (released)</b>", "<b>0.381 / 0.564</b>", "<b>0.420 / 0.590</b>", "best on validation"],
], [0.55 * inch, 1.9 * inch, 1.25 * inch, 1.0 * inch, 2.2 * inch], bold_rows=(9,))
P("The most informative diagnostic was evaluating on <b>held-out simulations of validation galaxies</b>. When the "
  "network does much better on sims than on real data, the simulator is wrong. When the two agree, the simulator "
  "is faithful and more data or capacity is the lever. After the forward-model fix the two agreed to 0.02.")
story.append(PageBreak())

# =====================================================================================
# 7 Results
# =====================================================================================
H1("7. Main results")
P("Released model: ensemble of E5 and E6, evaluated on the 2,073 test rows (galaxies never used in training or "
  "for simulation parameters).")
table([
    ["Metric", "g<sub>+</sub>", "g<sub>×</sub>"],
    ["Pearson r (raw posterior mean)", "0.420", "0.590"],
    ["Raw response 1 + m (posterior-mean shrinkage)", "0.19", "0.34"],
    ["σ<sub>eff</sub>, unweighted, response-corrected", "0.127", "0.079"],
    ["<b>σ<sub>eff</sub>, de-shrunk, inverse-variance weighted, calibrated</b>", "<b>0.097</b>", "<b>0.044</b>"],
    ["Multiplicative bias after validation calibration", "+0.014 ± 0.036", "+0.025 ± 0.016"],
    ["Additive bias after calibration", "−0.0019 ± 0.0023", "+0.0003 ± 0.0010"],
    ["Pull width (predicted σ vs actual error)", "0.96-1.04", "0.91-0.98"],
], [3.9 * inch, 1.5 * inch, 1.5 * inch], bold_rows=(4,))
table([
    ["cos i", "rows", "r(g<sub>+</sub>)", "σ<sub>eff</sub>(g<sub>+</sub>)", "g<sub>+</sub> ceiling", "r(g<sub>×</sub>)", "σ<sub>eff</sub>(g<sub>×</sub>)"],
    ["0.05-0.30 (edge-on)", "753", "0.33", "0.170", "0.34", "0.31", "0.181"],
    ["0.30-0.60", "678", "0.29", "0.191", "0.45", "0.63", "0.071"],
    ["0.60-0.95 (face-on)", "642", "0.66", "0.067", "0.85", "0.79", "0.044"],
], [1.55 * inch, 0.6 * inch, 0.75 * inch, 0.9 * inch, 0.9 * inch, 0.75 * inch, 0.9 * inch])
fig(FIGS2 / "summary_test.png", W, "Figure 7. <b>Left:</b> test correlation vs inclination with the perfect-data "
    "ceilings (dashed). <b>Centre:</b> inverse-variance-weighted ensemble shear in bins of true shear, after "
    "de-shrinking and validation calibration; both components follow the 1:1 line. <b>Right:</b> comparison with "
    "the previous best.")
fig(FIGS2 / "eval_test_ensemble.png", W, "Figure 8. Raw posterior-mean predictions vs truth (the response is linear "
    "across the full ±0.1 range, with shrinkage factor 0.19 / 0.34), correlation vs inclination, and the pull "
    "distribution of the predicted σ (std ≈ 1).")
P("<b>Interpretation.</b> Edge-on g<sub>+</sub> sits at its physical ceiling. Overall g<sub>+</sub> reaches about "
  "75% of the perfect-data correlation. The remaining g<sub>+</sub> headroom is in mid-inclination and face-on "
  "disks, and the g<sub>×</sub> headroom is in mid-inclination disks. With σ<sub>eff</sub>(g<sub>×</sub>) = 0.044, "
  "one galaxy carries as much g<sub>×</sub> information as about 35 galaxies in conventional shape-based lensing "
  "(0.26² / 0.044²); for g<sub>+</sub> the factor is about 7.")
story.append(PageBreak())

# =====================================================================================
# 8 Usage
# =====================================================================================
H1("8. Using the model in a shear pipeline")
P("The network outputs the <b>posterior mean</b> under the training prior (uniform |g| < 0.1) and a posterior σ. "
  "Posterior means are shrunk toward zero where the data are weak, so <b>averaging them does not give an unbiased "
  "shear</b>. For ensemble estimates, de-shrink each galaxy with the Gaussian approximation:")
EQ("R<sub>i</sub> = 1 − σ<sub>i</sub>² / σ<sub>prior</sub>²,&nbsp;&nbsp; ĝ<sub>i</sub> = (μ<sub>i</sub>/R<sub>i</sub> − c) / (1 + m),"
   "&nbsp;&nbsp; σ<sub>ĝ,i</sub> = σ<sub>i</sub> / (√R<sub>i</sub> (1 + m)),&nbsp;&nbsp; w<sub>i</sub> = 1/σ<sub>ĝ,i</sub>²")
P("Here σ<sub>prior</sub>² = 0.1²/3, and (m, c) per component come from the validation calibration "
  "(<font name='DVM'>models/shearnet_ensemble_calibration.json</font>). The weighted mean Σw<sub>i</sub>ĝ<sub>i</sub>/Σw<sub>i</sub> "
  "is unbiased within the errors quoted in Section 7. Galaxies with σ<sub>i</sub> ≈ σ<sub>prior</sub> get "
  "R<sub>i</sub> → 0, very large σ<sub>ĝ</sub> and negligible weight, as they should. <font name='DVM'>predict.py</font> "
  "writes all of these columns:")
P("python predict.py --ckpt models/shearnet_E5.pt models/shearnet_E6.pt \\<br/>"
  "&nbsp;&nbsp;&nbsp;&nbsp;--calib models/shearnet_ensemble_calibration.json --parquet galaxies.parquet --out shear.csv<br/>"
  "# columns: g_plus, g_cross, sigma_g_plus, sigma_g_cross (posterior)<br/>"
  "#&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ghat_plus, ghat_cross, sigma_ghat_plus, sigma_ghat_cross (de-shrunk, calibrated)", code)
P("The input parquet needs the six noisy <font name='DVM'>img.*</font> stamps, the <font name='DVM'>sigma_bkg.*</font> "
  "columns and the observable catalogue columns <font name='DVM'>fixed.z</font>, <font name='DVM'>pop.prior_vcirc_mu_kms</font>, "
  "<font name='DVM'>pop.logm_obs</font>, <font name='DVM'>pop.f_line_obs_cgs</font> and "
  "<font name='DVM'>pop.flux_obs_f129/f158_ujy</font>, in the v3 layout. CPU inference is about 1 ms per galaxy.")

# =====================================================================================
# 9 Caveats
# =====================================================================================
H1("9. Caveats and limitations")
bullets([
    "<b>Domain.</b> The model is trained for the v3 ×10 grism depth, the v3 selection, |g| < 0.1 uniform, the four "
    "rolls 0/10/167/177°, SCA 10 and the kl_roman_pipe source model (single exponential disks, arctan rotation "
    "curve, no bulges, neighbours or detector effects). Survey depth (×1), other priors or other roll sets need new "
    "simulations and retraining, which <font name='DVM'>gen_sims.py</font> supports.",
    "<b>Simulator dependence.</b> The network inherits the assumptions of <font name='DVM'>forward.py</font>, which "
    "deliberately match the mock renderer, including two empirical amplitude factors (line ×0.93, continuum ×0.95). "
    "Real Roman data will differ from both, so the 20% real-data fraction and the validation calibration only "
    "protect against differences between the mocks and the simulator.",
    "<b>Prior dependence.</b> The posterior mean and the calibration assume the uniform ±0.1 training prior. The "
    "de-shrinking formula is a Gaussian approximation; it is accurate here (response 0.92-1.01 before calibration) "
    "but should be re-checked for other priors.",
    "<b>Statistics.</b> Each test row has one noise realisation; errors on m are ±0.02-0.04. Testing the "
    "sub-percent bias required by Roman needs larger dedicated tests (Section 10).",
    "<b>Comparisons</b> with the earlier runs use different random splits (±0.02 in r). Their σ<sub>eff</sub> values "
    "are estimated from r, because those models give no per-galaxy uncertainty.",
    "<b>Oracle ceiling</b> is a thin-disk, small-shear idealisation evaluated with tree models. It is an "
    "approximate upper bound (±0.03), not a rigorous Cramér-Rao bound.",
    "<b>Compute.</b> Everything was run on 4 CPU cores without a GPU. Network size and training length were "
    "therefore modest.",
])
story.append(PageBreak())

# =====================================================================================
# 10 Next steps
# =====================================================================================
H1("10. Next steps")
P("In rough order of expected value:")
table([
    ["#", "Step", "Why / expected gain", "Effort"],
    ["1", "<b>Likelihood benchmark with forward.py</b>: port the renderer to torch/JAX (differentiable, batched) and "
     "fit each test galaxy (MAP + Laplace or MCMC), or compute Fisher matrices",
     "Gives the true per-galaxy bound on this data, showing how close the network is to optimal and where headroom remains. "
     "Could itself be the pipeline estimator.", "High"],
    ["2", "<b>GPU training at scale</b>: wider/deeper networks, 1-10 M simulations generated on the fly, longer schedules, larger ensembles",
     "Training loss was still above the noise floor and sims still helped. Expect gains mainly in g<sub>×</sub> at mid inclination and g<sub>+</sub> face-on.",
     "Medium"],
    ["3", "<b>Survey-depth (×1) and ×5 sets</b>: the x1/x5/x10 sets share galaxies and shears (match on db_row_id). "
     "Regenerate sims at those depths, or train depth-conditioned with the noise level as input",
     "The ×10 set isolates kinematic information; ×1 is the forecast-relevant case.", "Low-medium"],
    ["4", "<b>Constant-shear ensemble tests</b>: simulate 10<sup>4</sup>-10<sup>5</sup> galaxies with a common shear, "
     "measure m and c to < 0.5%", "Required before pipeline use; current tests resolve m to only ±2-4%.", "Low"],
    ["5", "<b>Full posteriors</b> (neural posterior estimation / normalising flow over (g<sub>+</sub>, g<sub>×</sub>)) "
     "with a broad training prior", "Removes the Gaussian de-shrinking approximation and prior dependence; gives "
     "non-Gaussian tails and the g<sub>+</sub>-g<sub>×</sub> correlation.", "Medium"],
    ["6", "<b>Robustness to model misspecification</b>: bulges, clumps, non-arctan rotation curves, PSF and SCA "
     "variation, neighbours, wavelength-dependent continuum", "Real galaxies are not single exponential disks; "
     "estimate the systematic shear bias each one induces.", "Medium"],
    ["7", "<b>Upstream support</b>: obtain the kl_roman_pipe renderer itself for simulation (instead of the "
     "reverse-engineered copy), or export its forward model", "Removes the last 1-2% mismatch and makes the "
     "simulator automatically track pipeline changes.", "Low"],
    ["8", "<b>Use the TF/thickness structure explicitly</b>: per-galaxy information is predictable from cos i "
     "and size; use it for selection and weighting in survey forecasts", "Most g<sub>+</sub> information comes from "
     "cos i > 0.6, which affects target selection.", "Low"],
], [0.25 * inch, 2.6 * inch, 3.25 * inch, 0.8 * inch])

# =====================================================================================
# Appendix
# =====================================================================================
H1("Appendix A. Files and reproduction")
table([
    ["File (shearnet/)", "Purpose"],
    ["prep_data.py", "parquet → noisy/clean/var .npy arrays and scalar table"],
    ["oracle_ceiling.py", "perfect-observable information ceiling (Section 4.2)"],
    ["forward.py, validate_forward.py, forward_validation_figure.py", "forward model and its χ² validation"],
    ["make_psf_table.py", "precomputed Roman PSF kernels (models/psf_os5.npz)"],
    ["gen_sims.py", "simulation sets (--split train for training, val/test for diagnostics)"],
    ["canon.py, kinfeat.py", "canonical frame, kinematic channels and moment statistics"],
    ["train2.py", "training (sims + real, fresh noise, EMA, σ head, auxiliary targets)"],
    ["predict_split.py, calibrate.py, evaluate.py, eval_sims.py", "val/test predictions, calibration, metrics, sim-domain checks"],
    ["predict.py", "inference / ensembling on new data"],
    ["models/", "shearnet_E5.pt, shearnet_E6.pt, shearnet_ensemble_calibration.json, psf_os5.npz"],
    ["docs/", "this report and the scripts that build it"],
], [2.9 * inch, 4.0 * inch])
P("python prep_data.py --parquet kl_roman_nn_v3_lsnr10x.parquet --outdir DATA<br/>"
  "python make_psf_table.py models/psf_os5.npz<br/>"
  "python gen_sims.py --data DATA --out SIMS_A --n 120000 --seed 11&nbsp;&nbsp;&nbsp;# ~45 sims/s on 4 cores<br/>"
  "python gen_sims.py --data DATA --out SIMS_B --n 120000 --seed 13<br/>"
  "python train2.py --data DATA --sims SIMS_A SIMS_B --real_frac 0.2 --kin --epochs 40 --steps_per_epoch 500 --out RUN<br/>"
  "python predict_split.py --ckpt RUN/model.pt --out RUN &amp;&amp; python calibrate.py RUN &amp;&amp; python evaluate.py RUN", code)
P("The released models were trained in stages: E3 (30 epochs on initial-model sims) → E4 (20 epochs, 60k corrected "
  "sims) → E5 (16 epochs, 120k) → E6 (20 epochs, 240k). A single run on all the corrected sims, as above, is the "
  "simplest equivalent. Each 500-step epoch takes about 100 s on 4 CPU cores.")
H1("Appendix B. Metric definitions")
bullets([
    "<b>r</b>: Pearson correlation between predicted posterior mean and true shear over the test rows.",
    "<b>Response 1 + m, additive c</b>: least-squares fit prediction = (1 + m) g<sub>true</sub> + c, weighted by "
    "1/σ<sub>ĝ</sub>² for the de-shrunk estimator.",
    "<b>σ<sub>eff</sub></b>: with residuals ε<sub>i</sub> = ĝ<sub>i</sub> − (1 + m)g<sub>i</sub> − c and weights "
    "w<sub>i</sub>, σ<sub>eff</sub> = √(N Σw<sub>i</sub>²ε<sub>i</sub>² / (Σw<sub>i</sub>)²) / (1 + m). This is the "
    "per-galaxy noise implied by the actual scatter of the weighted ensemble mean. For the unweighted posterior mean, "
    "σ<sub>eff</sub> = std(ε) / (1 + m) ≈ σ<sub>g</sub> √(1 − r²) / r.",
    "<b>Pull</b>: (prediction − truth) / predicted σ. Its standard deviation should be 1 for calibrated uncertainties.",
    "<b>χ² per stamp</b>: Σ<sub>pixels</sub> (model − noiseless)² / variance at the true parameters; pure noise "
    "would give about 1024 for a 32×32 stamp.",
])


def on_page(c, d):
    if d.page == 1:
        return
    c.saveState()
    c.setFont("DV", 7.5)
    c.setFillColor(colors.HexColor("#666"))
    c.drawString(0.8 * inch, 0.5 * inch, "Kinematic-lensing shear inference — shearnet report")
    c.drawRightString(letter[0] - 0.8 * inch, 0.5 * inch, f"{d.page}")
    c.setStrokeColor(colors.HexColor("#c8ccd2"))
    c.line(0.8 * inch, 0.65 * inch, letter[0] - 0.8 * inch, 0.65 * inch)
    c.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                        topMargin=0.75 * inch, bottomMargin=0.85 * inch,
                        title="Kinematic-lensing shear inference: shearnet report",
                        author="shearnet", subject="Roman KL shear regression")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("wrote", OUT)
