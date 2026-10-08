# Provenance

Built 2026-10-02 with kl_roman_pipe (KinematicLensing/kl_roman_pipe, branch
se/gpu-speedups) by `experiments/sweverett/nn_database/build_db_v2.py`:

    python build_db_v2.py --seeds 1 2 --out pilot_v2_lsnr10x --workers 8 --line-depth-scale 10

Realization 1 ran on the working tree at HEAD 141af32 (with uncommitted
ensemble-spec edits that were committed later the same day); realization 2
ran at commit a453447.
The baseline pilot was built at commit 163696f. The commits in between change the
ensemble spec, fitting and sampler code, not the mock rendering, noise or catalog
path used here. Check: for every row in both realizations the imaging stamps,
imaging variance maps, noiseless stamps and every truth, prior and catalog column
are bit-identical to the baseline pilot; the grism variance maps are exactly 100x
smaller; the realized line signal-to-noise is exactly 10x the baseline; and the
grism noise equals the baseline noise divided by 10 to float32 rounding.

Settings: identical to the baseline pilot `kl_roman_nn_pilot_v2` (observation
config `hlwas_medium_roman_rotac_rolls_poisson`: Roman WFI PSF, SCA 10; 0.11
arcsec pixels; grism 1.1 nm/pixel; rolls 0/10/167/177 deg; physical background
+ shot noise; render oversample 5; COSMOS2025 catalog with painted Halpha,
private join, 2026-07-28 delivery; selection z in [0.55, 1.9], labeled total line
SNR >= 5 at the survey depth, disk r50 >= PSF FWHM), except that the grism
background sigma is divided by 10 and the grism gain multiplied by 100
(`--line-depth-scale 10`, recorded as `line_depth_scale` in the meta and
`depth_snr_scale.line` per row).

Contact: Spencer Everett (severett@caltech.edu).
