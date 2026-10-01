# Provenance

Built 2026-09-30 with kl_roman_pipe (KinematicLensing/kl_roman_pipe, branch
se/gpu-speedups, commit 163696f) by `experiments/sweverett/nn_database/build_db_v2.py`:

    python build_db_v2.py --seeds 1 2 --out pilot_v2 --workers 8

Settings: observation config `hlwas_medium_roman_rotac_rolls_poisson`
(Roman WFI PSF, SCA 10; 0.11 arcsec pixels; grism 1.1 nm/pixel; rolls
0/10/167/177 deg; physical background + shot noise), render oversample 5,
COSMOS2025 catalog with painted Halpha (private join, 2026-07-28 delivery),
selection z in [0.55, 1.9], labeled total line SNR >= 5, disk r50 >= PSF FWHM.
Base ensemble spec: `configs/ensembles/cosmos25_census_v2_hfree_xu23_rolls.yaml`
with the DB overrides recorded in `meta_pilot_v2_all.json`.

Contact: Spencer Everett (severett@caltech.edu).
