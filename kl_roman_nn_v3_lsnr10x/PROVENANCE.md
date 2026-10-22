# Provenance

Built with kl_roman_pipe (branch se/gpu-speedups) by `experiments/sweverett/nn_database/build_db_v3.py` (code at commit da14c56), collated by `package_db_v3.py`.

Settings: observation config `hlwas_medium_roman_rotac_rolls_poisson` (Roman WFI PSF, SCA 10; 0.11 arcsec pixels; grism 1.1 nm/pixel; rolls [0.0, 10.0, 167.0, 177.0] deg; background + photon noise); render oversample 5; COSMOS2025 catalog with painted Halpha (2026-07-28 delivery); z in [0.55, 1.9]; disk r50 >= PSF FWHM; realized baseline total line SNR >= 10 (catalog-estimate prefilter >= 3.5); grism depth x10 (sigma_bkg / 10, gain x 100); shear uniform +/-0.1 per galaxy-frame component; realizations (seeds) [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].

Gates passed at packaging: {"n": 13481, "snr_line": [100.00369340714094, 103.35678621912626, 149.56620116897136, 378.7114202373203], "snr_F158": [18.663560878506715, 62.239490052931664, 202.85895220685612], "pull_bb_F129": [0.0002209652115095588, 0.9995938869004638], "sigma_bkg_bb_F129": 0.006845956541060461, "pull_bb_F158": [-0.0005001308519628267, 0.9999682258898065], "sigma_bkg_bb_F158": 0.006059427304940354, "pull_gr_roll0": [-0.00019964169162569519, 0.9999631392690584], "sigma_bkg_gr_roll0": 0.0558859433694698, "pull_gr_roll1": [-0.00017023949333934036, 1.0002449971087595], "sigma_bkg_gr_roll1": 0.0558859433694698, "pull_gr_roll2": [-0.0006597384436997047, 0.9997022516674157], "sigma_bkg_gr_roll2": 0.0558859433694698, "pull_gr_roll3": [7.541705306960755e-05, 0.9999159622932254], "sigma_bkg_gr_roll3": 0.0558859433694698, "n_galaxies": 3014}

Contact: Spencer Everett (severett@caltech.edu).
