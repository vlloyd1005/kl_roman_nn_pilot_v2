# shearnet: simulation-trained shear regression for the Roman KL mocks (v3, line SNR x10)

Predicts the galaxy-frame shear `truth.g_plus`, `truth.g_cross` **and a calibrated
per-galaxy uncertainty** from the six Roman stamps (F129, F158, 4 grism rolls) of
`kl_roman_nn_v3_lsnr10x`.

The design differs from the earlier `train_kl_grism*.py` runs in one main way:
**most of the training data are simulations from a forward model that was rebuilt
to reproduce the mock renderer** (validated pixel by pixel against the noiseless
stamps). The real training rows are mixed in at 15% of each batch.

RESULTS_PLACEHOLDER

## Why the earlier models stalled (diagnosis)

1. **g+ has a hard physical ceiling.** For a thin disk, a shear along the major
   axis is exactly degenerate with inclination in both the image shape and the
   velocity-field shape. Only the Tully-Fisher prior on v_circ (0.049 dex
   scatter) and the disk thickness break it. With *perfect, noiseless*
   observables (`oracle_ceiling.py`) the best possible correlation is
   r(g+) = 0.56 overall: 0.85 for cos i > 0.6, 0.45 for 0.3-0.6, and 0.34 for
   edge-on. No network can beat this on the v3 data.
   g_x has no such degeneracy (oracle r = 1.0). It is limited by noise and resolution.
2. **Data volume, not architecture, limited the networks.** The 13,481 rows hold
   only 3,014 distinct galaxies. A CNN trained on the real rows memorises them
   (train MSE 0.60 vs val 1.07 after 30 epochs, even with fresh noise every
   epoch) and never learns g_x. The hand-built kinematic preprocessing in
   `train_kl_grism_v3.py` was compensating for this.
3. **The fix is unlimited simulated galaxies.** `forward.py` re-implements the
   mock renderer. Each simulation reuses the physical parameters of a training
   galaxy but redraws orientation, shear, centroids, rotation speed (around the
   TF prior) and noise. So the network sees each galaxy in many configurations
   and cannot memorise the shear.

## Forward model (`forward.py`), validated against the noiseless stamps

Reverse-engineered conventions, each confirmed by chi^2 against `img.*_clean`
(chi^2 per stamp summed over 32x32 pixels with the stored variance; pure noise gives about 1024):

| component | convention | chi^2 per stamp |
|---|---|---|
| imaging | GalSim `InclinedExponential` (sech^2 thickness h/r), rotated by theta_int, sheared by (g1, g2), shifted, Roman WFI PSF SCA 10 (pupil_bin 4, band effective wavelength) | 0.1-15 |
| grism geometry | stamp j = sky rotated by **-roll_j**; dispersion +x, 1.1 nm/px, centred on Halpha (vacuum 656.461 nm) at z | |
| kinematics | thin disk, v_los = v0 + vcirc (2/pi) arctan(R/r_v) sin i cos(phi), velocity field **centred on the stamp origin** (not the Halpha centroid; this was the largest mismatch) | |
| line | Gaussian line profile (truth.Halpha.dispersion), amplitude x0.93 | |
| continuum | flat f_lambda over +-20 px (+-22 nm) of the line, amplitude x0.95 | |
| grism total (4 rolls) | median 13-30, mean 60-75 per 4 stamps (about 1.5% of the noise variance) | |

`validate_forward.py` reproduces these checks.

## Model (`train2.py`, `canon.py`, `kinfeat.py`)

* **Canonical frame**: the photometric position angle is measured from windowed
  moments of the imaging, and every stamp is rotated so the major axis lies
  along +x (grism stamp j by -(PA - roll_j)). The dispersion direction in that
  frame is passed as a scalar input.
* **Inputs (12 channels)**: canonical F129/F158, canonical continuum-subtracted
  grism (4), raw grism (4), and two **opposite-roll difference maps**
  (roll 0 - roll 177, roll 10 - roll 167). In these the continuum cancels and the
  rotation dipole survives.
* **Scalars**: z, Tully-Fisher prior mean, observed mass/line flux/band fluxes
  (observable catalogue values only, never truths), plus 10 moment-based
  kinematic statistics (`kinfeat.py`): the velocity-gradient vector solved from
  how each roll's second moments shear along its dispersion direction, and its
  angle to the photometric axis.
* **Network**: ResNet (6 blocks, 32-64-128 channels) with 3x3 spatial pooling,
  then an MLP. Outputs are the mean (g+, gx), log sigma, and 7 auxiliary truths
  (cos i, spin-2 PA, v sin i, sky g1/g2, h/r).
* **Loss**: MSE on the mean; Gaussian NLL for sigma on a detached mean (the
  uncertainty head cannot bias the estimate); auxiliary smooth-L1 with weight 0.3.
* **Every example gets fresh noise** each time it is drawn:
  `clean + sqrt(sigma_bkg^2 + clean/gain) * N(0,1)`.
* Validation and test use the provided noisy stamps of galaxies (split by
  `pop.id`) that appear in no training set and whose parameters are never used
  to make training simulations.

## Using the outputs in a shear pipeline

The network returns the **posterior mean** under the training prior (uniform
|g| < 0.1). It is shrunk toward 0 where the data are weak, which is why the
raw response (1+m) is 0.2-0.3. Do not average raw predictions as a shear
estimate. Use either:

* **De-shrunk per-galaxy estimate** (the one to use for an ensemble shear):
  `R = 1 - sigma^2 / sigma_prior^2` with `sigma_prior^2 = 0.1^2/3`;
  `g_hat = pred / R` with variance `sigma^2 / R`. Weight by `R / sigma^2`.
  On the test set its weighted response is about 0.9 (one global factor,
  calibrate on validation or simulations) and its pulls have std about 1.0.
* or the raw posterior mean plus a global response measured on simulations.

The per-galaxy sigma is calibrated (pull std 1.0-1.07 on test). Most of the
information comes from galaxies with cos i > 0.6, so weighting matters.

## Files

| file | purpose |
|---|---|
| `prep_data.py` | parquet -> `noisy/clean/var.npy` + scalar table |
| `oracle_ceiling.py` | information ceiling from perfect noiseless observables |
| `forward.py`, `validate_forward.py`, `make_psf_table.py` | forward model, chi^2 validation, PSF kernel table |
| `gen_sims.py` | simulated training sets (`--split train` only for training) |
| `canon.py`, `kinfeat.py` | canonicalisation, kinematic channels and statistics |
| `train2.py` | training (sims + real, fresh noise, EMA, sigma head) |
| `evaluate.py`, `eval_sims.py` | pipeline metrics and plots; sim-domain evaluation |
| `predict.py` | inference / ensembling for new data -> CSV of g+, gx, sigma |
| `common.py`, `train.py` | shared utilities; `train.py` = first plain-CNN baseline |

## Reproduce

```bash
python prep_data.py --parquet kl_roman_nn_v3_lsnr10x.parquet --outdir DATA
python make_psf_table.py DATA/../psf_os5.npz
python gen_sims.py --data DATA --out SIMS_A --n 60000 --seed 11
python train2.py --data DATA --sims SIMS_A --real_frac 0.15 --kin --epochs 30 --steps_per_epoch 400 --out RUN
python evaluate.py RUN
python predict.py --ckpt RUN/model.pt --parquet new.parquet --out pred.csv
```

Dependencies: numpy, pandas, pyarrow, torch (CPU is fine), galsim (Roman PSF),
scipy, scikit-learn, lightgbm, matplotlib.
