# Roman kinematic-lensing mock stamps, v3: Halpha line depth x10

Purpose: train a network to predict the shear of a disk galaxy from its Roman
imaging and grism stamps. Inputs are six 32 x 32 images per galaxy; the targets
are the two galaxy-frame shear components `truth.g_plus` and `truth.g_cross`.

## What changed from the v2 pilot

- **Selection on realized signal-to-noise.** Every row has a realized Halpha
  line signal-to-noise (all four grism exposures combined) of at least
  100: at least 10 at the real survey depth, times the depth boost
  of 10. In v2 the cut was made on a cruder catalog estimate, which scatters
  by a factor 0.35-2.4 around the realized value.
- **Realized signal-to-noise columns only.** `snr.*` are measured on the
  stored stamps (noiseless stamp against the stored variance map). The catalog
  estimates (`line_snr`, `broadband_snr_*`) are gone; in v2 they were the
  baseline values even in the boosted sets.
- **Wider shear.** `truth.g_plus` and `truth.g_cross` are uniform on
  [-0.1, 0.1] independently (v2: Gaussian of width 0.03).
- **Updated survey depth anchor.** The grism depth now refers to the 0.3 arcsec
  reference source of the Roman Observations Time Allocation Committee report
  (v2: 0.25 arcsec), so the same galaxy has a line signal-to-noise about 1.15x
  higher than in v2.
- **Independent noise per depth.** The x1, x5 and x10 sets hold the same
  galaxies, orientations and shears (match rows on `db_row_id`), with
  independent noise in every channel.

## Files

- `kl_roman_nn_v3_lsnr10x.parquet`: 13481 rows, one per simulated galaxy.
- `meta_kl_roman_nn_v3_lsnr10x.json`: units, image shapes, and the settings shared by every row.
- `SCHEMA.md`: every column with a one-line description.
- `format_sample_100/`: a 100-row subset with the same layout.
- `diagnostics/`: truth distributions, noise checks and example rows (noisy /
  noiseless / residual per channel) for realization 1.
- `PROVENANCE.md`: code version and settings.

## What one row is

A galaxy from the COSMOS2025 catalog with a painted Halpha line, placed at a
random inclination, position angle and shear, and observed as Roman's High
Latitude Wide Area Survey would see it: two imaging bands (F129, F158) at the
survey depth and four grism exposures at roll angles 0, 10, 167 and 177
degrees, with the grism noise 10x below the survey's (line signal-to-noise
x10, equivalent to 100x the exposure time per roll). Redshift 0.55 to 1.9.

Realized signal-to-noise (5th / 50th / 95th percentile):

- Halpha line, all rolls combined (`snr.line_total`): 103 / 150 / 379 (minimum 100.0)
- F158 imaging (`snr.F158`): 19 / 62 / 203

The imaging is not boosted, so the boost isolates the kinematic information;
this is not a forecast of a real deeper survey.

13481 rows hold 3014 distinct catalog galaxies (about 4.5 rows each).
Each appearance has a fresh orientation, shear, rotation speed and noise, and
0.1 dex changes to its mass, size and fluxes (redshift +/- 0.05). Split
training and validation by `pop.id` so validation galaxies are unseen.

## Inputs

Each of the 18 image columns is a flat list of 1024 float32 values, row-major;
reshape to (32, 32). Pixel scale 0.11 arcsec.

| column | content | units |
|---|---|---|
| `img.bb_F129`, `img.bb_F158` | noisy imaging stamps | uJy per pixel |
| `img.gr_roll0` .. `img.gr_roll3` | noisy grism stamps (line + continuum) | 1e-17 erg/s/cm2 per pixel |
| `img.<channel>_var` | per-pixel noise variance | squared stamp units |
| `img.<channel>_clean` | the same stamp without noise | stamp units |

Imaging stamps are in sky orientation. Grism stamp j is the sky rotated by
`roll_deg_j`; the spectrum is dispersed along the second array axis (columns).

Noise is Gaussian with a flat background per channel plus the source's own
photon noise: `var = sigma_bkg**2 + max(clean, 0) / gain`. The background
`sigma_bkg.<channel>` is the same for every row, so the noise floor carries no
information about the galaxy. The photon-noise term follows the source, as in
real data.

## Targets and other truths

- `truth.g_plus`, `truth.g_cross`: shear along and at 45 degrees to the
  galaxy's major axis, uniform on [-0.1, 0.1] independently.
  `truth.g1`, `truth.g2` are the same shear in sky coordinates.
- `truth.cosi`: cosine of inclination (0 edge-on, 1 face-on).
  `truth.theta_int`: position angle, radians from the +x axis, [0, 2 pi).
- `truth.z`, `truth.vel.vcirc` (km/s), `truth.vel.v0` (km/s), sizes in arcsec
  (`*.rscale`), fluxes, disk thickness `truth.h_over_r`. Some `truth.*`
  columns are copies of a shared value.
- `pop.*`: catalog properties after the jitter (mass, line flux, equivalent
  width, ...).

The selection keeps galaxies whose line is bright and compact enough, so the
truth distributions (inclination, size, rotation speed) are those of the
selected sample, not of the catalog. The shear does not enter the selection
(checked: no correlation between shear and realized signal-to-noise).

## Loading

```python
import json, numpy as np, pandas as pd

d = pd.read_parquet('kl_roman_nn_v3_lsnr10x.parquet')
meta = json.load(open('meta_kl_roman_nn_v3_lsnr10x.json'))
channels = ['bb_F129', 'bb_F158', 'gr_roll0', 'gr_roll1', 'gr_roll2', 'gr_roll3']

def stack(key):  # (n_rows, 32, 32) float32
    return np.stack(d[f'img.{key}'].to_numpy()).reshape(len(d), *meta['image_shapes'][key])

x = np.stack([stack(c) for c in channels], axis=1)            # (n, 6, 32, 32) noisy stamps
var = np.stack([stack(f'{c}_var') for c in channels], axis=1)  # matching variance maps
y = d[['truth.g_plus', 'truth.g_cross']].to_numpy()            # (n, 2) targets
```

Normalize per channel (for example divide by `sigma_bkg.<channel>`); the
imaging and grism units differ.

## What to expect

Even at x10 the per-galaxy shear is weakly constrained. Likelihood fits of
these mocks give a per-galaxy shear uncertainty of roughly 0.04-0.12 at x10
(depending on inclination, with an unknown disk thickness), comparable to the
target range. A network should therefore be judged on calibration over many
galaxies (bias and slope of predicted vs true shear) rather than on per-galaxy
R^2.

## Known limitations

- Stamps are 32 pixels (3.5 arcsec); a small fraction of large galaxies spill
  past the edge.
- The forward model is our own pipeline, not an image simulator: single
  exponential disks, no bulges, no neighbors, no detector effects beyond
  pixelization and the Roman PSF.
