# sample_100.parquet schema (identical to pilot_v2_all.parquet apart from the db_seed column)

2929 rows, 121 columns. Units and frames: see meta_seed00001.json.

| column | type | description |
|---|---|---|
| fit_id | large_string | ensemble fit id (sha1 of the index tuple) |
| galaxy_id | int64 | population row index within the seed |
| noise_seed | int64 | seed of the per-channel noise streams |
| line_snr | double | labeled per-pass line SNR (population proxy) |
| broadband_snr_F129 | double | labeled per-band SNR (population proxy) |
| broadband_snr_F158 | double | labeled per-band SNR (population proxy) |
| truth.vel.v0 | double | model truth parameter (dotted SourceModel key); shared geometry has no prefix |
| truth.vel.rscale | double | model truth parameter (dotted SourceModel key); shared geometry has no prefix |
| truth.Halpha.flux | double | model truth parameter (dotted SourceModel key); shared geometry has no prefix |
| truth.Halpha.rscale | double | model truth parameter (dotted SourceModel key); shared geometry has no prefix |
| truth.Halpha.dispersion | double | model truth parameter (dotted SourceModel key); shared geometry has no prefix |
| truth.Halpha.cont.flux_per_nm | double | model truth parameter (dotted SourceModel key); shared geometry has no prefix |
| truth.Halpha.cont.rscale | double | model truth parameter (dotted SourceModel key); shared geometry has no prefix |
| truth.F129.flux | double | model truth parameter (dotted SourceModel key); shared geometry has no prefix |
| truth.F129.rscale | double | model truth parameter (dotted SourceModel key); shared geometry has no prefix |
| truth.F129.h_over_r | double | model truth parameter (dotted SourceModel key); shared geometry has no prefix |
| truth.F129.x0 | double | model truth parameter (dotted SourceModel key); shared geometry has no prefix |
| truth.F129.y0 | double | model truth parameter (dotted SourceModel key); shared geometry has no prefix |
| truth.F158.flux | double | model truth parameter (dotted SourceModel key); shared geometry has no prefix |
| truth.F158.rscale | double | model truth parameter (dotted SourceModel key); shared geometry has no prefix |
| truth.F158.h_over_r | double | model truth parameter (dotted SourceModel key); shared geometry has no prefix |
| truth.F158.x0 | double | model truth parameter (dotted SourceModel key); shared geometry has no prefix |
| truth.F158.y0 | double | model truth parameter (dotted SourceModel key); shared geometry has no prefix |
| truth.Halpha.h_over_r | double | model truth parameter (dotted SourceModel key); shared geometry has no prefix |
| truth.Halpha.x0 | double | model truth parameter (dotted SourceModel key); shared geometry has no prefix |
| truth.Halpha.y0 | double | model truth parameter (dotted SourceModel key); shared geometry has no prefix |
| truth.Halpha.cont.h_over_r | double | model truth parameter (dotted SourceModel key); shared geometry has no prefix |
| truth.Halpha.cont.x0 | double | model truth parameter (dotted SourceModel key); shared geometry has no prefix |
| truth.Halpha.cont.y0 | double | model truth parameter (dotted SourceModel key); shared geometry has no prefix |
| truth.cosi | double | model truth parameter (dotted SourceModel key); shared geometry has no prefix |
| truth.theta_int | double | model truth parameter (dotted SourceModel key); shared geometry has no prefix |
| truth.g1 | double | model truth parameter (dotted SourceModel key); shared geometry has no prefix |
| truth.g2 | double | model truth parameter (dotted SourceModel key); shared geometry has no prefix |
| truth.z | double | model truth parameter (dotted SourceModel key); shared geometry has no prefix |
| truth.vel.vcirc | double | model truth parameter (dotted SourceModel key); shared geometry has no prefix |
| truth.h_over_r | double | model truth parameter (dotted SourceModel key); shared geometry has no prefix |
| pop.id | double | catalog / population column (jittered where applicable) |
| pop.snr_line_per_pass | double | catalog / population column (jittered where applicable) |
| pop.snr_line_total | double | catalog / population column (jittered where applicable) |
| pop.ew_rest_a | double | catalog / population column (jittered where applicable) |
| pop.logm | double | catalog / population column (jittered where applicable) |
| pop.logm_obs | double | catalog / population column (jittered where applicable) |
| pop.prior_vcirc_mu_kms | double | catalog / population column (jittered where applicable) |
| pop.prior_vcirc_sigma_dex | double | catalog / population column (jittered where applicable) |
| pop.sigma0_kms | double | catalog / population column (jittered where applicable) |
| pop.f_line_cgs | double | catalog / population column (jittered where applicable) |
| pop.f_line_obs_cgs | double | catalog / population column (jittered where applicable) |
| pop.f_line_sigma_cgs | double | catalog / population column (jittered where applicable) |
| pop.flux_f106_ujy | double | catalog / population column (jittered where applicable) |
| pop.flux_obs_f106_ujy | double | catalog / population column (jittered where applicable) |
| pop.flux_sigma_f106_ujy | double | catalog / population column (jittered where applicable) |
| pop.snr_bb_f106 | double | catalog / population column (jittered where applicable) |
| pop.flux_f129_ujy | double | catalog / population column (jittered where applicable) |
| pop.flux_obs_f129_ujy | double | catalog / population column (jittered where applicable) |
| pop.flux_sigma_f129_ujy | double | catalog / population column (jittered where applicable) |
| pop.snr_bb_f129 | double | catalog / population column (jittered where applicable) |
| pop.flux_f158_ujy | double | catalog / population column (jittered where applicable) |
| pop.flux_obs_f158_ujy | double | catalog / population column (jittered where applicable) |
| pop.flux_sigma_f158_ujy | double | catalog / population column (jittered where applicable) |
| pop.snr_bb_f158 | double | catalog / population column (jittered where applicable) |
| theta_flipped | bool | True if the population theta got the +pi flip |
| sigma_bkg.bb_F129 | double | flat background sigma per pixel for the channel |
| gain.bb_F129 | double | detected electrons per stamp flux unit |
| sigma_bkg.bb_F158 | double | flat background sigma per pixel for the channel |
| gain.bb_F158 | double | detected electrons per stamp flux unit |
| sigma_bkg.gr_roll0 | double | flat background sigma per pixel for the channel |
| gain.gr_roll0 | double | detected electrons per stamp flux unit |
| sigma_bkg.gr_roll1 | double | flat background sigma per pixel for the channel |
| gain.gr_roll1 | double | detected electrons per stamp flux unit |
| sigma_bkg.gr_roll2 | double | flat background sigma per pixel for the channel |
| gain.gr_roll2 | double | detected electrons per stamp flux unit |
| sigma_bkg.gr_roll3 | double | flat background sigma per pixel for the channel |
| gain.gr_roll3 | double | detected electrons per stamp flux unit |
| snr_eff.F129 | double | realized matched-filter SNR against the variance map |
| snr_eff.F158 | double | realized matched-filter SNR against the variance map |
| snr_eff.line_roll0 | double | realized matched-filter SNR against the variance map |
| snr_eff.line_roll1 | double | realized matched-filter SNR against the variance map |
| snr_eff.line_roll2 | double | realized matched-filter SNR against the variance map |
| snr_eff.line_roll3 | double | realized matched-filter SNR against the variance map |
| snr_eff.line_total | double | realized matched-filter SNR against the variance map |
| truth.g_plus | double | galaxy-frame tangential shear (derived) |
| truth.g_cross | double | galaxy-frame cross shear (derived) |
| roll_deg_0 | double | grism roll angle of roll j [deg] |
| roll_deg_1 | double | grism roll angle of roll j [deg] |
| roll_deg_2 | double | grism roll angle of roll j [deg] |
| roll_deg_3 | double | grism roll angle of roll j [deg] |
| jitter.z_cat | double | per-galaxy catalog jitter draws (z_cat = catalog z) |
| jitter.dz | double | per-galaxy catalog jitter draws (z_cat = catalog z) |
| jitter.dlogm | double | per-galaxy catalog jitter draws (z_cat = catalog z) |
| jitter.dlog_f_line | double | per-galaxy catalog jitter draws (z_cat = catalog z) |
| jitter.dlog_ew | double | per-galaxy catalog jitter draws (z_cat = catalog z) |
| jitter.dlog_size | double | per-galaxy catalog jitter draws (z_cat = catalog z) |
| jitter.dlog_band_flux | double | per-galaxy catalog jitter draws (z_cat = catalog z) |
| db_row_id | large_string | s<seed>_<fit_id> |
| prior.F129.flux.loc | double | fit prior parameter that varies per galaxy (constant ones in meta) |
| prior.F129.flux.scale | double | fit prior parameter that varies per galaxy (constant ones in meta) |
| prior.F158.flux.loc | double | fit prior parameter that varies per galaxy (constant ones in meta) |
| prior.F158.flux.scale | double | fit prior parameter that varies per galaxy (constant ones in meta) |
| prior.Halpha.dispersion.loc | double | fit prior parameter that varies per galaxy (constant ones in meta) |
| prior.Halpha.flux.loc | double | fit prior parameter that varies per galaxy (constant ones in meta) |
| prior.Halpha.flux.scale | double | fit prior parameter that varies per galaxy (constant ones in meta) |
| prior.vel.vcirc.loc | double | fit prior parameter that varies per galaxy (constant ones in meta) |
| fixed.z | double | fixed fit parameter |
| img.bb_F129 | fixed_size_list<element: float>[1024] | stamp, fixed-size list<float32> row-major; *_var variance map, *_clean noiseless truth; shape [32, 32] |
| img.bb_F129_var | fixed_size_list<element: float>[1024] | stamp, fixed-size list<float32> row-major; *_var variance map, *_clean noiseless truth; shape [32, 32] |
| img.bb_F158 | fixed_size_list<element: float>[1024] | stamp, fixed-size list<float32> row-major; *_var variance map, *_clean noiseless truth; shape [32, 32] |
| img.bb_F158_var | fixed_size_list<element: float>[1024] | stamp, fixed-size list<float32> row-major; *_var variance map, *_clean noiseless truth; shape [32, 32] |
| img.gr_roll0 | fixed_size_list<element: float>[1024] | stamp, fixed-size list<float32> row-major; *_var variance map, *_clean noiseless truth; shape [32, 32] |
| img.gr_roll0_var | fixed_size_list<element: float>[1024] | stamp, fixed-size list<float32> row-major; *_var variance map, *_clean noiseless truth; shape [32, 32] |
| img.gr_roll1 | fixed_size_list<element: float>[1024] | stamp, fixed-size list<float32> row-major; *_var variance map, *_clean noiseless truth; shape [32, 32] |
| img.gr_roll1_var | fixed_size_list<element: float>[1024] | stamp, fixed-size list<float32> row-major; *_var variance map, *_clean noiseless truth; shape [32, 32] |
| img.gr_roll2 | fixed_size_list<element: float>[1024] | stamp, fixed-size list<float32> row-major; *_var variance map, *_clean noiseless truth; shape [32, 32] |
| img.gr_roll2_var | fixed_size_list<element: float>[1024] | stamp, fixed-size list<float32> row-major; *_var variance map, *_clean noiseless truth; shape [32, 32] |
| img.gr_roll3 | fixed_size_list<element: float>[1024] | stamp, fixed-size list<float32> row-major; *_var variance map, *_clean noiseless truth; shape [32, 32] |
| img.gr_roll3_var | fixed_size_list<element: float>[1024] | stamp, fixed-size list<float32> row-major; *_var variance map, *_clean noiseless truth; shape [32, 32] |
| img.bb_F129_clean | fixed_size_list<element: float>[1024] | stamp, fixed-size list<float32> row-major; *_var variance map, *_clean noiseless truth; shape [32, 32] |
| img.bb_F158_clean | fixed_size_list<element: float>[1024] | stamp, fixed-size list<float32> row-major; *_var variance map, *_clean noiseless truth; shape [32, 32] |
| img.gr_roll0_clean | fixed_size_list<element: float>[1024] | stamp, fixed-size list<float32> row-major; *_var variance map, *_clean noiseless truth; shape [32, 32] |
| img.gr_roll1_clean | fixed_size_list<element: float>[1024] | stamp, fixed-size list<float32> row-major; *_var variance map, *_clean noiseless truth; shape [32, 32] |
| img.gr_roll2_clean | fixed_size_list<element: float>[1024] | stamp, fixed-size list<float32> row-major; *_var variance map, *_clean noiseless truth; shape [32, 32] |
| img.gr_roll3_clean | fixed_size_list<element: float>[1024] | stamp, fixed-size list<float32> row-major; *_var variance map, *_clean noiseless truth; shape [32, 32] |

## Constant columns (in meta constant_columns)

- `ring_member` = 0
- `ring_partner_id` = 
- `noise_rep` = 0
- `observation_config_id` = hlwas_medium_roman_rotac_rolls_poisson
- `db_seed` = 1
- `prior.F129.flux.dist` = truncated_normal
- `prior.F129.flux.low` = 0.05
- `prior.F129.flux.high` = 2000.0
- `prior.F129.rscale.dist` = truncated_lognormal
- `prior.F129.rscale.loc` = -1.7822008619773915
- `prior.F129.rscale.scale` = 0.3707161999720414
- `prior.F129.rscale.low` = 0.005
- `prior.F129.rscale.high` = 3.0
- `prior.F129.x0.dist` = truncated_normal
- `prior.F129.x0.loc` = 0.0
- `prior.F129.x0.scale` = 0.11
- `prior.F129.x0.low` = -0.5
- `prior.F129.x0.high` = 0.5
- `prior.F129.y0.dist` = truncated_normal
- `prior.F129.y0.loc` = 0.0
- `prior.F129.y0.scale` = 0.11
- `prior.F129.y0.low` = -0.5
- `prior.F129.y0.high` = 0.5
- `prior.F158.flux.dist` = truncated_normal
- `prior.F158.flux.low` = 0.05
- `prior.F158.flux.high` = 2000.0
- `prior.F158.rscale.dist` = truncated_lognormal
- `prior.F158.rscale.loc` = -1.7822008619773915
- `prior.F158.rscale.scale` = 0.3707161999720414
- `prior.F158.rscale.low` = 0.005
- `prior.F158.rscale.high` = 3.0
- `prior.F158.x0.dist` = truncated_normal
- `prior.F158.x0.loc` = 0.0
- `prior.F158.x0.scale` = 0.11
- `prior.F158.x0.low` = -0.5
- `prior.F158.x0.high` = 0.5
- `prior.F158.y0.dist` = truncated_normal
- `prior.F158.y0.loc` = 0.0
- `prior.F158.y0.scale` = 0.11
- `prior.F158.y0.low` = -0.5
- `prior.F158.y0.high` = 0.5
- `prior.Halpha.cont.flux_per_nm.dist` = truncated_lognormal
- `prior.Halpha.cont.flux_per_nm.loc` = -0.7920892719899517
- `prior.Halpha.cont.flux_per_nm.scale` = 0.7782737614319876
- `prior.Halpha.cont.flux_per_nm.low` = 0.005
- `prior.Halpha.cont.flux_per_nm.high` = 500.0
- `prior.Halpha.cont.rscale.dist` = truncated_lognormal
- `prior.Halpha.cont.rscale.loc` = -1.7822008619773915
- `prior.Halpha.cont.rscale.scale` = 0.3707161999720414
- `prior.Halpha.cont.rscale.low` = 0.005
- `prior.Halpha.cont.rscale.high` = 3.0
- `prior.Halpha.cont.x0.dist` = truncated_normal
- `prior.Halpha.cont.x0.loc` = 0.0
- `prior.Halpha.cont.x0.scale` = 0.12298373876248843
- `prior.Halpha.cont.x0.low` = -0.5
- `prior.Halpha.cont.x0.high` = 0.5
- `prior.Halpha.cont.y0.dist` = truncated_normal
- `prior.Halpha.cont.y0.loc` = 0.0
- `prior.Halpha.cont.y0.scale` = 0.12298373876248843
- `prior.Halpha.cont.y0.low` = -0.5
- `prior.Halpha.cont.y0.high` = 0.5
- `prior.Halpha.dispersion.dist` = truncated_normal
- `prior.Halpha.dispersion.scale` = 11.3
- `prior.Halpha.dispersion.low` = 5.0
- `prior.Halpha.dispersion.high` = 150.0
- `prior.Halpha.flux.dist` = truncated_normal
- `prior.Halpha.flux.low` = 5.0
- `prior.Halpha.flux.high` = 5000.0
- `prior.Halpha.rscale.dist` = conditional_lognormal
- `prior.Halpha.rscale.parent` = F129.rscale
- `prior.Halpha.rscale.loc` = 0.26236426446749106
- `prior.Halpha.rscale.scale` = 0.4605170185988092
- `prior.Halpha.rscale.low` = 0.005
- `prior.Halpha.rscale.high` = 3.0
- `prior.Halpha.x0.dist` = truncated_normal
- `prior.Halpha.x0.loc` = 0.0
- `prior.Halpha.x0.scale` = 0.11
- `prior.Halpha.x0.low` = -0.5
- `prior.Halpha.x0.high` = 0.5
- `prior.Halpha.y0.dist` = truncated_normal
- `prior.Halpha.y0.loc` = 0.0
- `prior.Halpha.y0.scale` = 0.11
- `prior.Halpha.y0.low` = -0.5
- `prior.Halpha.y0.high` = 0.5
- `prior.cosi.dist` = uniform
- `prior.cosi.loc` = None
- `prior.cosi.scale` = None
- `prior.cosi.low` = 0.05
- `prior.cosi.high` = 0.95
- `prior.g1.dist` = gaussian
- `prior.g1.loc` = 0.0
- `prior.g1.scale` = 0.2
- `prior.g1.low` = None
- `prior.g1.high` = None
- `prior.g2.dist` = gaussian
- `prior.g2.loc` = 0.0
- `prior.g2.scale` = 0.2
- `prior.g2.low` = None
- `prior.g2.high` = None
- `prior.h_over_r.dist` = lognormal
- `prior.h_over_r.loc` = -1.3862943611198906
- `prior.h_over_r.scale` = 0.32236191301916645
- `prior.h_over_r.low` = None
- `prior.h_over_r.high` = None
- `prior.theta_int.dist` = circular_uniform
- `prior.theta_int.loc` = None
- `prior.theta_int.scale` = None
- `prior.theta_int.low` = 0.0
- `prior.theta_int.high` = 6.283185307179586
- `prior.vel.rscale.dist` = conditional_lognormal
- `prior.vel.rscale.parent` = F129.rscale
- `prior.vel.rscale.loc` = -0.6931471805599453
- `prior.vel.rscale.scale` = 0.6907755278982137
- `prior.vel.rscale.low` = 0.005
- `prior.vel.rscale.high` = 3.0
- `prior.vel.v0.dist` = gaussian
- `prior.vel.v0.loc` = 0.0
- `prior.vel.v0.scale` = 125.0
- `prior.vel.v0.low` = None
- `prior.vel.v0.high` = None
- `prior.vel.vcirc.dist` = lognormal
- `prior.vel.vcirc.scale` = 0.11282666955670825
- `prior.vel.vcirc.low` = None
- `prior.vel.vcirc.high` = None
- `sampled_names` = F129.flux,F129.rscale,F129.x0,F129.y0,F158.flux,F158.rscale,F158.x0,F158.y0,Halpha.cont.flux_per_nm,Halpha.cont.rscale,Halpha.cont.x0,Halpha.cont.y0,Halpha.dispersion,Halpha.flux,Halpha.rscale,Halpha.x0,Halpha.y0,cosi,g1,g2,h_over_r,theta_int,vel.rscale,vel.v0,vel.vcirc
