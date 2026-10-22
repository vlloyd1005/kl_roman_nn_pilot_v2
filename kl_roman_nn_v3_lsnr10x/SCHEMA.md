# kl_roman_nn_v3_lsnr10x.parquet schema

13481 rows, 115 columns.

| column | type | description |
|---|---|---|
| fit_id | large_string | ensemble row hash |
| galaxy_id | int64 | population row index within the realization |
| noise_seed | int64 | seed of this row's noise streams |
| truth.vel.v0 | double | model truth parameter (dotted SourceModel key; shared geometry unprefixed) |
| truth.vel.rscale | double | model truth parameter (dotted SourceModel key; shared geometry unprefixed) |
| truth.Halpha.flux | double | model truth parameter (dotted SourceModel key; shared geometry unprefixed) |
| truth.Halpha.rscale | double | model truth parameter (dotted SourceModel key; shared geometry unprefixed) |
| truth.Halpha.dispersion | double | model truth parameter (dotted SourceModel key; shared geometry unprefixed) |
| truth.Halpha.cont.flux_per_nm | double | model truth parameter (dotted SourceModel key; shared geometry unprefixed) |
| truth.Halpha.cont.rscale | double | model truth parameter (dotted SourceModel key; shared geometry unprefixed) |
| truth.F129.flux | double | model truth parameter (dotted SourceModel key; shared geometry unprefixed) |
| truth.F129.rscale | double | model truth parameter (dotted SourceModel key; shared geometry unprefixed) |
| truth.F129.h_over_r | double | model truth parameter (dotted SourceModel key; shared geometry unprefixed) |
| truth.F129.x0 | double | model truth parameter (dotted SourceModel key; shared geometry unprefixed) |
| truth.F129.y0 | double | model truth parameter (dotted SourceModel key; shared geometry unprefixed) |
| truth.F158.flux | double | model truth parameter (dotted SourceModel key; shared geometry unprefixed) |
| truth.F158.rscale | double | model truth parameter (dotted SourceModel key; shared geometry unprefixed) |
| truth.F158.h_over_r | double | model truth parameter (dotted SourceModel key; shared geometry unprefixed) |
| truth.F158.x0 | double | model truth parameter (dotted SourceModel key; shared geometry unprefixed) |
| truth.F158.y0 | double | model truth parameter (dotted SourceModel key; shared geometry unprefixed) |
| truth.Halpha.h_over_r | double | model truth parameter (dotted SourceModel key; shared geometry unprefixed) |
| truth.Halpha.x0 | double | model truth parameter (dotted SourceModel key; shared geometry unprefixed) |
| truth.Halpha.y0 | double | model truth parameter (dotted SourceModel key; shared geometry unprefixed) |
| truth.Halpha.cont.h_over_r | double | model truth parameter (dotted SourceModel key; shared geometry unprefixed) |
| truth.Halpha.cont.x0 | double | model truth parameter (dotted SourceModel key; shared geometry unprefixed) |
| truth.Halpha.cont.y0 | double | model truth parameter (dotted SourceModel key; shared geometry unprefixed) |
| truth.cosi | double | model truth parameter (dotted SourceModel key; shared geometry unprefixed) |
| truth.theta_int | double | model truth parameter (dotted SourceModel key; shared geometry unprefixed) |
| truth.g1 | double | model truth parameter (dotted SourceModel key; shared geometry unprefixed) |
| truth.g2 | double | model truth parameter (dotted SourceModel key; shared geometry unprefixed) |
| truth.z | double | model truth parameter (dotted SourceModel key; shared geometry unprefixed) |
| truth.vel.vcirc | double | model truth parameter (dotted SourceModel key; shared geometry unprefixed) |
| truth.h_over_r | double | model truth parameter (dotted SourceModel key; shared geometry unprefixed) |
| pop.id | double | catalog / population column (after the per-realization jitter) |
| pop.ew_rest_a | double | catalog / population column (after the per-realization jitter) |
| pop.logm | double | catalog / population column (after the per-realization jitter) |
| pop.logm_obs | double | catalog / population column (after the per-realization jitter) |
| pop.prior_vcirc_mu_kms | double | catalog / population column (after the per-realization jitter) |
| pop.prior_vcirc_sigma_dex | double | catalog / population column (after the per-realization jitter) |
| pop.sigma0_kms | double | catalog / population column (after the per-realization jitter) |
| pop.f_line_cgs | double | catalog / population column (after the per-realization jitter) |
| pop.f_line_obs_cgs | double | catalog / population column (after the per-realization jitter) |
| pop.f_line_sigma_cgs | double | catalog / population column (after the per-realization jitter) |
| pop.flux_f106_ujy | double | catalog / population column (after the per-realization jitter) |
| pop.flux_obs_f106_ujy | double | catalog / population column (after the per-realization jitter) |
| pop.flux_sigma_f106_ujy | double | catalog / population column (after the per-realization jitter) |
| pop.flux_f129_ujy | double | catalog / population column (after the per-realization jitter) |
| pop.flux_obs_f129_ujy | double | catalog / population column (after the per-realization jitter) |
| pop.flux_sigma_f129_ujy | double | catalog / population column (after the per-realization jitter) |
| pop.flux_f158_ujy | double | catalog / population column (after the per-realization jitter) |
| pop.flux_obs_f158_ujy | double | catalog / population column (after the per-realization jitter) |
| pop.flux_sigma_f158_ujy | double | catalog / population column (after the per-realization jitter) |
| theta_flipped | bool | theta_int had pi added |
| sigma_bkg.bb_F129 | double | flat background sigma per pixel, stamp units (constant per channel) |
| gain.bb_F129 | double | detected electrons per stamp flux unit (shot-noise term) |
| sigma_bkg.bb_F158 | double | flat background sigma per pixel, stamp units (constant per channel) |
| gain.bb_F158 | double | detected electrons per stamp flux unit (shot-noise term) |
| sigma_bkg.gr_roll0 | double | flat background sigma per pixel, stamp units (constant per channel) |
| gain.gr_roll0 | double | detected electrons per stamp flux unit (shot-noise term) |
| sigma_bkg.gr_roll1 | double | flat background sigma per pixel, stamp units (constant per channel) |
| gain.gr_roll1 | double | detected electrons per stamp flux unit (shot-noise term) |
| sigma_bkg.gr_roll2 | double | flat background sigma per pixel, stamp units (constant per channel) |
| gain.gr_roll2 | double | detected electrons per stamp flux unit (shot-noise term) |
| sigma_bkg.gr_roll3 | double | flat background sigma per pixel, stamp units (constant per channel) |
| gain.gr_roll3 | double | detected electrons per stamp flux unit (shot-noise term) |
| snr.F129 | double | REALIZED matched-filter SNR of the noiseless stamp against the stored variance map (line_*: line-only template; line_total: quadrature sum over rolls) |
| snr.F158 | double | REALIZED matched-filter SNR of the noiseless stamp against the stored variance map (line_*: line-only template; line_total: quadrature sum over rolls) |
| snr.line_roll0 | double | REALIZED matched-filter SNR of the noiseless stamp against the stored variance map (line_*: line-only template; line_total: quadrature sum over rolls) |
| snr.line_roll1 | double | REALIZED matched-filter SNR of the noiseless stamp against the stored variance map (line_*: line-only template; line_total: quadrature sum over rolls) |
| snr.line_roll2 | double | REALIZED matched-filter SNR of the noiseless stamp against the stored variance map (line_*: line-only template; line_total: quadrature sum over rolls) |
| snr.line_roll3 | double | REALIZED matched-filter SNR of the noiseless stamp against the stored variance map (line_*: line-only template; line_total: quadrature sum over rolls) |
| snr.line_total | double | REALIZED matched-filter SNR of the noiseless stamp against the stored variance map (line_*: line-only template; line_total: quadrature sum over rolls) |
| truth.g_plus | double | TARGET: galaxy-frame shear along the major axis |
| truth.g_cross | double | TARGET: galaxy-frame shear at 45 deg to the major axis |
| roll_deg_0 | double | grism roll angle of the channel, deg |
| roll_deg_1 | double | grism roll angle of the channel, deg |
| roll_deg_2 | double | grism roll angle of the channel, deg |
| roll_deg_3 | double | grism roll angle of the channel, deg |
| jitter.z_cat | double | per-realization catalog jitter applied to this galaxy |
| jitter.dz | double | per-realization catalog jitter applied to this galaxy |
| jitter.dlogm | double | per-realization catalog jitter applied to this galaxy |
| jitter.dlog_f_line | double | per-realization catalog jitter applied to this galaxy |
| jitter.dlog_ew | double | per-realization catalog jitter applied to this galaxy |
| jitter.dlog_size | double | per-realization catalog jitter applied to this galaxy |
| jitter.dlog_band_flux | double | per-realization catalog jitter applied to this galaxy |
| depth_snr_scale.line | double | grism depth boost s: grism variance = survey / s^2 |
| db_row_id | large_string | unique row id |
| prior.F129.flux.loc | double | fit prior specification for this row (inference bookkeeping, not an input) |
| prior.F129.flux.scale | double | fit prior specification for this row (inference bookkeeping, not an input) |
| prior.F158.flux.loc | double | fit prior specification for this row (inference bookkeeping, not an input) |
| prior.F158.flux.scale | double | fit prior specification for this row (inference bookkeeping, not an input) |
| prior.Halpha.dispersion.loc | double | fit prior specification for this row (inference bookkeeping, not an input) |
| prior.Halpha.flux.loc | double | fit prior specification for this row (inference bookkeeping, not an input) |
| prior.Halpha.flux.scale | double | fit prior specification for this row (inference bookkeeping, not an input) |
| prior.vel.vcirc.loc | double | fit prior specification for this row (inference bookkeeping, not an input) |
| fixed.z | double | parameter held fixed in the fit prior |
| img.bb_F129 | fixed_size_list<element: float>[1024] | noisy stamp (INPUT), 32x32 float32 flattened row-major |
| img.bb_F129_var | fixed_size_list<element: float>[1024] | variance map, 32x32 float32 flattened row-major |
| img.bb_F158 | fixed_size_list<element: float>[1024] | noisy stamp (INPUT), 32x32 float32 flattened row-major |
| img.bb_F158_var | fixed_size_list<element: float>[1024] | variance map, 32x32 float32 flattened row-major |
| img.gr_roll0 | fixed_size_list<element: float>[1024] | noisy stamp (INPUT), 32x32 float32 flattened row-major |
| img.gr_roll0_var | fixed_size_list<element: float>[1024] | variance map, 32x32 float32 flattened row-major |
| img.gr_roll1 | fixed_size_list<element: float>[1024] | noisy stamp (INPUT), 32x32 float32 flattened row-major |
| img.gr_roll1_var | fixed_size_list<element: float>[1024] | variance map, 32x32 float32 flattened row-major |
| img.gr_roll2 | fixed_size_list<element: float>[1024] | noisy stamp (INPUT), 32x32 float32 flattened row-major |
| img.gr_roll2_var | fixed_size_list<element: float>[1024] | variance map, 32x32 float32 flattened row-major |
| img.gr_roll3 | fixed_size_list<element: float>[1024] | noisy stamp (INPUT), 32x32 float32 flattened row-major |
| img.gr_roll3_var | fixed_size_list<element: float>[1024] | variance map, 32x32 float32 flattened row-major |
| img.bb_F129_clean | fixed_size_list<element: float>[1024] | noiseless stamp, 32x32 float32 flattened row-major |
| img.bb_F158_clean | fixed_size_list<element: float>[1024] | noiseless stamp, 32x32 float32 flattened row-major |
| img.gr_roll0_clean | fixed_size_list<element: float>[1024] | noiseless stamp, 32x32 float32 flattened row-major |
| img.gr_roll1_clean | fixed_size_list<element: float>[1024] | noiseless stamp, 32x32 float32 flattened row-major |
| img.gr_roll2_clean | fixed_size_list<element: float>[1024] | noiseless stamp, 32x32 float32 flattened row-major |
| img.gr_roll3_clean | fixed_size_list<element: float>[1024] | noiseless stamp, 32x32 float32 flattened row-major |
| db_seed | int64 | realization (population seed) |
