# PyGLDA v2 — configuration guide

PyGLDA v2 is driven by three JSON files that live together in one *settings folder* (`settings/<case>/`),
plus a handful of attributes set in the driver script (`src_demo/RDA_demo*.py`). This document describes
every option, what it does, which values are allowed, and which ones you normally have to touch.

| file | purpose | edited by |
|---|---|---|
| `DA_setting.json` | the assimilation case: period, basin, ensemble, state vector, observations, filter method | you (basic block partly overwritten by `RDA.read_config_and_save()`) |
| `perturbation.json` | how the ensemble is generated: perturbation of forcing and model parameters | you, once per ensemble design |
| `Config_ReWaterGAP.json` | the WaterGAP 2.2e model itself: input paths, run options, which variables are written | rarely; mostly paths and output switches |

The sections below follow this order; a last section lists the driver attributes.

---

## 1. `DA_setting.json`

Loaded by `src_DA/configure_DA.py` (`config_DA.loadjson(...).process()`); every stage of `Regional_DA.RDA`
reads it. Missing keys fall back to the defaults in `configure_DA.py`, so an old file keeps working when new
keys are added. The file has four blocks: `basic`, `obs`, `model`, `method`.

### 1.1 `basic` — case, period, basin, directories

| key | type | meaning | set by |
|---|---|---|---|
| `fromdate`, `todate` | `YYYY-MM-DD` | first and last day of the open loop and of the assimilation. The spin-up period is **not** here (see driver). | you (the driver uses its own `RDA.sim_begin_time` / `sim_end_time`; keep both consistent) |
| `ensemble` | int | number of **perturbed** members. The run needs `ensemble + 1` MPI ranks: rank 0 is the unperturbed control member (`Ens_0`), rank 1 is the main thread that performs the update. 4 for technical tests, 20–30 for science runs. | `RDA.ens` (overwritten by `read_config_and_save`) |
| `case` | str | case name; results go to `Res/<case>/`, DA temp output to `DA_output/<case>/`, OL mean to `GRACE/OLmean/<case>_<basin>.hdf5` | `RDA.case` |
| `basin` | str | basin name; used for mask, observation and envelope file names (`<basin>_obs_GRACE.hdf5`, `state_envelope_<basin>.nc`, `Basin/mask/<basin>/`) | `RDA.basin` |
| `basin_shp` | path | shapefile of the (sub-)basins. Must have an integer column `ID` = 1..N (one row per sub-basin); a `NAME` column is used in figures. Build it with `src_auxiliary/Hydroshed.py` from HydroBASINS. | `RDA.shp_path` |
| `basin_mask` | path | the rasterised mask `<basin>_res_0.5.h5` produced by `RDA.config_basin_mask()` (datasets `basin`, `sub_basin_k`, 360×720). The 1° companion `<basin>_res_1.h5` is used for the mascon covariance. Rasterisation rule: `src_auxiliary/shp2mask.py` (`default_rule='area'`, `area_threshold=0.4`; `'centre'` = classic cell-centre rule). | derived from `RDA.external_data_path` and `RDA.basin` |
| `res_permanent` | path | root of the collected results: `Res/<case>/{OL,DA}/Ens_k/daily_output_<year>.nc`, `basin_ts_*.h5`, `figures/` | you (once) |
| `OL_output_temp_dir` | path | temporary daily **global** output of the open loop, `OL_output/Ens_k/daily_output_YYYY-MM-DD.nc`. Shared by all cases (WaterGAP runs globally) — a new basin can reuse an existing open loop if period and perturbation are the same. Clean with `RDA.clean_temp_output(Stage.OL)`. | you (once) |
| `DA_output_temp_dir` | path | temporary daily output of the assimilation, `DA_output/<case>/Ens_k/` (regional box), plus `threshold_log.json` per member | you (once) |
| `Ensemble_ini_dir` | path | restart states written by the spin-up and read by OL/DA (`Ensemble_Initialization`) | you (once) |
| `Ensemble_input_dir` | path | perturbed forcing and parameters written by `RDA.model_perturbation()` | you (once) |
| `Auxiliary_dir` | path | static auxiliaries: `smax.nc` (max. soil water), `state_envelope_<basin>.nc` (OL envelope for thresholds, rebuilt at every DA start) | you (once) |

### 1.2 `obs` — GRACE observations

| key | meaning |
|---|---|
| `name` | module name, always `src_OBS` |
| `dir` | where the final observation file is written: `<dir>/<basin>_obs_GRACE.hdf5` (epochs, windows, unperturbed + perturbed sub-basin TWS, covariance, sub-basin areas). Written by `RDA.DA_run()` → `generate_perturbed_GRACE_obs()`. |
| `GRACE` | sub-block, see below |

`obs.GRACE`:

| key | meaning | Mascon_monthly | TUD_5daily |
|---|---|---|---|
| `kind` | observation product / time sampling. Allowed: `Mascon_monthly` (CSR RL06 mascons, monthly), `TUD_5daily` (TU Delft 5-daily hybrid L3, gridded EWH + per-cell sigma), `SH_monthly`, `ESA_SING`, `ESA_SING_ESM3` (legacy / synthetic). | ✓ | ✓ |
| `EWH_grid_dir` | folder with the gridded product | `GRACE/SaGEA/signal_Mascon` (netCDF of CSR mascons) | `GRACE/Miguel_Tudelft` (both TUD netCDF files) |
| `aux_for_time_epochs` | folder from which the observation epochs / windows are read | same as `EWH_grid_dir` | same as `EWH_grid_dir` |
| `cov_dir` | folder with the monthly covariance samples (`SaGEA/sample_DDK3/<YYYY-MM>_0.npy`, 1°) | used | **not used** (covariance propagated from the per-cell sigma) |
| `preprocess_res` | output of `RDA.get_GRACE_obs()`: `<basin>_signal.hdf5` (sub-basin TWS), `<basin>_gridded_signal.hdf5`, `<basin>_cov.hdf5` | `GRACE/output` | `GRACE/output_TUD` (keep the two products apart) |
| `OL_mean` | folder of the open-loop temporal mean per case (`<case>_<basin>.hdf5`), the datum to which the GRACE anomalies are referenced | ✓ | ✓ |
| `ewh_file` | TUD only: file name of the EWH grid (`TUD-L3-5dayEWH-GRACEv2-Hybrid-2002_2017-0.5x0.5.nc`) | – | optional, default as shown |
| `unc_file` | TUD only: file name of the per-cell uncertainty grid. Use the latest file from TU Delft (`TUD-L3-5dayEWH_UNC-...`); the default is the older `TUD-L3Uncorr-...`. | – | optional |
| `unit` | TUD only: unit of the netCDF (`cm`, `m`, `mm`); converted to mm internally | – | optional, default `cm` |
| `corr_length_km` | TUD only: spatial error correlation length L assumed when the sub-basin covariance is propagated from the per-cell sigma, ρ = exp(−d/L). 0 = uncorrelated cells (very small sub-basin sigma), 300 = recommended, ∞ ≈ fully correlated (sigma = RMS of cell sigma). | – | optional, default 300 |
| `add_gia` | TUD only: add the static `ewh_gia` field to the EWH (normally `false`, GIA is a constant removed with the mean) | – | optional, default `false` |

`RDA.get_GRACE_obs(is_diagonal=False)`: `is_diagonal=True` keeps only the diagonal of the sub-basin
covariance (independent sub-basins); `False` keeps the full covariance (recommended — for mascons it carries the
leakage correlation between neighbouring units, for TUD the propagated spatial correlation).

### 1.3 `model` — hydrological model and state vector

| key | meaning |
|---|---|
| `name` | `WaterGap` (others are legacy W3RA entries) |
| `layer` | dictionary `storage -> true/false`: which WaterGAP storages form the **state vector** that the filter updates. All ten storages always enter the observation operator (TWS = sum of all), but only the `true` ones receive increments; the `false` ones are subtracted from the observation (they are "excluded stores"). |

Storages (all in mm = kg m⁻²): `groundwstor`, `soilmoist`, `riverstor`, `swe` (snow), `canopystor`,
`locallakestor`, `localwetlandstor`, `globallakestor`, `globalwetlandstor`, `reservoirstor`.

Recommendations from the Amazon and Danube tests:

* `riverstor` must be in the state wherever river storage carries the seasonal signal (Amazon: 75 % of the
  TWS variance). Without it the increments go into groundwater and produce a sawtooth there.
  Where the river is a small part of TWS (Danube) leave it `false`: in run12 its correction was only −2..+3 mm
  (basin, seasonal), it drained within the same window (increment +0.13 mm/day, drift −0.005 mm/day × 29 days)
  and negative increments clipped river storage at its floor in 5–7 % of the cells in two months. With `false`
  the river is subtracted from GRACE with each member's own value, its share of the mismatch goes to groundwater,
  soil and snow, and it responds to the corrected storages through the model's own runoff (Danube from run13).
* `soilmoist` and `groundwstor` always; `swe` in snow-affected basins (Danube), not needed in the tropics.
* lakes, wetlands, reservoirs, canopy: leave `false` for now. Their member spread is a fixed offset from the
  depth perturbation and GRACE cannot resolve them separately; see §2 for the consequence on the ensemble.

Physical bounds applied after every update (`src_DA/Threshold.py`, not configurable from JSON): soil 0 ≤ S ≤ smax;
snow ≥ 0 (no fixed upper limit; the filter limits snow relative to the forecast, see `snow_upper_bound`); river ≥ 1e-3 mm and inside the open-loop envelope [0.5·min, 1.5·max]; groundwater inside the OL
envelope widened by half its range (signed storage); canopy / wetlands / reservoirs ≥ 0; lakes unbounded. The
envelope `state_envelope_<basin>.nc` is recomputed from `Res/<case>/OL` at every DA start. Clipping statistics
are printed per member and saved to `DA_output/<case>/Ens_k/threshold_log.json`.

### 1.4 `method` — the filter

`fusion_method` selects the analysis:

| value | meaning |
|---|---|
| `EnKF_v0` (default) | stochastic EnKF with perturbed observations, no localization or inflation (`src_DA/EnKF.py`); for large ensembles |
| `EnKF_localized` | the EnKF assembled from four independent components (`src_DA/filter_factory.py`), each chosen with its own block below; recommended for small ensembles |

The components of `EnKF_localized` (code in `src_DA/`):

| component | file | classes |
|---|---|---|
| localization of the gain | `localization.py` | `NoLocalization`, `BlockLocalization`, `GaussianLocalization` |
| spread maintenance (inflation) | `inflation.py` | `NoInflation`, `MultiplicativeInflation`, `RTPS`, `AdditiveInflation`, `AdaptiveAdditiveInflation` |
| increment partition | `partition.py` | `EnKFPartition`, `NonNegativePartition` |
| physical bounds (window-aware) | `bounds.py` | `StateBounds` |

Analysis of one window: `A = inflation.prior(A)` → `K = (L_state∘Pxy)(L_obs∘Pyy + R)⁻¹` → `dX = partition.split(K(y − HX))` → `Xa = inflation.posterior(X + dX)`. Any localization can be combined with any inflation and partition.

#### `localization`

| kind | settings | meaning |
|---|---|---|
| `none` | — | the plain sample covariances; every observation updates every cell |
| `block` (default) | — | each sub-basin observation updates only the cells of its own sub-basin; the strongest choice, right for small ensembles |
| `gaussian` | `length_km` (300), `cutoff` (2.0) | weight exp(−½(d/ℓ)²) with d the distance of a cell to the observed sub-basin (0 inside), 0 beyond `cutoff · length_km`; boundary cells are also informed by the neighbour's observation; use with ≥ 20 members |

#### `inflation` (one scheme per run)

| kind | settings (defaults) | meaning |
|---|---|---|
| `none` (default) | — | no spread maintenance |
| `multiplicative` | `factor` (1.0) | forecast anomalies × factor before the gain. Danube run 4 (1.6): it inflates storages that cancel each other (soil vs groundwater), the TWS spread GRACE sees does not grow, and the hidden directions grow every month (40 % groundwater clipping). |
| `rtps` | `alpha` (0.7), `space` (`state`) | relaxation to prior spread (Whitaker & Hamill 2012): σ_a' = α σ_f + (1−α) σ_a. `state`: per element; with basin-mean observations the cell anomalies compensate and the sub-basin spread stays collapsed (Danube run 2). `obs`: one factor per sub-basin from the spread of the observation equivalents, bounded elements limited, the rest rescaled (Danube run 3). |
| `additive` | `sigma` ({storage: std in mm}), `months` (all), `seed` (none), `exact_spread` (false) | zero-mean random perturbations added before the gain: one number per member, sub-basin and storage, centred over the members; groundwater: the same value in every cell of the sub-basin (visible to GRACE); snow and other bounded storages: weighted by the cell's storage, at most 50 % of it (empty cells stay empty). The perturbations stay in the members and are propagated by WaterGAP (model error). Danube runs 5–7: `{"groundwstor": 15, "swe": 5}`. |
| `adaptive_additive` | `split` ({"groundwstor": 0.9, "swe": 0.1}), `weight_current` (0.5), `max_factor` (3.0), `min_sigma` (5 mm), `remove_bias` (true), `bias_memory` (0.05), `months`, `seed`, `exact_spread` | as `additive`, but the std of the added sub-basin spread is estimated every window from the innovations d = mean(obs) − mean(HX): s² = (1−w) s²_prev + w (d − bias)², σ_add² = s² − var(HX) − R_jj, clipped to [`min_sigma`, `max_factor`·√R_jj]; `split` gives the share of the added variance per storage (variance snow cannot take goes to groundwater). The current window is included (Anderson 2007, 2009), so an extreme month gets a high weight on GRACE in the same update. Danube run 8. Writes `adaptive_inflation_log.csv`. `split` may also be `"ol_spread"` or `"ol_variability"` (+ optional `split_source`, `min_share`, `split_storages`), see below. |

`months`: calendar months of the window in which the additive schemes act. `exact_spread`: rescale the centred random numbers so that their sample std equals σ in every window (improved sampling, Evensen 2004); removes the chance in the added spread of a 4-member ensemble.

#### `increment_partition`

| kind | meaning |
|---|---|
| `enkf` (default) | increments as given by the (localized) Kalman gain; with few members the vertical split comes from noisy sample covariances (compensating +GW / −river increments, Amazon tug of war) |
| `non_negative` | the sub-basin TWS increment of the Kalman update is kept exactly, but distributed with shares ∝ variance × max(correlation with sub-basin TWS, 0); no storage is emptied to fill another. Bound-aware: an element is not pushed across its bound (snow/soil/canopy/wetlands/reservoirs ≥ 0, river ≥ floor, soil ≤ smax, snow ≤ factor × forecast + offset), its share goes to the other elements of the sub-basin. The bounds are checked against each member's lowest/highest **daily** value in the window, so no day crosses a bound. Recommended for small ensembles. |
| `non_negative` + `max_weight_ratio` (e.g. 5; default none) | caps every element's weight at `max_weight_ratio` × the median weight of the same storage in the sub-basin, so that one cell with an extreme ensemble spread cannot absorb the whole sub-basin correction (Danube: the Vienna cell, whose groundwater drifts by > 100 mm/yr in the open loop because of WaterGAP's groundwater abstraction of 263 mm/yr, had ~10⁴ × the typical weight and got single-window increments of 5–10 m). The sub-basin increment is unchanged; only its distribution over the cells changes. `filter_summary.json` reports how often the cap acted and where the largest weight ratio occurred. A number caps every storage (run 10); a dict caps only the listed storages: `{"kind": "non_negative", "max_weight_ratio": {"groundwstor": 5}}` (**recommended**). Capping snow is harmful: most cells have little or no snow, so the median snow weight is ~0 and the real mountain snow cells lose their weight (run 10: snow correction halved, up to 94 % of a sub-basin's snow share removed). `filter_summary.json` → `capped_by_storage`. |
| `non_negative` + `spatial_blend` (e.g. `{"groundwstor": 0.8}`; default none) | mixes each cell's weight of the listed storage with the sub-basin mean of that storage, w' = (1−b)·w + b·mean(w); the storage's total share is unchanged, only its spread over the cells. GRACE has no information below the sub-basin scale, so b = 1 spreads the storage's correction evenly, b = 0 follows the cell-to-cell differences of the ensemble. Danube run 11a (no blend): 89 % of the groundwater correction of sub-basin 6 went to 7 of 76 cells (−150 to −270 mm in 13 years). For unbounded storages (groundwater) only: for snow or soil it would put water into snow-free or full cells. |
| `non_negative` + `max_cell_factor` (e.g. 3, or `{"swe": 3, "groundwstor": 3}`; default none) | no storage of any cell may change by more than f × the sub-basin TWS increment of that member, \|dX_i\| ≤ f·\|δ_j\|; the limit acts on each storage of each cell separately (a dict sets f per storage, unlisted storages unlimited). The excess goes to the other elements of the sub-basin like a bound, so the sub-basin increment is unchanged. Danube run 11a (no limit): one Alpine snow cell got +820 mm in one window. `filter_summary.json` → `cell_limited_by_storage`, `increment_not_placed_mm_max` (should be ~0). Order inside the partition: `max_weight_ratio` → `spatial_blend` → bounds and `max_cell_factor`. |

#### The inflation split (`adaptive_additive`)

GRACE constrains only the sum of the storages, so where the correction ends up is decided by the spread the ensemble carries in each storage, and with adaptive additive inflation that spread is mostly the added noise. `split` therefore largely decides which storage takes the correction.

| `split` | shares of the added variance |
|---|---|
| `{"groundwstor": 0.9, "swe": 0.1}` | fixed, the same every month and sub-basin (Danube runs 8–10: groundwater takes ~90 % of the correction) |
| `"ol_spread"` | per calendar month and sub-basin ∝ the **open-loop ensemble variance** of each storage: the model's own error estimate from the forcing / parameter perturbations (Danube: soil 0.8 in May–Oct, groundwater + snow in winter, Alps groundwater / snow) |
| `"ol_variability"` | per calendar month and sub-basin ∝ the **year-to-year variance of the open-loop monthly anomalies** of each storage (natural variability; independent of the perturbation design, gives groundwater and river more weight in spring) |

The two OL-based tables are computed once at the start from `Res/<case>/Res_OL.h5` (or `"split_source": "<path>"`) for the storages of the DA state; `"min_share": 0.05` puts a floor under every share. `"split_storages": ["groundwstor", "soilmoist", "swe"]` restricts the noise to these storages and re-normalises the shares over them: river storage stays in the DA state (the update still corrects it) but gets no random noise, which would otherwise flow out within days as noise in each member's discharge (default: all storages of the state). They are written to `filter_summary.json` (`inflation.split_table[storage][month][sub-basin]`), and the shares used in every window are columns `share_<storage>` of `adaptive_inflation_log.csv`.

Noise in a bounded storage is weighted per cell by the room to the nearer bound, min(S − lb, ub − S): snow by its amount (as before), soil by the distance to 0 or to smax, so a nearly full winter soil gets almost none; what a bounded storage cannot take goes to groundwater. Use soil in the split only with `"soil_upper_bound": true` (default). Example (run 11a / 11b):

```json
"inflation": {"kind": "adaptive_additive", "split": "ol_spread",
              "split_storages": ["groundwstor", "soilmoist", "swe"], "weight_current": 0.5, "max_factor": 3.0,
              "min_sigma": 5.0, "remove_bias": true, "bias_memory": 0.05, "seed": 42}
```

#### `obs_error_inflation`

`{ "k": factor }` multiplies the observation error **variance** of sub-basin k (1-based, as string) by `factor`, keeping the error correlations (R' = D R D); `{}` = off.

#### `obs_error_correlation`

`"full"` (default): R as delivered with the GRACE product (for the CSR mascons in demo_2 the DDK3-based covariance, with correlations up to 0.9 between neighbouring sub-basins). `"diagonal"`: sensitivity test without error correlations between sub-basins — R keeps its variances only, and the member observations are re-perturbed consistently: the perturbations ε = GRACE `ens_k` − `ens_0` are whitened with the Cholesky factor of the full covariance and rescaled with the diagonal standard deviations, so the same random numbers are used without their correlation.

#### `obs_perturbation_centering`

`false` (default) or `true`. Each member assimilates GRACE plus its own random perturbation (`ens_k` − `ens_0` in the observation file, or its decorrelated version with `"obs_error_correlation": "diagonal"`). With few members the perturbations do not average to zero (4 members: about σ/2, Danube 12 mm RMS per sub-basin), so every analysis is pulled towards a randomly shifted GRACE value. `true` removes the member mean of the perturbations, so the ensemble-mean observation is exactly GRACE; the spread of the members is unchanged. Standard practice for perturbed-observation EnKFs; with 30 members the effect is small (σ/√30), and square-root filters (ETKF/LETKF) do not perturb observations at all. `filter_summary.json` reports the mean shift that was removed.

#### `soil_upper_bound`

`true` (default) or `false`. With the `non_negative` partition, soil is kept at or below its capacity `smax` (`Auxiliary_dir/smax.nc`, the same field the post-update threshold uses) already inside the update: the window-aware limit smax − (daily max − window mean) keeps every day of the window ≤ smax, and the part of a sub-basin increment that soil cannot hold goes to the other storages of the sub-basin with their shares (mostly groundwater). Without it (`false`, runs ≤ 10) only the threshold clipped soil at smax after the update and that water was deleted (Danube run 9: about 12.5 m per member summed over cells and days, all at the upper bound). The sub-basin TWS increments are the same either way; only where the water goes changes. `filter_summary.json` → `bounds.upper_fields` lists the bounded cells and the smax range. No effect with the `enkf` partition, which ignores bounds.

#### `snow_upper_bound`

`{"factor": 2.0, "offset_mm": 20}` (default) or `false`. Upper limit for snow per member and window, relative to the member's own forecast: snow ≤ factor × forecast + offset (window-aware: every day of the window stays below factor × that member's highest day + offset). The analysis can at most double a snow pack and add a few cm where there is little snow. Snow errors are mostly relative (precipitation under-catch in the mountains, melt timing), so a fixed value has no physical meaning; the former fixed 1000 mm (cell mean) let one Alpine cell jump from 172 to 993 mm in one window (Danube run 11a). WaterGAP itself has no cell-mean snow limit; it only stops accumulating in one of the 100 sub-grid elevation levels of a cell once that level holds 1000 mm. The post-update threshold now only keeps snow ≥ 0. `filter_summary.json` → `bounds.relative_upper`.

#### `snow_lower_bound`

`{"factor": 0.5}` (default) or `false`. Lower limit for snow per member and window, relative to the member's own forecast: snow ≥ factor × forecast, i.e. one update removes at most (1 − factor) of a snow pack (window-aware: every day of the window keeps at least factor × its own forecast, LB = window mean − (1 − factor) × the member's lowest day). Without it only snow ≥ 0 applied, and the partition could empty a single cell: Danube run12 removed 496 mm from one cell of one member in one window (2019-02), although GRACE has no information below the sub-basin scale. The removal the bound refuses goes to the other cells and storages of the sub-basin (bound-aware `non_negative` partition), so the sub-basin TWS increment is unchanged. Counterpart of `snow_upper_bound` (which limits additions). `filter_summary.json` → `bounds.relative_lower`. No effect with the `enkf` partition, which ignores bounds.

#### Output

`DA_output/<case>/filter_summary.json` (one section per component; for `non_negative` also where the largest single-element increment occurred) and, for `adaptive_additive`, `adaptive_inflation_log.csv` (one row per window and sub-basin: innovation, bias, forecast spread, observation error, added spread).

#### Older settings

Files written before the restructuring (October 2026) still work; the flat keys are translated with a note in the log:

| old | new |
|---|---|
| `"inflation": 1.6` | `"inflation": {"kind": "multiplicative", "factor": 1.6}` |
| `"rtps_alpha": 0.7, "rtps_space": "obs"` | `"inflation": {"kind": "rtps", "alpha": 0.7, "space": "obs"}` |
| `"additive_inflation": {"groundwstor": 15, "swe": 5, "seed": 42}` | `"inflation": {"kind": "additive", "sigma": {"groundwstor": 15, "swe": 5}, "seed": 42}` |
| `"additive_inflation": {"mode": "adaptive", ...}` | `"inflation": {"kind": "adaptive_additive", ...}` |
| `"increment_partition": "non_negative"` | `"increment_partition": {"kind": "non_negative"}` |

If more than one old scheme is active (e.g. `rtps_alpha` > 0 and `inflation` ≠ 1), the run stops with an error.

#### Typical blocks

```json
"method": {                                   // large ensemble, classic
    "fusion_method": "EnKF_v0"
}

"method": {                                   // small ensemble (Danube demo_2, run 8)
    "fusion_method": "EnKF_localized",
    "localization": {"kind": "block"},
    "inflation": {"kind": "adaptive_additive", "split": {"groundwstor": 0.9, "swe": 0.1},
                  "weight_current": 0.5, "max_factor": 3.0, "min_sigma": 5.0,
                  "remove_bias": true, "bias_memory": 0.05, "seed": 42},
    "increment_partition": {"kind": "non_negative"},
    "obs_error_inflation": {},
    "obs_error_correlation": "full"
}
```

(JSON does not allow comments; they are only in this document.)

### 1.5 Two complete examples

**demo_3 — Amazon, TUD 5-daily, 18 sub-basins** (settings/demo_3/DA_setting.json)

```json
{
  "basic": { "fromdate": "2002-01-01", "todate": "2016-04-30", "ensemble": 4, "case": "demo_3", "basin": "Amazon",
             "basin_shp": ".../Basin/shp/Amazon/Amazon.shp", "basin_mask": ".../Basin/mask/Amazon/Amazon_res_0.5.h5",
             "res_permanent": ".../Res", "OL_output_temp_dir": ".../OL_output", "DA_output_temp_dir": ".../DA_output",
             "Ensemble_ini_dir": ".../Ensemble_Initialization", "Ensemble_input_dir": ".../Ensemble_input",
             "Auxiliary_dir": ".../Auxiliary" },
  "obs": { "name": "src_OBS", "dir": ".../GRACE/obs",
           "GRACE": { "EWH_grid_dir": ".../GRACE/Miguel_Tudelft", "cov_dir": ".../GRACE/SaGEA/sample_DDK3",
                      "preprocess_res": ".../GRACE/output_TUD", "OL_mean": ".../GRACE/OLmean",
                      "aux_for_time_epochs": ".../GRACE/Miguel_Tudelft", "kind": "TUD_5daily",
                      "ewh_file": "TUD-L3-5dayEWH-GRACEv2-Hybrid-2002_2017-0.5x0.5.nc",
                      "unc_file": "TUD-L3-5dayEWH_UNC-GRACEv2-Hybrid-2002_2017-0.5x0.5.nc",
                      "corr_length_km": 300 } },
  "model": { "name": "WaterGap",
             "layer": { "groundwstor": true, "soilmoist": true, "riverstor": true, "swe": false, "canopystor": false,
                        "locallakestor": false, "localwetlandstor": false, "globallakestor": false,
                        "globalwetlandstor": false, "reservoirstor": false } },
  "method": { "fusion_method": "EnKF_localized", "localization": {"kind": "block"},
              "inflation": {"kind": "rtps", "alpha": 0.7, "space": "state"},
              "increment_partition": {"kind": "non_negative"}, "obs_error_inflation": {} }
}
```

**demo_2 — Danube, CSR mascons monthly, 6 sub-basins**: identical structure with `case: demo_2`, `basin: Danube`,
`basin_shp: .../Basin/shp/Danube/Danube.shp`, `obs.GRACE.kind: Mascon_monthly`,
`EWH_grid_dir` = `aux_for_time_epochs` = `.../GRACE/SaGEA/signal_Mascon`, `preprocess_res: .../GRACE/output`,
`layer.swe: true`, and the `method` block of run 8 (`adaptive_additive`, see 1.4).

---

## 2. `perturbation.json` — ensemble generation

Read by `RDA.model_perturbation()` (`src_DA/Perturbation.py`); global, independent of the basin.

| key | meaning |
|---|---|
| `ensemble` | number of perturbed members (keep equal to `basic.ensemble`) |
| `dir.in`, `dir.out` | unperturbed WaterGAP input (`Input_data`) and where the perturbed forcing / parameter files are written (`Ensemble_input`, = `basic.Ensemble_input_dir`) |
| `correlation.forcing.isSpatialCorrelated / isTemporalCorrelated` | spatially correlated perturbation fields (true), AR(1) in time (currently false → white noise in time, i.e. monthly forcing perturbed independently) |
| `correlation.par.isSpatialCorrelated` | spatially correlated parameter perturbation |
| `forcing.<var>` | per forcing variable (`pr` precipitation, `tas` temperature, `rsds`, `rlds` radiation): `is_perturbed`, `perturb_method` (`multiplicative` / `additive`), `error_distribution` (`gaussian` / `triangle`), `coefficients` (std for gaussian: 0.3 = 30 % for `pr`, 2 K for `tas`) |
| `par.<parameter>` | per WaterGAP parameter: `is_perturbed`, `error_distribution` (`triangle`), `coefficients` = [min, mode, max] of the triangle, drawn once per member (constant in time). Perturbed in the current design: `pt_coeff_humid_arid`, `adiabatic_lapse_rate`, `max_daily_pet`, `critcal_gw_precipitation`, `swb_outflow_coeff`, `activelake_depth` [1, 5, 20] m, `activewetland_depth` [0.5, 2, 5] m, `max_canopy_storage_coefficient`. |

Consequences seen in the Amazon runs: the `activelake_depth` / `activewetland_depth` perturbation creates large
**constant offsets** between members (75 of 129 mm TWS spread) in storages that are not in the state; the filter
cannot correct them and the state storages end up compensating them. For a science ensemble reduce these two
ranges, keep `pr` perturbation (ideally with temporal correlation) and reduce the recharge-parameter spread, so
that the member differences come from dynamics rather than from fixed partition preferences.

Any change here requires rerunning `model_perturbation()`, `spin_up()`, `OL_run()` and the OL collection.

---

## 3. `Config_ReWaterGAP.json` — the WaterGAP model

Passed on the command line (`python RDA_demo.py Config_ReWaterGAP.json`). Keys that matter for PyGLDA:

| key | meaning / required value |
|---|---|
| `FilePath.inputDir.*` | climate forcing, water use, static land data, global parameter file (`WaterGAP_2.2e_global_parameters_gswp3_w5e5.nc`) |
| `FilePath.outputDir` | WaterGAP's own output directory (not used by PyGLDA, which writes `OL_output` / `DA_output`) |
| `RuntimeOptions.SimulationOption.AntNat_opts` | `ant: true` anthropogenic run with water use (`subtract_use`) and reservoirs (`res_opt`) |
| `RuntimeOptions.RestartOptions` | `restart`, `save_model_states_for_restart` are controlled by PyGLDA at run time; `save_and_read_states_dir` is the WaterGAP restart folder |
| `RuntimeOptions.SimulationPeriod` | overwritten by the driver (`spin_up_start/end`, `sim_begin/end_time`); `spinup_years` is also set from `RDA.spin_up_years` |
| `RuntimeOptions.SimulationExtent.run_basin` | **must be `false`**: WaterGAP runs globally, the basin is applied only when collecting results, so one open loop serves all basins |
| `RuntimeOptions.Calibrate WaterGAP.run_calib` | `false` |
| `OutputVariable.*Storages` | which storages WaterGAP writes daily. **Every storage that appears in `model.layer` (true or false) must be switched on here**, because TWS is their sum: `canopy_storage`, `snow_water_equiv`, `soil_moisture`, and the lateral storages (groundwater, lakes, wetlands, river, reservoirs). Fluxes are optional (`consistent_precipitation` is on for the water-balance check). |

---

## 4. Driver attributes (`src_demo/RDA_demo*.py`)

Set on the class `RDA` before calling the stages; several of them overwrite `basic` in `DA_setting.json`
through `RDA.read_config_and_save()`.

| attribute | meaning |
|---|---|
| `RDA.setting_dir` | the settings folder of the case (`settings/demo_3`) |
| `RDA.external_data_path` | root of the external data (`PyGLDA_v2_external_data`), used to derive `Basin/mask`, `GRACE/global_mask`, ... |
| `RDA.case`, `RDA.basin`, `RDA.shp_path`, `RDA.ens` | → `basic.case`, `basic.basin`, `basic.basin_shp`, `basic.ensemble`, `basic.basin_mask` |
| `RDA.spin_up_start`, `RDA.spin_up_end`, `RDA.spin_up_years` | spin-up period and number of repetitions (5 × 2000–2001) |
| `RDA.sim_begin_time`, `RDA.sim_end_time` | open-loop / assimilation period (`2002-01-01` … `2016-04-30`); must match `basic.fromdate/todate` and the collected OL |
| `RDA.map_style` | `'smooth'` or `'pixel'` (or a list) for the harmonic maps |

Stage order and parallelism:

| stage | call | MPI | needed again when… |
|---|---|---|---|
| external data check | `RDA.config_external_data()` | serial | new machine |
| write basics into JSON | `RDA.read_config_and_save()` | serial | case/basin/ensemble changed |
| perturbation | `RDA.model_perturbation()` | serial | `perturbation.json` changed |
| basin mask | `RDA.config_basin_mask()` | serial | shapefile or mask rule changed |
| GRACE observations | `RDA.get_GRACE_obs(is_diagonal=False)` | serial | basin, mask, product, period or uncertainty file changed |
| spin-up, open loop | `RDA.spin_up()`, `RDA.OL_run()` | `mpiexec -n ens+1` | perturbation or period changed (global → shared by all basins) |
| OL collection + statistics | `RDA.collect_and_statistics(Stage.OL)` | `mpiexec -n ens+1` | per case (crops the global OL to the basin, computes OL mean / stats) |
| assimilation | `RDA.DA_run()` | `mpiexec -n ens+1` | every experiment (envelope rebuilt automatically) |
| DA collection | `RDA.collect_and_statistics(Stage.DA)` | `mpiexec -n ens+1` | after every DA |
| post-processing, figures, diagnosis | `RDA.post_processing()`, `RDA.visualization()`, `RDA.increment_diagnosis()` | serial | after every DA |
| house-keeping | `RDA.clean_temp_output(Stage.OL/DA, dry_run=True)` | serial | when disk space is needed (deletes only collected daily files) |

Logs of the MPI stages are written to `PyGLDA_v2/parallel_logs/{OL,DA,collect}/rank_k.log`; the main thread
(rank 1) prints to the terminal, which is where the filter summary (`EnKF_localized: ...`, `Increment partition:
...`) appears.


### Note on the post-update threshold (`src_DA/Threshold.py`)
It is a safety net: with `"soil_upper_bound": true` and the `non_negative` partition the soil upper limit is already respected by the update, so the soil `water_removed_mm` in `threshold_monthly.csv` should be ~0 (it still acts for the `enkf` partition, multiplicative / RTPS inflation and rounding). Groundwater has no limit by default (`gw_lower=False`, `gw_upper=False`). The open-loop envelope (OL range widened by half the range on each side) was a safety net against metre-scale blow-ups of the 4-member standard partition (Amazon run 3); with the non-negative partition and sub-basin-coherent additive inflation it clipped real signal instead: the lower edge in the Danube droughts (2007, 2011–12) and the trend, the upper edge in the 2006 spring peak (20 % of cell-days). `gw_lower=True` / `gw_upper=True` restore either edge. River keeps its floor and envelope. Every clip is logged per month, sub-basin and bound in `threshold_monthly.csv` next to `threshold_log.json` in each `Ens_k` folder.
