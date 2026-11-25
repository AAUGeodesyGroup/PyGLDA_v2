# PyGLDA v2 — settings reference

Every key of `DA_setting.json` that the code reads, with its allowed values and its default. A key that is
missing takes the default. Each default is defined once in the code: top-level keys in `src_DA/configure_DA.py`
(`config_basic`, `config_obs`, `config_model`, `config_method`, `GRACE_DEFAULTS`), keys inside a block in the
component that reads it. `doc/configuration.md` explains the methods in more detail. Paths start with
`${DATA_ROOT}`; `prepare_settings` replaces it by the data root (`PYGLDA_DATA`) in the run copy.

Settings written for an older code version: `python -m src_auxiliary.translate_settings --check <old file or folder>`
(in v2.0 no key was removed or renamed, so older settings are read as they are).

## basic

| key | meaning | values | default |
|---|---|---|---|
| `fromdate`, `todate` | assimilation period | `YYYY-MM-DD` | set by the driver |
| `ensemble` | number of perturbed members (`ensemble + 1` MPI ranks) | integer | set by the driver |
| `case`, `basin`, `basin_shp`, `basin_mask` | case name, unit set, its shapefile and 0.5° mask | text / path | set by the driver |
| `res_permanent` | collected results `Res/` | path | required |
| `OL_output_temp_dir`, `DA_output_temp_dir` | daily output of the open loop / assimilation | path | required |
| `Ensemble_ini_dir`, `Ensemble_input_dir` | restart states / perturbed inputs of the members | path | required |
| `Auxiliary_dir` | `smax.nc`, state envelope | path | required |
| `DA_restart_dir` | where a DA run saves its end states | path | `null` = `<DA_output_temp_dir>/<case>/restart` |

## obs

| key | meaning | values | default |
|---|---|---|---|
| `dir` | folder of the observation file of a run | path | required |
| `withhold` | GRACE epochs not assimilated (held-out validation) | `{"every": n, "offset": k}` or `{"epochs": [...]}` | `null` (all assimilated) |
| `GRACE.kind` | GRACE product | `Mascon_monthly`, `SH_monthly` (monthly product filtered and gridded beforehand, e.g. a spherical-harmonic solution processed with SaGEA, as ONE CF NetCDF in `EWH_grid_dir`: `time`, `lat`, `lon`, EWH variable `nc_var` in mm/cm/m on the 0.5° grid, optional `time_bounds`; error samples in `cov_dir` and `earthquake_correction` as for the mascons; `resample`, `lake_leakage` ignored), `TUD_5daily` (regional only) | `Mascon_monthly` |
| `GRACE.EWH_grid_dir` | gridded product | path | required |
| `GRACE.aux_for_time_epochs` | folder from which epochs and windows are read | path | required |
| `GRACE.cov_dir` | mascons: monthly error samples (not used by `TUD_5daily`) | path | required for mascons |
| `GRACE.preprocess_res` | unit signal, gridded signal and covariance of `get_GRACE_obs` | path | required |
| `GRACE.OL_mean` | open-loop mean per case (datum of the GRACE anomalies) | path | required |
| `GRACE.resample` | mascons 0.25° → 0.5° | `subsample`, `block_mean` | `subsample` |
| `GRACE.earthquake_correction` | mascons and `SH_monthly`: co-/post-seismic signals removed | `{"cap_km", "taper_km", "events", "significance": {"n_placebo", "z_lo", "z_hi"}}` | `null` (off); inside: 1000, 200, the four events listed in `doc/configuration.md`, `significance` `null` (no placebo test; values inside default to 10, 2.0, 4.0) |
| `GRACE.lake_leakage` | mascons: leakage of water bodies outside the model domain subtracted | npz path (`months`, `leak_mm`) | `null` (off) |
| `GRACE.nc_file`, `nc_var` | `SH_monthly`: NetCDF file name in `EWH_grid_dir` and EWH variable | text | `null` (the only `*.nc` there), `tws` |
| `GRACE.ewh_file`, `unc_file` | `TUD_5daily`: file names | text | the TU Delft v2 hybrid files |
| `GRACE.unit`, `corr_length_km`, `add_gia` | `TUD_5daily`: grid unit, error correlation length, static GIA | `cm` / km / bool | `cm`, 300, `false` |

## model

| key | meaning | values | default |
|---|---|---|---|
| `name` | hydrological model | `WaterGap` | `WaterGap` |
| `layer.<storage>` | storage in the DA state (`true`) or subtracted from GRACE (`false`); all ten keys | bool | all `true` |

Storages: `groundwstor`, `soilmoist`, `swe`, `locallakestor`, `localwetlandstor`, `globallakestor`,
`globalwetlandstor`, `riverstor`, `reservoirstor`, `canopystor`.

## method

| key | meaning | values | default |
|---|---|---|---|
| `fusion_method` | filter | `EnKF_localized`, `LETKF` | `EnKF_localized` |
| `letkf.min_taper` | LETKF: smallest taper weight of a local observation | 0–1 | 0.0 |
| `localization.kind` | which cells an observation updates | `none`, `block`, `gaussian` (`length_km` 300, `cutoff` 2.0) | `block` |
| `inflation.kind` | spread maintenance | `none`, `multiplicative` (`factor` 1.0), `rtps` (`alpha` 0.7), `additive` (`sigma`, `seed`), `adaptive_additive` | `none` |
| `inflation` (adaptive) | `split_storages`, `split`, `weight_current`, `max_factor`, `min_sigma`, `remove_bias`, `bias_memory`, `seed` | `split`: `ol_spread`, `ol_seasonal` | all state storages, `ol_spread`, 0.5, 3.0, 5.0 mm, `true`, 0.05, `null` |
| `increment_partition.kind` | how a unit increment is spread | `enkf`, `non_negative` | `enkf` |
| `obs_error_inflation` | factor on the observation-error variance | > 0 | 1.0 |
| `obs_error_correlation` | error correlation between units | `full`, `diagonal` | `full` |
| `obs_perturbation_centering` | member perturbations shifted to zero mean | bool | `false` |
| `soil_upper_bound` | soil ≤ smax inside the partition | bool | `true` |
| `soil_daily_bounds` | every day of the window within the soil bounds | bool | `false` |
| `increment_release` | increment of fast storages released over the first days of the next window | `{"storages": [...], "days": n}` | `null` (off); inside: `["riverstor"]`, 5 |
| `snow_bounds` | snow limits | `lower_factor`, `upper_factor`, `upper_offset_mm`, `envelope_factor`, `envelope_offset_mm` (`null` = limit off) | 0.5, 2.0, 20, `null`, 10 |
| `wetland_bounds.upper` | wetlands ≤ WaterGAP's maximum wetland storage | bool | off |
| `window_scaling.reference_days` | per-window constants scaled for windows shorter than a month | days | off (30.44 when on) |
| `log_applied_increment` | monthly sums of the applied increment (diagnostic) | bool | `false` |

Options of `increment_partition` with `kind` `non_negative` (all off unless given):

| key | meaning | values / defaults inside |
|---|---|---|
| `max_weight_ratio` | cap of an element's weight, multiple of the storage's median weight in the unit | number or `{storage: number}` (> 1) |
| `spatial_blend` | weights mixed with their unit mean | `{storage: b}`, 0 < b ≤ 1 |
| `max_cell_factor` | no element changes by more than f × the unit increment | number or `{storage: f}` (≥ 1) |
| `storage_shares` | storage split from the open loop's seasonal amplitude | `storages` (`["soilmoist", "groundwstor"]`), `mix` (1.0) |
| `boundary_smoothing` | mass-preserving smoothing across unit boundaries | `storages` (required), `length_km` (50), `n_iter` (10), `alpha` (1.0), `cutoff` (2.0) |
| `bound_overflow` | storages that take no overflow of bounded elements | `exclude` (required), `fallback` (`false`) |
| `seasonal_split` | seasonal part of the mean increment kept out of slow storages | `storages_slow` (`["groundwstor"]`), `method` (`lowpass` / `climatology`, default `lowpass`), `tau_months` (12), `memory_years` (3), `fraction` (1.0), `init`, `save` (climatology only); a missing `init` file stops the run at start. `settings/GMD_Global` reads `Auxiliary/seasonal_split_init/GMD_Global_GlobalBasins_T9.npz`, which comes with the external-data package (Zenodo) |
| `hotspot_redistribution` | part of the groundwater increment moved to pumping hotspots | `field` (npz `pnag_mm_yr`, required), `threshold_mm_yr` (50), `p_ref_mm_yr` (20), `fraction` (1.0), `storage` (`groundwstor`) |
| `unit_storage_mask` | a storage receives increments only in the listed units | `{storage: npz with unit_flag (n_units,)}` |

## perturbation.json and Config_ReWaterGAP.json

`perturbation.json` (ensemble generation, `src_DA/perturbation_par.py`, `src_DA/perturbation_forcing.py`):
`ensemble` (overridden by the driver), `seed` (`null` = numpy's global generator), `dir.in` / `dir.out`,
`correlation.forcing` (`isSpatialCorrelated`, `isTemporalCorrelated`, `length_km`, `time_scale_days`, `customize`),
`correlation.par` (`isSpatialCorrelated`, `customize`), `forcing.<pr|tas|rsds|rlds>` and `par.<parameter>`
(`is_perturbed`, `perturb_method` / `type`, `error_distribution`, `coefficients`). `customize: true` calls the empty
hook `_perturb_*_customize` for user code.

`Config_ReWaterGAP.json` is WaterGAP's own configuration (see `doc/configuration.md` §3); PyGLDA overwrites the
simulation period and the restart options at run time.

## Sensitivity options

Options of the design and sensitivity experiments of the WRR and GMD papers. All are off unless set, and the default
templates do not use them. Each is described in the docstring of the module that reads it.

| key | what it does | status |
|---|---|---|
| `obs.operator` (`mass_consistent`, `contfrac_file`, `contfrac_var`) | continental-fraction weights in the unit means (`src_DA/obs_operator.py`); the open loop must be collected with the same operator | sensitivity option; not used in the default configuration |
| `method.innovation_bias` (`tau_months`, `fraction`, `mode`, `freeze`, `init`, `save`) | bias-aware reference: the running mean innovation per unit is removed (`src_DA/EnKF_localized.py`) | sensitivity option; not used in the default configuration |
| `method.reservoir_state` (`schedule`, `log`) | reservoirs in the state only during their filling years (`src_DA/reservoir_state.py`) | sensitivity option; not used in the default configuration |
| `method.excluded_storage` (`mode`, `clim`, `layers`, `member_deviation`, `log`) | climatology instead of the member value for storages outside the state (`src_DA/EnKF.py`) | sensitivity option; not used in the default configuration |
| `increment_partition.positive_to` (`mask`, `from`, `to`, `min_share`) | positive soil / wetland increments moved to groundwater in masked units | sensitivity option; not used in the default configuration |
| `increment_partition.storage_shares.residence` (`tau_ref_days`, `tau_min_days`, `tau_max_days`) | storage shares weighted by the open loop's residence times | sensitivity option; not used in the default configuration |
| `increment_partition.seasonal_split.interannual` (`tau_months`, `fraction`) | second low-pass split of the non-seasonal remainder | sensitivity option; not used in the default configuration |
| `increment_partition.bound_overflow.fallback_from` | with `fallback: false`, upper-bound overflow of the listed storages goes to the excluded storages | sensitivity option; not used in the default configuration |
