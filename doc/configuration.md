# PyGLDA v2 — configuration guide

An experiment is described by three JSON files and a short driver script:

| file | purpose |
|---|---|
| `settings/<case>/DA_setting.json` | the assimilation case: folders, GRACE observations, state vector, filter |
| `settings/<case>/perturbation.json` | how the ensemble is generated (perturbation of forcing and model parameters) |
| `settings/<case>/Config_ReWaterGAP.json` | the WaterGAP 2.2e model: input paths, run options, variables written |
| `src_demo/global_run.py`, `src_demo/regional_run.py` | the drivers: case name, basin, ensemble size, periods, stages |

Each case folder is self-contained and holds all three files. The repository ships three cases:

| case | units | description |
|---|---|---|
| `GDA_Global` | 737 units of GlobalBasins v2.0 | global experiment (`src_demo/global_run.py`) |
| `RDA_Danube` | 6 sub-basins | regional experiment (`src_demo/regional_run.py Danube ...`) |
| `RDA_Amazon` | 18 sub-basins | regional experiment (`src_demo/regional_run.py Amazon ...`) |

## 0. Paths and the run copy of the settings

All paths in the JSON files start with `${DATA_ROOT}`, the folder that holds the input data and receives the
output (`Input_data`, `Basin`, `GRACE`, `Ensemble_input`, `Ensemble_Initialization`, `OL_output`, `DA_output`,
`Res`, `Auxiliary`; see `doc/installation.md`). The drivers take the data root from the environment variable
`PYGLDA_DATA`.

At the start of every stage the driver calls `RDA.prepare_settings(settings/<case>)`. It copies the three files
to `${DATA_ROOT}/settings/<case>/`, replaces `${DATA_ROOT}` by the data root of the machine, and writes the case,
basin, shapefile, mask, ensemble size and period of the driver into `DA_setting.json`. All stages read this run copy (`RDA.setting_dir`); the files in the repository are never
modified, and the run copy documents exactly what a run used.

Defaults: a key that is missing in `DA_setting.json` takes the default of `src_DA/configure_DA.py` (top-level keys)
or of the component class (keys inside a block). The settings files list only what differs from these defaults.

---

## 1. `DA_setting.json`

Four blocks: `basic`, `obs`, `model`, `method`.

### 1.1 `basic` — folders

| key | meaning |
|---|---|
| `res_permanent` | collected results: `Res/<case>/{OL,DA}/Ens_k/daily_output_<year>.nc`, `basin_ts_*.h5`, `Res_OL.h5`, `Res_DA.h5`, evaluation reports and figures |
| `OL_output_temp_dir` | temporary daily **global** output of the open loop, `OL_output/Ens_k/`. WaterGAP always runs on the global grid, so one open loop serves every case with the same ensemble and period. |
| `DA_output_temp_dir` | temporary daily output of the assimilation, `DA_output/<case>/Ens_k/`, plus the filter and threshold logs |
| `Ensemble_ini_dir` | restart states written by the spin-up and read by the open loop and the assimilation |
| `Ensemble_input_dir` | perturbed forcing and parameters of every member, written by `model_perturbation()` |
| `Auxiliary_dir` | `smax.nc` (maximum soil water content) and `state_envelope_<basin>.nc` (open-loop envelope, rebuilt at every DA start) |

Set by the drivers (not in the repository files): `fromdate`, `todate` (assimilation period), `ensemble` (number of
perturbed members; a run needs `ensemble + 1` MPI ranks: rank 0 is the unperturbed member `Ens_0`, rank 1 computes
the analysis), `case`, `basin`, `basin_shp`, `basin_mask`.

### 1.2 `obs` — GRACE observations

| key | meaning |
|---|---|
| `dir` | where the observation file of a run is written: `<dir>/<basin>_obs_GRACE.hdf5` (epochs, windows, unperturbed and perturbed unit TWS, covariance) |
| `GRACE` | the product, see below |

`obs.GRACE`:

| key | meaning |
|---|---|
| `kind` | `Mascon_monthly` (CSR RL06.3 mascons, monthly) or `TUD_5daily` (TU Delft 5-daily hybrid product, gridded EWH and per-cell uncertainty) |
| `EWH_grid_dir` | folder of the gridded product |
| `aux_for_time_epochs` | folder from which the observation epochs and windows are read (normally the same as `EWH_grid_dir`) |
| `cov_dir` | mascons: monthly Monte-Carlo error samples of a DDK3-filtered spherical-harmonic solution (1°), used as the error proxy of the mascons. Not used for `TUD_5daily`. |
| `preprocess_res` | output of `get_GRACE_obs()`: `<basin>_signal.hdf5` (unit TWS), `<basin>_gridded_signal.hdf5`, `<basin>_cov.hdf5` |
| `OL_mean` | open-loop mean per case (`<case>_<basin>.hdf5`), the datum to which the GRACE anomalies are referenced |
| `resample` | mascons: how the 0.25° grid is brought to the 0.5° model grid, `subsample` (every second cell, default) or `block_mean` (mean of the four cells, recommended) |
| `earthquake_correction` | mascons, optional (default off): co- and post-seismic signals removed from the grid before averaging (`src_OBS/earthquake_correction.py`). Within `cap_km` (1000) of each rupture of `events` (default: Sumatra 2004, Nias 2005, Maule 2010, Tohoku 2011) every cell is fitted with bias, trend, acceleration, annual and semi-annual terms, a step and post-seismic exponentials (relaxation times of Tanaka and Heki 2014), tapered over `taper_km` (200). `significance` `{"n_placebo": 10, "z_lo": 2.0, "z_hi": 4.0}`: the step is refitted at `n_placebo` fake dates and applied with the weight clip((\|S\|/σ_p − z_lo)/(z_hi − z_lo), 0, 1), so that hydrological signals are not mistaken for deformation. |
| `ewh_file`, `unc_file`, `unit` (`cm`), `corr_length_km` (300), `add_gia` (`false`) | `TUD_5daily` only: file names, unit of the grids, spatial error correlation length used when the unit covariance is propagated from the per-cell uncertainty (ρ = exp(−d/L)), and whether the static GIA field is added |

`get_GRACE_obs(is_diagonal=False)` of the regional driver keeps the full unit covariance; `is_diagonal=True` keeps
only its diagonal.

### 1.3 `model` — state vector

| key | meaning |
|---|---|
| `name` | `WaterGap` |
| `layer` | `storage: true/false` for all ten WaterGAP storages. TWS is always the sum of all ten; the `true` storages form the **state vector** and receive increments, the `false` ones are subtracted from the observation with each member's own value. Keep all ten keys, in this order. |

Storages (mm): `groundwstor`, `soilmoist`, `swe`, `locallakestor`, `localwetlandstor`, `globallakestor`,
`globalwetlandstor`, `riverstor`, `reservoirstor`, `canopystor`. The shipped cases update groundwater, soil water,
snow, the two wetland storages and river storage; lakes, reservoirs and canopy respond to the corrected storages
through the model. Lakes can be added to the state: WaterGAP's lake floor (−maximum storage) is then used as their
lower bound.

### 1.4 `method` — the filter

| key | default | meaning |
|---|---|---|
| `fusion_method` | `EnKF_localized` | `EnKF_localized`: stochastic EnKF with perturbed observations. `LETKF`: deterministic local ensemble transform (Hunt et al. 2007) with the same components; it replaces only the raw increment K(y − HX), using the unperturbed observation and R-localization from the taper. `"letkf": {"min_taper": 0.0}` drops observations with a smaller taper weight from a cell's local set. |
| `localization` | `{"kind": "block"}` | see below |
| `inflation` | `{"kind": "none"}` | see below |
| `increment_partition` | `{"kind": "enkf"}` | see below |
| `obs_error_inflation` | `1.0` | factor on the observation-error variance of every unit, R' = f R |
| `obs_error_correlation` | `full` | `full`: R as delivered. `diagonal`: R without correlations between units, and the member observations are perturbed consistently with independent draws from N(0, diag R) (fixed seed). |
| `obs_perturbation_centering` | `false` | `true`: the observation perturbations of the members are shifted to zero mean, so the ensemble-mean observation is exactly the GRACE value |
| `soil_upper_bound` | `true` | soil ≤ smax (`Auxiliary_dir/smax.nc` × land fraction × the member's smax multiplier) inside the bound-aware partition: the water soil cannot hold goes to the other storages of the unit instead of being clipped after the update |
| `soil_daily_bounds` | `false` | `true`: the soil bounds act on the window mean, and the window-mean soil increment δ is distributed over the days in proportion to what each day can take (δ > 0: the room to capacity, δ < 0: the water available), so that every day stays within [0, smax] and mean(d_t) = δ. Without it every day gets the same increment, and a soil that touches a bound on one day blocks the increment of the whole window. |
| `increment_release` | off | `{"storages": ["riverstor"], "days": 10}`: the increment of the listed (fast) storages is not put into the restart state; it is added in equal parts on the first `days` days of the next window (incremental analysis update). The stored daily analysis keeps the full increment. Avoids a discharge pulse at every window start. |
| `snow_bounds` | see below | limits of the analysed snow |

Analysis of one window (`src_DA/EnKF_localized.py`):

    A  = inflation.prior(X_f − mean)
    K  = (L_state ∘ Pxy)(L_obs ∘ Pyy + R')⁻¹
    dX = partition.split(K (y − HX))
    Xa = inflation.posterior(X + dX)

The increments are computed for the window means and added to every day of the window; the physical bounds hold on
every day of the window (window-aware bounds, `src_DA/bounds.py`), and the post-update threshold
(`src_DA/Threshold.py`) clips what is left. Any localization can be combined with any inflation and partition.

#### `localization`

| kind | settings (default) | meaning |
|---|---|---|
| `none` | — | plain sample covariances; every observation updates every cell |
| `block` | — | each unit observation updates only the cells of its own unit |
| `gaussian` | `length_km` (300), `cutoff` (2.0) | weight exp(−½(d/ℓ)²) with d the distance of a cell to the unit (0 inside), 0 beyond `cutoff · length_km` |

#### `inflation` (one scheme)

| kind | settings (default) | meaning |
|---|---|---|
| `none` | — | no spread maintenance |
| `multiplicative` | `factor` (1.0) | forecast anomalies multiplied by `factor` before the gain (Anderson and Anderson 1999) |
| `rtps` | `alpha` (0.7) | relaxation to prior spread, element-wise (Whitaker and Hamill 2012), limited so that no member crosses a bound |
| `additive` | `sigma` ({storage: std in mm}), `seed` | zero-mean random perturbations added before the gain, one number per member, unit and storage, centred over the members. Unbounded storages (groundwater) get the same value in every cell of the unit; bounded storages are weighted by the room to the nearer bound (empty cells stay empty) and limited to 50 % of the storage. The perturbations stay in the members (model error). |
| `adaptive_additive` | `split_storages` (all state storages), `weight_current` (0.5), `max_factor` (3.0), `min_sigma` (5.0), `remove_bias` (`true`), `bias_memory` (0.05), `seed` | as `additive`, but the std of the added noise is estimated every window and unit from the innovations: s² = (1−w) s²_prev + w (d − b)², σ = clip(√(s² − var(HX) − R), `min_sigma`, `max_factor` √R), with d the innovation of the ensemble mean and b its slowly varying mean (bias memory). The added variance is shared among `split_storages` per calendar month and unit in proportion to the open-loop ensemble variance of each storage (computed once from `Res/<case>/Res_OL.h5`, so the open loop must be collected first). Variance a bounded storage cannot take goes to the unbounded ones. |

#### `increment_partition`

| kind | meaning |
|---|---|
| `enkf` | increments as given by the (localized) Kalman gain |
| `non_negative` | sign-consistent partition: the unit increment δ_j = (H dX)_j of the Kalman update is kept exactly, but distributed with shares ∝ variance × max(correlation with the unit TWS, 0), so that no storage is emptied to fill another. Bound-aware: an element is not pushed across its window-aware bound; its share goes to the other elements of the unit. |

Options of `non_negative` (default off):

| option | example | meaning |
|---|---|---|
| `max_weight_ratio` | `{"groundwstor": 5}` or `5` | caps an element's weight at this multiple of the median weight of the same storage among the active elements of the unit, so that one cell with an extreme ensemble spread cannot absorb the correction of the whole unit |
| `spatial_blend` | `{"groundwstor": 0.8}` | w' = (1−b) w + b mean(w) over the active elements of the storage in the unit; the storage's total share is unchanged. GRACE has no information below the unit scale. Meant for unbounded or sparse storages (groundwater, river, wetlands). |
| `max_cell_factor` | `{"soilmoist": 2}` or `3` | no element may change by more than f × the unit increment of its member, \|dX_i\| ≤ f \|δ_j\|; the excess is redistributed like a bound |
| `storage_shares` | `{"storages": ["soilmoist", "groundwstor"], "mix": 1.0}` | the total share of the listed storages in a unit increment is kept as the ensemble gives it, but re-split among them in proportion to each storage's seasonal amplitude in the open loop of that unit (`Res/<case>/Res_OL.h5`), s' = (1 − mix) s_ens + mix s_OL. Within a storage the cells keep their ensemble weights. GRACE sees only the sum; the ensemble's soil/groundwater covariance follows how the inflation noise is split rather than where the model's seasonal error is. |
| `boundary_smoothing` | `{"storages": ["groundwstor"], "length_km": 50}` | with one observation per unit, the increment field is built inside each unit from that unit's increment and steps at unit boundaries. Mass-preserving (pycnophylactic, Tobler 1979) smoothing of the listed storages: `n_iter` (10) passes of a Gaussian kernel of `length_km` (50, zero beyond `cutoff` (2.0) × `length_km`), each followed by restoring the unit mean of the storage; `alpha` (1.0) blends with the unsmoothed field. H dX = δ is kept exactly. Meant for unbounded storages: smoothing bounded storages moves water into cells at their limits. |

#### `snow_bounds`

| entry | limit | default |
|---|---|---|
| `lower_factor` f | snow ≥ f × the member's forecast (one window removes at most 1 − f of a pack) | 0.5 |
| `upper_factor` f, `upper_offset_mm` o | snow ≤ f × forecast + o | 2.0, 20 mm |
| `envelope_factor` f, `envelope_offset_mm` o | snow ≤ f × max_OL(cell, calendar month) + o, the largest snow of that month in the open loop | off, 10 mm |

`null` or `false` switches a limit off. All limits are window-aware, act inside the bound-aware partition and never
remove snow a member already has in its forecast. The envelope cap stops the slow growth of packs that never melt.

#### Output of the filter

`DA_output/<case>/filter_summary.json` (one section per component) and, for `adaptive_additive`,
`adaptive_inflation_log.csv` (one row per window and unit: innovation, bias, forecast spread, observation error,
added spread). Every member writes `threshold_log.json` and `threshold_monthly.csv` (what the post-update threshold
clipped, per storage, month and unit).

#### Post-update threshold (`src_DA/Threshold.py`)

A safety net after the update: soil 0 ≤ S ≤ smax; snow ≥ 0 (and the envelope cap, if on); river ≥ 10⁻³ mm and
within [0.5 min_OL, 1.5 max_OL] of the open-loop envelope; canopy, wetlands, reservoirs ≥ 0; groundwater and lakes
unbounded. With the `non_negative` partition and `soil_upper_bound` the update already respects these limits, so
little is clipped.

---

## 2. `perturbation.json` — ensemble generation

Read by `model_perturbation()` (`src_DA/perturbation_par.py` for the parameters, `src_DA/perturbation_forcing.py`
for the forcing).

| key | meaning |
|---|---|
| `ensemble` | number of perturbed members (the drivers pass their own value) |
| `seed` | integer: reproducible random numbers; `null`: numpy's global generator |
| `dir.in`, `dir.out` | unperturbed WaterGAP input (`Input_data`) and the output folder (`Ensemble_input`) |
| `correlation.forcing` | `isSpatialCorrelated`, `isTemporalCorrelated`, `length_km`, `time_scale_days`: one random field per member, variable and day with the Gaussian spatial correlation exp(−(r/L)²) and an AR(1) in time |
| `correlation.par.isSpatialCorrelated` | spatially correlated parameter perturbation |
| `forcing.<var>` | `pr`, `tas`, `rsds`, `rlds`: `is_perturbed`, `perturb_method` (`multiplicative` / `additive`), `error_distribution` (`gaussian`, `triangle`, `lognormal`), `coefficients` (std; coefficient of variation for the lognormal precipitation factor) |
| `par.<parameter>` | `is_perturbed`, `type` (`value`: one global value replaced per member; `multiplier`: factor on a cell field or on a quantity WaterGAP derives), `error_distribution` (`triangle` [min, mode, max], `uniform` [min, standard, max], `gaussian` [mean, std]), `coefficients`. The middle coefficient must equal the standard value. Drawn once per member, constant in time and space. |

The parameters PyGLDA can perturb follow Döll et al. (2024, Table 1). `smax_multiplier` must be one value per
member (the soil bound of the filter scales the soil capacity by it). A change here requires a new perturbation,
spin-up, open loop and open-loop collection.

---

## 3. `Config_ReWaterGAP.json` — the WaterGAP model

| key | meaning |
|---|---|
| `FilePath.inputDir.*` | climate forcing, water use, static land data, global parameter file |
| `FilePath.outputDir` | WaterGAP's own output folder (PyGLDA writes `OL_output` / `DA_output` instead) |
| `RuntimeOptions.SimulationOption.AntNat_opts` | `ant: true`: anthropogenic run with water use (`subtract_use`) and reservoirs (`res_opt`) |
| `RuntimeOptions.RestartOptions` | controlled by PyGLDA at run time; `save_and_read_states_dir` is WaterGAP's restart folder |
| `RuntimeOptions.SimulationPeriod` | overwritten by the driver (spin-up and simulation periods, `spinup_years`) |
| `RuntimeOptions.SimulationExtent.run_basin` | must be `false`: WaterGAP runs globally, the basin is applied when the results are collected |
| `OutputVariable.*Storages` | every storage of `model.layer` (true or false) must be written, because TWS is their sum |

---

## 4. Drivers and stages

The drivers set the case on the class `RDA` (`GDA` for the global case) and run the stages:

| attribute | meaning |
|---|---|
| `case`, `basin`, `shp_path`, `ens` | case name, basin / unit-set name (mask `Basin/mask/<basin>/<basin>_res_0.5.h5`), shapefile with an integer column `ID` = 1..N, number of perturbed members |
| `external_data_path` | the data root |
| `spin_up_start`, `spin_up_end`, `spin_up_years` | spin-up: the first year repeated `spin_up_years` times, then the whole period |
| `sim_begin_time`, `sim_end_time` | open loop and assimilation period |

| stage | call | processes |
|---|---|---|
| settings | `prepare_settings(settings/<case>)` | one |
| unit set / mask | `GDA.check_units()` / `RDA.config_basin_mask()` | one |
| GRACE observations | `get_GRACE_obs()` | one |
| perturbation, spin-up, open loop | `model_perturbation()`, `spin_up()`, `OL_run()` | `ens + 1` MPI ranks |
| open-loop collection | `collect_and_statistics(Stage.OL)` | `ens + 1` |
| assimilation | `DA_run()` | `ens + 1` |
| assimilation collection | `collect_and_statistics(Stage.DA)` | `ens + 1` |
| evaluation | `da_evaluation(tag)`, `visualization(...)` | one |
| house-keeping | `clean_temp_output(Stage.DA, dry_run=True)` | one |

Logs of the MPI stages are written to `parallel_logs/<stage>/rank_k.log`; one rank prints to the terminal.
