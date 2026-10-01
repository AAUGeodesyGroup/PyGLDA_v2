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
* `soilmoist` and `groundwstor` always; `swe` in snow-affected basins (Danube), not needed in the tropics.
* lakes, wetlands, reservoirs, canopy: leave `false` for now. Their member spread is a fixed offset from the
  depth perturbation and GRACE cannot resolve them separately; see §2 for the consequence on the ensemble.

Physical bounds applied after every update (`src_DA/Threshold.py`, not configurable from JSON): soil 0 ≤ S ≤ smax;
snow 0–1000 mm; river ≥ 1e-3 mm and inside the open-loop envelope [0.5·min, 1.5·max]; groundwater inside the OL
envelope widened by half its range (signed storage); canopy / wetlands / reservoirs ≥ 0; lakes unbounded. The
envelope `state_envelope_<basin>.nc` is recomputed from `Res/<case>/OL` at every DA start. Clipping statistics
are printed per member and saved to `DA_output/<case>/Ens_k/threshold_log.json`.

### 1.4 `method` — the filter

| key | type / values | default | meaning |
|---|---|---|---|
| `fusion_method` | `EnKF_v0`, `EnKF_v1`, `EnKF_v2`, `EnKF_localized` | `EnKF_v0` | `EnKF_v0`: stochastic EnKF with perturbed observations, no localisation (fine for large ensembles). `EnKF_localized`: the same update with the options below; recommended for ≤ 10 members. |
| `localization.kind` | `block`, `gaussian` | `block` | taper of the state–observation covariance. `block`: each sub-basin observation updates only the cells of its own sub-basin (taper 1 inside, 0 outside); the strongest choice, right for small ensembles. `gaussian`: weight exp(−½(d/ℓ)²) with d the distance of a cell to the observed sub-basin (0 inside), so boundary cells are also informed by the neighbour's observation; use with ≥ 20 members. |
| `localization.length_km` | float | 300 | ℓ of the gaussian taper (ignored for `block`) |
| `localization.cutoff` | float | 2.0 | gaussian weight set to 0 beyond `cutoff · length_km` |
| `inflation` | float ≥ 1 | 1.0 | multiplicative inflation of the forecast anomalies before the update (1 = off). Prefer RTPS. |
| `rtps_alpha` | 0 … 1 | 0.7 | relaxation to prior spread (Whitaker & Hamill 2012): after the update the spread of every state element is set to α·σ_forecast + (1−α)·σ_analysis. 0 = off; 0.7 stopped the spread collapse of the 4-member Amazon run (χ² 5.9 → 0.5). |
| `obs_error_inflation` | `{ "k": factor }` | `{}` | multiply the observation error **variance** of sub-basin k (1-based, as string) by `factor`, keeping the error correlations (R' = D R D). Use to down-weight a unit whose observation is suspect (e.g. coastal leakage); `{}` = off. |
| `increment_partition` | `enkf`, `non_negative` | `enkf` | how the sub-basin TWS increment is split over the cells and storages of the sub-basin. `enkf`: as given by the Kalman gain (vertical split from the sample covariance; with few members this produces compensating increments, e.g. +GW / −river, and a filter–model tug of war). `non_negative`: the sub-basin TWS increment of the Kalman update is kept exactly, but distributed with shares ∝ variance × max(correlation with sub-basin TWS, 0); no storage is emptied to fill another, single-element increments are bounded. Recommended for small ensembles; compare with `enkf` once the ensemble is ≥ 20. |

Typical blocks:

```json
"method": {                                  // large ensemble, classic
    "fusion_method": "EnKF_v0"
}

"method": {                                  // small ensemble (Amazon run 4, Danube demo_2)
    "fusion_method": "EnKF_localized",
    "localization": {"kind": "block", "length_km": 300, "cutoff": 2.0},
    "inflation": 1.0,
    "rtps_alpha": 0.7,
    "obs_error_inflation": {},
    "increment_partition": "non_negative"
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
  "method": { "fusion_method": "EnKF_localized", "localization": {"kind": "block", "length_km": 300, "cutoff": 2.0},
              "inflation": 1.0, "rtps_alpha": 0.7, "obs_error_inflation": {}, "increment_partition": "non_negative" }
}
```

**demo_2 — Danube, CSR mascons monthly, 6 sub-basins**: identical structure with `case: demo_2`, `basin: Danube`,
`basin_shp: .../Basin/shp/Danube/Danube.shp`, `obs.GRACE.kind: Mascon_monthly`,
`EWH_grid_dir` = `aux_for_time_epochs` = `.../GRACE/SaGEA/signal_Mascon`, `preprocess_res: .../GRACE/output`,
`layer.swe: true`, same `method` block.

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
