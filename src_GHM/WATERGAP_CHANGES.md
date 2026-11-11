# Changes to the original WaterGAP code (ReWaterGAP) in PyGLDA v2

Last update: 7 Nov 2026 (Cowork). Keep this file up to date whenever a file under `src_GHM/ReWaterGAP/` is changed, or when a WaterGAP module is copied into `src_GHM/Extension/`.

## Reference version
- Upstream: https://github.com/HydrologyFrankfurt/ReWaterGAP, commit **2ebcdd1** (14 Feb 2026, "Add files via upload"). The copy in `src_GHM/ReWaterGAP/` was taken on about 15 May 2026.
- Check: all files of `src_GHM/ReWaterGAP/` that were not touched afterwards are identical to 2ebcdd1 (line endings ignored). Upstream has changed since then, up to Sept 2026, e.g. the forcing, water-use and static-data handlers. Those upstream changes are **not** included here.
- Unified diffs of every change against 2ebcdd1 are in `doc/watergap_changes/diffs/`.

## A. Files changed in place in `src_GHM/ReWaterGAP/`

| file | md5 (now) | change | effect on a standard WaterGAP run |
|---|---|---|---|
| `model/lateralwaterbalance/lakes_wetlands.py` | d246193c (upstream 987dbaf6) | **7 Nov 2026, DA review 246.** New njit helper `_limit_ratio(petgwr_max, petgwr)`. It replaces the 7 divisions `petgwr_netabs_sw_max / petgwr_netabs_sw` in the limit branch of `lake_wetland_water_balance`, and returns 0 when `petgwr_netabs_sw <= 0` and the ratio clipped to [0, 1] otherwise. **Why:** a lake or wetland written back by the DA at, or a round-off below, its floor (−max_storage or 0) has zero area, so `petgwr = 0` and `petgwr_max < 0`. The original then divides 0 / 0 (`ZeroDivisionError` in numba). This was the cause of the V2lk / V2lk2 / V2lk3 crashes of 24 Oct 2026. | **None.** In a free run the storage never lies below its floor, so `petgwr_max >= 0`, the branch implies `petgwr > 0`, and the ratio is the same number. Tested: 4 000 random free-run calls (local / global lake, local / global wetland, limit branch reached) give bit-identical outputs. The original raises on the DA cases and the guarded version returns the floor (`diffs/test_lakes_wetlands_guard.py`). |
| `model/lateralwaterbalance/waterbalance_lateral.py` | fa4e5eb4 | 10 Oct 2026. Potential net abstraction from groundwater and surface water is multiplied by the ensemble multipliers `netabs_gw_multiplier` / `netabs_sw_multiplier` from the parameter file. | None: a multiplier of 1.0 is used when it is absent (see `parameters.py`). |
| `model/parameters.py` | 65651713 | 10 Oct 2026. Tuple `MULTIPLIERS` (lai, degree-day, smax, gw recharge factor, max gw recharge, net abstraction gw / sw). Any multiplier missing from the parameter file is added as 1.0 everywhere. The multipliers are written per ensemble member by `src_DA/perturbation_par.py`. | None with the original parameter file (all 1.0). |
| `model/verticalwaterbalance/waterbalance_vertical.py` | 1475b28f | 5 Oct 2026. `snow_water_storage_subgrid_out = snow_water_storage_subgrid` instead of `.copy()`. `snow_water_balance` updates each cell's sub-grid storage in place, so the output is the input array itself; the copy cost 207 MB per day. | None (same values; memory only). |
| `model/land_surfacewater_fraction.py` | 89fb4a21 | New function `update_setting()` that re-reads `cm.ant` and `cm.RESERVOIR_OPT` into the module globals, for runs whose configuration is set from Python. | None (only adds a function). |
| `controller/configuration_module.py` | f413f125 | The module-level configuration reading is wrapped in `init_config(config_input=None)`, which accepts a file name or a dict and sets the same module globals. It is still called once at import with the command-line file, as before. | None (same globals and values; allows PyGLDA to configure WaterGAP from Python). |
| `Config_ReWaterGAP.json` | bbcb024b | Local run settings, not code: restart true, period 2002-01, reservoir years 2002, no spin-up, fewer output variables. PyGLDA writes its own run copies of the configuration. | n/a |

Files whose time stamp changed but whose content equals 2ebcdd1: `run_watergap.py`, `controller/climateforcing_handler.py`, `controller/read_forcings_and_static.py`, `model/lateralwaterbalance/routing.py`, `model/verticalwaterbalance/waterbalance_vertical_init.py`, `view/createandwrite.py`.

## B. Modified copies of WaterGAP modules in `src_GHM/Extension/` (originals unchanged)

| PyGLDA file | md5 | derived from (2ebcdd1) | what differs |
|---|---|---|---|
| `fan_waterbalance_vertical_init.py` | 37c439af | `model/verticalwaterbalance/waterbalance_vertical_init.py` | Applies the ensemble multipliers: LAI (min and max), degree-day factor, Smax, groundwater recharge factor, maximum groundwater recharge. Forcing passed as DataArrays instead of Datasets. Caches `_per_contfrac` for the DA write-back of the vertical storages. |
| `fan_climateforcing_handler.py` | 5cfac829 | `controller/climateforcing_handler.py` | Reads monthly forcing files `YYYY-MM.nc` restricted to the simulation period (`load_nc_range`), with a daily `update(date)`. |
| `fan_read_forcings_and_static.py` | 51e7f2d4 | `controller/read_forcings_and_static.py` | Uses `fan_ClimateForcing`. |
| `fan_createandwrite.py` | bebd9ee6 | `view/createandwrite.py` | Optional crop to a bounding box; `save_netcdf_daily_single_file` (one file per day, zlib level 1, 36 × 72 tiles). |
| `fan_data_output_handler.py` | b8ad044f | `view/data_output_handler.py` | Keeps only the current day in memory (no time dimension). |
| `fan_restart_watergap.py` | e89a5565 | `model/utility/restart_watergap.py` | Separate read and save paths for restart states (ensemble members). |
| `fan_run_watergap.py` | 7a6153b3 | `run_watergap.py` | Stand-alone run with the fan_* modules and daily single-file output. The ensemble hot start is `src_GHM/Interface/Hotrun.py`. |

`src_GHM/Interface/` (DailyStepRun, Hotrun, Spinup, UnitConverter) is PyGLDA's own coupling code, not WaterGAP code.

## B2. Changes to WaterGAP static input data (proposed, not used by any run yet)

| Input | Change | Status |
|---|---|---|
| `land_water_fractions/watergap_22e_global_lake_area.nc`, `watergap_22e_glolak.nc4` (`PyGLDA_v2_external_data/Input_data/static_input/`) | **Aral Sea area (F3, DA reviews 243 / 256).** GLWD unit 2: outflow-cell area 66,114.125 km² (≈ the 1960s lake) → 17,160 km² (2004; primary) or 10,000 km² (2015; sensitivity), and glolak of the 48 unit-2 cells scaled by the same factor (uniform). contfrac unchanged, so the removed lake fraction becomes land. Max storage follows (area × 5 m). All other cells and files unchanged. Patched copies in `Input_data/static_input_aral/{a17160,a10000}/` with `make_aral_static.py` and README (md5, sources: Jin et al. 2017, Remote Sens. 9, 900). The originals are never overwritten. | Prepared 7 Nov 2026; open-loop test pending (review 256). A declared model (input) change: free runs are NOT bit-identical. |

## C. Rules
- Any change in `src_GHM/ReWaterGAP/` must leave a standard (non-DA) WaterGAP run bit-identical, unless the change is a declared model change. It needs a test, an entry in this file, and its diff in `doc/watergap_changes/diffs/`.
- Model changes requested by the DA (bounds, guards) belong in the DA code where possible. They go into WaterGAP only for numerical guards like `_limit_ratio`.
