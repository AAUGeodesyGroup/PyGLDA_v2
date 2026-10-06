<!-- Copy of the claude.ai project note "claude/ucloud_transfer_notes.md" (project "PyGLDA V2"), written by the Cowork session on the workstation so that the UCloud Claude Code session can read it (synced via Syncthing). The project note is the master copy; this file is refreshed when the note changes. Last copied: 2026-10-05 22:50 UTC -->

# PyGLDA v2 → UCloud: code map and transfer notes (4 Oct 2026, updated 5 Oct evening)

Code: `/media/user/My Book/Fan/PyGLDA_v2` (also on GitHub, AAUGeodesyGroup/PyGLDA_v2), synced with UCloud via Syncthing. External data on UCloud: `/work/PyGLDA_v2_external_data` (workstation copy: `/media/user/My Book/Fan/PyGLDA_v2_external_data`). Fan reaches UCloud from a terminal (SSH); a Claude Code session also runs on UCloud and launches jobs. UCloud quota: 7 TB.

## Status (5–6 Oct 2026)
Case **`demo_Danube`**: 30 members, 2002-01-01 – 2019-12-31 (GRACE + GRACE-FO, gap 2017-07 – 2018-05), method = run 11a2 baseline. Done on UCloud: config_external_data, read_config_and_save, get_GRACE_obs, model_perturbation (31 members, 2000-01 – 2019-12; NOTE: this run used the old SERIAL perturbation, 4 h 40 min — the MPI version is untested on UCloud and will first be used for the next case), spin-up (57 min), OL (2 h 23 min, ~1.3 s/model day; collect 16 min), DA **run12** (2 h 43 min; collect 10 min), post-processing, da_evaluation run12 (report in `Res/demo_Danube/evaluation/run12/`, analysis by the UCloud session in `analysis_run12.md`, copy on the workstation).

run12 results (2002–2019, 30 members): basin RMS DA–GRACE 7.5 mm (OL 36.8; run11a2 8.3), sub-basins 9.5–10.5 mm; MAM 9.6 (was 12.2); trend DA −4.55 vs GRACE −4.94 vs OL −1.73 mm/yr; gain 0.70; chi2 median 1.37 / mean 3.70; forecast spread 7.3 mm vs GRACE sigma 6.8. Cowork review: (a) the 'over-fitting' warning (RMS < 0.8 sigma) is likely a false alarm — for gain K≈0.7 the expected analysis residual is √(1−K)σ≈0.55σ, observed 0.65–0.75σ; the real question is diagonal R (correlated sub-basin errors) → test full R rather than R×1.5; GRACE fit is not independent validation (need held-out months / GRDC). (b) Main issue = systematic seasonal bias (same correction every year: +27 mm Jun, −25 mm Nov–Jan; innovations 25–29 mm in Apr–May/Oct vs 10–14 expected; tug of war: model drift undoes the monthly increment within the window; OL annual amplitude too small, peak too early), absorbed by adaptive inflation; 2/3 of the seasonal correction goes to groundwater via the OL-spread split — check monthly split shares. Fixes before run13/run14 — DONE 6 Oct (Cowork): `settings/demo_Danube/DA_setting.json` riverstor false + `"snow_lower_bound": {"factor": 0.5}`; new option `snow_lower_bound` (default {factor 0.5}, false = none) in bounds.py (`relative_lower`, window-aware LB = x_mean − (1−f)·x_min), EnKF_localized, filter_factory, configure_DA, doc/configuration.md §1.4 (+ river note: keep river in the state where it carries the seasonal signal, e.g. Amazon; false for the Danube); da_evaluation over-fitting warning now compares RMS(DA−GRACE) with the consistent-analysis expectation sqrt(mean(σ⁴/(sd_fc²+σ²))) (threshold 0.7×), new column 'expected' in summary.txt. Unit-tested the bound on synthetic windows; not yet run in a DA. Follow-up 6 Oct (UCloud finding, fixed by Cowork): the expected RMS was dominated by degraded GRACE months (sub-basin 1: 19.4 mm > sigma); `rms_expected` now uses only windows with sigma <= 3 x its median (same rule as the degraded-month warning) and compares with the actual RMS on the same windows (`rms_DA_regular`, `n_regular_windows`; column 'DA reg' in summary.txt). Sync incident: an intermediate draft with an undefined name (TH_print) briefly overwrote da_evaluation.py; the correct version was re-committed and verified (grep TH_print must give 0). Dropped: spread floor in the split (not needed — partition weights are var x max(corr,0), no division by spread; the reported ratio 4126 is BEFORE the groundwater cap of 5x median, which works; only improve the diagnostic), sigma cap for degraded GRACE months (Fan: don't do it); check forcing mid-2017 (OL residual jump), reduced gain after the GRACE-FO gap (2018-10 event −110..−143 mm GW in sub-basins 1–3); Vienna cell 48.25N 16.25E trend −151 mm/yr again (lake/reservoir drift). Next runs: run13 = run12 on 2002-01..2016-04, run14 = full R.

Next: collect_and_statistics(Stage.OL) (writes Res_OL.h5 + Harmonic_OL.nc; Res_OL.h5 needed by the DA for the `ol_spread` split) → DA_run → collect_and_statistics(Stage.DA) → visualization, increment_diagnosis, `da_evaluation(tag='run12', reference=DATA_ROOT + '/Res/demo_2/evaluation/run11a2')`.

**OL output must stay GLOBAL** (Fan: so the OL can be reused for DA in other regions). Do NOT crop the OL output and do NOT run `clean_temp_output(Stage.OL)` (it deletes the global daily OL files). Clean-up applies to the DA temp output only.

Local workstation: `RDA_local.py` now points to `settings/demo_2` (case demo_2, 4 members, 2002-01 – 2016-04, local paths); demo2 currently runs only `model_perturbation()` (used to test the MPI perturbation; local Ensemble_input Ens_0–4 for 2000-01 – 2004-04 was regenerated, the rest is from the old serial run).

## Performance changes (5 Oct)
- `model_perturbation()` MPI stage (tested on the workstation only; run12 still used the serial version): each rank writes its own Ens_k, rank 0 draws/broadcasts random numbers (same seed → identical to the old serial code). Forcing zlib level 1 (`perturbation.FORCING_COMPLEVEL`). Workstation: ~3.7 s per month for 5 members vs ~54 s before (~15x). Test script `temp/env_test/test_perturbation.py`.
- Daily output writer `src_GHM/Extension/fan_createandwrite.py` (`save_netcdf_daily_single_file`): zlib level 1 (was 5) and 36×72 lat/lon tiles (`CreateandWritetoVariables.COMPLEVEL`, `.TILE`; capped at the field size for cropped DA files). Writing ~2.3x faster (UCloud measurement: 0.72 → 0.16 s/day), files ~5–7% larger; reading a basin box 7–14x faster.
- **WaterGAP change** (recorded in README "Modifications to WaterGAP 2.2e", item 3 — Fan's rule: every change to the original WaterGAP code must be recorded there): `src_GHM/ReWaterGAP/model/verticalwaterbalance/waterbalance_vertical.py` line 70, removed the daily `.copy()` of `snow_water_storage_subgrid` (207 MB/day/member). Verified bit-identical on the workstation with `temp/env_test/check_snow_copy.py` (original vs modified routine, 10-day spin-up + 3-day resume, restart states and daily output identical); ~0.4 s saved per model day locally. Note: Fan's `src_GHM/Extension/fan_waterbalance_vertical_init.py` overrides only the wrapper class; it still calls this original numba routine.
- `statistical_analysis.py` `load_nc_files`: `open_mfdataset(parallel=False)` (parallel=True crashed in tests on non-thread-safe HDF5; no speed loss).
- Recommended launch with thread limits: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 mpiexec -n 31 python -u RDA_Ucloud.py Config_ReWaterGAP.json` (from src_demo).
- Remaining ideas (not done, Fan's call): profile one model day to split the ~3.9 s/day (workstation); snow subgrid array with bands last (~0.25 s/day, WaterGAP change, restart transpose); lookup tables for three whole-grid searches in routing (reservoir release, riparian redistribution, neighbouring-cell supply); DA run on basin + upstream cells only (5–10x per DA day, OL stays global; needs basin+upstream mask and a 1-year validation); step-2 basin averages read each variable once (11 s → ~5.6 s per member) and close the mask file; DA keeps window states in memory (~2%). Skipped by Fan: h5py reader for the merge step (gain now ~1.5 min per collect after tiling).

## Code review of the UCloud entry (5 Oct)
- `spin_up()` passes `RDA.spin_up_years` to `model_spinup`.
- Stage APIs take the `Stage` enum only (`from src_DA.EnumDA import Stage`); strings raise ValueError: `post_processing`, `harmonic_fit`, `clean_temp_output` (CLI converts with `Stage[name]`).
- `collect_and_statistics(stage, skip_collect=False, post_process=True)`, step 3: (3a) `RDA.harmonic_fit(stage, comm, root=1)` on all ranks (members spread over the ranks, rank 1 writes Harmonic_<stage>.nc); (3b) `RDA.post_processing(stages=[stage], harmonic=False)` on rank 1 (Res_<stage>.h5, GRACE files for the DA). A failure in step 3 is reported and does not abort the job; rerun serially with `RDA.post_processing(stages=[Stage.X])` or `RDA.harmonic_fit(Stage.X)`. If an OL-based split finds no Res_OL.h5, `split_table` stops with a message naming the call.
- Design rule (Fan): `RDA` / the entry scripts expose only the most important run parameters; secondary options keep defaults in lower modules (harmonic fit `HarmonicMapAnalysis(per_member=True)`, `run(n_workers=1)`; map style `visualization.harmonic_maps(style='smooth')`; `perturbation.FORCING_COMPLEVEL`; writer `COMPLEVEL`/`TILE`).
- Checked: forcing and water use (1901–2019) and CSR GRACE/GRACE-FO mascons + DDK3 covariance samples (2002-04 – 2024-03) cover the period. GRACE obs, OL-mean and state-envelope files are named per basin (shared by demo_2 and demo_Danube; the latest run's version wins).
- Report comparisons with run 11a2 cover different periods (2002–2016.4 vs 2002–2019).

## Agreed plan (4 Oct 2026)
1. **Data**: one-time rsync from the workstation (≈ 350 GB). Data folder is NOT synced continuously.
2. **Code**: synced both ways with Syncthing (not git). `.stglobalignore` in the repo root (`.git`, `.idea`, `__pycache__`, `*.pyc`, `/parallel_logs`, `/temp` except `/temp/env_test` minus its output/init/logs). Caution: two sessions (Cowork on the workstation, Claude Code on UCloud) edit the same repo — let only one edit a given file; once an edit from Cowork was overwritten by an older copy from UCloud within seconds (check Syncthing conflicts if a file reverts).
3. **Paths**: settings carry UCloud paths; `RDA_Ucloud.py` demo2 calls `config_external_data()` on rank 0 at every MPI start.
4. **Environment** on /work: pyglda_v2 conda env via `installation/setup_env.sh`; pygmt in a separate env.
5. **Claude Code on UCloud**: conda env on /work, `CLAUDE_CONFIG_DIR=/work/<folder>/.claude`. Start by reading this note — the UCloud session has no project access, so a copy is kept in the repo at `doc/ucloud_notes.md` (refreshed by Cowork whenever this note changes) — and `da_filter_restructure_plan.md`.
6. **Results to desktop / Cowork chat**: rsync `Res/<case>/evaluation/` back after each run.

## How a run works
- Entry: `src_demo/RDA_Ucloud.py` sets class attributes on `RDA` and calls stages; launched from `src_demo/` (`python RDA_Ucloud.py Config_ReWaterGAP.json` serial, `mpiexec -n 31 python -u RDA_Ucloud.py Config_ReWaterGAP.json` MPI). The argv is required and is opened relative to the cwd by WaterGAP's `controller/configuration_module.py` at import (WaterGAP exits with code 0 if it is missing).
- WaterGAP 2.2e is vendored in `src_GHM/ReWaterGAP/`; bare imports only (`from controller import configuration_module as cm`).
- MPI: `ens+1` ranks; rank 0 = unperturbed control, rank 1 = main thread; other ranks log to `parallel_logs/<stage>/rank_k.log`.
- Every rank runs WaterGAP globally. OL writes global daily files `OL_output/Ens_k/daily_output_YYYY-MM-DD.nc`; DA writes the cropped basin box to `DA_output/<case>/Ens_k/`.
- Res_<stage>.h5: `time` (decimal years, daily) + `basin/` and `sub_basin_k/` → `<storage>/<member>` daily series in mm, member '0' = unperturbed.

## Technical notes for UCloud
- `RDA.config_external_data()` rewrites every path in the three JSONs of `RDA.setting_dir` to `RDA.external_data_path`. Keep external-data folder names identical.
- `Ens_k` folders are created with `mkdir(parents=False)` → parent folders must exist.
- Environment: py3.10, numpy 1.23.5 ↔ numba 0.56.4, mpich 4.1, nompi hdf5/netcdf; use the env's own mpirun on a single node.
- Model speed: UCloud ~1.0 s per model day per rank (+0.16 s output writing in the OL); workstation ~3.9 s/day (spin-up, after the snow fix).

## Storage for the 30-member run 2002–2019 (31 ranks)
OL_output ≈ 6 575 days × ~5 MB × 31 ≈ 1.0–1.1 TB (kept permanently, global); Ensemble_input ≈ 0.5–0.6 TB; Ensemble_Initialization ≈ 26 GB; DA_output + Res ≈ 40–50 GB. Peak ≈ 2 TB of 7 TB. Clean up only the DA temp output (`clean_temp_output(Stage.DA)`, dry run first).

## Public data product (6 Oct 2026, implemented; not an official publication: no licence, repository or DOI)
- Script `src_postprocessing/export_product.py`, class `product_export` (Fan's style): `product_export(setting_dir, version)`
  `.configure_output(out_dir)` `.configure_baseline(begin, end)` -> `.run()`; demo1 (demo_2 local), demo2 (demo_Danube).
  Paths, case, basin, period and ensemble size come from `<setting_dir>/DA_setting.json`.
- Called from Regional_DA: `RDA.export_product(version='v1.0')` (only argument: version; options stay in the script).
  Entry scripts: commented line in demo1 after da_evaluation — `RDA_Ucloud.py`: `# RDA.export_product(version='v1.0')`,
  `RDA_local.py`: `# RDA.export_product(version='v0.1')`. Run only when the data are shared; needs collect_and_statistics of OL and DA.
- Output `Res/<case>/product/PyGLDA-v2_<case>_<version>/`:
  - `gridded/..._daily_<year>.nc`: for every variable `<v>` and stage OL/DA: `<v>_<OL|DA>_mean`, `<v>_<OL|DA>_spread`
    (std, ddof=1, members 1..N; member 0 excluded), daily, 0.5°, basin box, NaN outside the basin. Variables: tws,
    tws_anomaly (baseline 2004-01-01..2009-12-31 per stage), groundwater, soil_moisture, swe, surface_water (river + lakes
    + wetlands + reservoirs), canopy, discharge (dis); recharge (qr) and runoff (qtot) are picked up automatically once
    a run writes them (Fan: keep them in the product; switch on total_groundwater_recharge / total_runoff in
    Config_ReWaterGAP.json OutputVariable next time). No flags in these files (Fan removed grace_obs / grace_degraded).
  - `gridded/..._GRACE_monthly.nc`: gridded GRACE TWS on the same grid (from `<obs.GRACE.preprocess_res>/<basin>_gridded_signal.hdf5`,
    land mask GRACE/global_mask/GlobalLandMaskForGRACE.hdf5 if needed), `tws` as prepared and `tws_anomaly` on the same baseline.
  - `timeseries/..._basin_timeseries.nc`: basin and sub-basin daily mean/spread of the storages (from Res_<stage>.h5) and the
    GRACE TWS assimilated per epoch with sigma (basin sigma includes the sub-basin correlations).
  - `ancillary/..._ancillary.nc`: cell_area, basin_mask, sub_basin_id, tws_trend_OL/DA, cell_flag (|trend| > 20 mm/yr).
  - `README.md` (variables, method, limitations, contact). No settings/ or validation/ folders (Fan removed them).
- Tested on a synthetic case with the real file layout (means, spreads, anomaly, NaN outside basin, member 0 excluded,
  gridded GRACE placement incl. land mask, basin GRACE sigma, time series). Not yet run on real data.

## UCloud Claude Code session: changes and suggestions (5-6 Oct 2026; to be merged into the master note by Cowork)
Everything below was done from the desktop Claude Code session driving UCloud over ssh (edits on the desktop, synced,
launched remotely). Nothing here is in the master note yet unless marked.

### Done
- **Environments on UCloud** under `/work/PyGLDA_v2_external_data/envs`: `miniforge3`, `pyglda_v2` (from
  installation/environment.yml, mpich launcher), `pygmt` (python 3.13, pygmt 0.16, gmt 6.6, ghostscript, pandas<3,
  spec `installation/environment_pygmt.yml`), Claude Code 2.1.289 (native binary, `claude_home/`, login kept in
  `claude_config/` via CLAUDE_CONFIG_DIR). `installation/setup_env.sh` rewritten: `install --prefix DIR
  [--with-pygmt --with-claude]`, idempotent, bootstraps Miniforge, verifies imports / numba / 2-rank MPI / pygmt png,
  writes `DIR/activate.sh` and `DIR/ucloud_init.sh` (select the latter in the job form "Initialization": it adds the
  activation to every shell and switches on tmux mouse scrolling). Guide: `doc/installation.md`. The UCloud image's
  `/usr/local/bin/mpirun` is Open MPI: with it every rank is rank 0 - always use the env's mpiexec (activate.sh puts
  it first on PATH). Job limits are in /sys/fs/cgroup (32 cores / 89 GB on cpu-amd-zen5-32-vcpu; nproc shows 256).
- **Pipeline run12** driven stage by stage (perturbation serial 4 h 40 min, spin-up 57 min, GRACE obs 1 min,
  OL 2 h 23 min + collect 16 min, DA 2 h 43 min + collect 10 min, post-processing 2 min in the pygmt env).
  Timings per model day on 31 ranks: OL 1.1-1.3 s (with the level-1 writer), DA 1.5 s. Memory ~2.2 GB per rank.
  GRACE COV step: 180 GB of DDK3 samples in 53 s on WekaFS (18 GB/s) vs ~30 min on the desktop USB drive - same
  data, same result, checked.
- **Pop-ups off in the pipeline path**: `Visualization.py` defaults `allow_pop_up=False`, `Regional_DA.visualization`
  passes False, `GRACE_perturbation.visualization` saves a png instead of `fig.show()` (committed in 51498ef).
  HydroShed.py / Shape.py / shp2mask.py were left unchanged on Fan's request.
- **run12 re-evaluated** (6 Oct 00:0x) with the corrected `rms_expected` (regular windows only): verified on
  UCloud that the final da_evaluation.py arrived (no `TH_print`, `rms_DA_regular` present, md5 equal on both sides,
  no Syncthing conflict files). On sub-basin 1 the expectation is 11.4 mm on regular windows (19.4 with the degraded
  months) against an actual 9.8 / 9.2 mm - no over-fitting by that measure; the median-based variant gives 9.7.
- **run13 prepared, NOT launched** (Fan, 6 Oct 01:00: not planned for now; the driver stays set up for it): per Fan (6 Oct, chat) run13 = run12
  setup on the FULL period 2002-2019 with riverstor false and snow_lower_bound 0.5, reference run12 - NOT the
  2002..2016-04 variant written in "Next runs" above; the comparison with run11a2 is dropped. Driver: demo2 = DA_run +
  collect(DA); demo1 = visualization, increment_diagnosis, `da_evaluation(tag='run13', reference=.../demo_Danube/
  evaluation/run12)`. Launch (from src_demo, env's mpiexec, plain command as for run12; Fan 6 Oct: do NOT use the
  thread-limit variables, keep it simple): `mpiexec -n 31 python -u RDA_Ucloud.py Config_ReWaterGAP.json`.
  Note: the DA overwrites Res/demo_Danube/{DA/, Res_DA.h5, Harmonic_DA.nc, GRACE_*} and figures/; only
  evaluation/<tag>/ keeps a run (summary, figures, logs, analysis_<tag>.md).
- Working rules used: edit on the desktop only, poll the md5 on UCloud before launching; runs in tmux without log
  file (Fan's preference), progress from `parallel_logs/<stage>/rank_2.log` and the tmux pane; `/temp` and
  `/parallel_logs` are not synced (.stglobalignore), so notes for UCloud belong in doc/ (this file).

- **Product v1.0 of run12 exported** (6 Oct 01:01, `RDA.export_product(version='v1.0')` in the pyglda_v2 env, 2 min,
  exit 0; first run on real data): `Res/demo_Danube/product/PyGLDA-v2_demo_Danube_v1.0/` (246 MB: 18 daily files of
  ~14 MB, GRACE monthly 179 epochs, basin time series 7 units x 6574 days, ancillary with 1 flagged cell, README).
  Checked on the desktop copy (`Res/demo_Danube/product/` in the desktop data root): NaN outside the basin (43 % of the
  box), float32, units mm, anomaly baseline 2004-2009 averages to 0.01 mm, 26 time-series variables. Note for the
  README/limitations: the per-cell ensemble spread is almost the same in DA and OL (2010: 28.7 vs 29.3 mm) because the
  update constrains the sub-basin means while the cell-level member differences come from the parameter perturbation;
  the spread reduction is visible at basin scale (5.3 vs 16.5 mm), not per cell.

- **Performance figure set for the product** (6 Oct 01:1x, Fan's request): `Res/demo_Danube/product/
  PyGLDA-v2_demo_Danube_v1.0_figures/` (sibling of the product folder, not inside it, since the product has no
  validation/ folder): 9 figures renumbered 01-09, this experiment only (Fan: no comparison figures with earlier runs; fig1 and fig4 of the report dropped): innovation/residual/spread,
  trend-amplitude-phase maps, seasonal correction, groundwater correction, storage components OL/DA, filter health,
  increments, GMT TWS panels; PDFs where they exist) + `FIGURES.md` with captions, key numbers and the limitations
  the figures show. Copied from the evaluation/run12 and figures/ folders; on UCloud and in the desktop data root.

### Suggestions / open points from analysis_run12.md not yet in the master note
- 2018-10 re-entry after the GRACE-FO gap: largest event of the run (basin GW -21..-47 mm; -110..-143 mm in
  sub-basins 1-3). Beyond the forcing check, consider a reduced gain or a wider R for the first 1-2 windows after a
  gap, or a short free-run spin-down of the ensemble spread before the first GRACE-FO window.
- OL far too wet from mid-2017 (residual +70..80 mm in 2018-2019) - the abrupt start suggests the forcing, not the
  model; compare W5E5 precipitation 2017-2019 with the earlier years over the Danube before run14.
- Partition diagnostic: report the weight ratio AFTER the cap (the 4126 is before the 5x cap), otherwise the warning
  is not actionable.
- Over-fitting check: the scalar expectation ignores that the block update also uses the correlated neighbours of a
  sub-basin; treat ratios 0.7-1.0 as normal. The proper independent validation remains held-out GRACE months or GRDC.
- River storage out of the state (done): the DA figures of run13 will show whether the seasonal river correction
  (+3 mm in Jun) is reproduced by the model's own routing from the corrected storages.
- Perturbation: the MPI version is still untested on UCloud; first use at the next new ensemble, with a 2-month test
  (`spin_up_start` .. +1 month) before the full 240 months.
- Thread limits: not used for run12 (plain `mpiexec -n 31`, 30 of 32 cores busy, no problem) and, per Fan, not to be
  used at all; all launches stay the plain command.
