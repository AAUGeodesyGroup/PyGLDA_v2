<!-- Copy of the claude.ai project note "claude/ucloud_transfer_notes.md" (project "PyGLDA V2"), written by the Cowork session on the workstation so that the UCloud Claude Code session can read it (synced via Syncthing). The project note is the master copy; this file is refreshed when the note changes. Last copied: 2026-10-09 (Test 2 script added) -->

# PyGLDA v2 → UCloud: code map and transfer notes (4 Oct 2026, updated 8 Oct 2026)

Code: `/media/user/My Book/Fan/PyGLDA_v2` (also on GitHub, AAUGeodesyGroup/PyGLDA_v2), synced with UCloud via Syncthing. External data on UCloud: `/work/PyGLDA_v2_external_data` (workstation copy: `/media/user/My Book/Fan/PyGLDA_v2_external_data`). Fan reaches UCloud from a terminal (SSH); a Claude Code session also runs on UCloud and launches jobs. UCloud quota: 7 TB.

## READ FIRST — changes since run12, known issues, plan (Cowork, 8 Oct 2026)
Everything below was edited on the workstation by Cowork, tested in the Cowork sandbox (harnesses, synthetic data, Danube/Amazon/global masks), committed into the synced repo and md5-checked. **Nothing of it has run on UCloud yet.** Rule as before: only one session edits a given file; before launching, check that the files below arrived (md5 / timestamps, no Syncthing conflict copies).

### 1. Code changes since run12 (all shared by the regional and the global workflow)
Correctness fixes:
- `src_GHM/Interface/DailyStepRun.py` (**WaterGAP-side, recorded in README "Modifications to WaterGAP 2.2e"**): `passState` now syncs the 100 snow elevation sub-cells with the analysed cell snow (`_sync_snow_subgrid`: scale new/old, even split where old = 0, zero where new <= 0). Before, WaterGAP rebuilt cell snow from the sub-cells the next day and the **swe increment was lost**. Open loop untouched (bit-identical). Expect visible changes in snow sub-basins (Alps).
- `src_FlowControl/DA_GRACE.py` `gather_OLmean`: OL mean over the GRACE epochs present, each averaged over its observation window (was: all OL days of the period -> constant per-unit offset of the innovations when GRACE months are missing). Rebuilt automatically at every `DA_run`. `make_state_envelope` only when riverstor is in the DA state (skipped for demo_Danube/demo_Global now).
- `src_DA/EnKF_localized.py`: diagonal obs-error mode draws the perturbations from N(0, diag R) with a fixed seed (`OBS_PERTURBATION_SEED = 20021`); the old whitening with the sample covariance collapsed or blew up the spread for many units. `src_OBS/GRACE_perturbation.py` seeded too (SEED = 20021) -> reruns reproducible.
- Soil upper bound (`EnKF.py`, `Threshold.py`, `bounds.py`): capacity on disk = smax x land/continental fraction, per member, largest fraction of the window (smax.nc is per land area). Before, soil in cells with lakes/wetlands could be pushed above its real capacity. Expect less soil clipping.
- GRACE processing land mask = WaterGAP land (`RDA._land_mask()` -> `WaterGAPLandMask.hdf5`); 1-degree basin mask derived from the 0.5-degree one (`shp2mask.mask_05_to_1`). Takes effect only after `config_basin_mask()` + `get_GRACE_obs()` are rerun.
- `src_GHM/Interface/UnitConverter.py`: full-grid box for a global mask (regional unchanged).
- `Regional_DA.DA_run`: whole body in try/abort + barrier before rank 0 rewrites DA_setting.json (a failing rank no longer hangs the job).
- `clean_temp_output(Stage.OL)` refused when `OL_output/<case>` does not exist (protects the shared global OL) unless `force=True`.
- `statistical_analysis.run_GRACE` / `da_evaluation._warnings`: box from `load_mask`, lookups by coordinate.

Memory / speed (numerically identical, tested to 1e-12 mm):
- Sparse CSR design matrix (`ObsDesignMatrix`), sparse localization taper (`localization.py`, Gaussian with KD-tree), update `dX = Pxy @ solve(Pyy, d)` without the gain, sparse partition / RTPS (`partition.py`, `inflation.py`), cached `load_mask` (bool vectors), Threshold bookkeeping with bincount, GRACE covariance read per month (`observations.py`). Global filter: a few MB per rank (was 6-8 GB); worker rank ~ WaterGAP 2.2 GB + ~50 MB.
- Harmonic maps (`HarmonicMapAnalysis`): only cells with data, blocks of 8000 columns, MPI reduce on a common column set: 1.3-1.5 GB per rank for 730 global days (old code OOM).
- `state_envelope.py` without float64 copies.

New tools (global only):
- `src_FlowControl/Global_DA.py` (class GDA, inherits RDA): `check_units`, `get_GRACE_obs`, `da_evaluation(tag)`, `visualization(tag)`.
- `src_postprocessing/global_evaluation.py`: per-unit + grouped report -> `Res/<case>/evaluation_global/<tag>/` (unit_metrics.csv, group_metrics.csv, group_series.csv, summary.txt/json, logs/).
- `src_postprocessing/global_visualization.py` (PyGMT, needs the **pygmt env**): gfig1 unit maps, gfig2 region series, gfig3 harmonic maps OL|DA|GRACE (pixel mode, Fan's harmonic_maps style), gfig3b DA-OL, gfig4 filter health, gfig5 RMS reduction vs SNR/area; PNG + PDF.
- `src_demo/demo_global.py`: **UCloud driver** of the global pilot (see §3).

### 2. Known issues (open)
- Final restart is written before the last window's analysis and into the shared `Ensemble_Initialization/Ens_k` — Fan: not a problem, set aside.
- GRACE obs / OL-mean / envelope files are named by basin, not case (cases with the same basin overwrite each other); the obs covariance is read by every rank during the run (small, harmless).
- `localization: none` on the global mask would need ~30 GB (no dense branch) -> use `block` (demo_Global does).
- EnKF_localized refuses masks with a cell in two units; Pxy/W share index arrays with the taper; cached mask arrays are writable; `EnKF_fixed_time_old.py` dead code breaks with sparse H.
- `increment_diagnosis` is still the regional tool (not for 772 units).
- `obs_error_correlation: "diagonal"` (Fan's choice, kept): correlated GRACE errors of neighbouring units are ignored -> observations slightly over-weighted; full R is the test if chi2 / RMS look too good.
- Danube open points from run12 still open: 2018-10 re-entry after the GRACE-FO gap, OL too wet from mid-2017 (forcing?), partition diagnostic after the cap, Vienna cell drift.

### 3. Plan (in this order; Fan decides each step)
**Step A — Danube rerun (run13) to measure the impact of the changes.** Same 30-member setup and period 2002-2019 as run12. Note: the settings also changed since run12 (riverstor false, snow_lower_bound 0.5), so run13 vs run12 mixes setting and code changes.
0. data (the data folder is NOT synced): copy `GRACE/global_mask/WaterGAPLandMask.hdf5` (2.6 MB, made on the workstation 7 Oct) from the workstation data root to `/work/PyGLDA_v2_external_data/GRACE/global_mask/` (or rebuild it with `src_auxiliary/watergap_land_mask.py`). For step B also check that `Basin/shp/Global_v1.0/` and `Basin/mask/GlobalBasins_v1.0/` exist on UCloud.
1. serial: `src_demo/RDA_Ucloud.py` is prepared for run13 (8 Oct, Fan's go): `__main__` calls demo1, whose active steps are `RDA.config_basin_mask()` + `RDA.get_GRACE_obs(is_diagonal=False)` (new land mask; the after-run steps are commented out) -> `python RDA_Ucloud.py Config_ReWaterGAP.json`;
2. switch `__main__` to demo2 and run the plain `mpiexec -n 31 python -u RDA_Ucloud.py Config_ReWaterGAP.json`: `RDA.DA_run()` -> `RDA.collect_and_statistics(Stage.DA)` (OL is not rerun);
3. switch back to demo1: comment out the two before-steps, uncomment `RDA.visualization()`, `RDA.increment_diagnosis()`, `RDA.da_evaluation(tag='run13', reference=DATA_ROOT + '/Res/demo_Danube/evaluation/run12')` (figures in the pygmt env, as for run12).
Judge: basin/sub-basin RMS and correlation vs GRACE within ~10 % of run12, chi2 and spread in the same range, no new warnings, less soil clipping; larger differences should be explainable by the snow sync or the OL mean. Report back with summary.txt of both runs.

**Step B — global pilot (demo_Global, 2002-2003, 30 members, block localization; state = groundwstor, soilmoist, swe, riverstor — Fan 8 Oct: river storage on from the start, also in inflation split_storages; lakes/wetlands/reservoirs/canopy off)**, only after A is accepted. With riverstor in the state, DA_run rebuilds the OL envelope (`make_state_envelope`, rank 0, river only) at its start and Threshold bounds river storage by it. From `src_demo/`:
1. serial, `demo_global.py` demo1 (uncomment the 4 steps): `GDA.config_external_data()`, `GDA.read_config_and_save()`, `GDA.check_units()` (must print IDs equal, 0 cells in two units, md5 True/None), `GDA.get_GRACE_obs()` (first real run for 772 units: unit TWS, 772 x 772 COV, gridded TWS);
2. MPI, demo2: `mpiexec -n 31 python -u demo_global.py Config_ReWaterGAP.json` = `collect_and_statistics(Stage.OL)` (global yearly OL files + Res_OL.h5 + harmonic maps) -> `DA_run()` -> `collect_and_statistics(Stage.DA)`;
3. serial: `GDA.da_evaluation(tag='pilot1')`, then `GDA.visualization(tag='pilot1')` in the **pygmt env**; rsync `Res/demo_Global/evaluation_global/pilot1/` back to the workstation.
Watch: memory per rank (~2.3 GB expected), time per DA day, `parallel_logs/DA/rank_2.log`, chi2 / RMS reduction / clipping warnings in summary.txt.
Never run `clean_temp_output(Stage.OL)`; `GDA.clean_temp_output(Stage.DA, dry_run=True)` only after the report is checked.

**Step C — full global run 2002-2019**, after the pilot. Storage: UCloud shows 3.9 of 7 TB used (8 Oct). The full run adds ~3 TB at the peak (OL yearly global ~1.0 TB, DA daily ~1.0 TB, DA yearly ~1.0 TB; ~4.8 MB per member-day, ~1.8 GB per member-year) -> too tight. Before C: `du -sh /work/PyGLDA_v2_external_data/* /work/* | sort -h`, clean DA temp output of old runs, and/or ask for more quota; code options (Fan's call): merge+clean DA per year during the run, float32/compressed yearly files, OL statistics without global yearly OL files.

### 4. TASK for Claude Code — find out why run13 is ~2x slower than run12 (Fan, 9 Oct: must be solved BEFORE the global run)
**STATUS 9 Oct (Cowork, after Test 2 v2): RESOLVED — not the code.** current vs nosync equal within noise; the extra time sits in WaterGAP's unchanged routines (river_routing 1.5-1.8 s, vertical balance 0.4 s per model day); the code on that path is identical to run12 (git e04095aa); I/O fast. Cause: shared host without CPU set, memory-bandwidth contention (membench 1.95 s alone vs 5.3-7.6 s with 31 copies). Consequences: (1) before every long run, run `membench.py` (31 processes, 30 s) as a host check and compare with the reference numbers in "Slowdown run13"; if it is far above, resubmit to another node; (2) Fan to ask UCloud about a CPU set / whole-node reservation for the global run; (3) the global gate is lifted for the slowdown. Still open before Step B: the snow ratchet (cap proposal awaiting Fan's OK) (the ZeroDivisionError in river_routing of the 5-rank smoke test: Fan, 9 Oct — not a problem, no action).
Facts so far (your monitor, 8-9 Oct): run13 3.3 s per model day (98 s per GRACE window) vs run12 1.5 s (45 s per window), same node type, same 31 ranks, same OL; all cores look busy the whole window; memory per rank 1.9 GB (run12 2.2 GB). run13 has FEWER states than run12 (riverstor off).
Cowork's code review (no measurement): the per-day additions are cheap for the Danube (376 cells): snow sub-cell sync in `DailyStepRun.passState` returns after one array comparison on normal days; land-fraction tracking `EnKF._land_ratio_box` is one clip; `Threshold.threshold` / `_record_month` (bincount) and the per-member soil scale are per window; the analysis on rank 1 (1128 states x 6 obs x 30 members) is well below 1 s. Note: "all cores busy" does not exclude a serial phase — MPICH ranks waiting in gather/scatter spin at 100 % CPU. Hypotheses: (H1) file I/O on /work is slower (every window each member re-reads and rewrites ~30 daily state files + writes daily output: the burst you saw at the window end — that code did not change); (H2) something in the new code that only shows at full size; (H3) machine/storage load outside the job.

Rules for this task: do NOT edit anything under `src_*` or `settings/`; new test scripts only under `temp/env_test/` (synced back, Cowork can read them); report every result in this file (section "UCloud Claude Code session", sub-heading "Slowdown run13", with numbers) and stop when done — Cowork fixes the code on the workstation after Fan's OK.

**Test 1 — now, while run13 is running (no code, read-only):**
1. Pace inside one window: sample `parallel_logs/DA/rank_2.log` every 2 s (e.g. `while sleep 2; do date +%T; tail -c 120 parallel_logs/DA/rank_2.log | tr -d '\n' | tail -c 40; echo; done > temp/env_test/pace_run13.txt`) for one full window (~2 min), then stop it. Report: seconds between dots (model days) and the length of the pause around `|` (window end). Even dots ~3.3 s -> per-day work (model step / daily I/O); dots ~1.5 s + ~50 s pause at `|` -> window end (state rewrite, analysis, gather/scatter).
2. Storage: `cat /sys/fs/cgroup/io.pressure /sys/fs/cgroup/cpu.pressure` (and `/proc/pressure/*` if present) twice, 1 min apart; `dd if=/dev/zero of=/work/PyGLDA_v2_external_data/ddtest bs=1M count=2000 oflag=direct; rm /work/PyGLDA_v2_external_data/ddtest` (write speed; a few GB/s is normal, report the number).
3. Processes: `top -b -n 1 | head -50` — anything outside the 31 python ranks using CPU? state of the ranks (R vs D = waiting for disk).

**Test 2 — NOW (run13 is finished): the script is ready, written by Cowork: `temp/env_test/profile_da.py`** (read its docstring first). Cowork's finding before the test (9 Oct, git diff of commit e04095aa = run12 code against today): the only code on the per-day path that changed since run12 is `DailyModelRun._sync_snow_subgrid` (in passState) and `EnKF._land_ratio_box`; `EnsStates` (daily state files), the WaterGAP extension files and the writer are unchanged. The test therefore runs the same short DA twice:
```
cd src_demo
PROFILE_MODE=current mpiexec -n 31 python -u ../temp/env_test/profile_da.py Config_ReWaterGAP.json
PROFILE_MODE=nosync  mpiexec -n 31 python -u ../temp/env_test/profile_da.py Config_ReWaterGAP.json
```
(2002-01-01..2002-05-31 from the 2001-12-31 restart, 2 GRACE windows, ~10 min each; `nosync` switches the two additions off by monkeypatching, nothing in src_* changes). It writes `temp/env_test/profile/<mode>_timing.txt` (wall time and per-function time for every rank) and `<mode>_rank<r>.prof`, and does NOT touch `settings/demo_Danube` (**v2, 9 Oct ~07:10 UTC, after Code's report below**: it works on a copy `temp/env_test/settings_profile/` and skips `gather_OLmean`, keeping the OL-mean file of run13 — the short period cannot be checked against the 6574-day OL series; timing is unaffected). Before re-running: check that `settings/demo_Danube/DA_setting.json` has todate 2019-12-31 (Code restored it at 09:00 local). Report in this file: wall time of both modes, the timing table of rank 1 and rank 2 of both modes, and the top 25 lines of `python -c "import pstats; pstats.Stats('temp/env_test/profile/current_rank2.prof').sort_stats('tottime').print_stats(25)"` (and for rank 1). Interpretation: `nosync` much faster -> one of the two functions is the cause; both equally slow (~3.3 s per model day, `DailyModelRun.update` dominating) -> the model step itself is slower than in run12 -> environment (node/storage), then also report `lscpu | head -20`, `cat /sys/fs/cgroup/cpu.max /sys/fs/cgroup/cpu.stat` and the `dd` write test from Test 1.

**Test 3 — only if Test 2 is inconclusive:** same short run with the run12 settings (riverstor true in a copy of DA_setting.json written by the script and restored afterwards) to see whether the settings or the code make the difference. Ask Fan before running it.

Gate: the global pilot (Step B) does not start until the cause is found and fixed (Fan, 9 Oct).

### 5. Snow cap + envelope inside DA_run (Cowork, 9 Oct 2026) — NEXT TEST before the global pilot
Why: run13 showed a snow "ratchet" in the high Hohe Tauern cells (47.25N 12.75E, 47.25N 13.25E, 47.75N 12.75E): the relative
snow upper bound (2 x forecast + 20 mm) moves up with the pack, the sub-cell sync puts the increment into the highest bands
that never melt, and the snow spread (40-85 mm vs soil 1-20, GW 2-7 mm) then draws most of the increment.
Changed (committed + md5-verified 9 Oct; Cowork edits, UCloud does not):
- `src_DA/state_envelope.py`: parallel (member k on rank k%size, MPI MIN/MAX reduction, identical to serial), new
  `swe_max_month` (12 x lat x lon) = largest open-loop snow of each calendar month over all years and members.
- `Regional_DA.DA_run`: `da.make_state_envelope(comm=comm, force=True)` on ALL ranks at every DA start (always rebuilt;
  ~1-2 min for the Danube with 31 ranks, expect longer for the global OL).
- Snow settings merged into ONE block (10 Oct; old keys still read, results identical, `src_DA.configure_DA.snow_bounds()`):
  `"snow_bounds": {"lower_factor": 0.5, "upper_factor": 2.0, "upper_offset_mm": 20.0, "envelope_factor": 1.5,
  "envelope_offset_mm": 10.0}` in DA_setting.json of demo_Danube and demo_Global (envelope default off):
  snow cap per cell = 1.5 x monthly OL max + 10 mm (max over the months of the GRACE window). Applied window-aware in the
  increment partition (`bounds.set_window(upper_window=...)`) and as a safety clip in `Threshold` with bound
  max(cap, forecast) -> never removes forecast snow, only stops the growth.
- Files: EnKF.py, EnKF_localized.py, Threshold.py, bounds.py, configure_DA.py, filter_factory.py, state_envelope.py,
  DA_GRACE.py, doc/configuration.md, both DA_setting.json. Merge (10 Oct): configure_DA.py, EnKF_localized.py, filter_factory.py,
  Threshold.py, bounds.py, state_envelope.py, DA_GRACE.py, da_evaluation.py, doc/configuration.md, both DA_setting.json.
Sandbox tests: cap off bit-identical to the old code (72 cases); cap on: elements above the cap 1920 -> 0, no forced
removal, sub-basin TWS increment unchanged (5e-13 mm), snow share of the increment 98.8 % -> 73.8 % in the test case.
- Inflation (10 Oct): with the snow cap on, the snow noise of the adaptive additive inflation is weighted per cell by
  min(S, cap - S) instead of S (`inflation._room`, cap of the window kept in `EnKF_localized._snow_cap_now`): packs at
  or above their cap get no noise, the sub-basin snow noise goes to the seasonal cells (max 50 % of their snow).
  Cap off bit-identical; mock test: capped cells 20-28 -> 0-5 mm noise, the old mean shift by clipping at the cap
  (up to 18.6 mm) -> 0.7 mm. Files: inflation.py, EnKF_localized.py, doc/configuration.md.
- demo_Global (10 Oct, Fan): `riverstor` removed from the inflation's `split_storages` (now groundwstor, soilmoist, swe);
  riverstor stays in the DA state (`riverstor: true`) and is corrected by the update. River spread comes from the
  routed soil / groundwater perturbations; direct noise would jump discharge and mostly drain before the next window.
  Pilot check: river spread and river increments in the Amazon / large rivers - if the analysis hardly corrects them,
  river can be added back.
**Test on UCloud** = exp-1 in §6 below.

### 6. JOB LIST for Claude Code — exp-1, exp-2a, exp-2b, exp-4 (Fan, 10 Oct 2026). Work AUTONOMOUSLY until all are done
Fan's instruction: run the whole list below on your own, one job after the other, without waiting for him. Decide by
the rules here; write progress into §7 (below) after every step, so Fan and Cowork can follow. Stop only when the
list is done or a stop rule says so. The global pilot (exp-3) is NOT part of this list - Fan decides after the results.

**Prepared by Cowork (10 Oct, committed + md5-verified on the workstation, synced):**
- `src_demo/exp_Ucloud.py` — one driver for all three, chosen by environment variables `EXP` (exp1 | exp2a | exp2b)
  and `STEP` (prepare | run | evaluate). Each experiment is its OWN case (own settings folder, own Res/<case>), so no
  earlier result (run12/run13 in Res/demo_Danube) is touched.
- `settings/demo_Danube_exp1/`  = demo_Danube settings, case demo_Danube_exp1, 2002-01-01..2005-12-31
- `settings/demo_Amazon_A1/`    = demo_Global settings (riverstor in the state, NOT in the inflation), basin Amazon
- `settings/demo_Amazon_A2/`    = as A1, plus `riverstor` in the inflation `split_storages`
  All: 30 members, 2002-01-01..2005-12-31, snow_bounds with envelope 1.5 / 10 mm, block localization, diagonal R.
- Why own cases: `gather_OLmean` needs the collected OL of exactly the DA period, and da_evaluation compares Res_OL.h5
  with Res_DA.h5 of the same case. Each `run` therefore first collects the shared global daily OL (OL_output/Ens_k,
  only read) for its case and period. Note for the comparison with run13: exp1 anchors GRACE and OL on their
  2002-2005 means (run13: 2002-2019); the fit metrics use anomalies, so RMS / correlation are comparable, but
  the increments of the first windows are not identical for that reason alone.

**Rules for you (Code):**
- Do NOT edit `src_*`, `settings/`, or `exp_Ucloud.py`. Nothing in this list needs an edit. If something needs a code
  change, write the error + your diagnosis into §7 and continue with the next independent job (see stop rules).
- Never run `clean_temp_output(Stage.OL)`. No thread-limit environment variables. Plain `mpiexec -n 31`.
- Launch every step from `src_demo/` with `nohup ... > ../temp/exp_logs/<EXP>_<STEP>.out 2>&1 &` (create the folder),
  then poll every ~10 min (`tail` of that file and of `parallel_logs/DA/rank_2.log` / `parallel_logs/collect/rank_2.log`,
  `pgrep -f exp_Ucloud`) until the process has ended; check the exit (last lines, "Abort", "Traceback", "[ERROR]").
- Always `evaluate` right after `run`, before the next `run` (parallel_logs/DA, `<basin>_obs_GRACE.hdf5` and the
  basin's state envelope are overwritten by the next run).

**Job 0 — checks (once):**
1. The synced files: `src_demo/exp_Ucloud.py`, the three settings folders, and the code of 9-10 Oct
   (`src_DA/configure_DA.py` has `def snow_bounds`, `src_DA/inflation.py` uses `_snow_cap_now`). No Syncthing conflict
   copies (`find . -name "*sync-conflict*"`).
2. Shared open loop present for 2002-2005: `ls /work/PyGLDA_v2_external_data/OL_output/Ens_1 | grep -c "daily_output_200[2-5]"`
   = 1461, same for Ens_0 and Ens_30. Restarts of 2001-12-31 in Ensemble_Initialization/Ens_k.
3. `df -h /work` and note the free space (each experiment needs a few tens of GB at most).
4. Host check: `membench.py` alone and with 31 concurrent copies. If the 31-copy mean is > 3 s, wait 15 min and
   repeat; after 3 bad checks start anyway (speed does not change the results) and note it. Repeat this host check
   before every `run` step.

**Job 1 — exp1 (Danube: snow cap + cap-aware snow noise):**
    EXP=exp1 STEP=prepare  python -u exp_Ucloud.py Config_ReWaterGAP.json
    EXP=exp1 STEP=run      mpiexec -n 31 python -u exp_Ucloud.py Config_ReWaterGAP.json
    EXP=exp1 STEP=evaluate python -u exp_Ucloud.py Config_ReWaterGAP.json
Report in §7 (from Res/demo_Danube_exp1/evaluation/exp1/ and the rank logs):
- rank_0.log / rank_2.log: the snow cap is on (EnKF_localized print "snow cap"), the envelope line "swe monthly max"
- snow of the cells 47.25N 12.75E, 47.25N 13.25E, 47.75N 12.75E: ensemble mean and max member at the end of each
  year 2002-2005 (Res/demo_Danube_exp1/DA/Ens_k/daily_output_<year>.nc, swe) vs the same in run13
  (Res/demo_Danube/DA/...) and vs the cap 1.5 x swe_max_month + 10 (Auxiliary/state_envelope_Danube.nc). Expected: no
  growth year to year, at or below the cap.
- RMS / correlation DA vs GRACE, basin and sub-basins: exp1 summary.txt vs run13 restricted to 2002-2005 (compute it
  from run13's monthly_series.csv for 2002-01..2005-12 the same way as summary.txt). Expected within ~10 %.
- chi2, forecast spread, threshold clipping of snow and soil, warnings; snow shares in adaptive_inflation_log.csv vs
  run13 2002-2005; wall time of the run.

**Job 2 — exp2a (Amazon A1, riverstor in the state, NOT in the inflation = current global settings):**
    EXP=exp2a STEP=prepare  python -u exp_Ucloud.py Config_ReWaterGAP.json      (masks with the WaterGAP land mask +
                                                                                 GRACE obs of the Amazon, 18 sub-basins)
    EXP=exp2a STEP=run      mpiexec -n 31 python -u exp_Ucloud.py Config_ReWaterGAP.json
    EXP=exp2a STEP=evaluate python -u exp_Ucloud.py Config_ReWaterGAP.json
**Job 3 — exp2b (Amazon A2, riverstor also in the inflation):** the same three steps with EXP=exp2b (its prepare
does not redo the masks / GRACE obs; its evaluation uses exp2a as reference).
Report in §7, A1 vs A2:
- RMS / correlation vs GRACE, basin and the 18 sub-basins; chi2; forecast spread of TWS
- river storage: ensemble spread (Res_DA.h5 vs Res_OL.h5, sub-basin series), mean increment per window and its share
  of the TWS increment (increment diagnosis), clipping of riverstor (threshold logs in evaluation/<tag>/logs/Ens_k)
- split of the seasonal correction over river / soil / groundwater / snow (summary.txt, fig2)
- discharge smoothness at the outlet (Obidos, about 1.75S 55.75W, the cell with the largest mean discharge near
  there): ensemble-mean day-to-day change on the first day of each window vs the other days, A1 vs A2 (A2 expected to jump)
- inflation shares of river in A2 (adaptive_inflation_log.csv), warnings, wall times.

**Job 4 — exp4 (Danube, exp1 setup over the FULL period 2002-01-01..2019-12-31; added by Fan after exp1, 10 Oct):**
Purpose: the snow ratchet of run13 built up after ~2012, so only the full period shows whether the cap stops it. Same
settings as exp1 (case `demo_Danube_exp4`, `settings/demo_Danube_exp4/`), same anchoring and period as run13, so
exp4 vs run13 differs ONLY by the snow cap + cap-aware snow noise (and the run-to-run seed paths, which are fixed).
    EXP=exp4 STEP=prepare  python -u exp_Ucloud.py Config_ReWaterGAP.json
    EXP=exp4 STEP=run      mpiexec -n 31 python -u exp_Ucloud.py Config_ReWaterGAP.json
    EXP=exp4 STEP=evaluate python -u exp_Ucloud.py Config_ReWaterGAP.json
(`exp_Ucloud.py` now has a per-experiment end date; exp1/exp2a/exp2b unchanged. Check before exp4 that the synced
file has md5 eb57a1ce6fca93c116590d4868821d8d.)
Time: OL collection of 18 years ~10-15 min, DA 6574 days at 1.45-3.3 s/day = 2.6-6 h, collection ~10 min. BEFORE
starting the run step, check the remaining time of the UCloud job: if it is less than (6574 days x the s/day measured
in exp2a/exp2b) + 45 min, do NOT start; write "exp4 waiting for a new UCloud job" in §7 and stop (Fan starts a job
and tells you to continue with exp4 only).
Report in §7, exp4 vs run13 (full period; run13 report Res/demo_Danube/evaluation/run13):
- the three Hohe Tauern cells: swe ensemble mean and max member on 31 Dec of every year 2002-2019, exp4 vs run13,
  and the cap of December / of the maximum month; member-days above the month's cap and worst excess (as in exp1)
- a yes/no: does the ensemble mean of these cells still grow year to year after 2012 (run13: one member 139 -> 694 mm)?
- summary.txt exp4 vs run13 (the report's own comparison): RMS / correlation basin + sub-basins, by season, trends,
  gain, forecast spread, chi2, clipping, warnings; the 2018-10 re-entry event (sub-basins 1-3)
- inflation shares (mean per sub-basin) exp4 vs run13; wall time and s/day.

**Job 5 — snow innovation diagnostic, READ-ONLY (Fan, 10 Oct; after exp4).** Question: does snow in the high Alpine cells
take too much of the GRACE innovation? Data: Res/demo_Danube_exp4 (DA and OL yearly files, Ens_1..30, 2002-2019) and
the exp4 report. No DA runs, no edits of src_* / settings; write your own script under temp/exp_logs/ and run it in tmux.
Cells: C1 47.25N 12.75E, C2 47.25N 13.25E, C3 47.75N 12.75E (all in sub-basin 1); also sub-basin 1 as a whole and the
basin. Windows = the GRACE months of exp4 (179).
1. **Increment per window and storage.** The DA output adds the window increment D_k to every day of window k, so per
   cell, member and storage estimate D_k from the step at the window start: D_k ~ x(first day of k) - x(last day of
   k-1) - (typical one-day change: mean of the day-to-day changes of the two days before and the two days after).
   Check the method on groundwater against the increment diagnosis of the exp4 report (basin / sub-basin means should
   agree within ~10 %). Report: share of the sub-basin-1 TWS increment that goes to snow (area-weighted, ensemble mean
   of |D|, and of signed D), per calendar month; the same for C1-C3 alone (snow vs soil vs groundwater in those
   cells); and the cumulative signed snow increment 2002-2019 in C1-C3.
2. **Spread.** Ensemble std of swe per cell, monthly means, DA vs OL, per calendar month and per year: does the DA
   snow spread in C1-C3 grow above the OL spread over the years? Same for sub-basin 1 snow and for soil / groundwater of C1-C3.
3. **Correlation.** Per window, the correlation across the 30 members between the window-mean swe of C1 (C2, C3) and
   the window-mean TWS of sub-basin 1, in the DA output and in the OL; median and per calendar month.
   (The partition weight of a storage element is variance x max(correlation with the sub-basin TWS, 0).)
4. One figure (temp/exp_logs/job5_snow.png): for C1, OL mean, DA mean, DA spread, OL spread of swe 2002-2019, and the
   cumulative snow increment; plus a table in §7.
Expected effort: < 1 h. Report in §7 with a short verdict: "snow takes too much innovation in C1-C3: yes/no, because ...".

**Job 6 — which perturbed parameter drives the open-loop snow spread? READ-ONLY (Fan, 10 Oct).** Hypothesis (Cowork,
from WaterGAP's elevation bands): the per-member `adiabatic_lapse_rate` (triangle 0.004 / 0.006 / 0.010 K/m) changes
the top-band temperature of C1 / C2 / C3 by 7.9 / 6.8 / 5.5 K between members (lowland cells 0.5 K), so steep-lapse
members keep a perennial pack. Data: the perturbed parameter files
`Ensemble_input/Ens_k/parameters/WaterGAP_2.2e_global_parameters_gswp3_w5e5.nc` (k = 1..30; each perturbed parameter
is one value per member, read it at any land cell) and the open loop of `Res/demo_Danube_exp4/OL/Ens_k` (2002-2019).
No runs, no edits of src_* / settings; script under temp/exp_logs/, in tmux.
1. Per member k: the values of all perturbed parameters (adiabatic_lapse_rate, pt_coeff_humid_arid humid / arid,
   max_daily_pet, critcal_gw_precipitation, swb_outflow_coeff, activelake_depth, activewetland_depth,
   max_canopy_storage_coefficient).
2. Per member and cell C1 / C2 / C3 (OL swe): mean of the annual maxima, mean late-summer minimum (mean swe of
   15 Aug - 15 Sep, averaged over the years), number of years with late-summer swe > 10 mm ("perennial years"),
   and swe on 31 Dec 2019.
3. Table of the 30 members sorted by lapse rate with these numbers; Spearman rank correlation of each parameter with
   the late-summer minimum and with the annual maximum, per cell. Optional: the same for the 6 sub-basin means of swe.
Report in §7 with a one-line verdict: "the OL snow spread in C1-C3 is driven by <parameter>: yes/no".
Expected effort: < 30 min.

**Stop rules / failures:**
- A step fails: read the traceback, write it + your diagnosis into §7. If the cause is the host / a transient I/O
  problem, retry the step once. Otherwise skip to the next job that does not depend on it (exp2a and exp2b do not
  depend on exp1; exp2b depends on exp2a's `prepare` only), and stop when nothing is left.
- Disk below 200 GB free: stop and report.
- When all jobs are done (exp1, exp2a, exp2b, exp4): write a short overall summary at the top of §7 ("ALL DONE <date>"), then stop. Do not
  clean any output, do not start the global pilot.

### 7. Job log (Code writes here)
**ALL DONE 9 Oct 2026 16:30 (UCloud Claude Code, job 12417340); Job 5 (snow innovation diagnostic, read-only) added by Fan and done 17:40 - verdict YES, snow in C1-C3 takes 67-92 % of the cell increments via a variance-driven, self-reinforcing weight (see Job 5 below). Job 6 (parameter vs OL snow spread, read-only) done 17:30 - verdict YES for C1 / C2: adiabatic_lapse_rate (rho 0.89 / 0.75 with the late-summer swe, threshold >= 0.0076 K/m), not applicable for C3 (no OL pack in any member).** Job 0 checks passed; exp1, exp2a, exp2b and exp4 (added by
Fan while exp1-exp2b ran) each prepared,
run (plain `mpiexec -n 31`, nohup) and evaluated without any error; reports in Res/demo_Danube_exp1/evaluation/exp1,
Res/demo_Amazon_A1/evaluation/exp2a, Res/demo_Amazon_A2/evaluation/exp2b (+ figures/ of each case); run logs in
temp/exp_logs/. Wall: exp1 39 min, exp2a 31 min, exp2b 30 min (host fast all afternoon: 1.1-1.5 s per model day).
exp4 (Danube, cap over 2002-2019; run 14:07-16:14, DA 1.08 s/day): fit and filter identical to run13 to the second
decimal; the snow cap pins the max member at the cap (31 Dec 2019: 666 / 282 / 282 mm vs caps 667 / 282 / 300), halves the
member-days above the cap and cuts the worst excess 2-5 x (370 / 267 / 387 vs 884 / 1446 / 685 mm), removes the +21 mm/yr
trend warning; the ensemble mean stops growing in two of the three Hohe Tauern cells, but keeps growing (258 -> 486 mm
2012-2019) in 47.25N 12.75E where 1.5 x the large OL maximum leaves room -> a tighter envelope factor is the next knob.
Headlines: (1) exp1: snow cap on and respected (member-days above the cap a third of run13's, worst excess 40 vs
159 mm), GRACE fit within 10 % of run13 on 2002-2005, inflation snow share 0.15 vs 0.12; the long-term ratchet cannot
be judged on 4 years. (2) Amazon A1 vs A2: fit 2.9 vs 2.7 mm basin RMS (OL 20.4), filter indicators equal, river
carries ~30 % of the TWS increment in both; A2 puts 54 % of the added spread into river, clips 7.6 x more river water
and sits at the inflation cap in 7 sub-basins; the ~9 % discharge jump at the outlet on the first day of each window
exists in A1 and A2 alike (river increment, not noise). (3) Common Amazon issue: chi2 median 0.18 and a fit 4-5 x
tighter than the mascon sigma - over-dispersive ensemble / over-trusted obs in 17-18 sub-basins. Nothing cleaned, no
code or settings edited, global pilot not started. Details per job below.

**Job 0 - checks (9 Oct 11:51, job 12417340, host bit-c26a-07, 32 vCPU / 89 GB, expires 20:38):** PASSED.
- Sync: exp_Ucloud.py, the 9 settings files of demo_Danube_exp1 / demo_Amazon_A1 / demo_Amazon_A2 and the 9-10 Oct code files
  (configure_DA, inflation, state_envelope, EnKF_localized, Threshold, bounds, DA_GRACE, Regional_DA) md5-identical on both
  sides; `def snow_bounds` present, `_snow_cap_now` in inflation.py (2 hits); 0 Syncthing conflict copies.
- Shared OL 2002-2005: 1461 daily files in OL_output/Ens_0, Ens_1, Ens_30; 31 restarts of 2001-12-31.
- Disk: /work filesystem 1.8 PB free; data root 1.9 TB of the 7 TB quota.
- Host check (membench): alone 0.82 s, 31 concurrent mean 3.98 s, max 5.97 s (> 3 s rule -> repeat before the run step;
  the serial prepare is started meanwhile). Load average 133.
- Host check repeated 12:01: alone 0.82 s, 31 concurrent mean 2.30 s, max 3.92 s -> OK.

**Job 1 - exp1 (Danube, snow cap + cap-aware snow noise):**
- prepare (11:52, serial, nohup -> temp/exp_logs/exp1_prepare.out): exit clean, no errors; DA_setting.json of
  demo_Danube_exp1 now: case demo_Danube_exp1, Danube, 2002-01-01..2005-12-31, 30 members, state groundwstor/soilmoist/swe,
  snow_bounds {lower 0.5, upper 2.0 + 20 mm, envelope 1.5 + 10 mm}.
- run (12:02, `mpiexec -n 31`, nohup -> temp/exp_logs/exp1_run.out): started; collect OL -> DA 2002-2005 -> collect DA.
- run finished 12:41:21, exit clean, 0 errors/warnings in exp1_run.out and the rank logs. Wall 39 min: OL collection
  2002-2005 2 min (12:02-12:04), DA 1461 days 35 min (1.45 s/day - this host was fast at that time), DA collection +
  post-processing 1.5 min. 124 yearly DA files, Res_OL/DA.h5, Harmonic_OL/DA/GRACE.nc, GRACE_Danube.h5 in
  Res/demo_Danube_exp1. evaluate 12:41:39-12:41:57: report Res/demo_Danube_exp1/evaluation/exp1/ (reference run13).
- **Cap on:** rank_0/rank_2 banner: `snow cap (monthly open-loop maximum): 1.5 x max_OL(month) + 10 mm`, `snow lower
  bound: 0.50 x forecast`, `snow upper bound: 2.0 x forecast + 20 mm`; filter_summary.json bounds:
  windows_with_absolute_cap swe 42/42, snow_envelope {1.5, 10}. Envelope file Auxiliary/state_envelope_Danube.nc has
  `swe_max_month` (12 x 16 x 41). No separate "swe monthly max" line found in the logs (the banner above is the evidence).
- **Hohe Tauern cells, swe at 31 Dec, ensemble mean / max member [mm], cells (47.25N 12.75E, 47.25N 13.25E, 47.75N 12.75E):**
  ```
             exp1 mean            exp1 max member        run13 mean           run13 max member
  2002   128.3  60.6  26.4     337.6 133.0  53.4     134.9  63.6  25.0     371.2 154.5  40.5
  2003   170.8 103.3  43.7     478.8 189.6  64.6     161.9 107.3  43.4     370.7 214.3  73.4
  2004   123.7 105.3  40.9     457.4 214.2  67.1     162.8 137.2  40.1     517.5 295.1  60.5
  2005   217.2 181.9 155.8     482.9 259.1 194.6     235.3 200.7 155.3     503.9 318.4 198.1
  cap Dec (1.5 x OL monthly max + 10): 551 / 266 / 300 mm; cap of the maximum month: 711 / 384 / 603 mm;
  OL monthly maximum 2002-2019 (envelope): 467 / 249 / 395 mm.
  ```
  Member-days above the month's cap (> 0.5 mm) over 2002-2005, 30 members x 1461 days = 43830 per cell:
  exp1 91 / 697 / 528 (0.2 / 1.6 / 1.2 %), worst excess 36 / 40 / 39 mm (model growth inside a window, the cap is
  window-aware at the analysis and never removes forecast snow); run13 302 / 3716 / 512, worst 126 / 159 / 86 mm.
  So the cap holds (max member 259 mm at the 266 mm Dec cap in cell 2) and the excess is a third of run13's, but in
  2002-2005 the ensemble mean still rises from 2002 to 2005 in both runs (2005 is a big snow year: the OL monthly
  maxima are 467/249/395 mm) - the long-term ratchet of run13 built up after 2012, so the 4-year test cannot show
  "no growth year to year"; it shows "bounded by the OL climatology". A verdict on the ratchet needs the full period.
- **Fit vs run13 restricted to 2002-2005** (run13 numbers computed from its monthly_series.csv over 2002-01..2005-12
  with the series as aligned over 2002-2019; exp1 from its own summary; 42 windows each):
  ```
  unit          exp1 rmsOL rmsDA corr  spread chi2 | run13 rmsOL rmsDA corr  spread chi2
  basin          27.5   6.9 0.995   8.7  0.74 |  29.3   7.1 0.995   8.5  0.99
  sub_basin_1    25.0   8.1 0.995  11.5  0.38 |  25.1   8.0 0.995  11.7  0.33
  sub_basin_2    26.5  10.1 0.987  10.8  0.44 |  28.7  10.5 0.987  11.0  0.45
  sub_basin_3    33.0   8.1 0.996  12.9  0.56 |  33.1   8.2 0.996  13.3  0.48
  sub_basin_4    37.0   9.8 0.987  11.7  0.73 |  37.9   9.6 0.988  12.1  0.87
  sub_basin_5    41.3   9.7 0.994  13.7  0.27 |  42.1  10.4 0.994  13.3  0.47
  sub_basin_6    38.4  13.1 0.979  13.7  0.83 |  45.4  11.9 0.984  13.9  0.77
  ```
  Within 10 % everywhere except sub-basin 6 (+10 %, 13.1 vs 11.9 mm). (Re-aligning run13's GRACE on the 2002-05 OL
  mean instead would give 10.5 mm basin - the anchoring matters more than the code, as the §6 note anticipated.)
  Seasons exp1: DJF 5.5, MAM 9.7, JJA 4.6, SON 6.1 mm. Gain 0.65 (run13 0.71), forecast spread 8.7 (7.4), chi2
  median 0.74 / mean 2.34 (run13 1.36 / 3.84), innovation mean 2.0 mm, RMS 18.2.
- Clipping: soil 0 %, swe 0.7 % (as run13). Largest single-cell increment: swe -341 mm (cell 122, sub-basin 1,
  member 16, 2004-01; run13 +467). Partition weights capped 276 (largest ratio 575), water redistributed 1.2e6 mm.
- Inflation snow shares (adaptive_inflation_log.csv, mean over 42 windows x 6 sub-basins): exp1 groundwstor 0.208,
  soilmoist 0.640, swe 0.152; run13 2002-2005: 0.234 / 0.649 / 0.117. Added obs-space sigma 9.5 / 7.6 / 11.8 / 11.3 /
  13.3 / 17.3 mm (sub-basins 1-6); sub-basin 6 at the cap in 12 % of the windows (run13 11 %).
- Warnings 5: 108 cells with |trend| > 20 mm/yr (a 4-year trend, GRACE itself +5.5 mm/yr basin and -25 / +24 mm/yr
  in sub-basins 1 / 6 - not meaningful for this period); over-fitting flag on sub-basins 1, 3, 4, 5 (regular-window
  criterion, ratios 0.64-0.69); median chi2 < 0.5 in sub-basins 1, 2, 5; 276 weights capped; sub-basin 6 inflation cap.
- Verdict exp1: the snow cap and the cap-aware noise run cleanly, respect the OL-climatology cap and do not change the
  GRACE fit (within 10 %); the inflation puts a slightly larger share into snow (0.15 vs 0.12). The ratchet itself is
  not testable on 4 years.

**Job 2 - exp2a (Amazon A1: riverstor in the state, not in the inflation):**
- prepare (12:43-12:45, serial, nohup -> exp2a_prepare.out): exit clean (only the non-fatal PROJ database warning of
  GDAL). Masks Basin/mask/Amazon/Amazon_res_0.5.h5 (18 sub-basins, 2004 cells) and _res_1.h5 (536 cells, from the
  0.5-degree mask); GRACE/output/Amazon_{signal,gridded_signal,cov}.hdf5 for 42 epochs 2002-04-15..2005-12-15, cov
  42 x 18 x 18. DA_setting.json of demo_Amazon_A1: Amazon, 2002-01-01..2005-12-31, 30 members, state groundwstor /
  soilmoist / swe / riverstor, inflation split_storages groundwstor / soilmoist / swe, snow_bounds with envelope.
- Host check 12:46: load 11, alone 0.44 s, 31 concurrent mean 1.81 s, max 2.63 s -> OK.
- run (12:47, `mpiexec -n 31`, nohup -> exp2a_run.out): started.
- run finished 13:18:05, exit clean, 0 errors in exp2a_run.out. Wall 31 min: OL collection Amazon 2 min (12:47-12:49),
  DA 1461 days 27 min (1.1 s/day; filter 8016 states = 2004 cells x 4 storages, 18 observations, mean taper weight
  0.056), DA collection + post-processing 1.5 min (13:16:40-13:18:05). 124 yearly DA files, Res_OL/DA.h5,
  Harmonic_OL/DA/GRACE.nc, GRACE_Amazon.h5 in Res/demo_Amazon_A1. evaluate 13:18-13:19: report
  Res/demo_Amazon_A1/evaluation/exp2a/ (no reference). Analysis A1 vs A2 follows after exp2b.

**Job 3 - exp2b (Amazon A2: riverstor in the state AND in the inflation):**
- prepare (13:21, serial, nohup -> exp2b_prepare.out): exit clean, no errors; masks / GRACE obs of exp2a reused.
  DA_setting.json of demo_Amazon_A2: Amazon, 2002-01-01..2005-12-31, 30 members, state groundwstor / soilmoist / swe /
  riverstor, inflation split_storages groundwstor / soilmoist / swe / riverstor.
- Host check 13:25: load 17, alone 0.45 s, 31 concurrent mean 2.79 s, max 3.62 s -> OK (< 3 s).
- run (13:26, `mpiexec -n 31`, nohup -> exp2b_run.out): started.
- run finished 13:55:41, exit clean, 0 errors in exp2b_run.out. Wall 30 min: OL collection 1.5 min, DA 1461 days 28 min
  (1.1 s/day), DA collection + post-processing 1.5 min. 124 yearly DA files in Res/demo_Amazon_A2. evaluate 13:56-13:57:
  report Res/demo_Amazon_A2/evaluation/exp2b/ (reference exp2a).

**A1 vs A2 (Amazon, 2002-2005, 42 windows, 18 sub-basins; A1 = exp2a = river in the state only, A2 = exp2b = river
also in the inflation split):**
- Fit to GRACE (window means): basin RMS A1 2.90 mm, A2 2.72 mm (OL 20.4), corr 1.000 both; seasons A1/A2 DJF 2.0/1.8,
  MAM 2.8/2.7, JJA 1.7/1.7, SON 4.1/3.8. Sub-basins (RMS A1 -> A2 [mm]): 1 2.2->2.3, 2 4.0->3.8, 3 0.5->0.5, 4 1.9->1.9,
  5 2.4->2.4, 6 4.2->4.1, 7 8.0->8.4, 8 3.8->3.6, 9 4.1->4.2, 10 4.2->6.6, 11 8.8->8.9, 12 7.6->7.6, 13 8.4->8.8,
  14 6.4->6.6, 15 2.0->2.1, 16 9.2->7.6, 17 7.1->7.1, 18 3.6->2.7. Differences are within noise except sub-basin 10
  (+2.4 mm, worse) and 16 / 18 (-1.6 / -0.9, better). Trends identical (basin -20.5 vs GRACE -20.9, OL -18.9 mm/yr).
- Filter: gain 0.88 / 0.89; forecast TWS spread 16.1 / 16.8 mm vs GRACE sigma 3.8 mm; chi2 median 0.18 both, mean 0.80
  / 0.72 -> the Amazon ensemble is strongly over-dispersive relative to the mascon sigma (the fit is 4-5 x tighter
  than the GRACE error: all 17-18 sub-basins carry the over-fitting flag, and sub-basins 10 and 18 have chi2 medians of
  7-14, i.e. the obs error there is far below the innovations). Innovation mean 2.5 / 2.4 mm, RMS 16 mm.
- River storage spread (member std, time-mean, mm): basin OL 27.3, A1 DA 27.1, A2 DA 28.4; sub-basins with the largest
  change A2 vs A1: 6 23.3->29.3, 7 5.5->10.7, 16 9.7->17.5, 17 8.9->13.8, 13 4.9->8.3, 10 69.3->75.6. River spread in
  the OL is 7-95 mm; the DA leaves it where the river dominates (3, 4, 5, 15: DA > OL because the update adds member
  differences) and shrinks it in the small-river units (7, 11-14, 16, 17) in A1, which A2's river noise restores.
  TWS spread DA is 12 mm (basin) and 16-34 mm per sub-basin in both.
- River increments (increment diagnosis, basin-mean, mm/day at window start): A1 river +0.85, soil +2.22, GW -0.37,
  TWS +2.72 -> river carries 31 % of the TWS increment; A2 river +0.76, soil +1.84, GW -0.02, TWS +2.59 -> 29 %.
  Per sub-basin the river increment reaches +4.1 (A1) / +3.8 (A2) mm/day in sub-basin 6 and +1.7 / +2.0 in 3; the
  within-window drift undoes 1/4 of it (incr + 29 x drift: river +0.22 / +0.20). Largest single-element increment:
  river -1357 (A1) / -1364 mm (A2), cell 380, sub-basin 18, member 8, 2002-04 (first window, EnKF raw 6012 mm cut by
  the partition to 1364). Event windows: only 2002-04-01 in both (first update; river -58..+90 / -62..+106 mm over members).
- River clipping (Threshold, envelope bound): fraction of cells clipped A1 0.7 %, A2 1.0 % (all members); water added
  per member A1 19565 mm, A2 148209 mm (7.6 x), removed 648 / 4114 mm; Ens_1 threshold_log: 74003 vs 82778 clipped
  cells of 3.76 M checks, max change 58 vs 142 mm. The river noise of A2 pushes members below the envelope floor far
  more often.
- Seasonal correction DA-OL (basin, mm): A1 river -4.0 (Mar) .. +9.4 (Jun); A2 -8.0 (Oct) .. +12.9 (Jun); groundwater
  A1 -18 (Sep) .. +2.6 (Apr), A2 -14 (Sep) .. +2.2 (Dec); soil +11 (Jun) both; snow < 1 mm. A2 moves part of the
  autumn/winter correction from groundwater to river (river -7..-8 mm in Sep-Oct where A1 had -3).
- Inflation shares (adaptive_inflation_log.csv, mean over 42 x 18): A1 GW 0.26, soil 0.71, swe 0.03 (river not
  split); A2 GW 0.10, soil 0.34, river 0.54, swe 0.02 -> with river in the split, more than half of the added
  spread goes into river storage (its OL spread is the largest of the storages in most units). Added obs-space sigma
  16.1 / 16.4 mm mean; at the cap: A1 sub-basins 10 (59 %), 18 (57 %), 6 (29 %); A2 18 (64 %), 10 (57 %), 6 / 16 (24 %).
- Discharge at the outlet cell (-1.75, -53.25; largest mean discharge near Obidos, 166 000 units of the file):
  ensemble-mean |day-to-day change| on the first day of a window vs the other days: A1 13 461 vs 1 159 (ratio 11.6;
  8.6 % of the discharge, max 25 %), A2 14 864 vs 1 273 (ratio 11.7; 9.5 %, max 27 %); OL 649 vs 623. So the
  window-start discharge jump of ~9 % is produced by the river INCREMENT of the update (A1 already has it); the river
  noise of A2 adds only +1 % to the jump and 9.4 -> 9.8 % to the member spread of discharge. The expected "A2 jumps"
  is therefore not a distinguishing feature here; the jump is an A1 property too.
- Warnings 14 both: 816 / 793 cells with |trend| > 20 mm/yr (4-year trends in a basin with -20 mm/yr GRACE trend -
  not meaningful), 17 / 18 over-fitting flags, chi2 median 0.18 (spread too large), 4 sub-basins with chi2 outside
  [0.5, 2] (2, 5, 10, 18), largest increment 1357 / 1364 mm, 2669 / 2511 weights capped, inflation at the cap in
  3 / 7 sub-basins (A2 more: 6, 9, 10, 13, 14, 16, 18).
- Wall times: A1 run 31 min (DA 27), A2 run 30 min (DA 28); evaluate ~1 min each.
- Fit per unit, A1 (exp2a) and A2 (exp2b), window means 2002-2005 (RMS in mm; OL numbers identical in both runs by
  construction; corr OL from the monthly_series.csv of each report, the rest from summary.json; added 9 Oct 14:40 on
  Fan's request):

| unit | A1 RMS OL | A1 RMS DA | A1 red. % | A1 corr OL / DA | A2 RMS OL | A2 RMS DA | A2 red. % | A2 corr OL / DA |
|---|---|---|---|---|---|---|---|---|
| basin | 20.4 | 2.9 | 86 | 0.991 / 1.000 | 20.4 | 2.7 | 87 | 0.991 / 1.000 |
| sub 1 | 32.9 | 2.2 | 93 | 0.807 / 0.999 | 32.9 | 2.3 | 93 | 0.807 / 0.999 |
| sub 2 | 26.5 | 4.0 | 85 | 0.950 / 0.999 | 26.5 | 3.8 | 86 | 0.950 / 0.999 |
| sub 3 | 44.5 | 0.5 | 99 | 0.962 / 1.000 | 44.5 | 0.5 | 99 | 0.962 / 1.000 |
| sub 4 | 61.1 | 1.9 | 97 | 0.899 / 1.000 | 61.1 | 1.9 | 97 | 0.899 / 1.000 |
| sub 5 | 41.3 | 2.4 | 94 | 0.964 / 1.000 | 41.3 | 2.4 | 94 | 0.964 / 1.000 |
| sub 6 | 87.9 | 4.2 | 95 | 0.972 / 1.000 | 87.9 | 4.1 | 95 | 0.972 / 1.000 |
| sub 7 | 67.0 | 8.0 | 88 | 0.875 / 0.998 | 67.0 | 8.4 | 88 | 0.875 / 0.998 |
| sub 8 | 43.4 | 3.8 | 91 | 0.956 / 1.000 | 43.4 | 3.6 | 92 | 0.956 / 1.000 |
| sub 9 | 81.9 | 4.1 | 95 | 0.951 / 1.000 | 81.9 | 4.2 | 95 | 0.951 / 1.000 |
| sub 10 | 153.1 | 4.2 | 97 | 0.981 / 1.000 | 153.1 | 6.6 | 96 | 0.981 / 1.000 |
| sub 11 | 41.2 | 8.8 | 79 | 0.931 / 0.998 | 41.2 | 8.9 | 79 | 0.931 / 0.997 |
| sub 12 | 36.3 | 7.6 | 79 | 0.924 / 0.998 | 36.3 | 7.6 | 79 | 0.924 / 0.998 |
| sub 13 | 62.9 | 8.4 | 87 | 0.892 / 0.998 | 62.9 | 8.8 | 86 | 0.892 / 0.998 |
| sub 14 | 44.0 | 6.4 | 85 | 0.982 / 1.000 | 44.0 | 6.6 | 85 | 0.982 / 1.000 |
| sub 15 | 78.1 | 2.0 | 97 | 0.972 / 1.000 | 78.1 | 2.1 | 97 | 0.972 / 1.000 |
| sub 16 | 80.5 | 9.2 | 89 | 0.916 / 0.999 | 80.5 | 7.6 | 91 | 0.916 / 1.000 |
| sub 17 | 59.7 | 7.1 | 88 | 0.954 / 1.000 | 59.7 | 7.1 | 88 | 0.954 / 1.000 |
| sub 18 | 181.3 | 3.6 | 98 | 0.986 / 1.000 | 181.3 | 2.7 | 98 | 0.986 / 1.000 |

- Reading: A2 is marginally better in the fit (basin 2.7 vs 2.9 mm) and keeps the river spread in the small-river units,
  at the price of 7.6 x more river clipping water and inflation at the cap in 7 sub-basins; the discharge jump at the
  outlet is the same in both because it comes from the river increment, not from the noise. Neither run shows a
  shortage of river correction (river carries ~30 % of the TWS increment in both), so Fan's pilot check "river hardly
  corrected -> add river to the inflation" does not trigger for the Amazon; the stronger argument for or against A2 is
  the clipping and the spread at the cap. The Amazon over-dispersion (chi2 0.18, fit 4-5 x below sigma) is common to
  both and a bigger issue for the global run than the river question.

**Job 4 - exp4 (Danube, exp1 setup over 2002-2019; added by Fan on 10 Oct, found in §6 at 14:05):**
- Checks 14:06: exp_Ucloud.py md5 eb57a1ce6fca93c116590d4868821d8d on both sides (expected value from §6), settings
  demo_Danube_exp4 synced (DA_setting a3cffca5..., todate 2019-12-31, 30 members, snow_bounds with envelope, state
  groundwstor/soilmoist/swe), 0 conflict copies, OL_output/Ens_1 6574 daily files 2002-2019. Host check: load 9, alone
  0.45 s, 31 concurrent mean 2.26 s, max 3.33 s -> OK.
- Time rule: 6574 days x 1.15 s/day (exp2a/exp2b pace) = 126 min + 45 min = 171 min < 392 min remaining (job expires
  20:38) -> start allowed. (At run13's worst pace of 3.3 s/day the DA alone would take 6 h and overrun the job; the
  afternoon pace on this host has been 1.1-1.5 s/day.)
- prepare + run launched 14:08 (nohup -> exp4_prepare.out, exp4_run.out).
- run finished 16:14:51, exit clean, 0 errors. Wall 2 h 07 min: OL collection of 18 years 5 min (14:07-14:12), DA 6574
  days 118 min (1.08 s/day - the host stayed fast), DA collection + post-processing 4.5 min. 558 yearly DA files in
  Res/demo_Danube_exp4. evaluate 16:16-16:17 in tmux session `exp4_eval` (Fan, 15:50: always tmux from now on):
  report Res/demo_Danube_exp4/evaluation/exp4/ (reference run13). Banner: `snow cap (monthly open-loop maximum):
  1.5 x max_OL(month) + 10 mm`; filter_summary bounds windows_with_absolute_cap swe 179/179.
  Note: the envelope (Auxiliary/state_envelope_Danube.nc) is rebuilt by every DA_run from the OL of ITS case and
  period, so exp4's caps use the 2002-2019 maxima (Dec cap 667 / 282 / 300 mm, cap of the maximum month 944 / 465 /
  705; OL monthly max 623 / 303 / 463) and are larger than exp1's 2002-2005 caps (551 / 266 / 300). All run13 numbers
  below are against the current 2002-2019 caps.
- **Hohe Tauern cells (47.25N 12.75E, 47.25N 13.25E, 47.75N 12.75E), swe on 31 Dec, ensemble mean / max member [mm]:**
  ```
          exp4 mean              exp4 max member        run13 mean             run13 max member
  2002   134.9  63.6  25.0     371.2 154.5  40.5     134.9  63.6  25.0     371.2 154.5  40.5
  2003   162.0 102.3  43.4     371.3 183.3  73.4     161.9 107.3  43.4     370.7 214.3  73.4
  2004   163.3 124.5  40.2     518.5 272.0  60.8     162.8 137.2  40.1     517.5 295.1  60.5
  2005   236.5 195.4 155.3     509.6 284.1 198.1     235.3 200.7 155.3     503.9 318.4 198.1
  2006   113.4  73.2   9.4     526.7 201.2  24.1     115.7  80.1   9.3     539.9 260.7  24.0
  2007   212.3 152.7  93.0     616.8 265.2 129.0     214.4 168.4  92.8     643.5 355.2 129.2
  2008   256.4 187.7  76.2     599.4 282.1 115.3     259.3 219.1  76.0     634.7 375.4 115.9
  2009   152.9  98.5  38.0     450.7 278.8 100.9     160.3 131.4  37.5     622.5 473.4  97.8
  2010   289.9 223.1 123.1     628.9 320.8 214.5     305.6 305.0 125.6     999.7 685.0 211.2
  2011   399.9 287.9 158.2     738.0 398.7 298.4     417.0 392.0 161.8    1288.3 944.2 365.7
  2012   257.5 276.3 101.0     567.2 365.3 260.0     298.5 303.0 107.7    1238.0 921.9 242.2
  2013   307.7 226.9  71.5     641.7 316.9 225.8     346.0 283.2 107.3    1386.7 860.2 306.4
  2014   392.6 249.1 143.2     694.2 340.7 350.7     413.2 408.7 224.4    1161.6 1156.8 600.7
  2015   264.7 218.9  76.7     656.5 323.5 277.7     244.3 260.9 149.3     877.8 957.7 396.2
  2016   322.4 273.4  91.7     731.1 369.3 407.8     317.8 367.0 194.4    1201.0 1314.1 625.3
  2017   457.7 363.9 186.6     940.6 433.1 438.3     453.7 488.8 314.6    1361.8 1488.5 762.9
  2018   363.6 244.9 132.1     666.6 338.4 254.0     362.9 376.0 222.6     870.0 988.3 646.6
  2019   486.4 238.3  90.2     666.3 282.1 281.5     382.4 421.5 242.6     886.1 926.9 578.0
  ```
  Member-days above the month's cap (> 0.5 mm), 30 members x 6574 days = 197 220 per cell: exp4 5 690 / 38 840 /
  15 995, worst excess 370 / 267 / 387 mm; run13 11 997 / 60 782 / 31 075, worst 884 / 1 446 / 685 mm (ensemble-mean
  days above the cap in run13: 0 / 2 505 / 1 215 of 6 574). On 31 Dec 2019 exp4's max member sits exactly at the
  December cap in all three cells (666 / 282 / 282 vs cap 667 / 282 / 300): the cap is the binding limit; exceedances
  that remain come from snowfall inside a window (the cap never removes forecast snow) and from the per-month cap
  being lower in late autumn than in spring.
- **Open loop in the same cells (Res/demo_Danube_exp4/OL, Ens_1..30; added 9 Oct 16:45 on Fan's request), swe on 31 Dec,
  ensemble mean / max member [mm]:**
  ```
          OL mean                OL max member
  2002   103.9  42.5  23.4     261.7  89.6  39.3
  2003   116.8  64.9  39.5     284.1 108.0  55.6
  2004    79.9  59.3  36.8     268.1 113.9  57.6
  2005   182.6 139.1 153.9     360.8 170.7 193.5
  2006    54.0  19.5   8.2     256.2  65.5  22.1
  2007   139.7  97.9  90.0     348.0 153.8 122.4
  2008   161.1 106.7  66.6     367.8 149.5 100.1
  2009    80.0  34.8  29.8     303.9  82.9  49.3
  2010    97.6  60.6  79.3     332.3 107.9  98.7
  2011   143.0  68.8 111.9     375.0 116.1 147.6
  2012   109.3  70.8  73.5     350.8 122.0 100.8
  2013    94.9  59.3  22.6     365.2 125.6  35.2
  2014    90.9  53.1  58.6     379.3 118.0  76.1
  2015    39.9  19.6   7.2     317.5  80.8  15.9
  2016    45.7  24.1   6.6     339.8  96.3  14.1
  2017   146.9 103.4 106.8     434.9 172.3 138.5
  2018   121.2  72.1  69.2     407.5 138.7 112.0
  2019   138.4  99.1  33.1     432.7 181.4  48.0
  ```
  Answer to Fan's question: NO. The OL mean of 47.25N 12.75E does not grow 2012-2019 - it stays between 40 and 147 mm
  (109 in 2012, 138 in 2019; the 2015-2016 minima of 40-46 mm show that the OL pack melts out in warm years), with no
  trend; only the max member drifts upward slowly (351 -> 433 mm, i.e. one member with a perturbed parameter set keeps
  a perennial pack). exp4's mean in that cell (258 -> 486 mm) is therefore entirely a DA product: the analysis adds
  snow that the model never melts, and the 1.5 x cap on the OL maximum of 623 mm (reached by that single member in
  the OL) does not constrain the ensemble mean. The same holds for the other two cells (OL mean 71 -> 99 and 74 -> 33 mm
  vs exp4 276 -> 238 and 101 -> 90): the OL carries 25-150 mm on 31 Dec with large year-to-year swings, the DA
  2-3 x more and smoother. Implication: a cap tied to the OL *maximum member* is set by the most snow-prone perturbed
  parameter set; a cap on the OL ensemble mean (or a smaller factor) would hold the DA mean near the model climatology.
- **Yes/no - does the ensemble mean still grow after 2012?** Cell 47.25N 12.75E: YES - 258 (2012) -> 486 mm (2019),
  the same level as run13 (382) and above it in 2019, because its cap is generous (OL max 623 mm -> cap 944 in the
  peak month, 667 in December; the pack stays below the cap and the sync/spread mechanism keeps adding). Cell 47.25N
  13.25E: NO - 276 -> 238 mm (run13 303 -> 422; the max member is pinned at 282). Cell 47.75N 12.75E: NO - 101 ->
  90 mm (run13 108 -> 243). So the cap stops the runaway in the two cells where the OL climatology is modest and
  converts the ratchet into saturation at 1.5 x the OL maximum; where the OL itself has a large pack, 1.5 x + 10 mm
  leaves room for a slow build-up that is not distinguishable from run13 at the ensemble mean. A factor of 1.0-1.2
  (or a cap on the ensemble mean rather than per member) would be the next knob.
- **Fit and filter vs run13 (report section 7): identical.** Basin RMS 6.91 vs 6.92 mm, corr 0.996, seasons within
  0.05 mm, sub-basins within 0.17 mm (sub-basin 1: 7.82 vs 7.65), trends identical (-4.54 mm/yr), gain 0.71, chi2
  median 1.33 / mean 3.80 (1.36 / 3.84), forecast spread 7.40 (7.39), innovation mean 1.7 (1.5) mm, key months within
  0.1 mm. Clipping soil 0 %, swe 0.7 % (same). Largest single-cell increment +405 mm swe (cell 111, sub-basin 4,
  member 7, 2006-03) vs +467. Partition: 1198 weights capped (1184), largest ratio 1405 (1405), water redistributed
  6.0e6 (7.5e6) mm. Warnings 5 (5): same list (Vienna cell, 5 degraded GRACE months, over-fitting flags on sub-basins
  1 and 3, weight caps, sub-basin 6 inflation cap); the exp4-specific trend warning of run13 at 47.25N 13.25E (+20.9
  mm/yr) is GONE - only the Vienna reservoir cell remains.
- **2018-10 re-entry event** (first GRACE-FO window): identical to run13 - member 11 basin GW -52 mm, sub-basins 1 / 2 /
  3: GW -141 / -59 / -157 mm, innovation -95 / -88 / -72 mm (run13 -142 / -59 / -157). The event is groundwater, not
  snow, so the cap cannot and does not touch it. Event list (7 windows) identical to run13.
- **Inflation shares** (mean per sub-basin, gw / soil / swe): identical to run13 to three decimals - 1: 0.426 / 0.416 /
  0.158, 2: 0.194 / 0.679 / 0.127, 3: 0.261 / 0.583 / 0.156, 4: 0.191 / 0.709 / 0.101, 5: 0.190 / 0.709 / 0.102,
  6: 0.160 / 0.712 / 0.129; added sigma 11.8 / 12.1 / 15.9 / 13.5 / 18.4 / 18.9 mm. Expected: the shares come from the
  OL spread split, the cap-aware weighting only moves the snow noise between cells inside a sub-basin.
- Figures (16:35, Fan's request): the experiment driver's evaluate makes no GMT figures, so `temp/env_test/viz_exp.py`
  (new test script: imports exp_Ucloud.py with EXP=exp4 to configure RDA, then calls `RDA.visualization()`) was run in
  tmux `exp4_viz` in the pygmt env: Res/demo_Danube_exp4/figures/ now has Harmonic_maps_tws_pixel.png/.pdf (OL | DA |
  GRACE trend, annual amplitude, phase), DA_tws, Components_OL/DA, DA_eval_basin (+ DA_eval_stats.json). Copied to the
  desktop data root. Note the map style is now `pixel` (visualization default after Cowork's change; run12/run13 had
  `smooth`).
- Seasonal correction (basin): swe +1.4..+5.5 mm all year (run13 +1.8..+6.0), groundwater and soil as run13 within
  0.5 mm. Verdict exp4: the snow cap works as designed (max member pinned at the cap, exceedances halved, worst excess
  2-5 x smaller, trend warning gone) at zero cost to the GRACE fit and filter statistics; it does not remove the
  perennial snow where the OL climatology itself is large, and the basin-scale +2..+5 mm snow offset vs the OL stays.

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
- Every rank runs WaterGAP globally. OL writes global daily files `OL_output/Ens_k/daily_output_YYYY-MM-DD.nc`; DA writes the cropped basin box to `DA_output/<case>/Ens_k/` (global unit mask, e.g. demo_Global: the full 360x720 grid).
- Res_<stage>.h5: `time` (decimal years, daily) + `basin/` and `sub_basin_k/` → `<storage>/<member>` daily series in mm, member '0' = unperturbed.

## Technical notes for UCloud
- `RDA.config_external_data()` rewrites every path in the three JSONs of `RDA.setting_dir` to `RDA.external_data_path`. Keep external-data folder names identical.
- `Ens_k` folders are created with `mkdir(parents=False)` → parent folders must exist.
- Environment: py3.10, numpy 1.23.5 ↔ numba 0.56.4, mpich 4.1, nompi hdf5/netcdf; use the env's own mpirun on a single node.
- Model speed: UCloud ~1.0 s per model day per rank (+0.16 s output writing in the OL); workstation ~3.9 s/day (spin-up, after the snow fix).

## Storage for the 30-member run 2002–2019 (31 ranks)
OL_output ≈ 6 575 days × ~5 MB × 31 ≈ 1.0–1.1 TB (kept permanently, global); Ensemble_input ≈ 0.5–0.6 TB; Ensemble_Initialization ≈ 26 GB; DA_output + Res ≈ 40–50 GB. Peak ≈ 2 TB of 7 TB (Danube case; for the global DA see READ FIRST §3 step C). Clean up only the DA temp output (`clean_temp_output(Stage.DA)`, dry run first).

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

## UCloud Claude Code session: changes and suggestions (5-6 Oct 2026; merged into the master note on 8 Oct)
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

- **Step A done: run13 (8-9 Oct, job 12416808, port 2957).** Step 0: all 15 changed files md5-identical on UCloud, no
  conflict copies, WaterGAPLandMask.hdf5 present. Step 1 (config_basin_mask + get_GRACE_obs, new land mask): 3 min,
  exit 0 (a non-fatal PROJ database warning from GDAL). Step 2 (DA_run + collect DA, plain mpiexec -n 31): DA loop
  22:30-04:29 = 5 h 59 min (run12 2 h 43 min; 98 s per window vs ~45 s, all cores busy -> per-day work on every rank,
  not the analysis; py-spy blocked by ptrace), collect 15 min, max RSS 2.07 GB (run12 2.2), exit 0, no warnings in the
  logs. Step 3 (visualization, increment_diagnosis, da_evaluation run13 vs run12, pygmt env): 3 min, exit 0.
  Step 4: `Res/demo_Danube/evaluation/run13/analysis_run13.md` (also on the desktop). Summary: basin RMS 6.9 mm (7.5),
  sub-basins 7.7-10.6 (9.5-10.5), MAM 7.7 (9.6), SON 6.3 (5.2), trends identical, gain 0.71, chi2 median 1.36, spread
  7.4 mm, inflation unchanged, soil never clipped, river clipping gone, innovation mean 4.1 -> 1.5 mm (OL mean over the
  epochs). Warnings 5 (6). **New: the snow increment is now effective (+0.63 mm/day at window start, largest storage)
  and builds perennial snow in 3 Hohe Tauern cells (47.25N 12.75/13.25E, 47.75N 12.75E; DA swe trend 14-58 mm/yr,
  cell 47.25N 13.25E member 1: 139 -> 694 mm, never melts); basin-mean swe DA-OL +2..+6 mm all year (Jul-Sep OL 0.2 mm,
  DA 2.3), plateau ~3-4 mm since 2006. The relative upper bound (2x forecast) plus the new lower bound act as a
  ratchet on a pack that never melts. Suggested (not applied): absolute per-cell cap from the OL climatology, or no snow
  increment in snow-free months/cells, and a per-sub-basin snow-increment line in the report.** Verdict: Step A
  accepted on the criteria; settle the snow ratchet and the 2x runtime before Step B.

### Slowdown run13 (Test 2, 9 Oct, job 12416808 on host bit-c26a-08; plain `mpiexec -n 31` from src_demo, pyglda_v2 env)
**Result: the two per-day additions are NOT the cause (together < 1 s of 460 s). The model step itself is ~2x slower
than in the OL/run12, and the host's memory bandwidth under 31 concurrent ranks is the only measured factor of that
size. Code is unchanged on the per-day path (git diff e04095a..HEAD under src_GHM: only Interface/DailyStepRun.py and
UnitConverter.py). Details below; v2 of profile_da.py worked on a settings copy, `settings/demo_Danube/DA_setting.json`
md5 66f6813d... before and after (identical: True).**

First attempt (08:50, v1 of the script) aborted in `gather_OLmean` (OL series 6574 days vs 151-day period) and left
DA_setting.json with todate 2002-05-31 and a contaminated backup; restored from the desktop copy (md5 66f6813d...).
v2 (settings copy, gather_OLmean skipped) ran at 09:00-09:17.

**Wall time, 151 model days (2002-01-01..05-31), 2 GRACE windows, 31 ranks:** current 460-461 s (3.05 s/day),
nosync 517 s (3.42 s/day) -> nosync is 12 % SLOWER: run-to-run noise of the node is >= 10 %.

Timing tables (per-function wall time, sum over the run / calls):
```
current, rank 1 (wall 461.0 s)                        nosync, rank 1 (wall 516.8 s)
EnKF.predict                 393.24 s  151  2.604/call  EnKF.predict                 460.59 s  151  3.050/call
DailyModelRun.update         393.23 s  151  2.604/call  DailyModelRun.update         460.58 s  151  3.050/call
EnsStates.load_state_dict     37.18 s  363  0.102/call  EnsStates.load_state_dict     39.33 s  363  0.108/call
EnsStates.save_to_nc           1.30 s   61  0.021/call  EnsStates.save_to_nc           0.83 s   61  0.014/call
DailyModelRun.passState        0.98 s  150  0.007/call  DailyModelRun.passState        0.65 s  150  0.004/call
DailyModelRun._sync_snow_subgrid 0.48 s 150 0.003/call  _sync_snow_subgrid             0.00 s  (no-op)
EnKF_localized.update          0.06 s    2  0.031/call  EnKF_localized.update          0.03 s    2
model_state_threshold.threshold 0.06 s  61  0.001/call  model_state_threshold.threshold 0.04 s  61
EnKF._land_ratio_box           0.01 s   61  0.000/call  EnKF._land_ratio_box           0.00 s  (ones)

current, rank 2 (wall 460.9 s)                        nosync, rank 2 (wall 516.9 s)
EnKF.predict                 387.42 s  151  2.566/call  EnKF.predict                 450.50 s  151  2.983/call
DailyModelRun.update         387.42 s  151  2.566/call  DailyModelRun.update         450.49 s  151  2.983/call
EnsStates.load_state_dict     39.11 s  363  0.108/call  EnsStates.load_state_dict     38.36 s  363  0.106/call
EnsStates.save_to_nc           1.22 s   61  0.020/call  EnsStates.save_to_nc           0.75 s   61
DailyModelRun.passState        0.84 s  150  0.006/call  DailyModelRun.passState        0.54 s  150
_sync_snow_subgrid             0.37 s  150  0.002/call  _sync_snow_subgrid             0.00 s
model_state_threshold.threshold 0.06 s  61               model_state_threshold.threshold 0.04 s  61
EnKF._land_ratio_box           0.01 s   61               EnKF._land_ratio_box           0.00 s
```
Per model day on every rank: `DailyModelRun.update` 2.6-3.0 s, of which (pstats) `river_routing` 1.53-1.84 s,
`vert_water_balance` 0.41-0.46 s, `aggregate_potnetabs` 0.36-0.42 s/day equivalent (11-12 s once per month),
netCDF4 `filters` (daily state files, 7587 calls) 0.19 s, `load_state_dict` 0.25 s/day (363 calls = 3 per day).
Rank 1 additionally waits 25 s in `gather` and the workers 20-30 s in `scatter` (2 windows) = the serial analysis
phase, ~12 s per window - small. For comparison: the OL on 5 Oct (job 12411786, host bit-c26a-09) ran the whole model
day incl. output at 1.32 s/day (rank_2.log, 6574 days in 2 h 25 min); run12's DA at 1.49 s/day.
All numba caches were reused (routing.river_routing-39: 4 specializations from 10/11 Sep and 5 Oct, nothing
recompiled on 8/9 Oct); all array layouts are C-contiguous (no slow 'A'-layout specialization) - checked by reading
the .nbi index files. `river_routing`, `vert_water_balance`, `aggregate_potnetabs` are `@njit(cache=True)`, not
parallel, so NUMBA_NUM_THREADS (default 256 here) plays no role.

pstats top 25 by tottime (full text also in temp/env_test/profile/pstats_top25.txt):
```
##### current rank 1  (sorted by tottime, top 25)
Fri Oct  9 09:07:57 2026    temp/env_test/profile/current_rank1.prof
         23946635 function calls (23527518 primitive calls) in 460.940 seconds
   Ordered by: internal time
   List reduced from 5782 to 25 due to restriction <25>
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      151  236.064    1.563  236.069    1.563 /work/PyGLDA/src_GHM/ReWaterGAP/model/lateralwaterbalance/routing.py:39(river_routing)
      151   63.059    0.418   63.063    0.418 /work/PyGLDA/src_GHM/ReWaterGAP/model/verticalwaterbalance/waterbalance_vertical.py:22(vert_water_balance)
        5   53.602   10.720   53.602   10.720 /work/PyGLDA/src_GHM/ReWaterGAP/model/lateralwaterbalance/aggregate_net_abstraction.py:22(aggregate_potnetabs)
     7587   27.406    0.004   27.406    0.004 {method 'filters' of 'netCDF4._netCDF4.Variable' objects}
       14   25.416    1.815   25.417    1.816 {method 'gather' of 'mpi4py.MPI.Comm' objects}
     8931    4.935    0.001    4.935    0.001 {method 'acquire' of '_thread.lock' objects}
    25546    3.221    0.000    3.612    0.000 /work/PyGLDA_v2_external_data/envs/pyglda_v2/lib/python3.10/site-packages/xarray/backends/file_manager.py:206(_acquire_with_cache_info)
     3020    2.907    0.001    2.970    0.001 {method 'createVariable' of 'netCDF4._netCDF4.Dataset' objects}
43222/40297    2.820    0.000    3.002    0.000 {built-in method numpy.core._multiarray_umath.implement_array_function}
    14695    2.226    0.000    2.226    0.000 {method 'copy' of 'numpy.ndarray' objects}
      151    1.978    0.013  294.367    1.949 /work/PyGLDA/src_GHM/ReWaterGAP/model/lateralwaterbalance/waterbalance_lateral.py:511(calculate)
      151    1.492    0.010   72.728    0.482 /work/PyGLDA/src_GHM/Extension/fan_waterbalance_vertical_init.py:170(calculate)
      151    1.148    0.008    3.984    0.026 /work/PyGLDA/src_GHM/Extension/fan_createandwrite.py:265(base_units)
     3645    1.002    0.000    1.222    0.000 {built-in method _operator.getitem}
       61    0.862    0.014    1.299    0.021 /work/PyGLDA/src_DA/ExtractStates.py:131(save_to_nc)
      538    0.809    0.002    0.809    0.002 {method 'close' of 'netCDF4._netCDF4.Dataset' objects}
      151    0.784    0.005    3.121    0.021 /work/PyGLDA/src_GHM/ReWaterGAP/model/land_surfacewater_fraction_init.py:210(adapt_glores_storage)
211236/152116    0.772    0.000    1.109    0.000 /work/PyGLDA_v2_external_data/envs/pyglda_v2/lib/python3.10/site-packages/xarray/core/indexing.py:512(shape)
     3020    0.767    0.000    0.767    0.000 {method 'setncatts' of 'netCDF4._netCDF4.Variable' objects}
        1    0.695    0.695    0.695    0.695 {built-in method _pickle.dump}
3128053/3127934    0.641    0.000    0.956    0.000 {built-in method builtins.isinstance}
     2720    0.620    0.000    0.620    0.000 {method '__deepcopy__' of 'numpy.ndarray' objects}
    56833    0.476    0.000    0.476    0.000 {built-in method posix.stat}
     3020    0.453    0.000    1.148    0.000 /work/PyGLDA_v2_external_data/envs/pyglda_v2/lib/python3.10/site-packages/xarray/backends/netCDF4_.py:67(__setitem__)
   128063    0.436    0.000    1.303    0.000 /work/PyGLDA_v2_external_data/envs/pyglda_v2/lib/python3.10/site-packages/xarray/core/variable.py:230(as_compatible_data)

##### current rank 2  (sorted by tottime, top 25)
Fri Oct  9 09:07:57 2026    temp/env_test/profile/current_rank2.prof
         23931122 function calls (23513682 primitive calls) in 460.841 seconds
   Ordered by: internal time
   List reduced from 5679 to 25 due to restriction <25>
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      151  231.013    1.530  231.017    1.530 /work/PyGLDA/src_GHM/ReWaterGAP/model/lateralwaterbalance/routing.py:39(river_routing)
      151   61.223    0.405   61.227    0.405 /work/PyGLDA/src_GHM/ReWaterGAP/model/verticalwaterbalance/waterbalance_vertical.py:22(vert_water_balance)
        5   55.421   11.084   55.422   11.084 /work/PyGLDA/src_GHM/ReWaterGAP/model/lateralwaterbalance/aggregate_net_abstraction.py:22(aggregate_potnetabs)
        2   29.510   14.755   29.510   14.755 {method 'scatter' of 'mpi4py.MPI.Comm' objects}
     7587   28.881    0.004   28.881    0.004 {method 'filters' of 'netCDF4._netCDF4.Variable' objects}
     8931    5.028    0.001    5.028    0.001 {method 'acquire' of '_thread.lock' objects}
    25546    3.168    0.000    3.555    0.000 /work/PyGLDA_v2_external_data/envs/pyglda_v2/lib/python3.10/site-packages/xarray/backends/file_manager.py:206(_acquire_with_cache_info)
42639/39805    2.826    0.000    3.007    0.000 {built-in method numpy.core._multiarray_umath.implement_array_function}
     3020    2.623    0.001    2.678    0.001 {method 'createVariable' of 'netCDF4._netCDF4.Dataset' objects}
    14675    1.982    0.000    1.982    0.000 {method 'copy' of 'numpy.ndarray' objects}
      151    1.907    0.013  291.027    1.927 /work/PyGLDA/src_GHM/ReWaterGAP/model/lateralwaterbalance/waterbalance_lateral.py:511(calculate)
      151    1.499    0.010   71.144    0.471 /work/PyGLDA/src_GHM/Extension/fan_waterbalance_vertical_init.py:170(calculate)
      151    1.183    0.008    4.122    0.027 /work/PyGLDA/src_GHM/Extension/fan_createandwrite.py:265(base_units)
     3645    1.060    0.000    1.303    0.000 {built-in method _operator.getitem}
       61    0.810    0.013    1.222    0.020 /work/PyGLDA/src_DA/ExtractStates.py:131(save_to_nc)
211236/152116    0.806    0.000    1.157    0.000 /work/PyGLDA_v2_external_data/envs/pyglda_v2/lib/python3.10/site-packages/xarray/core/indexing.py:512(shape)
      151    0.805    0.005    3.171    0.021 /work/PyGLDA/src_GHM/ReWaterGAP/model/land_surfacewater_fraction_init.py:210(adapt_glores_storage)
      538    0.801    0.001    0.801    0.001 {method 'close' of 'netCDF4._netCDF4.Dataset' objects}
3126413/3126294    0.655    0.000    0.974    0.000 {built-in method builtins.isinstance}
     3020    0.650    0.000    0.650    0.000 {method 'setncatts' of 'netCDF4._netCDF4.Variable' objects}
     2720    0.622    0.000    0.622    0.000 {method '__deepcopy__' of 'numpy.ndarray' objects}
    34985    0.491    0.000    0.491    0.000 {method 'getncattr' of 'netCDF4._netCDF4.Variable' objects}
   128063    0.459    0.000    1.347    0.000 /work/PyGLDA_v2_external_data/envs/pyglda_v2/lib/python3.10/site-packages/xarray/core/variable.py:230(as_compatible_data)
    56835    0.448    0.000    0.448    0.000 {built-in method posix.stat}
     2718    0.434    0.000    0.443    0.000 /work/PyGLDA_v2_external_data/envs/pyglda_v2/lib/python3.10/site-packages/xarray/core/indexing.py:1265(__setitem__)

##### nosync rank 1  (sorted by tottime, top 25)
Fri Oct  9 09:16:42 2026    temp/env_test/profile/nosync_rank1.prof
         23942498 function calls (23523503 primitive calls) in 516.771 seconds
   Ordered by: internal time
   List reduced from 5784 to 25 due to restriction <25>
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      151  277.441    1.837  277.446    1.837 /work/PyGLDA/src_GHM/ReWaterGAP/model/lateralwaterbalance/routing.py:39(river_routing)
      151   69.615    0.461   69.620    0.461 /work/PyGLDA/src_GHM/ReWaterGAP/model/verticalwaterbalance/waterbalance_vertical.py:22(vert_water_balance)
        5   63.110   12.622   63.111   12.622 /work/PyGLDA/src_GHM/ReWaterGAP/model/lateralwaterbalance/aggregate_net_abstraction.py:22(aggregate_potnetabs)
     7587   29.590    0.004   29.590    0.004 {method 'filters' of 'netCDF4._netCDF4.Variable' objects}
       14    9.216    0.658    9.217    0.658 {method 'gather' of 'mpi4py.MPI.Comm' objects}
     8931    5.216    0.001    5.216    0.001 {method 'acquire' of '_thread.lock' objects}
43037/40234    4.531    0.000    4.758    0.000 {built-in method numpy.core._multiarray_umath.implement_array_function}
    25546    3.386    0.000    3.828    0.000 /work/PyGLDA_v2_external_data/envs/pyglda_v2/lib/python3.10/site-packages/xarray/backends/file_manager.py:206(_acquire_with_cache_info)
     3020    3.378    0.001    3.452    0.001 {method 'createVariable' of 'netCDF4._netCDF4.Dataset' objects}
      151    3.315    0.022    7.534    0.050 /work/PyGLDA/src_GHM/ReWaterGAP/model/land_surfacewater_fraction_init.py:210(adapt_glores_storage)
    14695    2.630    0.000    2.630    0.000 {method 'copy' of 'numpy.ndarray' objects}
      151    2.299    0.015  346.179    2.293 /work/PyGLDA/src_GHM/ReWaterGAP/model/lateralwaterbalance/waterbalance_lateral.py:511(calculate)
      151    1.722    0.011   80.394    0.532 /work/PyGLDA/src_GHM/Extension/fan_waterbalance_vertical_init.py:170(calculate)
      151    1.477    0.010    4.968    0.033 /work/PyGLDA/src_GHM/Extension/fan_createandwrite.py:265(base_units)
     3645    1.447    0.000    1.724    0.000 {built-in method _operator.getitem}
      538    0.981    0.002    0.981    0.002 {method 'close' of 'netCDF4._netCDF4.Dataset' objects}
     3020    0.897    0.000    0.897    0.000 {method 'setncatts' of 'netCDF4._netCDF4.Variable' objects}
211236/152116    0.837    0.000    1.206    0.000 /work/PyGLDA_v2_external_data/envs/pyglda_v2/lib/python3.10/site-packages/xarray/core/indexing.py:512(shape)
3127929/3127810    0.749    0.000    1.129    0.000 {built-in method builtins.isinstance}
     2720    0.740    0.000    0.740    0.000 {method '__deepcopy__' of 'numpy.ndarray' objects}
     3020    0.558    0.000    1.377    0.000 /work/PyGLDA_v2_external_data/envs/pyglda_v2/lib/python3.10/site-packages/xarray/backends/netCDF4_.py:67(__setitem__)
       61    0.555    0.009    0.824    0.014 /work/PyGLDA/src_DA/ExtractStates.py:131(save_to_nc)
     2718    0.551    0.000    0.561    0.000 /work/PyGLDA_v2_external_data/envs/pyglda_v2/lib/python3.10/site-packages/xarray/core/indexing.py:1265(__setitem__)
    16352    0.530    0.000    0.530    0.000 {method 'reduce' of 'numpy.ufunc' objects}
   128063    0.507    0.000    1.504    0.000 /work/PyGLDA_v2_external_data/envs/pyglda_v2/lib/python3.10/site-packages/xarray/core/variable.py:230(as_compatible_data)

##### nosync rank 2  (sorted by tottime, top 25)
Fri Oct  9 09:16:43 2026    temp/env_test/profile/nosync_rank2.prof
         23926879 function calls (23509561 primitive calls) in 516.897 seconds
   Ordered by: internal time
   List reduced from 5679 to 25 due to restriction <25>
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      151  272.628    1.805  272.633    1.806 /work/PyGLDA/src_GHM/ReWaterGAP/model/lateralwaterbalance/routing.py:39(river_routing)
      151   70.177    0.465   70.181    0.465 /work/PyGLDA/src_GHM/ReWaterGAP/model/verticalwaterbalance/waterbalance_vertical.py:22(vert_water_balance)
        5   58.782   11.756   58.782   11.756 /work/PyGLDA/src_GHM/ReWaterGAP/model/lateralwaterbalance/aggregate_net_abstraction.py:22(aggregate_potnetabs)
     7587   29.199    0.004   29.199    0.004 {method 'filters' of 'netCDF4._netCDF4.Variable' objects}
        2   20.094   10.047   20.094   10.047 {method 'scatter' of 'mpi4py.MPI.Comm' objects}
     8931    5.472    0.001    5.472    0.001 {method 'acquire' of '_thread.lock' objects}
42455/39743    4.569    0.000    4.760    0.000 {built-in method numpy.core._multiarray_umath.implement_array_function}
      151    3.512    0.023    7.865    0.052 /work/PyGLDA/src_GHM/ReWaterGAP/model/land_surfacewater_fraction_init.py:210(adapt_glores_storage)
    25546    3.134    0.000    3.545    0.000 /work/PyGLDA_v2_external_data/envs/pyglda_v2/lib/python3.10/site-packages/xarray/backends/file_manager.py:206(_acquire_with_cache_info)
     3020    2.858    0.001    2.922    0.001 {method 'createVariable' of 'netCDF4._netCDF4.Dataset' objects}
    14675    2.371    0.000    2.371    0.000 {method 'copy' of 'numpy.ndarray' objects}
      151    2.335    0.015  337.017    2.232 /work/PyGLDA/src_GHM/ReWaterGAP/model/lateralwaterbalance/waterbalance_lateral.py:511(calculate)
      151    1.804    0.012   80.885    0.536 /work/PyGLDA/src_GHM/Extension/fan_waterbalance_vertical_init.py:170(calculate)
      151    1.497    0.010    5.054    0.033 /work/PyGLDA/src_GHM/Extension/fan_createandwrite.py:265(base_units)
     3645    1.388    0.000    1.634    0.000 {built-in method _operator.getitem}
      538    0.818    0.002    0.818    0.002 {method 'close' of 'netCDF4._netCDF4.Dataset' objects}
211236/152116    0.815    0.000    1.170    0.000 /work/PyGLDA_v2_external_data/envs/pyglda_v2/lib/python3.10/site-packages/xarray/core/indexing.py:512(shape)
3126289/3126170    0.760    0.000    1.126    0.000 {built-in method builtins.isinstance}
     3020    0.722    0.000    0.722    0.000 {method 'setncatts' of 'netCDF4._netCDF4.Variable' objects}
     2720    0.694    0.000    0.694    0.000 {method '__deepcopy__' of 'numpy.ndarray' objects}
     2718    0.567    0.000    0.578    0.000 /work/PyGLDA_v2_external_data/envs/pyglda_v2/lib/python3.10/site-packages/xarray/core/indexing.py:1265(__setitem__)
    16010    0.545    0.000    0.545    0.000 {method 'reduce' of 'numpy.ufunc' objects}
       61    0.507    0.008    0.750    0.012 /work/PyGLDA/src_DA/ExtractStates.py:131(save_to_nc)
   128063    0.507    0.000    1.456    0.000 /work/PyGLDA_v2_external_data/envs/pyglda_v2/lib/python3.10/site-packages/xarray/core/variable.py:230(as_compatible_data)
    55674    0.480    0.000    1.653    0.000 <frozen importlib._bootstrap_external>:1536(find_spec)
```

Environment of job 12416808 (host bit-c26a-08, 2 x AMD EPYC 9535, 128 cores / 256 threads, SMT on, 72 % scaling
MHz ~3.17 GHz under load): NO cpuset - the job's processes may run on all 256 logical CPUs (`taskset` 0-255); only
`cpu.max` = 32 cores caps the usage (cpu.stat: 340 throttled periods of 339341, 0.5 % of the time); host load average
170-200 all day (other tenants). Storage: `dd` 2 GB direct write 546 MB/s, a 5.5 MB OL daily file read at 2.4 GB/s,
100 x 1 MB create+write+fsync 2.9 ms/file, read 0.3 ms/file -> I/O is not the bottleneck (H1 rejected: the daily state
files cost 0.1 s per load, 0.02 s per save). No non-python process uses CPU (ucmetrics 10 %).
Benchmarks alone vs 31 concurrent single-threaded processes on this host (envs/tools/cpubench.py, membench.py):
pure-python loop 0.24 s -> 0.29 s mean (+20 %, worst 0.35); BLAS matmul 0.18 -> 0.11-0.15 s; **memory-bandwidth loop
(2 x 320 MB arrays, 10 passes) 1.95 s alone -> 5.28 s mean with 31 concurrent, range 2.7-7.6 s (2.7x slower)**; repeated at 09:40 (load 162): alone 1.98 s, 31 concurrent mean 2.52 s, max 3.57 s, min 1.40 s - the contention varies by a factor 2 within 20 minutes, i.e. it follows the other tenants' activity. Job 12417340 (9 Oct 11:38-11:50, host bit-c26a-07, same product, load 100-127, no cpuset): alone 0.53 s (3.7x faster than on bit-c26a-08 with the same CPU model), 8 concurrent mean 0.66 s, 31 concurrent mean 9.53 / 5.31 / 2.24 s (max 15.2 / 6.9 / 3.7 s) in three runs within 12 minutes; 31 copies pinned one per physical core across both sockets (taskset) mean 3.62 s, max 8.9 s - pinning does not help, the variance comes from the other tenants' memory traffic. Conclusion: the per-rank bandwidth of a 32-vCPU job on these shared 256-thread hosts swings by a factor 5-10 within minutes; the DA pace of run13 reflects that, not the code.
WaterGAP's routing and vertical balance stream global 360x720 (and 100x360x720 snow) arrays every day: they are
bandwidth-bound, so 31 ranks on a host whose other tenants already load ~170 threads get a third of the bandwidth.
This matches the observed 2-3x on exactly the model routines, with the small per-day additions unaffected (H3:
machine/storage load outside the job; H2 rejected by nosync; H1 rejected by the I/O numbers).
Not explained with certainty: why the 5 Oct job did not suffer (same product, host bit-c26a-09; its load at 16:44 was
not recorded; the 4-vCPU job's host showed load 630 at 10:03 and still ran the smoke test fine, but with 5 ranks).
Suggested checks (no code): (a) repeat the profile at a quiet hour or on a job with a different host and compare
`DailyModelRun.update` per call - if it drops to ~1.3 s the host is the cause; (b) ask UCloud whether a cpuset /
whole-node reservation is possible for the 32-vCPU product; (c) for the global run, bandwidth per rank matters more
than cores: fewer ranks per host (e.g. 2 x 16-vCPU jobs) is NOT possible with one mpiexec, so prefer a quiet host or a
larger product; (d) a 31-process `membench.py` before each long run is a 30-s check of the host (alone ~2 s,
concurrent mean < 3 s = good host).

Side finding (smoke test rerun 09:3x, `mpiexec -n 5 python -u run_resume.py` in temp/env_test, 5 days): all_ok False -
ranks 1 and 3 (perturbed members) abort inside `river_routing` (ZeroDivisionError, traceback: DailyStepRun.py:560
hot_run_with_daily_timestep -> :442 update -> waterbalance_lateral.py:708 calculate -> rt.river_routing); ranks 0, 2, 4
fine, 5 days in ~18-27 s per rank incl. start-up (5 Oct: 31 s on the 4-vCPU job). Likely cause: the smoke test's own
restart states (temp/env_test/init, 5 Oct 10:11) predate the regenerated Ensemble_input of 5 Oct 11:45 (new parameter
draws), so restart and parameters no longer match; run13's full DA had no such error. Not pursued (test-only data);
worth a look at which perturbed parameter can reach 0 in the routing denominator.


**Job 5 - snow innovation diagnostic (READ-ONLY on Res/demo_Danube_exp4; 9 Oct 17:00-17:40, tmux `job5`, script
`temp/exp_logs/job5_snow.py`, results `job5_snow_results.txt/.json`, figure `job5_snow.png`; all four files also on the
desktop in temp/exp_logs/).** 30 members, 179 GRACE windows 2002-04..2019-12; C1 = 47.25N 12.75E, C2 = 47.25N 13.25E,
C3 = 47.75N 12.75E (all in sub-basin 1, 62 cells); area weights cos(lat); a window = the days from one GRACE month
start to the next (gap months are lumped into the preceding window).
- **1a Validation of the step estimator** (D_k = x(first day of k) - x(last day of k-1) - mean one-day change of the
  two days before and after), ensemble-mean signed D per window [mm], basin, vs the exp4 increment diagnosis (whose
  tug-of-war "increment" is per window, not per day as labelled - it is the jump of DA-OL at the first day):
  tws +0.68 (DA-OL jump +0.99; report +0.989), swe +0.55 (+0.60; +0.603), soilmoist +0.08 (+0.31; +0.305),
  groundwstor +0.05 (+0.05; +0.055); sub-basin 1 groundwstor -0.42 (-0.40; -0.402). Snow and groundwater agree
  within 10 %; the step estimator loses 25-30 % for soil and TWS because the subtracted "typical change" is noisy
  for soil; the DA-OL jump reproduces the report. At the three largest events the estimator gives basin TWS +42.6
  (2006-04), +34.7 (2013-04), -36.3 mm (2018-10) against jumps of +44.8 / +40.5 / -39.3 in the DA-OL series.
- **1b Share of the sub-basin-1 TWS increment taken by snow** (sum over members and windows of |D_swe| / |D_tws|, and
  signed), per calendar month; mean D [mm per window] of tws / swe / soil / gw:
  ```
  month  n   |D| share  signed share   D_tws   D_swe  D_soil  D_gw
    1   15     0.40        0.17         -7.9    -1.4   -0.4   -5.8
    2   14     0.66       -0.09          2.1    -0.2   -0.1    1.7
    3   15     0.49        0.40         11.2     4.5   -0.1    6.5
    4   16     0.44        0.39         10.7     4.2    0.3    6.6
    5   15     0.27        0.23          5.9     1.4    2.3    2.2
    6   14     0.10        0.17          3.2     0.5    0.9    1.1
    7   15     0.04        0.01         -8.2    -0.1   -5.1   -3.0
    8   15     0.03        0.02         -3.5    -0.1   -2.2   -0.6
    9   14     0.04        0.03        -12.5    -0.3   -6.4   -5.4
   10   14     0.04        0.01        -18.5    -0.1   -5.1  -13.6
   11   16     0.12        0.06          4.9     0.3    2.3    2.2
   12   16     0.19        0.43          2.3     1.0   -0.3    1.4
   all 179     0.22  (signed share undefined: the signed TWS sum is ~0)
  ```
  Snow takes 40-66 % of the absolute sub-basin-1 TWS increment in Jan-Apr (signed: 40 % of the positive spring
  increments go into snow), under 5 % in Jul-Oct; 22 % over the year.
- **1c In the cells C1-C3** (share of the cell's own TWS increment, |D|): C1 swe 0.92 / soil 0.01 / gw 0.18, C2 0.77 /
  0.05 / 0.29, C3 0.67 / 0.09 / 0.38 (signed snow share 1.03 / 1.19 / 1.22, i.e. snow receives more than the net TWS
  increment while groundwater is reduced); the snow share stays 0.8-1.0 in every month in C1, and 0.5-0.97 in C2 / C3
  outside Jul-Oct. Mean D_swe +7.1 / +4.1 / +5.5 mm per window (D_tws +6.8 / +3.4 / +4.5).
- **1d Cumulative signed snow increment** (ensemble mean, mm, end of year): C1 105 (2002), 399 (2005), 519 (2009),
  956 (2011), 1142 (2014), 1244 (2017), 1268 (2019); C2 76, 357, 437, 771, 776, 889, 725; C3 26, 248, 331, 692, 880,
  1018, 979. The DA has pumped 0.7-1.3 m of water equivalent into the snow of each cell over 18 years, most of it in
  2003-2005 and 2010-2014; after 2014 the cap (exp4) flattens it (C2 and C3 decline 2017-2019, C1 still +24 mm).
- **2 Spread** (ensemble std of swe [mm], mean of the daily values): C1 DA 131-165 by calendar month vs OL 56-75;
  by year DA 76 (2002), 110 (2004), 138 (2008), 175 (2011), 202 (2014), 204-209 (2017-2018), 173 (2019) vs OL 53-70
  flat. C2 DA 56-72 vs OL 13-28; by year DA 30 -> 68-82 (2010-2019) vs OL 17-24. C3 DA 37-61 vs OL 0-23; by year DA
  15 (2002) -> 85-90 (2014-2016) -> 73 (2019) vs OL 5-17. Sub-basin-1 mean swe: DA 4-10 vs OL 1-7 by month, DA 4.6-9.7
  vs OL 2.6-5.1 by year (no trend). Soil in C1-C3: DA spread smaller than OL in summer (C1 1.3 vs 5.1 in July);
  groundwater: DA 8-14 vs OL 1-11 (DA 3-5 x the OL in C1 / C2, +30 % in C3), without a trend. So YES: the DA snow spread
  in C1-C3 is 2-8 x the OL spread and grows over the years (C1 x 2.7 from 2002 to 2018, C3 x 6) while the OL spread is
  flat; the sub-basin-1 snow spread does not grow - the growth is confined to the perennial-snow cells.
- **3 Correlation across members per window** (window-mean swe of the cell vs window-mean TWS of sub-basin 1): DA
  median 0.03 / 0.03 / 0.09 (positive in 60 / 58 / 61 % of windows), OL -0.16 / -0.09 / 0.02 (13 / 34 / 50 %). By
  month, DA: 0.2-0.4 in Jan-Apr (C3 up to 0.40 in January), ~0 or negative in Jun-Oct; OL: negative in Jun-Dec,
  C3 0.3-0.5 in Jan-Mar. The correlation the partition weights rely on is weak: snow in these cells explains little
  of the sub-basin TWS spread, except moderately in winter / spring.
- **4 Figure** temp/exp_logs/job5_snow.png (C1): DA mean swe ~2 x the OL with a rising floor (summer minimum 25 mm in
  2002 -> 200-300 mm after 2014), DA spread 50 -> 200-240 mm vs OL 45-90 mm, cumulative snow increment a staircase of
  spring windows to +1268 mm.
- **Verdict: snow takes too much innovation in C1-C3: YES**, because (i) 67-92 % of the absolute TWS increment of
  these cells and 40-66 % of the sub-basin-1 increment in Jan-Apr are routed into snow, cumulatively +0.7..+1.3 m per
  cell over 18 years, (ii) the weight comes from the ensemble variance, not from information: the DA snow spread in
  the cells is 2-8 x the OL spread and keeps growing with the pack (each increment widens the member differences,
  which raises the next weight - a self-reinforcing loop), while the member correlation with the sub-basin TWS is
  only 0.03-0.09 in median and 0.2-0.4 in the snow season, and (iii) the sub-basin-wide snow spread and the OL show no
  such growth, so it is a property of the analysis in the perennial-snow cells, not of the model. The cap of exp4
  limits the level (max member pinned at the cap) but not the share: the increments still go to snow until the cap
  binds. Remedies to consider (Fan / Cowork): bound the partition weight of a snow cell by the OL spread (or cap the
  DA snow spread at k x the OL spread), use the OL variance instead of the DA variance in the snow split, or exclude
  cells with a perennial DA pack (summer swe > OL summer max) from the snow increment.


**Job 6 - which perturbed parameter drives the OL snow spread? (READ-ONLY on Ensemble_input/Ens_k/parameters and
Res/demo_Danube_exp4/OL; 9 Oct 17:26-17:28, tmux `job6`, 95 s; script `temp/exp_logs/job6_param_snow.py`, results
`job6_results.txt/.json`, figure `job6_param_snow.png`, console `job6.out`; all on both machines.)** Each perturbed
parameter is one value per member (checked: a single non-NaN value per file, two for pt_coeff humid / arid, split by the
Ens_0 values 1.26 / 1.74). Nominal (Ens_0): lapse 0.006, pt 1.26 / 1.74, max_daily_pet 15, critcal_gw_precip 12.5,
swb_outflow 0.01, lake depth 5, wetland depth 2, canopy 0.3. OL swe metrics: annual max = mean (2002-2019) of the
yearly maximum; late = mean swe 15 Aug-15 Sep averaged over the years; per = years (of 18) with late > 10 mm;
Dec19 = swe on 2019-12-31. Table sorted by lapse rate (sb1 = sub-basin-1 mean):
```
 k   lapse   pt_hum pt_arid  petmax  gwcrit  swbout  lake_d  wet_d   canopy | C1: amax  late  per  Dec19 | C2: amax  late  per  Dec19 | C3: amax  late  per  Dec19 | sb1: amax late
15  0.00439  1.308  1.992   10.79    5.11  0.0128   14.76    2.81  0.643 |    217     2   1     65 |    154     1   1     48 |    181     0   0     14 |    53  0.1
 3  0.00503  1.134  1.926   14.62   19.02  0.0386    6.44    2.76  0.802 |    231     2   1    113 |    164     1   1     95 |    195     0   0     41 |    56  0.1
21  0.00503  0.996  1.667    7.91   13.03  0.0672    5.81    2.90  0.399 |    232     1   1    130 |    163     1   0    101 |    197     0   0     35 |    57  0.1
 6  0.00524  1.341  1.934   16.53   15.65  0.0368   12.94    1.33  0.848 |    226     2   1     99 |    157     1   0     59 |    186     0   0     32 |    53  0.1
29  0.00528  1.276  1.681    9.56   16.63  0.0140    8.65    2.13  0.458 |    224     2   1     87 |    159     1   1     48 |    186     0   0     19 |    54  0.1
27  0.00538  1.342  1.807   19.50   10.93  0.0386    4.07    2.62  0.818 |    223     2   1    114 |    155     1   1     83 |    183     0   0     19 |    53  0.1
10  0.00543  1.380  1.747   17.60   10.75  0.0519   12.99    4.09  0.345 |    228     2   1    128 |    158     1   0     80 |    189     0   0     33 |    56  0.1
11  0.00553  1.149  1.713    9.44   12.82  0.0254    2.79    2.50  0.284 |    225     2   1    161 |    159     1   1    134 |    177     1   0     27 |    52  0.2
23  0.00561  1.509  1.901   14.10   12.73  0.0237    9.65    3.03  0.936 |    222     2   1    112 |    154     1   1    101 |    178     0   0     23 |    53  0.1
16  0.00570  1.040  1.473   16.77    7.41  0.0871    6.63    2.99  0.377 |    228     2   1    105 |    161     1   1     77 |    182     0   0     30 |    53  0.1
18  0.00576  1.341  1.786   16.19   11.74  0.0075    4.45    1.98  0.132 |    229     2   1    117 |    163     1   0     86 |    187     0   0     27 |    57  0.1
24  0.00582  1.269  1.770   13.75    2.58  0.0641   11.00    3.25  0.641 |    220     2   1    124 |    152     1   0     97 |    174     0   0     43 |    51  0.1
12  0.00592  1.190  1.673   16.49   16.40  0.0117   15.31    2.64  0.398 |    234     2   1     76 |    166     1   1     58 |    188     0   0     20 |    56  0.1
13  0.00596  1.070  1.725   11.49   13.55  0.0526    8.17    1.17  0.893 |    226     2   1    135 |    160     1   1    126 |    188     0   0     37 |    57  0.1
 5  0.00609  1.437  1.501   18.15   16.55  0.0291   17.32    0.81  0.241 |    225     2   1    138 |    156     1   1    100 |    176     0   0     38 |    52  0.1
28  0.00615  1.359  1.651   12.07    7.92  0.0260    4.15    4.34  0.525 |    216     2   0    120 |    151     1   0     83 |    186     0   0     37 |    55  0.1
22  0.00667  1.398  1.672   15.13   16.92  0.0079   17.54    1.85  0.614 |    234     4   2    115 |    161     1   0     85 |    188     0   0     23 |    58  0.2
20  0.00694  1.327  1.822   15.88   17.63  0.0186    6.74    1.77  0.350 |    234     8   6    117 |    163     1   1     86 |    187     0   0     36 |    54  0.2
 2  0.00697  1.065  1.730   14.80   11.62  0.0346   12.66    4.63  0.796 |    243    12  10    126 |    166     1   0    104 |    192     0   0     36 |    58  0.3
 8  0.00703  1.016  1.522   11.26   12.41  0.0598   10.75    3.57  0.341 |    243    13  11    141 |    165     1   1    109 |    196     0   0     39 |    57  0.3
26  0.00709  1.465  1.689   14.80   11.88  0.0126   11.18    1.52  0.307 |    242    18  13    117 |    160     1   0    111 |    186     0   0     48 |    56  0.4
 9  0.00715  1.101  1.471   13.33   10.01  0.0400    6.04    1.27  0.735 |    254    27  16    146 |    163     2   1    109 |    181     0   0     40 |    54  0.6
 4  0.00762  1.534  1.758   18.42   13.60  0.0067    8.83    0.97  0.513 |    258    28  18    157 |    168     2   1    105 |    197     0   0     34 |    57  0.6
19  0.00770  1.228  1.674    8.43    9.94  0.0177    6.54    2.03  0.499 |    266    35  18    140 |    171     8   7    103 |    197     1   1     34 |    58  0.9
30  0.00770  1.255  1.497   14.52   13.85  0.0688   14.07    1.30  0.833 |    277    36  18    140 |    182    10   9    111 |    194     0   0     37 |    59  0.8
14  0.00805  1.380  1.938   18.59   15.96  0.0243    5.02    1.81  0.657 |    288    58  18    135 |    180    17  18    108 |    182     0   0     32 |    56  1.3
 7  0.00807  1.177  1.734    9.47   10.22  0.0385    2.19    2.85  0.402 |    309    72  18    137 |    195    26  18    116 |    197     0   0     48 |    59  1.7
17  0.00815  1.218  1.858   16.91    9.92  0.0563   17.05    3.85  0.575 |    302    68  18    147 |    185    18  18    115 |    191     0   0     39 |    59  1.5
 1  0.00851  1.249  1.720   18.79   12.15  0.0536    3.37    2.41  0.453 |    388   161  18    275 |    206    43  18    157 |    185     0   0     34 |    59  3.5
25  0.00900  1.525  1.632   16.33    5.05  0.0205    3.61    2.36  0.554 |    477   254  18    433 |    218    53  18    181 |    189     0   0     40 |    63  5.2
```
- **Ensemble range (min / median / max):** C1 annual max 216 / 233 / 477, late-summer 1 / 2 / 254, perennial years
  0 / 1 / 18, Dec-2019 65 / 127 / 433; C2 annual max 151 / 163 / 218, late 1 / 1 / 53, perennial 0 / 1 / 18,
  Dec19 48 / 101 / 181; C3 annual max 174 / 187 / 197, late 0 / 0 / 1, perennial 0 / 0 / 1, Dec19 14 / 35 / 48.
- **Spearman rho (30 members) with the late-summer swe / annual max:** adiabatic_lapse_rate C1 +0.89 / +0.83,
  C2 +0.75 / +0.74, C3 +0.24 / +0.29 (p < 0.001 in C1 / C2, p = 0.2 / 0.13 in C3); with the perennial-years count
  +0.87 / +0.57 / +0.18, with Dec-2019 swe +0.70 / +0.71 / +0.49. Every other parameter is |rho| <= 0.28 with p > 0.1
  (pt_humid, pt_arid, max_daily_pet, critcal_gw_precipitation, swb_outflow_coeff, activelake_depth, activewetland_depth,
  max_canopy_storage_coefficient), the only exceptions being activewetland_depth vs C3 late-summer (-0.47, p = 0.009)
  and pt_arid vs C3 late (-0.36, p = 0.048), which concern a metric that is 0-0.7 mm in every member (noise; 2 hits in
  54 tests at the 5 % level is the expected false-positive count). After removing a linear fit on the lapse rate
  (R2 0.55 / 0.53 / 0.09 in C1 / C2 / C3) no residual correlation exceeds |0.36| in C1 / C2.
- **Forcing control** (2010, C1, per-member mean of the daily perturbations vs Ens_0): tas offset -0.26..+0.25 K
  (median 0.00), pr factor 0.95..1.04. The forcing noise is drawn per day with no temporal correlation, so a member
  carries no persistent temperature or precipitation bias (expected 2 K / sqrt(365) = 0.1 K) and cannot explain a
  perennial pack; rho of the 2010 offsets with the snow metrics is -0.3..+0 (n.s.).
- **Shape of the relation** (job6_param_snow.png): a threshold, not a line. C1 has no perennial pack for lapse <=
  0.0062 K/m (16 members, late-summer 1-2 mm), a partial one for 0.0067-0.0072 (2-16 perennial years), and a
  pack in all 18 years for >= 0.0076 (8 members, late 28-254 mm); the two steepest members (0.0085, 0.0090) carry
  161 / 254 mm all summer and an annual max of 388 / 477 mm against 216-234 for the lower half. C2 switches at
  0.0077 (17-53 mm for the 6 members >= 0.0080). C3 never builds a perennial pack in the OL (late-summer <= 0.7 mm
  in all members) and its annual-max spread is only +-6 mm; its perennial DA pack (exp4 mean 243 mm at the end of
  2019) is entirely a product of the increments (Job 5), not of the OL spread.
- **Sub-basin means** (6 snowiest sub-basins by annual max; mean +- spread, rho with the lapse rate for late / annual
  max): sb1 55.9 +- 2.8 mm, late 0.64 +- 1.12, +0.88 / +0.66; sb6 46.0 +- 1.7, late 0, +0.77 / +0.06; sb3 39.1 +- 2.2,
  +0.28 / +0.35; sb2 38.2 +- 1.6, late 0.18, +0.87 / +0.61; sb5 34.8 +- 1.8, +0.75 / +0.38; sb4 34.7 +- 1.3,
  +0.71 / +0.35; basin 37.9 +- 1.5, late 0.13, +0.89 / +0.59. The lapse rate is the strongest parameter for the annual
  max in every snowy sub-basin but sb6 (where the spread is tiny and no parameter reaches |0.3|).
- **Verdict: the OL snow spread in C1-C3 is driven by adiabatic_lapse_rate: YES for C1 and C2** (rho 0.89 / 0.75
  with the late-summer swe, 0.83 / 0.74 with the annual max, no other parameter or the forcing noise correlates;
  Cowork's elevation-band hypothesis is confirmed: members with lapse >= 0.0076 K/m, 8 of 30 under the triangle
  0.004 / 0.006 / 0.010, keep a perennial pack in C1 and the two steepest members alone produce the OL annual-max
  spread of 60-70 mm), **and NOT APPLICABLE for C3**, whose OL has no perennial pack and almost no spread in any
  member - there the pack is made by the DA. Implications for Fan / Cowork: (i) the triangle's right tail (0.010 K/m,
  well above the 0.0065 standard) is what creates the perennial-snow members; narrowing it to e.g. 0.005 / 0.0065 /
  0.008 would remove the OL spread that the snow partition weight feeds on in C1 / C2; (ii) because the OL max
  member is itself perennial in C1 / C2, the envelope cap (1.5 x OL monthly max + 10 mm) is set by the two steepest
  members (477 mm annual max in C1 -> cap ~ 725 mm) and therefore binds late; (iii) C3 shows that the cap and the
  lapse-rate tail are not the whole story - the variance-driven weight loop of Job 5 creates a pack even where the OL
  has none.

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
