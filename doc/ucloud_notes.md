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

**Job 7 — check of the new parameter perturbation code (step 1 of the Döll et al. 2024 ensemble; Fan, 9 Oct; added
after Job 6).** Cowork committed (md5-verified on the workstation; sync first, then compare these md5s):
```
e2555d56e8dc85a65ac3ee913ec3293c  src_DA/Perturbation.py
5c60a59cf33d5b591c404b2b5ec28ce2  src_GHM/ReWaterGAP/model/parameters.py                         (CRLF)
f5052d4c9e05ef3513a643148b1567db  src_GHM/Extension/fan_waterbalance_vertical_init.py           (CRLF)
fa4e5eb490f1ecfd14996d2dc2e679d7  src_GHM/ReWaterGAP/model/lateralwaterbalance/waterbalance_lateral.py (CRLF)
48ee58844bb8af23fd29cc2c6611a92b  src_DA/EnKF.py
d10b9be54254ff206f516c1a50ebcb6f  settings/demo_Danube/perturbation.json  (= settings/demo_Global/perturbation.json)
2e0a6b2cb6522d5333ddf92c86238746  README.md
5906846b4c08fd6463f90ff37a3c80ce  doc/configuration.md
```
What changed: every parameter of Döll et al. (2024, Table 1) except the forcing multipliers is now an entry of
perturbation.json (`type` value | multiplier, distribution triangle | gaussian | uniform, optional `seed`). Seven
parameters that are not in the parameter file are new multiplier fields (default 1 everywhere when absent):
`lai_multiplier`, `degreeday_multiplier`, `smax_multiplier`, `gw_recharge_factor_multiplier`,
`max_gw_recharge_multiplier` (applied in fan_waterbalance_vertical_init.py), `netabs_gw_multiplier`,
`netabs_sw_multiplier` (applied in waterbalance_lateral.py). The `par` block of perturbation.json (and of `save_default_json`) now holds ONLY these 20
Döll parameters; the 11 other entries (all off: openwater_albedo, snow_albedo_thresh, runoff_frac_builtup,
areal_corr_factor, reduction_exponent_lakewet, gw_recharge_constant, lake_out_exp, wetland_out_exp,
swb_drainage_area_factor, reduction_exponent_res, stat_corr_fact) were removed (same members with a fixed seed). parameters.py adds the missing ones as ones, so the
EXISTING Ensemble_input files (without these fields) must give exactly the old model. EnKF reads the member's
`smax_multiplier` for the soil bound. The current perturbation.json perturbs the same 8 parameters as before
(all new entries are off), so nothing about the current ensemble changes. Tested by Cowork (perturbation files old
vs new identical with a fixed seed; vertical init bit-identical without multipliers and scaled exactly with them).
NOT yet tested: a running model, in particular the net-abstraction multipliers -> this job.

Rules: READ-ONLY on everything that exists (Ensemble_input, Ensemble_Initialization, OL_output, Res). Write ONLY
under `$W = /work/PyGLDA_v2_external_data/temp_job7/` and your scripts/logs under `temp/exp_logs/job7/`. Do not edit
`src_*`, `settings/`, `exp_Ucloud.py`. Your own driver (in temp/exp_logs/job7/) uses `OpenLoop` exactly as
`RDA.OL_run` does (settings dir `settings/demo_Global`, `configure_time('2002-01-01', '2002-01-31')`,
`configure_Ens_output`, `configure_Ens_input`, `model_resume()`), with ONE difference: in
`configure_ini_for_resume` READ from the real restarts but SAVE into `$W/init_save` (never into
Ensemble_Initialization; OL_run saves into the read folder). OpenLoop's `mkdir(parents=False)`: create the parent
folders yourself. tmux session `job7`, plain `mpiexec -n <N>`.

7a — no-change check (new code + existing member files must reproduce the old open loop):
1. `mpiexec -n 4`: members Ens_0..Ens_3, input = the real `Ensemble_input`, read init = the real
   `Ensemble_Initialization` (2001-12-31), output `$W/7a/Ens_k`, Jan 2002.
2. Compare every output variable of every day of Jan 2002 with `OL_output/Ens_k` (k = 0..3): max |difference|,
   expected exactly 0. If not 0: run the same with the code before this commit (Fan's last GitHub commit, in a
   separate checkout under `$W/oldcode`, not in the working tree) and report which of the two differs from
   OL_output (a difference that already exists with the old code is not caused by step 1).

7b — each new multiplier acts, alone, where expected:
1. Build `$W/7b_input/Ens_k`, k = 0..7: `parameters/WaterGAP_2.2e_global_parameters_gswp3_w5e5.nc` = a copy of the
   real Ens_0 file, plus for k = 1..7 ONE field set to a constant (xarray, `full_like(ds['max_daily_pet'], value)`,
   NaN pattern of max_daily_pet): 1 lai_multiplier 1.5, 2 degreeday_multiplier 1.5, 3 smax_multiplier 1.5,
   4 gw_recharge_factor_multiplier 0.5, 5 max_gw_recharge_multiplier 0.5, 6 netabs_gw_multiplier 2,
   7 netabs_sw_multiplier 2; k = 0 unchanged. `monthly_climate_forcing` of every k = symlink to the real
   `Ensemble_input/Ens_0/monthly_climate_forcing`.
2. Read init: `$W/7b_init/Ens_k` = symlink to the real `Ensemble_Initialization/Ens_0` for every k (all start from
   the same state). `mpiexec -n 8`, output `$W/7b/Ens_k`, Jan 2002.
3. Check: Ens_0 equals 7a Ens_0 exactly; every k = 1..7 runs without error and differs from Ens_0. Report per k the
   global (land-area-weighted) and Danube-mean difference vs Ens_0 on 31 Jan for the storages (canopy, snow, soil,
   groundwater, surface water, TWS) and, if in the output, the fluxes. Plausibility (direction, not size): 1 LAI up
   -> canopy storage up; 2 degree-day up -> snow down where it melts; 3 smax up -> soil storage up; 4 / 5 less
   recharge -> groundwater down, soil or runoff up; 6 groundwater abstraction x2 -> groundwater down in irrigated
   regions (e.g. NW India, North China Plain, US High Plains); 7 surface-water abstraction x2 -> surface water / river
   storage down in irrigated regions. Flag any multiplier whose run is identical to Ens_0 or goes the wrong way.

Report in §7 with a one-line verdict ("7a identical: yes/no; 7b all seven act as expected: yes/no, exceptions ...").
Then stop; keep `$W` until Fan says otherwise. Expected effort: two 1-month runs (4 and 8 ranks), < 1 h.

**Job 8 — parameter sensitivity test and range optimisation for the GLOBAL GRACE DA ensemble (Fan, 9 Oct; added after
Job 7). Work AUTONOMOUSLY through all rounds (8a -> 8b ... -> 8z) until the stop condition, then report.**
Goal (Fan): starting from the reference table below, find the parameter set and ranges that are realistic at both ends,
matter for GRACE TWS, and give an ensemble whose spread fits the OL-GRACE mismatch. You do NOT change `settings/`,
`src_*` or `exp_Ucloud.py`: every parameter file you test is written by your own script into the job folder; at the
end you propose the final table in §7 and Cowork / Fan put it into `settings/demo_Global_v2/perturbation.json`.

Reference table (= `settings/demo_Global_v2/perturbation.json`, md5 58d8568578344af5e38db1984d4e2f5d; all triangular,
middle = standard WaterGAP value; "x" = multiplier of a field of the parameter file, "new" = multiplier written as a
new variable, applied by WaterGAP, default 1; "value" = the one global value of the file is replaced, NaN kept):
```
 id  parameter (json key)                         kind   min      standard  max      acts on
 P1  pt_coeff_humid_arid  humid (cells == 1.26)   value  0.885    1.26      1.65     ET -> soil
 P2  pt_coeff_humid_arid  arid  (cells == 1.74)   value  1.365    1.74      2.115    ET -> soil
 P3  max_daily_pet                                value  6        15        22       ET (arid) -> soil
 P4  smax_multiplier                              new    0.5      1         2        soil capacity
 P5  gamma                                        x      0.5      1         2        runoff vs soil
 P6  snow_melt_temp  (K)                          value  269.4    273.15    276.9    snow
 P7  degreeday_multiplier                         new    0.5      1         2        snow melt rate
 P8  critcal_gw_precipitation                     value  2.5      12.5      20       gw recharge (arid)
 P9  gw_recharge_factor_multiplier                new    0.5      1         2        gw recharge
 P10 gw_dis_coeff                                 value  0.001    0.01      0.02     gw recession
 P11 netabs_gw_multiplier                         new    0.5      1         1.5      gw pumping
 P12 river_roughness_coeff_mult                   x      1/3      1         5/3      river storage
 P13 activewetland_depth (m)                      value  0.5      2         5        wetlands
```
Global DA state: groundwater, soil, snow, river, local + global wetlands (lakes, reservoirs, canopy are NOT in the
state; each member's excluded storages are subtracted from its GRACE value).

**Common set-up of every model run (W = /work/PyGLDA_v2_external_data/temp_job8):**
- Driver: your own script in `temp/exp_logs/job8/`, using `OpenLoop` exactly as in Job 7 (rank r = member r), with
  `setting_dir = settings/demo_Global` (only its `Config_ReWaterGAP.json` is read). Never write into
  Ensemble_input, Ensemble_Initialization, OL_output, Res or settings.
- Input per member: `$W/<round>/input/Ens_k/parameters/WaterGAP_2.2e_global_parameters_gswp3_w5e5.nc` (built from
  the ORIGINAL `Input_data/Parameters/...nc`, see "Parameter files") and `monthly_climate_forcing` = symlink to the
  UNPERTURBED `Ensemble_input/Ens_0/monthly_climate_forcing` (all members the same forcing: only the parameter acts).
- PERIOD (final, Cowork 9 Oct 22:05; Fan let Cowork decide, short enough for many rounds by 10:00): spin-up 3 years
  per member from cold start (no restart read): `configure_time('2000-01-01', '2001-12-31')`,
  `model_spinup(spinup_years=1)` (= one cycle of 2000 + 2000-2001); restarts saved to `$W/<round>/init/Ens_k` (create
  parents; OpenLoop uses mkdir(parents=False)). Then 5 diagnostic years 2002-2006 (GRACE 2002-04 .. 2006-12, ~57
  months; includes the 2003 European and 2005 Amazon droughts). 8 model years per run ~ 60 min at 1.2 s per day.
  NOTE: 3 years do not fully equilibrate slow groundwater (gw_dis_coeff 0.001 1/d: ~3-year e-folding) - see the drift
  flag below.
- Run 2002-01-01 .. 2006-12-31 YEAR BY YEAR (one mpiexec per year, read and save restarts in `$W/<round>/init`):
  after each year compute the MONTHLY means of all output variables per member into
  `$W/<round>/monthly/Ens_k/<year>.nc`, then delete that year's daily files (they are ~0.35 GB per member-month).
  Delete the spin-up daily output too. Check free disk before each year; stop and report below 300 GB.
- tmux session `job8`, plain `mpiexec -n <members>`, no thread-limit variables.

**Parameter files (your script):** copy the original file and change ONE parameter (rounds 8a/8b) or all
(ensemble round): value type -> replace the non-NaN cells (P1: only cells == 1.26, P2: only cells == 1.74; assert the
file holds the standard value as in perturbation_par); "x" -> multiply the field (NaN kept); "new" -> add the variable
`xr.full_like(ds['max_daily_pet'], value, dtype=float)`. For the ensemble round use the code itself:
`src_DA/perturbation_par.perturbation_par` and `src_DA/perturbation_forcing.perturbation_forcing` with a COPY of
`settings/demo_Global_v2/perturbation.json` in `$W` whose `dir` points into `$W` and whose `par` holds the round's
table (do not edit the original).

**Diagnostics (one script, the same in every round; write CSV tables + a short markdown into `$W/<round>/report/`):**
Per member k, from the monthly means 2002-01 .. 2006-12 (GRACE months from 2002-04):
- per DA unit: the global DA is done on the GlobalBasins_v1.0 units (Fan, 9 Oct: "those 727 units"; the frozen
  mask `Basin/mask/GlobalBasins_v1.0/GlobalBasins_v1.0_res_0.5.h5` of settings/demo_Global/DA_setting.json holds 772
  units per the project notes - use the mask as it is, report the number of units you find), area-weighted with
  `shp2mask.unit_weights` (cos-lat, WaterGAP land), ALL medians / percentages below are over these units; and for
  global land: monthly TWS (total_water_storage) and each storage (soil, gw, snow, river, local+global
  wetlands, lakes+reservoirs, canopy).
- GRACE: the monthly mascon grid the DA uses (`obs.EWH_grid_dir` of DA_setting.json) averaged on the same units, the
  same months. Anomalies: subtract each series' mean over the common months. GRACE error per unit: the diagonal of the
  observation covariance the DA uses for the global units (run `GDA.get_GRACE_obs` (src_FlowControl/Global_DA.py) on a COPY of
  settings/demo_Global in `$W` if the global obs file does not exist; if that fails use 20 mm and say so).
- Metrics per unit and member: RMS(TWSA_k - GRACE), correlation, seasonal amplitude ratio A_k / A_GRACE (annual
  harmonic), linear trend; S_k = RMS(TWSA_k - TWSA_0) (sensitivity), the same S per storage; mean level of each
  storage minus member 0; trend of each storage minus member 0 (drift); snow: number of "perennial" cells (Aug 15 -
  Sep 15 mean swe > 10 mm in >= 3 of the 5 years) and the 2006 minus 2002 change of the global snow mass; discharge:
  for each GRACE unit the cell with the largest mean streamflow: mean flow change (%) and the shift of the
  climatological peak month vs member 0.
- Flags per member: NaN / negative storages; drift: |trend_k - trend_0| > 5 mm/yr in any storage over > 5 % of the
  land area (groundwater excepted for P9, P10 and P11: with 3 spin-up years their groundwater is still adjusting, and
  P11 SHOULD change the trend - report these groundwater trends separately instead of flagging); perennial snow cells > 1.2 x
  member 0 or global snow mass growing > 2 %/yr; outlet mean flow change > 50 % in > 10 % of the units.

**Round 8a — one-at-a-time screening, 27 members:** member 0 = standard file; members 2p-1 / 2p = parameter Pp at its
min / max (p = 1..13, order of the table). One spin-up (3 years) + run 2002-2006 with `mpiexec -n 27`.

**Decision rules (apply them yourself after each round; write every decision with its numbers into §7):**
For each end e (min or max) of each parameter, over the GRACE units (median over units unless said otherwise):
1. UNREALISTIC end if any flag above, OR the median RMS vs GRACE is > 25 % above member 0's, OR the median amplitude
   ratio moves outside 0.5 .. 2 while member 0 is inside. -> shrink that end halfway towards the standard value
   (multipliers and P10 halfway in log space: new = sqrt(end x standard); others arithmetic) and re-test only that end
   in the next round.
2. INSENSITIVE parameter if at BOTH ends the median S < 3 mm and S < 10 mm in > 90 % of the units AND the S of the
   storage it acts on is < 3 mm in > 90 % of the units -> candidate for removal (mark it; it is removed only if it
   stays insensitive in the ensemble round).
3. An end is ACCEPTED when it passes rule 1. A parameter whose end has been halved 3 times and still fails -> fix that
   end at the last value that passed, or remove the parameter if neither end passes (say which).
4. Never widen beyond the reference table (no range is extended in this job).
**Round 8b, 8c, ... — re-test only the ends changed by rule 1** (members = 0 plus the changed ends; reuse member 0 of
8a; same spin-up / run / diagnostics). Repeat until no end changes or after 4 rounds.

**Step 8g — generate ALL perturbed forcing ONCE, then reuse it in every forcing test (Fan, 9 Oct 22:15: "produce the
perturbed forcing field first ... use the same forcing field for all the tests to save time").** Run it once, before
8f (it may run while no model round is using the CPUs). Generate with `perturbation_forcing` (correlated mode, `par`
all off, `seed` 2026, months 2000-01 .. 2006-12, MPI over the members) into
`$W/forcing/<config>/Ens_k/monthly_climate_forcing/` (k = 1..n; Ens_0 = symlink to the unperturbed
Ensemble_input/Ens_0 forcing, do not copy it):
- F0 (reference: pr 0.5, tas 2, 300 km, 1 day): n = 30 members (8f uses Ens_1..8, 8z uses all 30);
- F1, F2, F5, F6, F8: n = 8 members each (F3, F4, F7, F9 dropped by Fan).
Never regenerate: 8f and 8z (and its seed-2027 rerun, which then reuses F0 too - drop the second seed if it would need
new forcing) only symlink `monthly_climate_forcing` of their members to these folders. If 8f picks a config other than
F0, generate the 22 missing members of that config once (k = 9..30, same seed, same settings: the first 8 stay
identical) and use them in 8z. Rounds 8a / 8b ... keep the UNPERTURBED forcing (only the parameter acts).
Before generating, measure the size of one member-month and check: (30 + 5 x 8 = 70 members) x 84 months must fit
with > 300 GB left; if not, reduce F1-F8 to 6 members and say so.

**Round 8f — FORCING sensitivity (Fan, 9 Oct 21:50: "forcing field should be included as well"). Run it right after
8a, before 8b.** The forcing noise is random, so it is tested with small forcing-only ensembles, not min / max runs.
Parameters: all standard (the original file, member 0 of 8a). Forcing: the members generated ONCE in step 8g
(`$W/forcing/<config>/Ens_1..8`, symlinked; nothing is generated in 8f). Same 3 + 5 year period and diagnostics.
Configs (8 members each in 8f; F0 is the reference = settings/demo_Global_v2):
```
 config  pr (lognormal std)  tas (gaussian K)  length_km  time_scale_days
 F0      0.5                 2                 300        1
 F1      0.25                2                 300        1
 F2      0.75                2                 300        1
 F5      0.5                 2                 100        1
 F6      0.5                 2                 1000       1
 F8      0.5                 2                 500        1
```
(48 runs: 2 batches of <= 27 ranks, F0-F2 / F5, F6, F8, 8 members each.) Diagnostics per config and unit: forcing-only spread
(std of the members' TWSA, time mean) and its split per storage; bias = RMS(ensemble mean TWSA - member 0 of 8a)
(should be ~0: lognormal mean 1); forcing-only spread-skill R_F = spread / sqrt(max(MSE(member 0 - GRACE) -
sigma_GRACE^2, (5 mm)^2)); the flags above for every member (perennial snow with tas 3 K!).
Decision rules for the forcing: (a) a config is UNREALISTIC if its bias median > 5 mm or any flag; (b) forcing alone
must not over-disperse: keep the largest pr / tas that gives median R_F <= 0.7 and R_F <= 1 in >= 90 % of the units
(parameters and inflation add the rest); (c) L and tau: keep 300 km / 1 day unless F5 / F6 / F7 changes the median
spread by > 30 % - then report it and use the setting closer to median R_F = 0.5 within the bounds of (b); (d) never go
beyond the configs above. Carry the chosen forcing into 8z (and say so in §7).

**Ensemble round 8z — the candidate table as a real ensemble:** 30 members + Ens_0 (31 ranks), triangular with the
accepted ranges, insensitive candidates still in, `seed` 2026, forcing perturbation as in
settings/demo_Global_v2 (correlated mode: pr lognormal 0.5, tas 2 K, 300 km, 1 day), generated with the code (see
Parameter files), period: spin-up 3 years as above, run 2002-2006. Diagnostics per unit: ensemble-mean RMS vs GRACE (vs
member 0), ensemble spread (std of the members' TWSA, time mean), spread-skill ratio
R = spread / sqrt(max(MSE(ens mean - GRACE) - sigma_GRACE^2, (5 mm)^2)); share of the spread per storage; flags as
above for every member. Then:
- remove the insensitive candidates whose removal changes the median spread by < 5 % (from the per-member S, no rerun);
- if the median R > 1.5: the ensemble is too wide -> shrink all multiplier / value ranges by the same factor towards
  the standard (log space for multipliers) so that R ~ 1.2, and rerun 8z ONCE; if the median R < 0.5: report it (the
  adaptive inflation of the DA covers missing spread; do not widen).
**Stop condition:** 8z (or its one rerun) without new flags. Then write at the top of §7: "JOB 8 DONE <date>", the final
table (parameter, kind, min / standard / max, kept / removed, one-line reason), the round-by-round history, median R,
RMS vs GRACE of the ensemble mean vs member 0, and the share of the spread per storage. Keep `$W/*/monthly` and the
reports, delete only daily files. Expected effort: ~1 h per run (8 model years); 8a + 8f (3 batches) + 1-2 re-test rounds + 8z ~ 7-8 h.
**Time frame (Fan, 9 Oct evening): keep Job 8 running without stopping until Saturday 10 Oct 2026, 10:00 (Danish time,
CEST).** If the stop condition is reached earlier, use the remaining time, in this order: (i) rerun 8z with a second
seed (2027) and report how much R, the spread shares and the RMS change (robustness); (ii) nothing else - keep to
the 3 + 5 year period, do not extend it. If the UCloud
job would expire before 10:00, write a resume note in §7 (round, member, year, restart folder) so it can be continued.
Update §7 after every round, so Fan and Cowork can follow during the night.
**Sync with Cowork during the night:** Cowork reads §7 about every 2 hours (until 10:00) and may add instructions in
the block "Job 8 — Cowork review" right below. BEFORE starting each new round (and before 8z), re-read that block
from the synced file; its instructions override the decision rules above. Answer each review point in §7.

**Job 8 — Cowork review (Cowork writes here; newest first):**
- 11 Oct 12:20 (Cowork): thanks - fixed: settings/GDA_Ucloud/perturbation.json dir.out = .../Ensemble_input, new md5
  1421d1eb23668ffcb08c255128a1530e (committed, syncing). Rerun the stage chain from model_perturbation once it has
  synced. Your two code gaps (mkdir parents for preprocess_res / Ensemble_input / Ensemble_Initialization / OL_output
  / DA_output) are noted for Fan; the empty folders you re-created are the right workaround for now.
- 11 Oct 12:00 (Cowork, ORDER FROM FAN - read before continuing Job 9): **NO '_v2' / version suffixes anywhere; the case
  is renamed GDA_Ucloud.** Fan: outputs always go to the SAME folders so the next run overwrites them (saves space);
  this applies to settings as well. Done on the workstation (synced): new `settings/GDA_Ucloud/` = production
  perturbation.json (md5 24faa2cb), DA_setting.json (md5 83e729c1: case GDA_Ucloud, OL_output, DA_output,
  Ensemble_input, Ensemble_Initialization, GRACE/output, GRACE/OLmean - plain names) and Config_ReWaterGAP.json
  (56a3dcbf); new driver `src_demo/GDA_Ucloud.py` (md5 22b0fc62, case GDA_Ucloud, otherwise = demo_global.py).
  Fan deletes settings/demo_* (all of them, incl. demo_Global and demo_Global_v2) and src_demo/demo_global.py himself.
  FOR JOB 9: (1) stop the running step 1 (tmux job9) if it still writes into GRACE/output_v2; delete
  GRACE/output_v2 and any other *_v2 folder you created; (2) wait until settings/GDA_Ucloud/ and
  src_demo/GDA_Ucloud.py have synced with the md5s above; (3) rerun Job 9 from step 1 with `GDA_Ucloud.py` and
  setting_dir settings/GDA_Ucloud - everything else in the Job 9 spec unchanged (read 'demo_Global_v2' there as
  'GDA_Ucloud' and drop every '_v2'). Results: Res/GDA_Ucloud/, GRACE obs: GRACE/output/GlobalBasins_v1.0_*.
  Rule from now on for all jobs: never create versioned copies of output folders; overwrite in place.
- 10 Oct 16:55 (Cowork, ORDER FROM FAN, extends 16:40): **also clear the DEMO files**: every folder or file under
  /work (any depth, outside Input_data / GRACE/output* / Basin) whose name contains 'demo' or 'Demo' (demo data
  downloads, demo case outputs, demo products), and the public data product output of 6 Oct. List them with sizes
  in the same table before deleting. Still KEEP: Input_data, GRACE/output*, Basin, Res, temp/exp_logs.
- 10 Oct 16:40 (Cowork, ORDER FROM FAN, extends 15:35): **also DELETE ALL OUTPUT OF THE PREVIOUS EXPERIMENTS**
  (Danube, Amazon, Job 1-7 tests). Delete under /work/PyGLDA_v2_external_data: temp_job8 (as 15:35), every
  OL_output*, DA_output*, Ensemble_input*, Ensemble_Initialization*, every temp_* / test folder, and GRACE/OLmean*
  (regenerable). KEEP: Input_data, GRACE/output* (processed observations), Basin, Res (the small result series of
  run12 etc. - the paper figures come from them; Fan may order their deletion separately), temp/exp_logs (reports).
  Procedure: `du -sh` of each folder before, delete, then write here the table 'folder | size | deleted / kept',
  the total freed and 'ALL OUTPUT DELETED <time>'. Then STOP; do not extend the job.
- 10 Oct 15:35 (Cowork, ORDER FROM FAN: UCloud storage is full): **STOP after this and DELETE ALL JOB 8 OUTPUT.**
  1. Make sure nothing is running (kill the tmux session job8 if any window is still active).
  2. Record `du -sh /work/PyGLDA_v2_external_data/temp_job8` (one number, before).
  3. Delete /work/PyGLDA_v2_external_data/temp_job8 COMPLETELY: all rounds (inputs, init / restarts, monthly files,
     reports copies), all forcing sets under 8g (F0-F11, the seed-2027 sets), everything. The forcing for the
     production run is regenerated by the code from settings/demo_Global_v2/perturbation.json when the DA starts.
  4. KEEP: temp/exp_logs/job8/ (scripts, markdown reports, CSV tables - the evidence for the paper; it is small) and
     the project folders Ensemble_input, Ensemble_Initialization, OL_output*, DA_output*, Res, GRACE, Input_data,
     Basin - do not touch them.
  5. Write here: the size freed, `du -sh` of the remaining top-level folders under /work/PyGLDA_v2_external_data,
     and 'JOB 8 OUTPUT DELETED <time>'. Then STOP; do not start anything else and do not extend the job.
- 10 Oct 13:50 (Cowork; river check + 8z5 read): (1) RIVER: accepted as is - the big rivers carry 0.5-0.95 of the
  seasonal spread with river spread 1.2-2.7 x sigma_GRACE; the weak floodplain units are a WaterGAP storage limit, not
  a P12 range question. P12 stays 1/3 / 1 / 5/3. (2) 8z5: **log-symmetric table ADOPTED**; I agree that the outlet-flow
  criterion is soft for joint ensembles (designed for OAT ends; the same members sit at 8-13 % in 8z4 unflagged).
  Applied to settings/demo_Global_v2/perturbation.json: P1 humid 1.073 / 1.26 / 1.48, P4 smax 0.707 / 1 / 1.414,
  P10 gw_dis_coeff 0.005 / 0.01 / 0.02, P13 wetland depth 1 / 2 / 4 m; forcing F11 (pr lognormal 1.0, tas 1 K,
  300 km, 1 d); seed 2026. Local test: 30 members, all 13 values in range. This is the PRODUCTION TABLE; please
  rewrite the 'Proposed table' in the JOB 8 DONE block to it (kept / reason columns updated) and add a final line
  'production = 8z5 table + F11; expected: spread ~22 mm, R_state ~0.49, K_eff ~0.56, mean shift ~ -1 mm, snow bias
  ~ -1 mm'. (3) After 8fL: report the outlet-flow metric for the 8z5 members at the > 15 % level for the record, then
  write 'JOB 8 CLOSED' and stop; nothing else is needed. Thanks - excellent work through the night.
- 10 Oct 12:45 (Cowork, Fan's request): RIVER-STORAGE CHECK, diagnostics only (from the 8z4 monthly files, no new
  run; do it as soon as a window is free, before or alongside the L test). Fan: 'river storage should have a
  reasonable spread, large enough for the flooding areas'. Report, per unit for (a) the flood units (wetland
  amplitude >= 10 mm) and (b) the large-river units - the unit containing the outlet / main stem of Amazon, Congo,
  Ganges-Brahmaputra, Mekong, Mississippi, Parana, Niger, Zambezi, Ob, Yenisei, Lena, Danube, Yangtze (name them by
  the mask id and the river you identify from the largest mean streamflow): member-0 river seasonal amplitude (mm),
  GRACE seasonal amplitude, river spread (time-mean member std, mm), river spread / river amplitude, river share
  of the SEASONAL spread (covariance-consistent), the same for wetlands, and the ratio river spread / GRACE sigma.
  Also the distribution over the 178 flood units: median and p10 / p90 of river spread (mm) and of the river share,
  and how many flood units have river + wetlands >= 0.5 of the seasonal spread. One table in section 7 plus a CSV in
  $W/8z4/report/. Decision afterwards is Cowork's (candidate lever: P12 river roughness range).
- 10 Oct 12:30 (Cowork; 8z4 read): DECISION OVERRIDES the 10:10 rule: **tas 1 K is ADOPTED** - it removes most of
  the snow bias (-3.9 -> -1.2 mm) at zero cost in spread, R, K and RMS, and the long-term snow share is a
  precipitation effect that no tas value changes (your correction noted, thanks). Applied to
  settings/demo_Global_v2/perturbation.json (tas coefficients [1.0]; tested locally: offset std 0.996 K, pr
  unchanged, Ens_0 unchanged). 8z5 continues with F10 as launched (the tas amplitude does not touch spread / R /
  mean shift, so its decision rule holds); the production forcing is F11 whatever 8z5 decides. The L test stays
  as queued. Also please add to the 8z5 report: the snow-unit long-term snow share and the wetlands flood-unit
  share, so the final table has all T1-T5 numbers in one place.
- 10 Oct 12:07 (Cowork): job extension to 18:26 seen, thanks. YES to the optional L test after 8z5 (F5 / F6 / F8 vs
  F2, forcing-only, 8 members each): report per L the median TWS spread, R_F, and the spread share of the DA state
  vs the excluded storages; no decision rule, it is documentation for the paper (300 km stays unless a run shows a
  flag or a clearly better spread-per-realism trade-off, which I will judge). Waiting for 8z4.
- 10 Oct 10:25 (Cowork; 8z3 read): (1) SYNC NOTE: an earlier version of this file briefly overwrote your 8z3 section
  (10:07) - restored above/below from my copy; please check it is complete. (2) 8z4 (tas 1 K, applied table) as
  started at 10:17: fine, keep the 10:10 rule. (3) ONE MORE ROUND AFTER IT, **8z5 = log-symmetric table** (your 8z3
  suggestion (b); the +4 / +15 mm mean shift and the two seed-2027 flags are a bias the filter would 'correct'
  every window), seed 2026, forcing = whichever tas passes in 8z4 (F11 if adopted, else F10), 30 members, same
  period / spin-up / diagnostics. Changed ends (all others as applied): P1 PT humid max 1.65 -> 1.48
  (= 1.26^2 / 1.073); P4 smax max 2 -> 1.414; P10 gw_dis_coeff min 0.00316 -> 0.005; P13 wetland depth 0.5 / 2 / 5
  -> 1 / 2 / 4 m (P5, P7, P9 already log-symmetric; P2, P6, P11, P12 arithmetic-symmetric; P3, P8 arid only,
  unchanged). DECISION RULE 8z5 vs 8z4: adopt if ALL hold: (i) ensemble-mean minus member-0 TWS level <= 5 mm
  median and <= 25 mm at p90; (ii) median TWS spread and R_state each drop by <= 15 %; (iii) T2 and T4 still pass;
  (iv) no flag. Report the mean shift per storage (soil, gw, wetlands, snow) and the 8z2-style table in section 7.
  If it fails on the mean shift alone, say so: the fallback is a log-triangular draw in the code (Fan's decision).
  Cowork applies whatever passes to perturbation.json. Fan has been asked to extend the job.
- 10 Oct 10:10 (Cowork, Fan: 'do what you think is right'): ONE MORE ROUND IF THE JOB IS STILL ALIVE (Fan is asked to
  extend it; if it expires at 10:27 before you can start, write 'NOT RUN' here and stop). Round 8z4 = final table
  (seed 2026, table as applied in settings/demo_Global_v2/perturbation.json) with forcing F11 = pr lognormal 1.0,
  tas gaussian 1 K (instead of 2 K), L 300 km, tau 1 d, 30 members, same period / spin-up / diagnostics as 8z2.
  Reason: the long-term snow share (0.26 in the snow units, target <= 0.15) and the -3.9 mm snow bias come from the
  tas noise through the rain/snow and melt thresholds; parameters contribute 0.07 only. Generate F11 under $W/8g
  with the code (copy of the json, tas coefficients [1.0]). DECISION RULE (8z4 vs 8z2): adopt tas 1 K if ALL hold:
  (i) long-term snow share in the snow units <= 0.15 and snow bias improved; (ii) median TWS spread and R_state each
  drop by < 5 %; (iii) seasonal snow share stays within 0.10-0.40; (iv) no new flag. Otherwise keep 2 K. Report the
  same table as for 8z2 in section 7. Cowork applies the forcing change to perturbation.json itself if it passes.
  Still first: append the 8z3 (seed 2027) result. Optional, only after 8z4 and only if time remains: L test F6
  (500 km, 8 members, forcing-only) vs F2.
- 10 Oct 08:46 (Cowork): JOB 8 DONE read. The proposed final table (8z2: P1 humid 1.073/1.26/1.65, P4 0.707/1/2,
  P6 272.2/273.15/274.1 K, P10 0.00316/0.01/0.02, all others as the reference; pr lognormal 1.0, tas 2 K, L 300 km,
  tau 1 d; seed 2026) is APPLIED to settings/demo_Global_v2/perturbation.json (md5 7310531413de75b43b6ad50d7b4b8e94,
  tested locally: 30 members in range, forcing factor mean 0.997 / std 0.994). Please append the 8z3 (seed 2027)
  result to section 7 when it is done; no further rounds needed. Open points (tas 2 K, gw long-term noise in the
  inflation, R_state 0.53, mask path) go to Fan.
- 10 Oct 07:10 (Cowork, on F10): agreed, 8z2 with F10 goes ahead. DECISION RULE F2 vs F10 for the final table
  (8z1 vs 8z2, joint ensembles): take F10 only if ALL hold: (i) median R_state at least 15 % higher than with F2;
  (ii) median ensemble-mean RMS vs GRACE at most 3 % above member 0; (iii) systematic ensemble-mean TWS bias (after
  removing the 30-member sampling part) <= 5 mm and seasonal amplitude ratio within 0.95-1.05; (iv) no new flag.
  Otherwise F2. Report the numbers of both side by side in the summary; Cowork applies the chosen table.
- 10 Oct 05:35 (Cowork, on 8z-P / decomposition - very useful, thanks):
  1. Since the snow LONG-TERM share that fails T3 comes entirely from the 2 K temperature noise (parameter part passes
     T3), narrowing P7 cannot fix it. Let 8z1 run (it documents the effect), but for the FINAL table keep P7 degree-day
     0.5 / 1 / 2 unless 8z1 shows a clear gain (snow long-term share down by > 0.03 or snow bias closer to 0); P6
     stays symmetric 272.2 / 273.15 / 274.1 K. T3 long-term is reported as a forcing-temperature effect; Cowork asks
     Fan whether tas 2 K stays (it is fixed by Fan). If 8z2 (F10) runs, build it on the table WITHOUT the P7 change.
  2. T1: the long-term gw spread is forcing-driven (precipitation noise through recharge) - reported; no parameter
     lever. Keep as the final statement for the report.
  3. Final report (JOB 8 DONE): include this decomposition table, the history of every end, and the final table with
     the chosen forcing (F2 or F10).
- 10 Oct 04:00 (Cowork, on 8c / 8z / 8z1 - please apply before 8z1 starts):
  1. 8c: P6 max 274.1 K accepted, symmetric 272.2 / 273.15 / 274.1 K for the reruns: agreed.
  2. Negative-storage flag: excluding LAKES is right (WaterGAP lets lake storage fall to -S_max). WETLANDS: keep them
     excluded from the flag for now, but REPORT for member 0 the land fraction and the number of units with negative
     local / global wetland storage (min monthly) - wetlands are in the DA state with a lower bound 0
     (src_DA/bounds.py), so if WaterGAP wetlands can be negative, that bound must be revisited before the pilot.
  3. 8z1 - CHANGE: keep the snow levers (P6 symmetric + P7 degree-day 0.7 / 1 / 1.4) but DROP the smax / gamma shrink.
     Reason: R_state is 0.44 (spread far below the mismatch) and Fan's first aim is a high weight on GRACE; shrinking
     soil-side ranges lowers the total spread further and buys the gw share only by removing soil spread. T1 is
     treated as a model-structure result (in WaterGAP the interannual TWS spread sits in soil / wetlands / lakes);
     Cowork takes the gw long-term question to Fan as a DA-side choice (e.g. temporally correlated groundwater noise in
     the inflation). 8z1 = 8z with only P6 + P7 changed.
  4. New forcing test F10 = pr lognormal 1.0 (tas 2 K, L 300 km, tau 1 d), 8 members, generated with idle cores during
     8z-P / 8z1 and run as soon as cores are free (member 0 + 8). Rule (a) as before (no flag, systematic bias < 5 mm
     after removing the sampling part, R_F <= 1.5 in >= 90 % of units). If it passes, 8z2 = 8z1 with F10 forcing
     (30 members generated then) - this is the one lever left for T5 within realistic limits; report R_state, K_eff and
     the shares for 8z2 vs 8z1.
  5. From 8z-P, report which source carries the LONG-TERM spread of each storage (parameters vs forcing), especially gw.
- 10 Oct 02:25 (Cowork, on 8b / 8z):
  1. 8b decisions agreed (P1 min 1.073, P4 min 0.707, P10 min 0.00316 accepted; P6 max -> 8c at 274.1 K).
  2. Keep 8z running as launched (it is the baseline of the reruns). But P6 is now very lopsided: 269.4 / 273.15 /
     274.1 K has its mean at 272.2 K, i.e. the ensemble melts ~0.9 K too early on average -> less snow in the
     ensemble mean than in the standard model (a snow bias the DA would have to remove). RULE for P6 from the first
     rerun on: make it SYMMETRIC around 273.15 K with the half-width d = the accepted distance of the max end
     (274.1 K passes in 8c -> 272.2 / 273.15 / 274.1; if 8c fails -> 273.6 K, d = 0.45). Report the snow bias of the
     8z ensemble mean (snow units, mean storage minus member 0) so the effect of the lopsided range is documented.
     This also acts in the direction of T3 (less snow spread).
  3. Same check for the other ranges (report only, no change): ensemble-mean minus member-0 TWS level and seasonal
     amplitude per unit in 8z (median, p10 / p90), to see whether the asymmetric multipliers (gamma, recharge
     factor, Smax 0.707-2, PT humid 1.073-1.65) shift the ensemble mean.
- 10 Oct 01:20 (Cowork, on 8f):
  1. F2 (pr lognormal 0.75) for 8z: AGREED (largest realistic spread, systematic bias 3.5 mm, R_F 0.41; daily 75 %
     precipitation error at 0.5 deg is plausible; Fan wants a high weight on GRACE). Your bias / sampling-noise split is right.
  2. ADD for 8z / 8z-P / 8z1...: lakes, reservoirs and canopy are NOT in the DA state; each member's excluded storages
     are subtracted from its GRACE value, so their spread acts as extra observation noise. Report, per unit, the spread
     of the STATE part (gw + soil + snow + river + wetlands) s_state, of the EXCLUDED part s_excl, and the effective
     weight on GRACE K_eff = s_state^2 / (s_state^2 + sigma_GRACE^2 + s_excl^2) next to K. Judge T5 (R) on s_state.
     The forcing puts a lot of long-term spread into lakes (dominant in 180 units): flag units where s_excl > s_state.
  3. Snow (T3 high in the snow units already from the forcing: 0.44 seasonal, 0.31 long-term): judge it on 8z with the
     snow levers. If the levers cannot bring the snow share below ~0.45 seasonal / 0.3 long-term, do NOT change tas
     (Fan fixed 2 K); report it and Cowork asks Fan. Also report the snow share for the "seasonal-snow" units only
     (exclude units whose member-0 snow never melts out in Aug-Sep) - the covariance share may be legitimately high
     where snow IS the seasonal storage.
  4. T1 (gw long-term) must come from the parameters - expected; 8z-P will show it.
- 10 Oct 00:45 (Cowork): re-evaluated 8a table and the 4 retests - agreed. Snow finding (inside the units +0.7 %/yr,
  144 perennial cells, median snow-unit trend +0.27 mm/yr): good, no action. One check because of the builder bug you
  caught: verify that the 8a members 1-4 (P1 / P2 ends) changed ONLY their own cells (member 1/2: cells == 1.26 changed,
  cells == 1.74 untouched; member 3/4 the reverse) by reading the 8a parameter files; one line in §7 is enough.
- 10 Oct 00:15 (Cowork, on the 8a result - please apply BEFORE 8b starts and rebuild 8b from the re-evaluated decisions):
  1. Relative snow flags and the restricted negative-storage flag: CONFIRMED.
  2. (a) TWS is NOT counted as a separate storage in the drift test (it is the sum; each storage is judged on its own).
     -> P1 max, P5 min and P5 max are accepted unless another flag fires.
  3. (b) DRIFT RULE REDEFINED - the 3-point slope 2004-2006 cannot tell spin-up adjustment from a larger response to the
     2002-2006 interannual climate (a deeper soil, larger wetlands or faster groundwater responds more to the 2003 / 2005
     droughts; that is wanted long-term spread, not a defect). For each member, storage and cell use the difference
     series d(y) = annual mean_k - annual mean_0, y = 2002..2006 (steps s1..s4 = d(2003)-d(2002) .. d(2006)-d(2005)):
     - INTERANNUAL response (not flagged): the steps change sign.
     - SPIN-UP ADJUSTMENT (not flagged, but reported with the e-folding time from the step ratio, tau = -1 / ln(s4/s1)^(1/3)
       years): all steps the same sign AND |s4| < 0.5 |s1| (decaying).
     - DRIFT (flagged): all steps the same sign AND |s4| >= 0.5 |s1| AND |s4| > 5 mm/yr, over > 5 % of the land area.
     Apply to all storages (gw still excepted for P9 / P11). The outlet-flow and perennial-snow flags stay as they are.
     Re-run the 8a decisions with this rule, write the new table, and rebuild 8b from it (expected: fewer retests; P1
     min stays flagged by the outlet flow, P6 max by perennial snow, P10 min by its non-decaying fill).
  4. P10 min -> 0.00316 retest: agreed (no long spin-up needed for it).
  5. FINDING to keep for Fan (not part of the decisions): member 0 grows snow by 10 %/yr with 2859 perennial cells.
     Report how much of that growth and how many of those cells lie INSIDE the 772 DA units (glacier units were removed
     from the unit set), and the snow trend of the median snow unit of member 0. Fan wants no snow trend in the DA.
- 9 Oct 23:05 (Cowork, on your 22:48 / 22:52 entries):
  1. Negative-storage flag relative to member 0: agreed.
  2. DRIFT: judge it on 2004-2006 only (trend of the annual means 2004-2006, member k minus member 0). An adjustment that
     is present in 2002-2003 and gone by 2004-2006 is spin-up, not a parameter problem - report it but do not flag it.
     A drift that persists in 2004-2006 is flagged as specified (gw still excepted for P9 and P11; P10 is judged with
     its 9-year spin-up from m19).
  3. INSENSITIVE rule (rule 2) - CHANGE: medians over all 772 units hide parameters that act only regionally. P2 (arid
     PT), P8 (critical gw precipitation) and P11 (gw net abstraction) act in arid / irrigated units only, so a median of
     ~0 is expected. New rule: a parameter is insensitive only if, at BOTH ends, S_tws < 3 mm in >= 98 % of the units
     AND the maximum S_tws over all units < 10 mm (same for the S of the storage it acts on). Otherwise keep it and
     report the units where it acts (count, the 10 largest S with unit IDs / names). Expect P11 to grow with time
     (+-50 % pumping in the North China Plain / NW India / High Plains is ~+-10 mm/yr): judge it on 2006, not 2002.
  4. 8g check (pr mean 1.005, std 0.50 / 0.76; tas 2.0 K): good. 303 GB for the forcing - write the free disk in §7
     with every round.
- 9 Oct 22:35 (Cowork, answers to your feedback points 1-7 of ~22:00):
  1. Mask path: do NOT move folders or edit the json now; Job 8 keeps using the existing file. Cowork / Fan fix it
     before the global pilot (the workstation layout has no extra `mask/` level, so the UCloud folder is the odd one).
  2. GRACE obs: good - keep it in $W/grace for now; it will be copied to GRACE/output_v2 when the pilot is set up.
  3. Perennial-snow flag on monthly Aug + Sep means: accepted.
  4. CPU sharing with 8g: accepted.  5. / 6.: agreed.
  7. Spread shares: use the COVARIANCE-CONSISTENT split as the primary measure, share_s = cov(x_s, TWS) / var(TWS)
     per unit and time scale (shares sum to 1 exactly; a storage can be negative when it compensates). This is
     exactly how the Kalman gain distributes a TWS increment over the storages, so T1-T4 are judged on it. Report the
     variance shares next to it for information.
  Also: good catch on the stale expiresAt and on the running-script overwrite; keep both rules.
- 9 Oct ~22:00 (Fan: "I can continue to extend, so no rush to make a summary; keep doing the job until you find the
  optimized solution"). NO DEADLINE any more - this replaces the 10:00 time frame and the "two reruns" limit:
  * Keep iterating 8z reruns (8z1, 8z2, 8z3, ...) with the allowed levers until the STOP CONDITION: all targets T1-T5
    pass with no flag in any member, OR no allowed lever is left (then the best rerun is the result). Each rerun
    changes as few things as possible and records what changed and why.
  * After the stop condition: 8z-P for the final table (if its parameters changed since the last 8z-P), the remaining
    8f batches (F5 / F6 / F8) if L could still matter, then the seed-2027 robustness rerun of the final 8z.
  * Before each new round write in §7 the current expiresAt; if it is < 2 h away, write a resume note and tell Fan in
    §7 (he extends the job). Never start a round you cannot resume.
  * Write "JOB 8 DONE" only when all of this is finished.
- 9 Oct (Fan, correction): the UCloud job was extended by 12 hours (not 10), so it ends around 10:30 CEST, after the
  10:00 deadline. Check expiresAt with `ucloud introspect job` and write it in §7. At 10:00 write the summary at the top
  of §7 (proposed table from the rounds completed) and a resume note (round, members, year, restart folders).
- 9 Oct 23:00 (Fan: "snow should gain something but not much, no snow trend; wetlands should gain water over the flooding
  areas; adjust the parameters / forcing in those directions"). FULL TARGET SET for 8z (pass / fail, median over the
  units of each group; seasonal / long-term as defined in the 22:50 point):
  T1 groundwater >= 50 % of the LONG-TERM spread (all units);
  T2 soil + river + wetlands + snow >= 70 % of the SEASONAL spread (all units);
  T3 SNOW units (member 0 snow seasonal amplitude >= 20 mm): snow 10-40 % of the SEASONAL spread, <= 15 % of the
     LONG-TERM spread; and no member with the snow flags (perennial cells, snow mass trend);
  T4 FLOOD units (member 0 local + global wetland seasonal amplitude >= 10 mm): wetlands >= 20 % of the SEASONAL spread;
  T5 R (spread-skill) 1.0-2.0 and the expected weight on GRACE K as in the 22:40 point.
  LEVERS you may use (only shrinking, except W below; each change tested in an 8z rerun 8z1, 8z2 - up to TWO reruns
  now, in place of the single R-rerun; if two fail, report the best one):
  - T3 fails high (snow too much) or snow flags -> snow_melt_temp to 271.15 / 273.15 / 275.15 K (+-2 K) and
    degreeday_multiplier to 0.7 / 1 / 1.4; still high -> +-1 K and 0.8 / 1 / 1.25.  T3 low: leave it (forcing tas adds).
  - T1 fails -> shrink the soil-side multipliers that put LONG-TERM spread into soil (from the OAT table: smax,
    gamma; e.g. to 0.75 / 1 / 1.5) - never the groundwater parameters.
  - T4 fails -> candidates W (the only widening allowed, needs a test first): run an extra OAT round 8c-w (members 0 +
    4) with activewetland_depth max 10 m, and swb_outflow_coeff 0.003 / 0.03 (it also acts on LAKES: accept only if the
    lake + reservoir mean level change is <= 5 mm in the median flood unit and no flag); put the passing ones into the
    next 8z rerun and say so (Fan decides the final table).
  - Forcing: if T1-T4 pass but R < 1, use the larger-spread realistic forcing from 8f (pr 0.75 or a larger L).
  ORDER now: 8a -> 8g -> 8f F0-F2 -> 8b (+ 8c-w if the 8a OAT table already shows wetlands < 20 % in flood units) ->
  8z -> 8z-P -> 8z1 -> 8z2 -> 8f F5, F6, F8 -> 8z seed 2027.
- 9 Oct 22:50 (Fan): expected behaviour of the DA: LONG-TERM dynamics (interannual, trend) go to GROUNDWATER;
  SEASONALITY goes to SOIL or RIVER storage depending on the region. The filter splits an increment by the ensemble
  covariances, so the ensemble must carry spread on these time scales in these storages. Add a TIME-SCALE diagnostic:
  per member, unit and storage split the monthly anomaly 2002-2006 into (i) SEASONAL = mean annual cycle (12-month
  climatology) and (ii) LONG-TERM = 12-month running mean of the deseasonalised series (incl. trend). Report:
  a. OAT rounds (8a / 8b): S per storage per time scale for every parameter end (which parameter moves which storage
     on which time scale).
  b. Ensembles (8f configs, 8z-P, 8z): ensemble spread per storage per time scale, as shares of the total; per unit
     the dominant storage of each time scale, and the medians over units.
  Target (used as a TIE-BREAKER between realistic choices and reported as a pass / fail): in 8z the groundwater share
  of the LONG-TERM spread >= 50 % in the median unit, and soil + river (+ wetlands + snow) >= 70 % of the SEASONAL
  spread. If 8z fails it, report which parameter (from the OAT S-table) would fix it within the tested ranges and
  apply that choice in the 8z rerun (it may replace the R-based rerun if only one rerun fits before 10:00).
- 9 Oct 22:45 (Fan): temperature std 2 K and tau = 1 day are FIXED - no tests of them. Forcing configs left: F0 (ref),
  F1 / F2 (pr 0.25 / 0.75), F5 / F6 / F8 (L 100 / 1000 / 500 km). Order: 8a -> 8g -> 8f F0-F2 -> 8b -> 8z -> 8z-P ->
  8f F5, F6, F8 -> 8z seed 2027. In the rules, "L / tau" now means L only.
- 9 Oct 22:40 (Fan: "the DA should trust GRACE more, as in the Amazon / Danube runs: correlation close to 0.99 and a
  large RMS reduction"). SPREAD TARGET CHANGED - these points override the decision rules for the forcing (8f rule b)
  and for 8z:
  a. 8f: among the REALISTIC configs (no flag, bias median <= 5 mm) choose the one with the LARGEST median forcing-only
     spread, provided R_F <= 1.5 in >= 90 % of the units (forcing alone may now reach the full OL-GRACE mismatch).
     L / tau: prefer the setting that gives the larger spread if it passes the same test.
  b. 8z target: median R between 1.0 and 2.0 (ensemble spread at least as large as the OL-GRACE mismatch, so the
     filter gives GRACE a high weight). Shrink only if median R > 2.5 (same common factor as before, to R ~ 2). If
     median R < 1.0: do NOT widen beyond the tested ranges; report per unit how far below 1 it is (the DA inflation
     has to supply it).
  c. Parameter ends: rule 1 (unrealistic) stays as it is - wide but unphysical members are still removed.
  d. Report in addition, per unit, the "expected Kalman weight on GRACE" K = spread^2 / (spread^2 + sigma_GRACE^2)
     (median, 10th / 90th percentile) for 8f-F0, the chosen 8f config, 8z-P and 8z.
- 9 Oct 22:25 (Fan: "do whatever you think is best to find the best perturbation strategy; I need it before the global
  DA"). Cowork's plan, these points OVERRIDE the text above:
  1. ORDER and PRIORITY (if time runs short, drop from the end): 8a -> 8g (generate forcing) -> 8f with F0-F4 ->
     8b (re-test ends) -> 8z (joint ensemble) -> 8z-P (below) -> 8f with F5-F9 -> 8z seed 2027.
  2. SLOW GROUNDWATER: a member whose gw_dis_coeff k < 0.003 1/d gets a longer spin-up so that 3/k days have passed:
     spinup_years = min(8, ceil(3 / (365 k)) - 2) cycles of 2000 before 2000-2001 (k = 0.001 -> 7 cycles = 9 years,
     95 % settled). In 8a this is member 19 (P10 min); in 8z every member drawn below 0.003. All other members keep
     spinup_years = 1. With this, the groundwater of P10 IS flagged for drift like every other storage (P9 / P11 stay
     excepted).
  3. FORCING TESTS: F8 (L 500 km) and F9 (tau 3 days) added; 8 members per config (F0: 30 generated, 8 used in 8f).
  4. New round 8z-P = the 8z parameter ensemble with the UNPERTURBED forcing (30 members + Ens_0, same parameter
     files). With 8f's F-config (forcing only) and 8z (both) it splits the spread into parameter / forcing / joint
     parts per unit and storage - report this decomposition (needed for the paper).
- 9 Oct 22:15 (Fan): perturbed forcing is generated ONCE (new step 8g, before 8f) and reused by 8f and 8z.
- 9 Oct 22:05 (Cowork; Fan: "you decide the time period"): FINAL PERIOD for all rounds = spin-up 3 years
  (spinup_years=1 on 2000-2001), run 2002-2006 (text above updated). Replaces the 2 + 3 years of 21:40. If 8a is
  running with any other period, stop it and restart 8a with this one. Units: 772 is right ("727" was a typo, Fan).
- 9 Oct 21:55 (Fan): the FORCING is part of the test: new round 8f (above, after 8a, before 8b); 8z uses the forcing
  chosen in 8f instead of the fixed one.
- 9 Oct 21:40 (Fan): PERIOD changed for all rounds: spin-up 2 years (2000-2001 from cold start), run 2002-2004; the
  Job 8 text above is updated accordingly. If you already started with the old 10 + 11 years, stop and restart round 8a.


**Stop rules / failures:**
- A step fails: read the traceback, write it + your diagnosis into §7. If the cause is the host / a transient I/O
  problem, retry the step once. Otherwise skip to the next job that does not depend on it (exp2a and exp2b do not
  depend on exp1; exp2b depends on exp2a's `prepare` only), and stop when nothing is left.
- Disk below 200 GB free: stop and report.
- When all jobs are done (exp1, exp2a, exp2b, exp4): write a short overall summary at the top of §7 ("ALL DONE <date>"), then stop. Do not
  clean any output, do not start the global pilot.

**Job 9 — GLOBAL PILOT DATA ASSIMILATION with the production perturbation (Fan, 10 Oct 17:15: "after all is done,
start the global pilot DA experiment with the recommended perturbations"). Start only after the clean-up orders
(15:35 / 16:40 / 16:55) are confirmed in §7. Work autonomously; write progress into §7 after every step.**
Case `demo_Global_v2` (settings/demo_Global_v2: DA_setting.json + perturbation.json = production table; the
Config_ReWaterGAP.json and the driver update come from Cowork, see "GATE" below). 30 members + Ens_0 (31 ranks),
GlobalBasins_v1.0 (772 units), state = groundwater, soil, snow, river, local + global wetlands; EnKF_localized,
block localization, adaptive additive inflation (split gw / soil / snow), non-negative partition, diagonal R,
snow cap + envelope as in the json. UCloud job: book >= 10 h (estimate below).
0. Mask path (UCloud data side, no json edit): `mv /work/PyGLDA_v2_external_data/Basin/mask/mask/GlobalBasins_v1.0
   /work/PyGLDA_v2_external_data/Basin/mask/GlobalBasins_v1.0` (keep the `mask/mask/GlobalBasins` copy or delete it,
   your call); verify the md5 of GlobalBasins_v1.0_res_0.5.h5 and _res_1.h5 against Basin/mask/GlobalBasins_v1.0/
   _settings.txt afterwards. Also make sure GRACE/global_mask/WaterGAPLandMask.hdf5 exists (md5 9a21d88b...).
GATE: **driver ready (Cowork 17:35): src_demo/demo_global.py md5 00886009f8208c27e1db4d23ba59dd7d, settings/demo_Global_v2/Config_ReWaterGAP.json md5 56a3dcbf090877bd6ca354a9f49c91e9** (spin_up_years = 8 -> 10 model years, see the driver comment). Fan is deleting the other settings/demo_* folders (only demo_Global_v2 remains); the old wording of this gate was: wait until this block says "driver ready (md5 ...)" - Cowork will add it once Fan approves the edits of
   src_demo/demo_global.py (case demo_Global_v2, period 2002-01-01..2006-12-31, spin-up 2000-2001 x 5 cycles,
   steps model_perturbation / spin_up / OL_run enabled) and the copy of Config_ReWaterGAP.json into
   settings/demo_Global_v2. Check the synced md5 before running.
1. serial, demo1 (from src_demo, pyglda_v2 env): `GDA.config_external_data()`, `GDA.read_config_and_save()`,
   `GDA.check_units()` (IDs equal, 0 cells in two units, md5 True), `GDA.get_GRACE_obs()` -> GRACE/output_v2/
   GlobalBasins_v1.0_{signal,cov,gridded_signal}.hdf5 (2002-01 .. 2006-12 at least; the full GRACE period is fine).
2. MPI, demo2, plain `mpiexec -n 31 python -u demo_global.py Config_ReWaterGAP.json` in tmux `job9`:
   `GDA.model_perturbation()` (parameters + correlated forcing from the production json, Ensemble_input_v2; the
   forcing must cover the spin-up years 2000-2001 AND 2002-2006), `GDA.spin_up()` (2000-01-01..2001-12-31,
   spin_up_years = 5 -> 10 model years per member with ITS OWN parameters and forcing; restarts to
   Ensemble_Initialization_v2), `GDA.OL_run()` (2002-2006, OL_output_v2), `GDA.collect_and_statistics(Stage.OL)`
   (yearly files, Res_OL.h5, OL mean for GRACE -> OLmean_v2, harmonic maps), `GDA.DA_run()`,
   `GDA.collect_and_statistics(Stage.DA)`.
   Spin-up rationale (Cowork): slowest state = groundwater with gw_dis_coeff down to 0.005 1/d (e-folding 200 d,
   settling ~600 d); 10 years = 18 e-foldings for the slowest member, and enough for the global wetlands / lakes;
   cost ~70 min (730 d x 5 x 1.1 s/d). Do NOT reuse any restart of the old demo_Danube global OL (deleted anyway).
3. serial: `GDA.da_evaluation(tag='pilot1')`, `GDA.visualization(tag='pilot1')` in the pygmt env; results in
   Res/demo_Global_v2/evaluation_global/pilot1/ (rsync to the workstation).
Time estimate (31 ranks, 1.1-3.3 s/day on the shared host): perturbation ~30 min, spin-up ~70 min, OL 1826 d
   ~35-100 min, OL collection ~15 min, DA 1826 d ~45-170 min + 60 analyses, DA collection ~15 min: 4-8 h.
Storage: ~4.8 MB per member-day -> OL daily ~270 GB + DA daily ~270 GB + yearly files; check free quota before
   step 2 (stop and report below 1 TB free).
Watch and report in §7: memory per rank, s/day for OL and DA, analysis time per window, first-window log
   (parallel_logs/DA/rank_2.log), then summary.txt: per-unit RMS / correlation DA vs GRACE vs OL (median, p10 / p90,
   by continent), chi2, gain, forecast spread vs sigma_GRACE, inflation factors (how much the inflation adds on
   top of the 0.49 open-loop ratio), increment per storage per time scale (seasonal vs 12-month mean: does the
   long-term increment go to groundwater, the seasonal to soil / river, how much to snow and to wetlands in the
   flood units), clipping warnings by region, Hohe Tauern-type snow cells, units with chi2 > 3 or RMS > sigma.
Never run `clean_temp_output(Stage.OL)`; `clean_temp_output(Stage.DA, dry_run=True)` only after the report.

### 7. Job log (Code writes here)
**JOB 8 CLOSED 10 Oct 2026 15:05 (production table = 8z5 log-symmetric + F11, applied by Cowork; 8z4 tas test, 8z5, river check and the L test 8fL are documented below). JOB 8 DONE 10 Oct 2026 08:50 (UCloud Claude Code, job 12417913; rounds 8a, 8b, 8c, 8f F0-F2 + F10, 8g, 8z, 8z-P, 8z1,
8z2, 8z3 seed-2027 robustness rerun complete 10:05: spread / R_state / K_eff / shares reproduced within 5 %, the
ensemble-mean level shift and an outlet-flow flag of 2 members are draw-dependent (asymmetric ranges) - see 8z3 below; all data under
/work/PyGLDA_v2_external_data/temp_job8/<round>/{input,monthly,report} (daily files deleted), scripts and reports in
temp/exp_logs/job8/ on both machines; details round by round further down in this section).**
**PRODUCTION TABLE (Cowork 13:50, applied to settings/demo_Global_v2/perturbation.json; = the 8z5 log-symmetric
table + forcing F11; triangular, middle = standard WaterGAP, seed 2026):**
```
 id  parameter (json key)              kind   min       standard  max      kept  reason / history
 P1  pt_coeff_humid_arid humid (1.26)  value  1.073     1.26      1.48     yes   min 0.885 failed (outlet flow > 50 % in 18 % of units), 1.073 accepted (8b); max 1.65 accepted (8a), set to 1.48 = log-symmetric (8z5)
 P2  pt_coeff_humid_arid arid (1.74)   value  1.365     1.74      2.115    yes   both accepted (8a); acts only in arid units (max S_tws 320 mm)
 P3  max_daily_pet                     value  6         15        22       yes   both accepted (8a)
 P4  smax_multiplier                   new    0.707     1         1.414    yes   min 0.5 failed (outlet flow > 50 % in 10 % of units), 0.707 accepted (8b); max 2 accepted (8a), set to 1.414 = log-symmetric (8z5)
 P5  gamma                             x      0.5       1         2        yes   both accepted (8a); log-symmetric already
 P6  snow_melt_temp (K)                value  272.2     273.15    274.1    yes   max 276.9 (perennial snow x1.67) -> 275.0 (x1.31) -> 274.1 (x1.16, accepted 8c); symmetric per Cowork 02:25
 P7  degreeday_multiplier              new    0.5       1         2        yes   both accepted (8a); 0.7-1.4 tested in 8z1: no gain on the snow long-term share
 P8  critcal_gw_precipitation          value  2.5       12.5      20       yes   both accepted (8a); acts in arid units only (max S_tws 37 mm)
 P9  gw_recharge_factor_multiplier     new    0.5       1         2        yes   both accepted (8a); log-symmetric already
 P10 gw_dis_coeff (1/d)                value  0.005     0.01      0.02     yes   min 0.001 failed even with a 9-year spin-up (gw drift > 10 % of land), 0.00316 accepted (8b), set to 0.005 = log-symmetric (8z5); max accepted
 P11 netabs_gw_multiplier              new    0.5       1         1.5      yes   both accepted (8a); acts in irrigated units only (max S_tws 290 mm)
 P12 river_roughness_coeff_mult        x      1/3       1         5/3      yes   both accepted (8a); river check 13:05: big rivers carry 0.5-0.95 of the seasonal spread, weak floodplain units are a storage limit, not a P12 question
 P13 activewetland_depth (m)           value  1         2         4        yes   both ends accepted (8a) at 0.5 / 5; set to 1 / 4 = log-symmetric (8z5)
 forcing F11 (correlated mode): pr lognormal std 1.0 (F10 test 06:56, 8z2 vs 8z1 by the 07:10 rule), tas gaussian 1 K (8z4: same spread / R / K, snow bias -1.2 instead of -3.9 mm; Cowork 12:30), L 300 km, tau 1 day (L test 8fL for documentation only)
```
**production = 8z5 table + F11; expected: spread ~22 mm, R_state ~0.49, K_eff ~0.56, mean shift ~ -1 mm, snow bias
~ -1 mm** (8z5 with tas 2 K: spread 22.2, R_state 0.49, K_eff 0.56, TWS level -1.2 mm, snow bias -3.8; tas 1 K
brings the snow bias to ~ -1 mm without changing the rest, see 8z4). Nothing removed: no parameter is insensitive by
the 23:05 rule. The asymmetric-range history (8z2 / 8z4: soil +8-9 mm, TWS +4-5 mm mean shift; seed 2027: +15 mm) is
why the production table is log-symmetric.
**Reference ensemble of the summary numbers below (8z2: asymmetric table with F10, 30 members, 2002-2006, 772 units; the production table 8z5 + F11 differs as stated above):** TWS spread median 24.0 mm (p10 9.0,
p90 44.3; sa 38, na 30, af 24, as 22, eu 21, ar 20, si 18, au 12); DA-state spread 21.1, excluded (lakes + reservoirs
+ canopy) 1.9 mm; **R_state median 0.53** (89 % of the units below 1; the DA inflation must supply the rest - no
widening beyond the tested ranges was allowed); R on the total spread 0.62; **K median 0.72, K_eff 0.61** (p10 0.16,
p90 0.85); ensemble-mean RMS vs GRACE 41.2 mm vs member 0 41.0, correlation 0.83; ensemble mean minus member 0: TWS
+4.3 mm (soil +8.7, gw -5.2, wetlands +2.4, snow in the snow units -3.9), seasonal amplitude ratio 1.02; no member
flagged. Spread shares (covariance-consistent, median over units): SEASONAL soil 0.46, river 0.08, gw 0.05,
wetlands 0.01 (soil dominant in 432 units, river 122, snow 86, wet 66); LONG-TERM soil 0.21, gw 0.15, river 0.07,
lake 0.07, wet 0.05 (dominant: soil 243, wet 163, lake 151, snow 83, gw 69). Targets: T2 (0.91) and T4 (wetlands
0.26 of the seasonal spread in the flood units) pass; **T1 fails (gw 0.15 of the long-term spread, >= 0.5 in 6 % of
the units)**, **T3 fails on the long-term snow share (0.26 > 0.15; seasonal 0.32 passes)**, **T5 fails low (0.53)**.
**Where the spread comes from (8z-P parameters only vs 8f-F2 forcing only vs 8z joint, variance shares of the joint
variance, medians over units):** TWS total: parameters 0.21, forcing 0.67, interaction +0.08; SEASONAL: parameters
0.63 (soil 0.83, gw 1.08 with -0.24 interaction, river 0.71, wetlands 0.67), forcing 0.34; LONG-TERM: parameters
0.08, forcing 0.75 (+0.13 interaction) - gw long-term 0.33 parameters / 0.69 forcing, soil 0.17 / 0.50, river 0.15
/ 0.78, lakes 0.07 / 0.82, snow 0.02 / 0.72 (snow units 0.07 / 1.07). So: the parameters make the SEASONAL spread
(soil, river, wetlands - T2 / T4), the precipitation noise makes the LONG-TERM spread (incl. groundwater), the
temperature noise makes the snow long-term share (T3) - neither T1 nor T3 can be fixed by parameter ranges within
the tested limits (Cowork 05:35: T1 = model-structure result; T3 = tas 2 K question for Fan).
**Round-by-round:** 8a (27 OAT members, 21:52-23:31): member 0 RMS vs GRACE 41.0 mm, corr 0.83, amp 0.87; strongest
parameters P1, P3 min, P4, P13, P10; snow-mass growth of the standard run (10 %/yr) lies outside the DA units
(inside +0.7 %/yr, 144 perennial cells, median snow-unit trend +0.27 mm/yr); drift rule redefined by Cowork (00:15)
-> 4 ends to retest. 8b (00:53-02:02): P1 min 1.073, P4 min 0.707, P10 min 0.00316 accepted, P6 max 275.0 failed.
8c (02:03-03:29): P6 max 274.1 accepted. 8f F0 / F1 / F2 (23:31-00:53): forcing-only spread 11.6 / 6.3 / 16.4 mm,
R_F 0.30 / 0.16 / 0.41, no flags; F2 chosen (22:40 rule). 8z (02:04-03:40, F2, lopsided P6): spread 20.4, R_state
0.44, snow bias -7.3. 8z-P (03:41-05:17): parameters alone 9.1 mm. 8z1 (05:18-07:04, P6 symmetric + P7 0.7-1.4):
20.3 mm, snow bias -3.6, snow long-term share unchanged. 8fF10 (05:17-06:56): pr 1.0 forcing-only 20.3 mm, R_F 0.53,
systematic bias 4.7 mm, passes (a). 8z2 (07:08-08:33, F10, P7 0.5-2): 24.0 mm, R_state 0.53 -> final. 8z3 (seed
2027, 08:40-): running. Forcing sets generated (8g): F0 30, F1 8, F2 30, F5 / F6 / F8 8 (L 100 / 1000 / 500 km, pr
0.75, NOT run - no time; ready under $W/8g), F10 30, F2s2027 30, F10s2027 30 members.
**Open points for Fan / Cowork:** (1) tas 2 K noise drives the snow long-term share (T3) and a -3 to -4 mm snow bias
in the snow units; (2) the gw long-term share (T1) is a WaterGAP-structure result - a DA-side remedy (temporally
correlated gw noise in the inflation) is Cowork's suggestion; (3) R_state 0.53 means the OL ensemble explains about
half of the OL-GRACE mismatch in the median unit (a factor 1.9 short; 11 % of the units are at or above 1): the
adaptive inflation has to supply the rest, or Fan allows a wider precipitation noise than 1.0 (not tested, beyond
the configs); (4) the asymmetric multipliers shift the ensemble mean (soil +9 mm, gw -5 mm); (5) L-test F5 / F6 / F8
not run; (6) the mask path in settings/demo_Global*/DA_setting.json must be fixed before the pilot (feedback point 1).

**ALL DONE 9 Oct 2026 16:30 (UCloud Claude Code, job 12417340); Job 5 (snow innovation diagnostic, read-only) added by Fan and done 17:40 - verdict YES, snow in C1-C3 takes 67-92 % of the cell increments via a variance-driven, self-reinforcing weight (see Job 5 below). Job 6 (parameter vs OL snow spread, read-only) done 17:30 - verdict YES for C1 / C2: adiabatic_lapse_rate (rho 0.89 / 0.75 with the late-summer swe, threshold >= 0.0076 K/m), not applicable for C3 (no OL pack in any member). Job 7 (new parameter-perturbation code, step 1) done 9 Oct 19:40 - 7a identical: YES (new code + existing member files reproduce OL_output Jan 2002 bit-exactly, 2232 pairs); 7b all seven multipliers act as expected: YES, no exception (the net-abstraction multipliers scale a signed field, see the note).** Job 0 checks passed; exp1, exp2a, exp2b and exp4 (added by
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


**Job 7 - check of the new parameter perturbation code (step 1; 9 Oct 19:10-19:40; tmux `job7`; scripts and logs in
`temp/exp_logs/job7/` on both machines: driver.py, build_7b.py, compare.py, verify.py, run_job7.sh, job7.out,
compare_7a/7b.txt/.json, verify.out, <tag>_rank_k.log; data under `$W = /work/PyGLDA_v2_external_data/temp_job7/`
(5.0 GB: 7a/, 7b/, 7b_input/, 7b_init/, init_save/7a, init_save/7b), kept.)**
- **Sync / md5:** GitHub `origin/main` = local HEAD d629a81, nothing newer; Cowork's changes are the uncommitted
  working-tree modifications (10 files). All 8 md5s of the job list match on the desktop AND on UCloud (Perturbation.py
  e2555d56, parameters.py 5c60a59c, fan_waterbalance_vertical_init.py f5052d4c, waterbalance_lateral.py fa4e5eb4,
  EnKF.py 48ee5884, demo_Danube = demo_Global perturbation.json d10b9be5, README 2e0a6b2c, configuration.md 5906846b).
  Config_ReWaterGAP.json of demo_Global is identical to demo_Danube's (md5 56a3dcbf), so the OL_output Jan 2002
  (written by the demo_Danube OL run on 5 Oct) is the right reference.
- **Driver:** `temp/exp_logs/job7/driver.py`, run from src_demo as `mpiexec -n N python -u ../temp/exp_logs/job7/driver.py
  Config_ReWaterGAP.json`; OpenLoop(ensemble_id=rank, settings/demo_Global), configure_time 2002-01-01..2002-01-31,
  configure_Ens_output, configure_Ens_input, configure_ini_for_resume(read = the given init dir, save =
  $W/init_save/<tag>); the driver asserts cm.read_state_path / cm.save_state_path before model_resume. (Note: the
  SingleModel version of configure_ini_for_resume swaps read and save; OpenLoop overrides it correctly, and that is
  what RDA.OL_run and this driver use.) First launch aborted in `conda activate` because my launcher had `set -u`
  (gdal-activate.sh: GDAL_DATA unbound) - removed, relaunched; nothing had been written.
- **7a (4 ranks, 19:26:06-19:26:43, 37 s incl. model init; ~1 s per day):** $W/7a/Ens_0..3 vs OL_output/Ens_0..3,
  all 18 output variables (canopystor, swe, soilmoist, groundwstor, local/global lake and wetland stor, riverstor,
  reservoirstor, tws, consistent-precipitation, dis, dis-from-upstream, atotusegw, atotusesw, atotuse, evap-total),
  all 31 days, 4 members = 2232 (variable, day, member) pairs: NaN patterns identical, **global max |difference| =
  0.0**. The old-code checkout of the fallback was therefore not needed. The restart written at the end went to
  $W/init_save/7a/Ens_k/restartwatergap_2002-01-31.pickle only; no file under Ensemble_Initialization, Ensemble_input
  or OL_output is newer than the job's scripts (find -newer: none).
- **7b input:** $W/7b_input/Ens_k = copy of the real Ens_0 parameter file (24 variables) + one field for k = 1..7
  written exactly as Perturbation.py does it, `xr.full_like(ds['max_daily_pet'], value, dtype=float)` - NOTE this gives
  the constant on every cell, no NaN (nan count 0), which is also what the real ensemble files will contain, since
  full_like does not keep the NaN pattern of the template; harmless, the multipliers are only applied to land
  quantities; forcing = symlink to the real Ens_0 forcing; $W/7b_init/Ens_k -> real Ensemble_Initialization/Ens_0.
- **7b (8 ranks, 19:27:14-19:27:49, 35 s):** all 8 members ran without error. Ens_0 (no field) == 7a Ens_0 exactly
  (max |diff| 0 in every variable on 31 Jan). Each k = 1..7 differs from Ens_0. Differences Ens_k - Ens_0 on
  2002-01-31, cos(lat)-weighted means over the non-NaN cells [storages kg m-2 = mm; dis m3 s-1], with the cell counts
  of positive / negative difference (the full 18-variable tables incl. NW India 70-80E 22-32N, North China Plain
  113-120E 32-40N, US High Plains 103-97W 32-42N are in compare_7b.txt):
  ```
  k  multiplier                      variable      global    Danube   NW India  N China Pl  US High Pl  max|d|  n>0    n<0    expected -> seen
  1  lai x1.5                        canopystor   +0.055    +0.115   +0.0025    +0.0014     +0.031      0.72  28868      9  canopy up      -> UP
                                     soilmoist    +0.14     +0.005                                       12.3  14874  17699  (more interception: gw -0.23, river -0.19, dis -2.9)
  2  degreeday x1.5                  swe          -0.40     -3.32    -0.023     -0.011      +0.0000     205     300   9720  snow down where it melts -> DOWN
                                     soilmoist    +0.11     +0.60                                        25.7  10191   2590  (melt water: river +0.06 / Danube +2.1, dis Danube +57)
  3  smax x1.5                       soilmoist    +9.81    +20.5     +0.43      +0.80       +3.03       193   38678     36  soil up        -> UP
                                     groundwstor  -1.79     -6.68    -0.013     -0.25       -0.92        50    1155  31065  (less recharge/runoff: river -1.7, dis -34)
  4  gw_recharge_factor x0.5         groundwstor  -2.66     -4.93    -0.024     -1.36       -0.50        47    2361  29784  gw down        -> DOWN
                                     riverstor    +1.09     +4.28    +0.11      +0.47       +0.17       211   27497   8495  runoff up      -> UP (dis +21.8; soil -0.05, i.e. the water leaves as runoff)
  5  max_gw_recharge x0.5            groundwstor  -2.48     -3.15    -0.004     -0.53       -0.22        71     372  14966  gw down        -> DOWN
                                     riverstor    +0.99     +2.77    +0.07      +0.15       +0.08       264   11730   6700  runoff up      -> UP (dis +19.4)
  6  netabs_gw x2                    groundwstor  +0.028    -0.45    +0.47      -1.97       -0.10       132    6100  45578  gw down in irrigated regions -> DOWN in N China Plain, US High Plains, Danube (45578 cells down);
                                                                                                                             UP in NW India, see note
                                     atotusegw    (flux doubled: 45587 cells changed, max 6.2e-5 kg m-2 s-1 = 5.3 mm/d)
  7  netabs_sw x2                    riverstor    -0.11     +0.21    -2.16      +0.31       -0.0008     114   14189  40177  surface water down in irrigated regions -> DOWN in NW India (river -2.2, local lake -0.40,
                                     tws          -0.26     +0.21    -3.82      +0.15       -0.018      218   13521  37119  global wetland -0.37, reservoir -0.32, dis -15.9); 40177 cells down; UP in Danube / N China Plain river, see note
                                     atotusesw    (flux doubled: 39830 cells changed, max 2.8e-4 kg m-2 s-1 = 24 mm/d)
  ```
  consistent-precipitation is unchanged in every k (0 everywhere), as it must be (same forcing).
- **Note on the net-abstraction multipliers (k = 6, 7):** they scale a SIGNED quantity. WaterGAP's net abstraction is
  withdrawal minus return flow and is negative where return flows dominate (groundwater: return flow from
  surface-water irrigation; surface water: return flow from groundwater-fed use). In Ens_0 on 31 Jan the
  area-weighted net groundwater abstraction of the NW India box is negative (-0.0025 mm/d; 83 of 400 cells < 0, the
  Punjab canal-irrigation cells) and the net surface-water abstraction of the Danube is negative (-0.008 mm/d, 309 of
  373 cells < 0) and mixed in the North China Plain (145 cells < 0, 59 > 0; mean +0.004). Doubling therefore
  INCREASES groundwater in NW India (+0.47 mm) and river storage in the Danube (+0.21, dis +4.9) - the correct
  behaviour of a multiplier on a signed field, not a bug; where the net abstraction is positive (North China Plain
  gw -1.97, US High Plains gw -0.10, NW India surface water -3.8 mm TWS) the storages go down as expected. Two
  consequences for Fan / Cowork: (i) a sanity check of the multipliers by region must use the sign of the local net
  abstraction; (ii) the DOELL2024 table in Perturbation.py gives the netabs multipliers the triangle [-2, 1, 2]: a
  negative draw turns every withdrawal into an injection (and every return flow into a withdrawal) over the whole
  globe. If that is what Döll et al. (2024) prescribe, fine, but it is a strong perturbation; a draw of, e.g., 0.5-2
  would be the conservative alternative.
- **Side observations (not defects):** k = 1 (LAI) changes soil, groundwater and river as well (more interception ->
  less throughfall), k = 6 / 7 change canopy / snow / soil by < 1e-4 mm only through the neighbouring-cell demand
  satisfaction (actual consumption enters evap-total). The fluxes in the output are the 31-Jan daily values, so the
  atotusegw / atotusesw differences in k = 6 / 7 are the doubled net abstractions (0.0086 mm/d per 1e-7 kg m-2 s-1).
- **Verdict: 7a identical: YES (2232 pairs, max |diff| 0); 7b all seven act as expected: YES, no exception** (the
  only directions that differ from the naive expectation, k = 6 in NW India and k = 7 in the Danube / North China
  Plain, follow from the negative sign of the local net abstraction). Nothing outside $W and temp/exp_logs/job7/
  was written; src_*, settings/, exp_Ucloud.py untouched; $W kept.


**Job 8 - parameter sensitivity test and range optimisation for the global GRACE DA ensemble (started 9 Oct 21:30,
job 12417913, host bit-c26a-05, 32 vCPU / 96 GB; tmux `job8`; scripts and logs in `temp/exp_logs/job8/`, data under
`$W = /work/PyGLDA_v2_external_data/temp_job8/`). Progress log, newest round last.**
- **Set-up checks (21:28):** all new files in sync (md5 desktop = UCloud: src_DA/perturbation_par.py 3bd6eaa8,
  perturbation_forcing.py 0f816b17, settings/demo_Global_v2/perturbation.json 58d85685 = the reference table, DA_setting
  e6f33c98, shp2mask a2c92870, Global_DA fbb9c680, OpenLoop 3c2f3fb2). Standard values in the ORIGINAL parameter file
  confirmed: pt 1.26 (45544 cells) / 1.74 (21876), max_daily_pet 15, snow_melt_temp 273.15, critcal_gw_precip 12.5,
  gw_dis_coeff 0.01, activewetland_depth 2, gamma field 0.1-5.0 (10593 values), river_roughness_coeff_mult field
  {1, 2, 3, 10}. **Mask path:** `settings/demo_Global/DA_setting.json` points to
  `Basin/mask/GlobalBasins_v1.0/GlobalBasins_v1.0_res_0.5.h5`, which does NOT exist on UCloud; the frozen mask is at
  `Basin/mask/mask/GlobalBasins_v1.0/GlobalBasins_v1.0_res_0.5.h5` (772 units, 55461 land cells, attrs
  area_threshold 0.4, model_mask 1; an identical copy under mask/mask/GlobalBasins/). I use that file as it is (772
  units, not 727) and report over all 772. **GRACE:** no global unit obs file exists (GRACE/output has only Amazon /
  Brahmaputra / Danube), so GDA.get_GRACE_obs runs on a settings copy in $W/settings_grace (window `grace` of tmux
  job8, output $W/grace); fallback 20 mm if it fails.
- **Job expiry warning (21:30):** `ucloud introspect job` reports startedAt 21:26:54 and **expiresAt 22:26:54**
  (1 h) although the specification says timeAllocation 13 h. Reported to Fan; every round is resumable (restarts in
  $W/<round>/init, monthly files per year, `run_round.sh <round> <N> [first_year]` skips what exists).
- **Round 8a launched 21:31:54:** 27 members built by `build_params.py 8a spec_8a.json` from the ORIGINAL parameter
  file (Ens_0 standard; 2p-1 / 2p = P_p min / max, exactly the reference table: P1 humid cells==1.26 -> 0.885 / 1.65,
  P2 arid cells==1.74 -> 1.365 / 2.115, P3 6 / 22, P4 smax new 0.5 / 2, P5 gamma x0.5 / x2, P6 269.4 / 276.9 K, P7
  degreeday new 0.5 / 2, P8 2.5 / 20, P9 gw_recharge_factor new 0.5 / 2, P10 0.001 / 0.02, P11 netabs_gw new 0.5 / 1.5,
  P12 roughness x1/3 / x5/3, P13 0.5 / 5 m), forcing = symlink to the unperturbed Ensemble_input/Ens_0 forcing.
  Chain `run_round.sh 8a 27`: spin-up (cold start, 10 cycles of 2000 + 2000-2001, restarts -> $W/8a/init/Ens_k) ->
  years 2002..2012 one mpiexec each -> monthly means ($W/8a/monthly/Ens_k/<year>.nc, all 18 output variables) ->
  daily files deleted; free-disk check (> 300 GB) before every year. Speed 1.21 s per model day with 27 ranks, cgroup
  memory 36.5 GB -> spin-up ~ 90 min, 11 years ~ 80 min + averaging: model part of 8a expected ~ 00:30.
- **21:50 Fan: stop 8a, final period** (Cowork review 22:05 point): the 10-cycle run was killed at 21:50 (it was in the
  first spin-up cycle; nothing written except the inputs) and **8a restarted 21:52:13** with `model_spinup(spinup_years=1)`
  (1 cycle of 2000 + 2000-2001) and years 2002-2006 (`run_round.sh 8a 27`, LASTY 2006); diagnostics default 2002-2006,
  GRACE months from 2002-04, perennial snow = late-summer swe > 10 mm in >= 3 of 5 years, groundwater drift excepted
  for P9 / P10 / P11 (reported separately). cgroup memory 38 GB anon + 57 GB file cache; 1.0-1.3 s per day.
- **Cowork review read 21:56 (points 21:40-23:00) - actions:**
  - 22:25 point 2 (slow groundwater): member 19 (P10 min, k = 0.001) needs 7 cycles of 2000 before 2000-2001. 8a was
    already running with 1 cycle for all, so member 19 is re-run SEPARATELY in parallel (tmux window `m19`, 1 rank,
    `JOB8_OFFSET=19 JOB8_SPINUP_MAP=19:7`, folder $W/8a_m19, started 21:55:30; ~9 model years of spin-up + 5 years
    ~ 2.5-3 h); its monthly files replace $W/8a/monthly/Ens_19 when done, and the 8a diagnostics are rerun for member 19.
    The driver now takes a per-member cycle map; in 8z the members drawn below 0.003 get the same treatment (separate
    parallel chain so that the 31-rank spin-up is not delayed). With this, P10's groundwater IS drift-flagged (P9 / P11
    excepted). (The first m19 launch died at once on a JSON-quoting error of the env map - fixed, relaunched 21:55:30.)
  - 22:15 / 22:45 points (8g once, tas 2 K and tau 1 d fixed): 8g started 21:54:20 in tmux window `8g`, concurrent
    with 8a (4 ranks): F0 (30 members, pr lognormal 0.5, tas 2 K, L 300 km, tau 1 d, seed 2026), then F1 (pr 0.25)
    and F2 (pr 0.75) with 8 members each, months 2000-01..2006-12, into $W/8g/<config>/input/Ens_k with the project
    code (perturbation_forcing, correlated mode, par all off). F5 / F6 / F8 (L 100 / 1000 / 500 km) are generated
    later, before their 8f batch. F3, F4, F7, F9 dropped (tas / tau fixed).
  - ORDER followed (23:00 point): 8a -> 8g -> 8f F0-F2 (8 members each + member 0 = 25 ranks) -> 8b (+ 8c-w if the
    8a OAT table shows wetlands < 20 % of the seasonal spread in the flood units) -> 8z -> 8z-P -> 8z1 -> 8z2 ->
    8f F5 / F6 / F8 -> 8z seed 2027.
  - 22:50 / 23:00 points (time-scale diagnostic, targets T1-T5, K, R, spread decomposition): being added to the
    diagnostics (`diagnostics.py` OAT: S per storage per time scale; new `ensemble_diag.py` for 8f / 8z / 8z-P:
    spread per storage per time scale, shares, dominant storage per unit, bias, R, R_F, K, T1-T5 pass / fail, and the
    parameter / forcing / joint decomposition from 8z-P + 8f-F + 8z). Reported per round below.
  - 22:40 points a-d: 8f choice = largest median forcing-only spread among the realistic configs with R_F <= 1.5 in
    >= 90 % of the units; 8z target median R 1.0-2.0, shrink only if > 2.5; rule 1 unchanged; K reported.

- **Review block re-read 22:00 (new points ~22:00: no deadline, iterate 8z reruns until T1-T5 pass or no lever
  left; expiry check before each round; "job extended by 12 h"):** acknowledged. BUT `ucloud introspect job` at 22:00
  still reports **expiresAt 09 Oct 22:26** in its `status` field - BUT the job's update list shows the extensions
  (newTimeAllocation 9 h, 10 h, 11 h, 12 h, 13 h at 21:27-21:28; specification.timeAllocation 13 h), so the `expiresAt`
  field is simply stale and the job ends at **21:27 + 13 h = 10 Oct ~10:27 CEST** (Fan's "around 10:30"). Resolved; I
  read the end time from the last newTimeAllocation update from now on. If the job stopped unexpectedly:
  resume note: 8a is in the spin-up (year 2000-2001 run after the one cycle; restarts not yet written), m19 in its
  7-cycle spin-up, 8g at month 14 / 84 of F0. Everything restarts from scratch with `run_round.sh 8a 27`,
  `JOB8_OFFSET=19 JOB8_SPINUP_MAP=19:7 run_round.sh 8a_m19 1`, `run_8g.sh` (the forcing generator overwrites; the
  parameter inputs under $W/8a/input are complete and reusable). I will write expiresAt before every round and a
  resume note whenever < 2 h remain.
- **FEEDBACK / open points for Cowork (please read; answer in the review block if an action is needed):**
  1. **Mask path:** `settings/demo_Global/DA_setting.json` (and demo_Global_v2) point to
     `Basin/mask/GlobalBasins_v1.0/...` which does not exist on UCloud; the frozen 772-unit mask is under
     `Basin/mask/mask/GlobalBasins_v1.0/` (+ a copy `mask/mask/GlobalBasins/`). Either move the folder up one level on
     UCloud or fix the json before the global pilot. Job 8 uses the existing file directly.
  2. **GRACE obs for the global units did not exist** (GRACE/output has only the three regional basins); made with
     GDA.get_GRACE_obs on a settings copy -> $W/grace/GlobalBasins_v1.0_{signal,cov,gridded_signal}.hdf5, 2002-01..
     2012-12 (122 GRACE months; 57 in 2002-04..2006-12), ~5 min; can be copied to GRACE/output_v2 for the pilot
     (it is exactly what the DA would build). GRACE sigma per unit = mean over months of sqrt(diag cov).
  3. **Perennial-snow flag** is evaluated on the MONTHLY means (Aug + Sep mean swe > 10 mm in >= 3 of 5 years), not on
     15 Aug - 15 Sep daily values, because only monthly files are kept (daily deleted). Slightly stricter than the spec.
  4. **CPU:** 8a (27 ranks) + m19 (1) + 8g (4 ranks) = 32 processes on 32 vCPU: the model slowed from 1.1 to 1.3-2 s per
     day while 8g runs (8g F0 ~ 45 min, F1 / F2 ~ 25 min). Accepted: 8f needs both anyway. Memory is not an issue
     (38 GB anon for 27 ranks).
  5. **Slow-groundwater members:** handled as separate parallel single-rank chains (member 19 in 8a; in 8z all
     members drawn with gw_dis_coeff < 0.003), so the main mpiexec is not held up by 9-year spin-ups; the chain
     writes into its own folder and its monthly files are copied over the member's folder before the diagnostics.
  6. **8f member count:** 8 members per config as in the 22:25 point (F0: 30 generated, 8 used); F0-F2 in one batch
     of 25 ranks (member 0 + 24). F5 / F6 / F8 generated later (8 members each, L 100 / 1000 / 500 km).
  7. **Spread shares** in the ensemble diagnostics are VARIANCE shares per storage (covariances between storages
     ignored), both for the seasonal and the long-term part; the dominant storage per unit is the largest share.
     Say if you prefer std shares or a covariance-consistent split (tws variance = sum over storages incl. covariances).

- **22:05 GRACE reference ready and diagnostics tested on it:** 772 units, 54 GRACE months in 2002-04..2006-12, unit
  sigma (sqrt diag of the DA covariance, mean over months) median 16.3 mm (p10 10.0, p90 19.4); GRACE unit anomaly
  std median 63 mm (sa 118, as 93, af 83, eu 59, na 55, si 53, ar 41, au 35); split of the GRACE anomaly: seasonal
  std median 41 mm, long-term (12-month running mean) 27 mm. 8a: spin-up cycle done 22:00, 2000-2001 running
  (1.3 s/day), years 2002-2006 expected 22:20-23:05; 8g F0 at 17/84 months at 22:01.

- **22:19 8a spin-up done** (EXIT 0, 27 restarts 2001-12-31, 27 min for 3 model years at 27 ranks + the concurrent
  8g). **Incident:** the chain stopped after the spin-up with "unexpected EOF while looking for matching quote" at
  line 36 of run_round.sh - not a syntax error (the file parses), but I had overwritten run_round.sh by scp at 21:54
  while the 21:52 chain was executing it (bash reads a script incrementally by byte offset, so the replaced bytes were
  parsed as garbage). Relaunched 22:27:27 with the same command (spin-up skipped, restarts reused); 8 min lost.
  Rule from now on: never overwrite a running chain script; new versions get a new file name. 8g F0 finished 22:26
  (EXIT 0; 30 members, 84 months), F1 running.

- **22:35 review answers read 22:33:** points 1-6 as agreed; point 7 implemented: the ensemble diagnostics now use the
  covariance-consistent split share_s = time-mean cov_members(x_s, TWS) / time-mean var_members(TWS) per unit and
  time scale (sums to 1, negative = compensating) as the primary measure for T1-T4 and for the dominant storage;
  the variance shares are reported next to it ("vshare" columns).

- **22:48 dry run of the OAT diagnostics on the finished year 2002 (27 members; 9 GRACE months) - pipeline works
  end to end** (member processing 27 x ~40 s with 8 workers, metrics, decisions, report). Two things learned and fixed
  before the real 5-year evaluation: (1) **negative storages**: in the STANDARD run (member 0) the minimum monthly
  groundwater storage is negative over ~13 % of the land area (lakes + reservoirs ~10 %, tws ~13 %) - WaterGAP 2.2e
  allows negative groundwater storage under net abstraction, so an absolute "negative storage" flag fires for every
  member. The flag now means NEW negative cells relative to member 0 over > 1 % of the land area (member 0's own
  fractions are reported). (2) The drift flag on one year is meaningless (seasonality) - with the 5 years it is the
  specified test; expect it to catch the post-spin-up adjustment of the soil-side parameters (P1-P5), which is a
  property of the 3-year spin-up rather than of the parameter; I will report the drift fractions and judge per the
  rules (gw excepted for P9-P11). First look (2002 only, not final): P10 min (k = 0.001) moves TWS by 144 mm median
  (its 7-cycle spin-up is running, m19 at year ~5 of 9); P2 (arid PT), P8 (critical gw precipitation) and P11
  (net abstraction) are near-insensitive at both ends (median S_tws 0.0-0.3 mm); P1, P3 min, P4, P13 and P10 are the
  strong ones (S_tws 15-35 mm); P6 min (snow melt 269.4 K) and P13 max act mostly in si / ar / na.

- **22:52 8g done** (F0 30 members 22:26, F1 22:38, F2 22:52; 303 GB under $W/8g; EXIT 0 each). Check of the generated
  forcing against the unperturbed files (2003-07): F0 member 1 pr ratio mean 1.005, std 0.50; tas offset mean +0.03 K,
  std 2.03 K; F2 member 3 pr std 0.76, tas std 2.00 - as configured (lognormal mean 1, min 0). **8f input built**
  (24 members: 1-8 F0, 9-16 F1, 17-24 F2, parameters = original file, forcing symlinked to 8g; Ens_0 = standard).
  **8f queued** in tmux window `8f`: starts automatically (`run_round.sh 8f 25`) the moment 8a's model runs are complete
  (CPU: 27 + 25 ranks would oversubscribe the 32 vCPU). 8a: year 2003 done 22:51 (11.5 min per year), 2004 running;
  8a model runs expected complete ~23:28, then OAT diagnostics (~5 min) and the 8a report here.

- **23:05 review points read 23:06 - implemented:** (2) drift flag = slope of the ANNUAL means 2004-2006 (member k
  minus member 0) > 5 mm/yr over > 5 % of the land area; the all-months slope (incl. the 2002-2003 adjustment) is
  reported next to it as information; gw excepted for P9 / P11; P10 judged with m19's 9-year spin-up. (3) insensitive
  = at both ends S_tws < 3 mm in >= 98 % of the units AND max S_tws < 10 mm (same for the acting storage); the
  decisions table now gives %<3, max, the number of units with S_tws >= 3 and the 10 largest-S units with ids /
  names, plus S over the last year (2006) for P11. (4) free disk will be written with every round (now 1871 TB free
  of 7.2 PB; $W = 8a inputs + 8g 303 GB).

**Round 8a - result (model runs 21:52-23:31, 27 ranks, 1.0-1.3 s per day; diagnostics 23:33 and, with two flag
fixes, 23:40; full report `temp/exp_logs/job8/8a_report_v1b.md` = $W/8a/report/report.md, CSVs unit_metrics /
member_summary / scale_S / decisions in $W/8a/report/; member 19 (P10 min) still with the 1-cycle spin-up, its
9-year version lands ~23:55 and is re-evaluated automatically; free disk 1871 TB; expiresAt 10 Oct ~10:27).**
- **Member 0 (standard WaterGAP, unperturbed forcing) vs GRACE, 772 units, 54 months 2002-04..2006-12:** median RMS
  of the TWS anomaly 41.0 mm (p90 85), correlation 0.83, seasonal amplitude ratio model / GRACE 0.87, trend -0.35
  mm/yr; RMS by region sa 59, as 54, af 49, na 41, eu 37, au 37, ar 33, si 31. Snow units (member-0 snow seasonal
  amplitude >= 20 mm) 222, flood units (wetland amplitude >= 10 mm) 178.
- **FINDING for Cowork (standard run):** the global snow mass of member 0 GROWS by 10.1 %/yr over 2002-2006 and it
  has 2859 perennial-snow cells - WaterGAP accumulates snow in high-mountain / glacier cells from the cold start
  (no glacier model; the pack grows towards the model's cap), independent of any parameter. An absolute "snow mass
  growing > 2 %/yr" flag therefore fired for all 27 members; the flag is now RELATIVE (growth_k - growth_0 > 2 %/yr),
  perennial cells were already relative (> 1.2 x member 0). Likewise "negative storage" is now limited to the
  storages that must stay >= 0 (soil, snow, river, wetlands, lakes, canopy): groundwater (and hence TWS) is negative
  in ~13 % of the land in the standard run (net abstraction) and new negative gw cells (1-7 % of land for P4 max,
  P5 max, P8 max, P9 min, P10 max, P11) are reported but not flagged. Please confirm or override both.
- **OAT sensitivity S = RMS(unit TWSA_k - TWSA_0), median over units (mm) | share of units with S < 3 mm | max:**
  P1 humid PT min 35.9 / 14 % / 593, max 23.8; P3 max_daily_pet min 24.4, max 4.3; P4 smax min 20.4, max 33.5 (soil
  S 26 / 47); P5 gamma 8.8 / 6.8; P6 snow melt T 1.6 / 1.5 (snow units: tws S 13.8 / 7.5, snow S 16 / 13); P7
  degree-day 0.4 / 0.3 (snow units 4.4 / 3.5); P9 gw recharge factor 12.4 / 6.9; P10 gw_dis_coeff min 190 (1-cycle
  spin-up; max S 1798!), max 15.1; P12 roughness 8.0 / 6.4 (river S 7.8 / 6.3, max 1071 in large rivers); P13 wetland
  depth 17.8 / 27.6 (wet S 16 / 25). Regionally weak but locally strong: P2 arid PT median 0.4 / 0.3 but max 323 /
  321 mm (265 / 216 units >= 3 mm, af / au); P8 critical gw precipitation 0.1 / 0.0, max 37 / 18 (90 / 54 units, af
  / as); P11 net abstraction 0.2 / 0.2, max 278 / 286 (176 / 180 units; as 8-9 mm median: North China Plain / NW India).
  **No parameter is insensitive by the 23:05 rule** (every one has max S_tws >= 10 mm).
- **Time scales (median RMS of the seasonal | long-term part, tws):** P1 7.0 | 2.5, P3 min 6.2 | 2.2, P4 7.8-9.8 |
  3.0-4.6 (soil seasonal 10-13), P5 2.4-3.4 | 1.3-1.6, P9 1.7-2.1 | 0.9-1.4 (gw), P10 min 6.3 | 24.0 (gw long-term
  21), P10 max 2.0 | 1.7, P12 2.3 | 0.8 (river), P13 1.1 | 0.8-1.2 (wet; flood units 7.1-5.8 | wet 7.4-6.1), P6 in snow
  units 13.8 / 7.5 seasonal. So: the soil-side parameters (P1, P3, P4, P5) carry most of the SEASONAL TWS
  sensitivity; the long-term sensitivity sits in gw_dis_coeff (P10) and, weaker, P9 and P4; wetland depth acts
  seasonally in the flood units; snow parameters only in the snow units.
- **Flags and decisions (rule 1; drift = annual-mean slope 2004-2006 vs member 0 > 5 mm/yr over > 5 % of land; TWS
  counted as a storage - see question below):**
  ```
  P1  humid PT      min 0.885  UNREALISTIC drift soil 7.9 % tws 17.4 %, outlet flow >50 % in 18 % units -> 1.073 | max 1.65 UNREALISTIC drift tws 13.5 % -> 1.455
  P2  arid PT       min 1.365  ACCEPTED | max 2.115 ACCEPTED
  P3  max_daily_pet min 6      UNREALISTIC drift soil 5.0 % tws 13.0 % -> 10.5 | max 22 ACCEPTED
  P4  smax          min 0.5    UNREALISTIC drift soil 11.9 % tws 18.4 %, outlet 10 % -> 0.707 | max 2 UNREALISTIC drift soil 23.8 % tws 28.2 % -> 1.414
  P5  gamma         min 0.5    UNREALISTIC drift tws 10.0 % -> 0.707 | max 2 UNREALISTIC drift tws 9.2 % -> 1.414
  P6  snow melt T   min 269.4  UNREALISTIC drift snow 7.2 % tws 13.0 % -> 271.3 | max 276.9 UNREALISTIC perennial cells 4787 vs 2859 (x1.67), drift tws 6.6 % -> 275.0
  P7  degree-day    min 0.5    ACCEPTED (perennial 3365 = x1.18) | max 2 ACCEPTED
  P8  crit gw prec  min 2.5    ACCEPTED | max 20 ACCEPTED
  P9  gw rech fact  min 0.5    ACCEPTED (gw drift excepted) | max 2 ACCEPTED
  P10 gw_dis_coeff  min 0.001  pending m19 (1-cycle: drift gw 51.5 %, S 190 mm, RMS 48.9) | max 0.02 UNREALISTIC drift gw 7.0 % tws 8.2 % -> 0.01414
  P11 netabs gw     min 0.5    ACCEPTED | max 1.5 ACCEPTED
  P12 roughness     min 1/3    ACCEPTED | max 5/3 ACCEPTED
  P13 wetland depth min 0.5    UNREALISTIC drift wet 7.5 % tws 8.3 % -> 1.25 | max 5 UNREALISTIC drift wet 8.8 % tws 9.5 % -> 3.5
  ```
  No end fails the RMS rule (largest median RMS 44.3 = 1.08 x 41.0 for P1 max; P10 min 48.9 = 1.19 x) or the amplitude
  rule (ratios 0.74-1.02 vs 0.87). -> **8b = member 0 + 12 retests** (13 if P10 min stays flagged with the 9-year
  spin-up), runs after 8f (CPU).
- **Questions for Cowork before 8b starts (~00:45):** (a) the drift flag counts TWS as a storage; P1 max, P5 min and
  P5 max are flagged ONLY by the TWS drift (their storages stay below 5 % each). If TWS should not count, those three
  ends are ACCEPTED and 8b has 9-10 retests. (b) The soil drifts of P1 / P3 / P4 in 2004-2006 (5-24 % of land) are
  the slow wetting / drying of the soil column towards the new equilibrium in dry regions after only 3 spin-up years;
  they are adjustment, not a model defect, but they persist beyond 2003, so rule 1 shrinks the ends. Alternative:
  accept the ends and add spin-up cycles for the soil-side members (like P10). Your call; default = the rule.
- **8c-w trigger:** cannot be decided from OAT alone (needs the ensemble shares); in the flood units the wetland-depth
  ends move the seasonal TWS by 6-7 mm and the soil-side ends by 8-12 mm, so wetlands will carry roughly 20-35 % of the
  seasonal spread there - I will decide 8c-w after 8z-P (first ensemble shares).

- **23:58 member 19 (P10 min, k = 0.001) with the 9-year spin-up (7 cycles + 2000-2001, m19 chain 21:55-23:57)
  swapped in and re-evaluated:** RMS vs GRACE 42.2 (1-cycle: 48.9), corr 0.82, amp 0.82; S_tws median 230 mm (max
  2140), groundwater level +278 mm above member 0, long-term S 11.3 mm; the 2004-2006 gw drift still covers 47 % of
  the land (1-cycle 52 %): with k = 0.001 1/d the aquifer is still filling at years 9-14 of the run in cells where the
  equilibrium level is hundreds of mm higher. -> **P10 min UNREALISTIC -> retest 0.00316** (log halfway; that member
  then needs only the standard spin-up). Side effect of the longer spin-up: +19 mm more snow (perennial cells 2938 vs
  2859, growth 5.9 vs 10.1 %/yr) - the snow pack is further along its accumulation, not a P10 effect.
- **8b built and queued (00:02):** member 0 + 13 retests (P1 1.073 / 1.455, P3 min 10.5, P4 0.707 / 1.414, P5 0.707
  / 1.414, P6 271.3 / 275.0 K, P10 0.00316 / 0.01414, P13 1.25 / 3.5 m), `run_round.sh 8b 14` starts automatically
  when 8f's model runs complete (~00:45); its OAT diagnostics (member 0 reused from 8a) run automatically after.
  8f: spin-up done 23:54 (23 min), year 2002 running. If Cowork overrides the TWS-drift question before 8b starts,
  I rebuild 8b (spec + inputs) before the launch.

- **00:15 review (Cowork) read 00:11 - applied before 8b started (8b withdrawn and rebuilt):**
  1 / 2 / 4: confirmed; TWS no longer counted as a storage in the drift test. 3: **drift rule redefined and implemented**
  (per member, storage, cell: d(y) = annual mean_k - annual mean_0, steps s1..s4; DRIFT = all steps the same sign and
  |s4| >= 0.5 |s1| and |s4| > 5 mm/yr over > 5 % of land; ADJUSTMENT = same sign and decaying, reported with the median
  e-folding time; otherwise interannual response). **8a decisions re-run with it (final 8a table,
  `temp/exp_logs/job8/8a_report.md`):**
  ```
  P1  humid PT      min 0.885  UNREALISTIC outlet flow > 50 % in 18 % of the units -> 1.073 | max 1.65  ACCEPTED
  P2  arid PT       min / max  ACCEPTED
  P3  max_daily_pet min / max  ACCEPTED (min: adjustment of gw over 1 % of land, tau ~2 yr; no drift)
  P4  smax          min 0.5    UNREALISTIC outlet flow > 50 % in 10 % of the units -> 0.707 | max 2 ACCEPTED (drift soil 1 %, gw 1 %)
  P5  gamma         min / max  ACCEPTED
  P6  snow melt T   min 269.4  ACCEPTED | max 276.9 UNREALISTIC perennial cells 4787 vs 2859 -> 275.0
  P7, P8, P9        all ACCEPTED
  P10 gw_dis_coeff  min 0.001  UNREALISTIC drift gw 10.5 % of land (9-year spin-up; adjustment over another 9 %) -> 0.00316 | max 0.02 ACCEPTED
  P11, P12, P13     all ACCEPTED (P13 max: wet drift 1 %)
  ```
  Drift fractions are now 0-2 % everywhere except P10 min; adjustment fractions 0-1 % (P10 min 9 % gw) with e-folding
  times 1.1-2.8 yr - the 2004-2006 slopes flagged before were interannual responses, as Cowork suspected.
  -> **8b = member 0 + 4 retests** (P1 min 1.073, P4 min 0.707, P6 max 275.0 K, P10 min 0.00316), `run_round.sh 8b 5`
  queued behind 8f (CPU), diagnostics queued. (Bug caught while rebuilding: the retest builder applied the P1 value to
  the ARID cells (1.74) - fixed, rebuilt, verified "cells==1.26 (45544) -> 1.073".)
  5: **Snow finding for Fan (member 0, standard WaterGAP):** of the 10.1 %/yr global snow-mass growth, almost nothing
  lies inside the 772 DA units: snow mass inside the units grows +0.7 %/yr (land-mean 10.4 mm) vs +14.6 %/yr outside
  (22.0 mm; the removed glacier units, Greenland, Antarctic margin, high Asia); perennial cells 144 inside vs 2715
  outside; the snow trend of member 0 in the 222 snow units: median +0.27 mm/yr (p10 -3.6, p90 +4.4), all units
  median 0.00 mm/yr. So the unit set already excludes the accumulating cells; the DA snow has no systematic trend in
  the standard run.
- 8f: year 2003 running at 00:12 (10 min per year, 25 ranks) -> model runs complete ~00:48, ensemble diagnostics ~00:55.

- **00:45 review read 00:46 - check done:** 8a parameter files read back: member 1 / 2 (P1) cells==1.26 -> 0.885 /
  1.65 with cells==1.74 untouched (1.74); member 3 / 4 (P2) cells==1.74 -> 1.365 / 2.115 with cells==1.26 untouched
  (1.26); NaN pattern unchanged in all four; 8b member 1: cells==1.26 -> 1.073, cells==1.74 = 1.74. The 8a OAT files
  were correct (the bug was only in the retest builder, caught before any 8b run).

**Round 8f - forcing sensitivity, result (runs 23:31-00:53, 25 ranks; diagnostics 00:54; report
`temp/exp_logs/job8/8f_report.md`, per-unit CSVs $W/8f/report/F?_units.csv; free disk 1870 TB).** 8 members per
config, parameters standard, 2002-2006; spread = time-mean ensemble std of the unit TWS anomaly; shares =
covariance-consistent cov(x_s, TWS) / var(TWS); R_F = spread / sqrt(max(MSE(member 0 - GRACE) - sigma^2, 25)).
```
config  pr std  spread median (p10 / p90)  bias median (p90)  R_F median  R_F<=1 / <=1.5  K median (p10 / p90)  RMS ens-mean  flags
F0      0.50    11.6 (4.4 / 21.9)          4.7 (9.2)          0.30        99 / 100 %      0.38 (0.08 / 0.67)    41.5 (m0 41.0) none
F1      0.25     6.3 (2.3 / 11.7)          2.6 (5.3)          0.16       100 / 100 %      0.15 (0.02 / 0.38)    41.0          none
F2      0.75    16.4 (5.9 / 30.8)          6.8 (13.7)         0.41        96 /  99 %      0.55 (0.14 / 0.81)    41.9          none
```
- Spread scales linearly with the precipitation std (6.3 / 11.6 / 16.4 mm); by region it is largest in sa (F2 25
  mm), na (21), ar / af (16), smallest in au (8). Forcing alone reaches only R_F ~ 0.3-0.4 of the OL-GRACE mismatch
  even at pr 0.75 - far from over-dispersive (R_F <= 1.5 in 99 % of the units).
- **"Bias" is mostly 8-member sampling noise:** RMS(ensemble mean - member 0) is expected to be spread / sqrt(8) =
  2.2 / 4.1 / 5.8 mm for F1 / F0 / F2 from sampling alone; the systematic part sqrt(bias^2 - noise^2) is 1.4 / 2.3 /
  3.5 mm - all below the 5 mm limit of rule (a). (The lognormal factor has mean 1, so no systematic wet bias; with 30
  members the sampling part drops to 1-3 mm.)
- Where the forcing puts the spread (same in all three configs): SEASONAL covariance shares soil 0.30, gw 0.11,
  river 0.11, wet 0.03, lake 0.03, snow 0.00 (median; soil dominant in 324 units, snow 133, river 131, wet 89, lake
  84); LONG-TERM shares soil 0.17, gw 0.18, lake 0.08, river 0.07, wet 0.04 (dominant: lake 180, soil 178, wet 133,
  gw 110, snow 103). Targets on the forcing-only ensemble: T1 (gw >= 50 % of long-term) FAILS at 0.18 - the forcing
  noise feeds soil / lakes / wetlands on the long time scale, the groundwater share must come from the parameters;
  T2 passes (0.82); T3 fails high in the snow units (snow 0.44 of the seasonal, 0.31 of the long-term spread; the 2 K
  temperature noise moves the snow pack) - the snow levers of 8z will have to compensate; T4 passes (wetlands 0.28 of
  the seasonal spread in the flood units -> no 8c-w needed on this evidence); T5 fails (R 0.3-0.4, parameters to add).
- **Decision (22:40 rule a, 22:45: tas 2 K and tau 1 d fixed): F2 (pr lognormal 0.75, tas 2 K, L 300 km, tau 1 d)
  is carried into 8z** - largest median spread of the realistic configs, no flag, systematic bias 3.5 mm < 5 mm, R_F
  <= 1.5 in 99 % of the units. L (F5 / F6 / F8) is tested after 8z per the order. The 30-member F2 forcing is being
  generated now (8 ranks, tmux `8gF2`, members 1-8 identical to the 8f ones by construction of the seeds) in parallel
  with 8b. If Cowork prefers F0 (the reference), say so before 8z starts (~02:00) - the F0 30-member set exists.
- **8b started 00:53** (5 ranks: member 0 + P1 min 1.073, P4 min 0.707, P6 max 275.0 K, P10 min 0.00316); model runs
  expected complete ~01:55, diagnostics automatic.

- **01:20 review read 01:22 - implemented in the ensemble diagnostics (and 8f re-evaluated with it, see below):** per
  unit s_state (spread of gw + soil + snow + river + wetlands), s_excl (lakes + reservoirs + canopy), R_state =
  s_state / sqrt(max(MSE - sigma^2, 25)) used for T5, K_eff = s_state^2 / (s_state^2 + sigma^2 + s_excl^2), count of
  units with s_excl > s_state; snow shares also for the SEASONAL-snow units (snow units minus the units whose member-0
  Aug-Sep snow stays > 10 mm in all 5 years). F2 confirmed for 8z.

- **8f addendum (01:25, state / excluded split):** F0: s_state 10.2 mm, s_excl 1.0, R_state 0.24, K_eff 0.27 (p10
  0.05 / p90 0.60); F1: 5.5 / 0.6 / 0.13 / 0.11; **F2: s_state 14.2 mm, s_excl 1.4 mm, R_state 0.34 (<= 1 in 97 % of
  the units), K_eff 0.41 (p10 0.08 / p90 0.75)**; units with s_excl > s_state: 73 / 69 / 71 (the lake / reservoir
  units, where the forcing noise mostly moves the excluded storages). Seasonal-snow units (217 of the 222 snow units;
  5 units never melt out): snow shares 0.44 seasonal / 0.30 long-term - the same as for all snow units, so the high
  snow share of the forcing-only spread is not a perennial-snow artefact.

- **01:25:** F2 30-member forcing complete (Ens_0..30 under $W/8g/F2). Idle cores used for the later L test: F5 / F6 /
  F8 (L 100 / 1000 / 500 km) are being generated now (8 members each, tmux `8gL`) WITH the chosen pr 0.75 (not the
  0.5 of the original F5 / F6 rows), so that the L comparison refers to the forcing that goes into 8z. 8b: year 2002
  done 01:23 (8 min per year at 5 ranks) -> complete ~01:58.

**Round 8b - result (runs 00:53-02:02, 5 ranks; diagnostics 02:03; `temp/exp_logs/job8/8b_report.md`; free disk
1870 TB; job ends ~10:27).** Member 0 reused from 8a.
```
P1  humid PT min  1.073   ACCEPTED  RMS 39.6 (m0 41.0), amp 0.88, S_tws median 15.3 (max 436), no flag (outlet flow > 50 % now in 1 % of the units)
P4  smax min      0.707   ACCEPTED  RMS 41.3, amp 0.80, S_tws 11.5 (soil S 14.8), drift / adjustment 1 % soil, no flag
P6  snow melt max 275.0 K UNREALISTIC perennial cells 3733 = 1.31 x member 0 (limit 1.2) -> retest 274.1 K (arithmetic halfway; 8c)
P10 gw_dis_coeff min 0.00316 ACCEPTED RMS 40.6, S_tws 58.5 (gw S 58, max 508), gw drift 3 % of land (< 5 %), adjustment 4 % (tau 1.8 yr)
```
- **8c launched 02:03** (member 0 reused + P6 max 274.1 K, 2 ranks, in parallel with 8z; ~1 h, diagnostics automatic).
  If 274.1 K fails too, the third halving (273.6 K) would be the last by rule 3; the T3 snow lever of 8z may change
  the snow ends anyway.
- **8z candidate table (table_8z.json; triangular, seed 2026, 30 members + Ens_0, forcing F2 = pr lognormal 0.75, tas
  2 K, L 300 km, tau 1 d):**
  ```
  P1  pt humid (cells 1.26)       value  1.073 / 1.26 / 1.65      min from 8b
  P2  pt arid  (cells 1.74)       value  1.365 / 1.74 / 2.115
  P3  max_daily_pet               value  6 / 15 / 22
  P4  smax_multiplier             mult   0.707 / 1 / 2            min from 8b
  P5  gamma                       x      0.5 / 1 / 2
  P6  snow_melt_temp              value  269.4 / 273.15 / 274.1   max PROVISIONAL (8c running; 275.0 failed by perennial snow)
  P7  degreeday_multiplier        mult   0.5 / 1 / 2
  P8  critcal_gw_precipitation    value  2.5 / 12.5 / 20
  P9  gw_recharge_factor_mult     mult   0.5 / 1 / 2
  P10 gw_dis_coeff                value  0.00316 / 0.01 / 0.02    min from 8b (no member below 0.003 -> no long spin-up)
  P11 netabs_gw_multiplier        mult   0.5 / 1 / 1.5
  P12 river_roughness_coeff_mult  x      1/3 / 1 / 5/3
  P13 activewetland_depth         value  0.5 / 2 / 5
  ```
  No parameter removed (none insensitive by the 23:05 rule). Parameter files written by the project code
  (perturbation_par on a copy of settings/demo_Global_v2/perturbation.json with dir -> $W/8z, par = this table);
  slow-groundwater members: none (min k = 0.00316).
- **8z launched 02:04** (`launch_8z.sh 8z table_8z.json F2`: 31 ranks, 3-year spin-up + 2002-2006, ~1 h 15; ensemble
  diagnostics automatic, groups_8z.json). **8z-P queued** (same parameter files, UNPERTURBED forcing, 31 ranks) to
  start when 8z's model runs complete (~03:15), diagnostics automatic -> parameter / forcing / joint decomposition
  with 8f-F2.

- **02:25 review read 02:27:** 1. noted. 2. P6 symmetric rule applied from 8z1 on (272.2 / 273.15 / 274.1 K if 8c
  passes, else 273.6 K / d = 0.45); 8z keeps running as the baseline. 2 / 3. added to the ensemble diagnostics (tested on
  8f): ensemble-mean minus member-0 LEVEL per unit for tws / soil / gw / snow / river / wetlands (median, p10, p90),
  the snow level bias in the snow units, and the TWS seasonal-amplitude ratio ensemble mean / member 0 - reported for
  8z, 8z-P and every rerun. **8f-F0 check (forcing alone, 8 members): ensemble mean minus member 0: tws -2.0 mm
  [p10 -8.9, p90 +2.9], gw -0.9 [-6.8, +0.2], soil +0.1, river +0.1, wetlands 0.0; SNOW in the snow units -3.5 mm
  [-8.0, -1.5]; seasonal amplitude ratio 0.99 [0.95, 1.03].** So the forcing noise is not quite neutral: the 2 K
  temperature noise melts more than it adds (nonlinear melt threshold) -> a small negative snow bias in the snow
  units, and the groundwater mean is slightly lower (lognormal precipitation with the recharge threshold). To be
  compared with the 8z values (which add the lopsided P6).

**Round 8c - result (02:03-03:29, 2 ranks alongside 8z; `temp/exp_logs/job8/8c_report.md`):** P6 snow_melt_temp max
274.1 K **ACCEPTED** - perennial cells 3307 = 1.16 x member 0 (limit 1.2), no other flag, RMS 40.9, snow level +3.0
mm. -> P6 for the reruns (Cowork 02:25 rule, symmetric): **272.2 / 273.15 / 274.1 K** (d = 0.95 K). History of the
end: 276.9 (x1.67) -> 275.0 (x1.31) -> 274.1 (x1.16) accepted.

**Round 8z - joint ensemble, result (runs 02:04-03:40, 31 ranks, 1.4 s/day with 8c alongside; diagnostics 03:42;
`temp/exp_logs/job8/8z_report.md`, per-unit $W/8z/report/8z_units.csv; free disk 1873 TB; job ends ~10:27).** 30
members, table_8z.json (P6 lopsided 269.4 / 273.15 / 274.1), forcing F2.
- **Spread:** TWS anomaly spread median 20.4 mm (p10 7.7, p90 37.1; sa 32, na 25, af 21, as 19, eu 18, ar 17, si 16,
  au 10); per storage (median) soil 9.5, gw 4.4, river 2.7, wetlands 1.6, lakes 1.6, snow 0.1; DA-state part s_state
  17.8 mm, excluded part s_excl 1.6 mm (61 units with s_excl > s_state, lake / reservoir units).
- **Skill:** ensemble-mean RMS vs GRACE 41.0 mm = member 0 (41.0), corr 0.83; ensemble-mean bias RMS 5.9 mm (30
  members: sampling part 3.7, systematic 4.6 mm - the asymmetric ranges, see below). **R_state median 0.44** (R on
  the total spread 0.51): 734 of 772 units (95 %) below 1, by a factor 2.3 in the median unit (p10 1.3, p90 4.9); 457
  units below 0.5; by region R_state sa 0.56, na 0.50, si 0.49, eu 0.43, ar 0.39, af 0.39, as 0.37, au 0.34. K median
  0.64 (p10 0.21, p90 0.85), **K_eff 0.53 (0.13 / 0.80)**.
- **Time scales (covariance shares, median over units):** SEASONAL soil 0.47, river 0.07, gw 0.05, wet 0.01, snow
  0.00 (dominant: soil 433 units, river 119, snow 106, wet 54, lake 41, gw 19); LONG-TERM soil 0.21, gw 0.16, river
  0.07, lake 0.07, wet 0.05 (dominant: soil 239, wet 164, lake 149, gw 85, snow 77, river 58). gw >= 50 % of the
  long-term spread in only 43 units (6 %); p10 / p90 of the gw share 0.02 / 0.37.
- **Targets:** T1 gw long-term share 0.16 >= 0.5: **FAIL**; T2 soil+river+wet+snow seasonal 0.93: pass; T3 snow units:
  seasonal snow share 0.39 (in 0.10-0.40: pass, p10 0.12 / p90 0.85), long-term 0.23 <= 0.15: **FAIL** (146 of 222
  snow units above 0.15), no snow flag (seasonal-snow units: 0.39 / 0.23 as well); T4 wetlands 0.22 of the seasonal
  spread in the flood units: pass (barely); T5 R_state 0.44 in 1-2: **FAIL low** - no widening allowed, so this is
  reported (the DA inflation has to supply a factor ~2.3 in the median unit).
- **Ensemble-mean minus member 0 (level; median [p10, p90]):** TWS +2.8 mm [-11.8, +26.2], soil +9.0 [+1.3, +28.8],
  gw -4.9 [-19.6, 0.0], wetlands +2.0 [0.0, +21.7], river -0.4, snow 0.0 [-10.4, 0.0]; **snow in the snow units -7.3
  mm [-16.4, -3.5]** (forcing alone gave -3.5: the lopsided P6 adds ~-4); TWS seasonal amplitude ratio 1.02 [0.94,
  1.10]. So the asymmetric multipliers do shift the mean: smax 0.707-2 and PT humid 1.073-1.65 wet the soil (+9 mm)
  and dry the groundwater (-5 mm); activewetland_depth 0.5-5 adds +2 mm (up to +22) of wetland water.
- **Flags:** after excluding lakes / wetlands from the negative-storage flag (WaterGAP lets lake and wetland storage go
  negative down to -S_max, evaporation below the outflow level; 8 members had new negative LAKE cells over 1.0-1.5 %
  of land, member 0 has them over ~10 %) -> 0 flagged members, 0 snow flags. Cowork: please confirm this
  exclusion (soil, snow, river, canopy remain flagged).
- **8z-P running since 03:41** (same parameter files, unperturbed forcing, 31 ranks; complete ~05:00; decomposition
  parameter / forcing / joint with 8f-F2 automatic after it).
- **8z1 prepared and queued behind 8z-P (table_8z1.json, levers of the 23:00 point, smallest first step for each
  failing target):** P6 symmetric 272.2 / 273.15 / 274.1 K (02:25 rule) + P7 degree-day 0.7 / 1 / 1.4 (T3 high);
  P4 smax 0.75 / 1 / 1.5 and P5 gamma 0.75 / 1 / 1.5 (T1: less long-term spread in soil; the groundwater parameters
  untouched). All else as in 8z. If the 8z-P decomposition shows the long-term soil spread comes mostly from the
  forcing, the smax / gamma shrink will not raise the gw share much - then T1 is a forcing / structure limit and I
  report it. Expected: 8z1 launch ~05:00, result ~06:20.

- **04:00 review read 04:02 - applied:** 1. ok. 2. lakes excluded, wetlands excluded for now; **member-0 check of
  negative storages (minimum monthly value 2002-2006):** local wetlands: 0 cells, 0 units; global wetlands: 0 cells,
  0 units - WaterGAP wetlands never go negative in the standard run, so the DA lower bound 0 of src_DA/bounds.py is
  consistent (no action). Local lakes: negative over 11.2 % of the land (6505 cells, 384 of 772 units, down to -4998
  mm); global lakes: 0.5 % (309 cells, 127 units, down to -89709 mm per outflow cell, the lake storage is booked on
  one cell) - lakes / reservoirs are not in the DA state, so this only matters through the excluded-storage
  subtraction. 3. **8z1 table changed: only P6 symmetric 272.2 / 273.15 / 274.1 K + P7 0.7 / 1 / 1.4; smax and gamma
  back to 0.707-2 and 0.5-2** (table_8z1.json rewritten before the queued launch reads it). T1 reported as a
  model-structure result. 4. **F10 (pr lognormal 1.0)**: 30 members being generated now (4 ranks, tmux `F10`; members
  1-8 = the 8f-F10 test set), the test round 8fF10 (member 0 + 8, 9 ranks) starts automatically when 8z-P's model
  runs complete, concurrently with 8z1 (40 ranks on 32 cores for ~1 h, accepted), ensemble diagnostics automatic;
  if it passes rule (a), 8z2 = 8z1 with F10. 5. will be reported from the 8z-P decomposition.

**Round 8z-P - parameter-only ensemble, result (runs 03:41-05:17, 31 ranks; diagnostics 05:18; decomposition
05:20; `temp/exp_logs/job8/8zdecomp_report.md` + decomp_scales.txt; free disk 1873 TB; job ends ~10:27).** Same 30
parameter files as 8z, unperturbed forcing.
- **Parameters alone:** TWS spread median 9.1 mm (p10 3.5, p90 17.5; sa 13, af 10, as 10, eu 10, na 9, si 7, ar 5, au
  4); soil 6.0, gw 3.2, river 1.7, wet 0.9, lake 0.6, snow 0.0; s_state 9.1, s_excl 0.6; R_state 0.22, K_eff 0.25;
  no flags. Shares: SEASONAL soil 0.56, river 0.04, gw 0.02 (soil dominant in 451 units); LONG-TERM soil 0.19, gw
  0.18, river 0.03, wet 0.03 (dominant: soil 250, gw 168, wet 160, lake 90). Targets for the parameter part: T1 fail
  (0.18), T2 pass (0.97), **T3 pass (snow units: seasonal 0.39, long-term 0.15)**, T4 fail (wetlands 0.12 of the
  seasonal spread in the flood units), T5 fail (0.22). Ensemble-mean bias: soil +9.2, gw -2.9, wet +1.5, tws +4.3 mm,
  snow in the snow units -4.3 (the lopsided P6 of 8z: -4 mm), amplitude ratio 1.04.
- **Decomposition of the 8z spread (variance shares of the joint 8z variance, medians over units; parameters = 8z-P,
  forcing = 8f-F2 (8 members), interaction = rest):**
  ```
             TOTAL                    SEASONAL                  LONG-TERM
             param  forc  inter      param  forc  inter        param  forc  inter   spread param/forc/joint (mm)
  tws        0.21   0.67  +0.08      0.63   0.34  +0.02        0.08   0.75  +0.13   3.1 / 10.0 / 11.7
  soil       0.45   0.40  +0.14      0.83   0.13  +0.02        0.17   0.50  +0.28   1.4 /  2.6 /  3.6
  gw         0.64   0.52  -0.15      1.08   0.14  -0.24        0.33   0.69  -0.06   1.3 /  2.0 /  2.5
  snow       0.05   0.60               0.11   0.20               0.02   0.72           (snow units: 0.07 / 1.07)
  river      0.35   0.64  -0.01      0.71   0.28  -0.02        0.15   0.78  +0.06   0.4 /  1.0 /  1.1
  wet        0.31   0.41  +0.16      0.67   0.19  +0.02        0.18   0.44  +0.26   0.4 /  0.6 /  0.9  (flood units seasonal: 0.64 / 0.28)
  lake       0.14   0.77  +0.03                                0.07   0.82  +0.06
  ```
  **Answer to Cowork 04:00 point 5:** the SEASONAL spread is carried by the parameters (63 % of the TWS seasonal
  variance; soil 83 %, gw 108 % with a negative interaction, river 71 %, wetlands 67 %), the LONG-TERM spread by the
  forcing (75 % of the TWS long-term variance; gw 69 % forcing vs 33 % parameters; snow 100 % forcing; river 78 %,
  lakes 82 %, soil 50 % + 28 % interaction). So the interannual groundwater spread the DA needs on the long time
  scale comes mainly from the precipitation noise acting through recharge, not from the parameter ranges; the
  parameter part of the long-term gw spread (1.3 mm median) is small because the groundwater parameters (P8-P11)
  mostly change the LEVEL and the seasonal recession, not the interannual variability. The snow long-term share that
  fails T3 in 8z (0.23) is entirely a forcing (2 K temperature noise) effect: the parameter part alone passes T3.
- **8z1 launched 05:18** (31 ranks; 8z with P6 272.2 / 273.15 / 274.1 K, P7 0.7 / 1 / 1.4 only; forcing F2),
  **8fF10 test started 05:17** (member 0 + 8 members pr lognormal 1.0, 9 ranks, concurrently: 40 ranks on 32 cores),
  both with automatic diagnostics; expected ~06:45 (8z1) and ~06:30 (F10). Then 8z2 = 8z1 with F10 if F10 passes.

- **05:35 review read 05:37 - applied:** 1. 8z1 keeps running (documents the P7 effect); the final table keeps P7
  0.5 / 1 / 2 unless 8z1 shows a snow long-term share lower by > 0.03 or a snow bias closer to 0; **table_8z2.json
  rewritten = 8z table with P6 symmetric 272.2 / 273.15 / 274.1 K only** (P7 0.5 / 1 / 2), read by the queued 8z2
  launch. T3 long-term reported as a tas-noise effect. 2. T1 statement kept. 3. noted for the final report.

- **06:25 time plan vs job end (10:27):** 8z1 and the F10 test run at 13-16 min per model year with 40 ranks on 32
  cores -> both complete ~07:05; 8z2 (if F10 passes; alone, 31 ranks) 07:10-08:35 + diagnostics 08:40; the seed-2027
  robustness rerun of the final table would then run 08:40-10:00 and its diagnostics would end at ~10:05 - too close
  to the job end for a safe write-up. Plan: I write the JOB 8 summary (final table, history, decomposition, R / K /
  shares) right after the 8z2 diagnostics (~08:45) and start the seed-2027 rerun immediately after; if the job is
  extended by ~1 h (Fan), the seed-2027 result is appended by ~10:10, otherwise the summary stands without it and a
  resume note gives the folders. Expiry check before 8z2: expiresAt = 21:27 + 13 h = 10 Oct 10:27 (from the last
  newTimeAllocation update; unchanged).

**Round 8fF10 - forcing test pr lognormal 1.0 (runs 05:17-06:56, 9 ranks alongside 8z1; `temp/exp_logs/job8/8fF10_report.md`):**
8 members: spread median 20.3 mm (p10 7.4, p90 38.8) vs F2 16.4; s_state 17.6, s_excl 1.7 (74 units with s_excl >
s_state); R_F 0.53 (<= 1 in 90 %, <= 1.5 in 98 % of the units), R_state 0.43; K 0.65 (0.20 / 0.87), K_eff 0.51 (0.11 /
0.82); RMS of the ensemble mean vs GRACE 42.3 (member 0 41.0); no flags. Bias: RMS(ensemble mean - member 0) median
8.6 mm, of which the 8-member sampling part is 20.3 / sqrt(8) = 7.2 -> **systematic part 4.7 mm < 5 mm: rule (a)
passed, but only just**; and the ensemble mean sits BELOW member 0: TWS -4.8 mm [p10 -22.4, p90 +6.5], gw -2.7
[-19.2, +0.5], snow in the snow units -3.0, amplitude ratio 0.97 - at pr std 1.0 the lognormal noise dries the mean
response (skewed factor, recharge / runoff thresholds), F2 had -2.0 / -0.9. Shares as F2 (seasonal soil 0.3, river
0.1; long-term soil / gw 0.17 / 0.16; snow units 0.45 / 0.30; flood units wetlands 0.30). -> **8z2 = 8z2 table (P6
symmetric, P7 0.5-2) with F10 proceeds** as queued behind 8z1; F2 vs F10 is decided on 8z1 vs 8z2 (R_state, K_eff,
bias) - the -4.8 mm mean shift of F10 is the price of the extra spread; Cowork / Fan to weigh it.
- Seed-2027 forcing (F2 and F10 settings, 30 members each) is being generated now on 4 cores (tmux `s2027`,
  ~35 min each) so that the robustness rerun of the final table can start right after 8z2.

**Round 8z1 - rerun 1 (8z + P6 symmetric 272.2 / 273.15 / 274.1 K + P7 0.7 / 1 / 1.4; runs 05:18-07:04, 31 ranks
with the F10 test alongside; diagnostics 07:06; `temp/exp_logs/job8/8z1_report.md`; free disk 1872 TB).**
```
                        8z          8z1         change
spread median (mm)      20.4        20.3
s_state / s_excl        17.8 / 1.6  17.6 / 1.6
R_state median          0.44        0.44        (95 % of units < 1 in both)
K / K_eff median        0.64 / 0.53 0.64 / 0.53
RMS ens-mean vs GRACE   41.0        40.8        (member 0 41.0)
bias RMS median         5.9         5.4
ens-mean - m0: tws      +2.8        +5.0        soil +9.0 -> +9.1, gw -4.9 -> -4.0, wet +2.0 -> +2.2
snow bias, snow units   -7.3        -3.6        [p10 -8.2, p90 -1.5]   <- the symmetric P6 (lopsided P6 cost ~ -4 mm)
amplitude ratio         1.02        1.03
snow units seasonal     0.39        0.27        <- narrower degree-day range (P7 0.7-1.4)
snow units long-term    0.23        0.25        no gain (+0.02; it is the tas-noise part)
T1 / T2 / T3 / T4 / T5  F/P/F/P/F   F/P/F/P/F   (T3 still fails on the long-term share 0.25 > 0.15)
flags                   0           0
```
- **P7 decision (05:35 rule):** the snow long-term share did not fall (0.23 -> 0.25) and the snow bias improvement
  (-7.3 -> -3.6 mm) is the symmetric P6, not P7 (8z2, with P6 symmetric and P7 back at 0.5 / 1 / 2, will separate the
  two). -> **final table keeps P7 0.5 / 1 / 2** unless 8z2's snow bias is clearly worse than -3.6 mm.
- **07:10 review read 07:12:** F2 vs F10 decision rule (i)-(iv) noted; both will be tabulated side by side (8z1 vs
  8z2) in the summary. 8z2 (P6 symmetric, P7 0.5-2, F10) launched 07:08, 31 ranks alone -> complete ~08:30,
  diagnostics ~08:35. The seed-2027 robustness rerun (8z3) is armed: it starts as soon as I write the chosen forcing
  into final_forcing.txt after the 8z2 diagnostics (table_8z3.json = final table with seed 2027; forcing F2s2027 or
  F10s2027, being generated now).

**Round 8z2 - rerun 2 (P6 symmetric 272.2 / 273.15 / 274.1 K, P7 0.5 / 1 / 2, forcing F10 = pr lognormal 1.0; runs
07:08-08:33, 31 ranks alone, 1.1 s/day; diagnostics 08:37; `temp/exp_logs/job8/8z2_report.md`; free disk 1872 TB).**
```
                              8z1 (F2, P7 0.7-1.4)   8z2 (F10, P7 0.5-2)   rule 07:10
spread median (p10 / p90)     20.3 (7.6 / 37.1)      24.0 (9.0 / 44.3)
s_state / s_excl              17.6 / 1.6             21.1 / 1.9
R_state median                0.44 (95 % < 1)        0.53 (89 % < 1)       (i) +20 % >= +15 %   PASS
K / K_eff median              0.64 / 0.53            0.72 / 0.61 (p10 0.16, p90 0.85)
RMS ens-mean vs GRACE         40.8                   41.2 (member 0 41.0)  (ii) +0.5 % <= 3 %    PASS
bias RMS median / sampling    5.4 / 3.7 -> syst 3.9  6.0 / 4.4 -> syst 4.1 (iii) <= 5 mm         PASS
amplitude ratio ens / m0      1.03 [0.97, 1.10]      1.02 [0.97, 1.10]     (iii) 0.95-1.05       PASS
ens-mean - m0 level: tws      +5.0                   +4.3   (soil +8.7, gw -5.2, wet +2.4)
snow bias, snow units         -3.6                   -3.9 [-9.4, -1.6]     -> P7 0.5-2 costs 0.3 mm only: keep 0.5 / 1 / 2
snow units seasonal / long    0.27 / 0.25            0.32 / 0.26
T1 / T2 / T3 / T4 / T5        F / P / F / P / F      F / P / F / P / F     T4 0.26 (8z1 0.24)
flags                         0                      0                     (iv) PASS
```
-> **F10 (pr lognormal 1.0) is the final forcing** by the 07:10 rule (all four conditions hold), P7 stays 0.5 / 1 / 2
(05:35 rule), P6 symmetric. The final table = table_8z3.json (seed 2027 copy for the robustness run; the seed-2026
version is the same table with "seed": 2026).
- **8z3 (seed 2027 robustness rerun of the final table, forcing F10s2027) launched 08:40** (31 ranks; the first launch
  at 08:38 failed because the queued launcher evaluated the forcing file name too early - killed, rebuilt, relaunched
  within 2 min). Expected complete ~10:05, diagnostics ~10:10; **job ends 10:27** (13 h allocation confirmed again at
  08:39). If the job stops first: resume note below.
- **Resume note (if the job is cut):** everything is under /work/PyGLDA_v2_external_data/temp_job8/: 8z3 restarts in
  8z3/init/Ens_k (written after the spin-up), monthly files 8z3/monthly/Ens_k/<year>.nc, chain `run_round.sh 8z3 31`
  (skips what exists), diagnostics `ensemble_diag.py 8z3 groups_8z3.json`; forcing sets 8g/F2, F10, F2s2027, F10s2027
  (30 members each), F0, F1, F5, F6, F8 (8 members); every round's report in <round>/report/ and temp/exp_logs/job8/.

- **08:46 review read 08:48:** Cowork applied the final table to settings/demo_Global_v2/perturbation.json; verified on
  the desktop copy (md5 73105314...: seed 2026, pr lognormal 1.0, tas 2 K, L 300 km, tau 1 d; P1 humid 1.073 / 1.26 /
  1.65, P4 0.707 / 1 / 2, P6 272.2 / 273.15 / 274.1, P10 0.00316 / 0.01 / 0.02, P7 0.5 / 1 / 2, gamma 0.5 / 1 / 2).
  No further rounds; the 8z3 result will be appended when its diagnostics finish (~10:10).

- **10:10 review read 10:16 - 8z4 (tas 1 K) started as a chain at 10:17:** F11 forcing (pr lognormal 1.0, tas 1 K,
  L 300 km, tau 1 d, 30 members, seed 2026) is being generated (8 ranks, ~20 min), then `launch_8z.sh 8z4
  table_8z4.json F11` (final table, seed 2026) starts automatically with its diagnostics; expected result ~12:00 IF
  the job is extended (allocation at 10:16 still 13 h = ends 10:27). If the job expires before: **8z4 NOT RUN** - the
  chain is resumable after a restart: F11 under $W/8g/F11 (complete when 31 member folders hold 84 files), then
  `bash launch_8z.sh 8z4 table_8z4.json F11` from temp/exp_logs/job8 (run_round.sh skips finished parts). The
  8z4-vs-8z2 decision table (rule (i)-(iv) of 10:10) will be written here when the diagnostics are done.

**Round 8z3 - seed-2027 robustness rerun of the final table (parameters AND forcing redrawn with seed 2027; runs
08:40-10:01, 31 ranks; diagnostics 10:05; `temp/exp_logs/job8/8z3_report.md`; free disk 1872 TB; job ends 10:27).**
```
                              8z2 (seed 2026, final)   8z3 (seed 2027)
spread median (p10 / p90)     24.0 (9.0 / 44.3)        24.5 (9.4 / 45.6)
s_state / s_excl              21.1 / 1.9               21.3 / 1.9
R_state median (% < 1)        0.53 (89 %)              0.54 (88 %)
K / K_eff median              0.72 / 0.61              0.72 / 0.61 (p10 0.17, p90 0.86)
RMS ens-mean vs GRACE         41.2                     40.7   (member 0 41.0)
shares SEASONAL soil/gw/river 0.46 / 0.05 / 0.08       0.45 / 0.07 / 0.08
shares LONG-TERM soil/gw      0.21 / 0.15              0.22 / 0.15
snow units seasonal / long    0.32 / 0.26              0.28 / 0.24
flood units wetlands seasonal 0.26                     0.31
T1-T5                         F / P / F / P / F        F / P / F / P / F
ens-mean - m0: tws / soil / gw / wet   +4.3 / +8.7 / -5.2 / +2.4    +14.8 / +13.6 / -3.8 / +4.9  (p90 tws +60, wet +45)
snow bias, snow units         -3.9                     -3.7
flags                         0                        2 members (m10, m11): outlet mean flow change > 50 % in 15 / 12 % of the units
```
- **Robust:** spread, R_state, K_eff, the storage shares on both time scales, the targets and the snow bias change by
  < 5 % between the two seeds - the spread statistics of the final table are not a property of one draw.
- **Draw-dependent:** the ensemble-mean level shift (TWS +4 mm with seed 2026, +15 mm with seed 2027; soil +9 / +14,
  wetlands +2 / +5, p90 up to +45-60 mm) and the outlet-flow flag of two seed-2027 members (extreme combinations of
  the accepted ends, e.g. low PT humid + low smax + high gamma). Both come from the ASYMMETRIC ranges of the
  multipliers / depths (smax 0.707-2, wetland depth 0.5-5, PT humid 1.073-1.65, gamma 0.5-2 all have their standard
  value below the arithmetic mid-range). Suggestion for Fan / Cowork (not applied; the table is Cowork's): (a) check
  the production members (seed 2026) for the outlet-flow flag before the pilot (8z2 itself had none), (b) if the mean
  shift matters for the DA (it is a bias the filter will "correct" with GRACE), make the multiplier ranges symmetric
  in log space (e.g. smax 0.707 / 1 / 1.414, wetland depth 1 / 2 / 4, PT humid 1.073 / 1.26 / 1.48) - at the price of
  ~10-15 % less spread.
- All rounds complete; nothing more is running. tmux session `job8` keeps the windows for inspection; $W kept (daily
  files deleted; monthly files, inputs, forcing sets 8g/ and reports kept).

- **10:25 review read 10:27:** (1) the 8z3 section and the JOB 8 DONE header line are complete after the sync restore
  (checked). (2) 8z4 chain running (F11 generation, then the run). (3) **8z5 (log-symmetric table) prepared:**
  table_8z5.json = applied table with P1 humid max 1.48, P4 smax max 1.414, P10 min 0.005, P13 1 / 2 / 4 m (P6
  272.2 / 273.15 / 274.1, seed 2026); its launcher is armed and starts automatically after 8z4's model runs with the
  forcing I write into final_forcing_8z5.txt after the 8z4 diagnostics (F11 if tas 1 K passes the 10:10 rule, else
  F10). Decision rule 8z5 vs 8z4 (i)-(iv) noted; the mean shift per storage will be reported.

- **10:30 job extended: ends 10 Oct 18:26** (newTimeAllocation 21 h). Plan: 8z4 (tas 1 K) result ~12:00 -> decision ->
  8z5 (log-symmetric) ~12:05-13:30 -> decision -> the optional L test (F5 / F6 / F8 = 100 / 1000 / 500 km, 8 members
  each + member 0, 25 ranks, forcing-only, vs F2 at 300 km) ~13:35-14:50 if Cowork still wants it.

**Round 8z4 - applied table with tas 1 K (F11 = pr lognormal 1.0, tas 1 K, L 300 km, tau 1 d; runs 10:31-12:00, 31
ranks; diagnostics 12:04; `temp/exp_logs/job8/8z4_report.md`; free disk 1871 TB; job ends 18:26).**
```
                              8z2 (tas 2 K, applied)   8z4 (tas 1 K)          10:10 rule
spread median (p10 / p90)     24.0 (9.0 / 44.3)        24.0 (9.1 / 44.3)      (ii) drop 0 % < 5 %        PASS
s_state / s_excl              21.1 / 1.9               21.1 / 1.9
R_state median (% < 1)        0.53 (89 %)              0.53 (89 %)            (ii) drop 0 %              PASS
K / K_eff median              0.72 / 0.61              0.72 / 0.61
RMS ens-mean vs GRACE         41.2                     41.2
snow units seasonal share     0.32                     0.33                   (iii) 0.10-0.40            PASS
snow units LONG-TERM share    0.26                     0.26                   (i) <= 0.15                FAIL (unchanged)
snow bias, snow units         -3.9 [-9.4, -1.6]        -1.2 [-3.9, -0.1]      (i) improved               (yes)
ens-mean - m0: tws/soil/gw/wet +4.3 / +8.7 / -5.2 / +2.4  +4.7 / +8.3 / -5.0 / +2.3
T1-T5                         F / P / F / P / F        F / P / F / P / F
flags                         0                        0                      (iv)                       PASS
```
-> **tas 2 K is kept** (rule (i) fails: the long-term snow share does not move). **Correction of the earlier
interpretation:** halving the temperature noise leaves the long-term snow share in the snow units at 0.26, so that
share is NOT a temperature effect - it comes from the precipitation noise through the snowfall amounts (the
parameter-only ensemble had 0.15; 8z-P + F2 decomposition: snow long-term variance 100 % forcing - and the forcing
component is pr, not tas). The temperature noise drives only the snow BIAS (-3.9 -> -1.2 mm in the snow units with
1 K). Consequence for Fan: tas 1 K would remove most of the snow bias at no cost in spread or R, but it does not
help T3; if the bias matters, 1 K is the cheaper choice (Cowork's rule says keep 2 K - kept).
- **8z5 (log-symmetric table, forcing F10) launched 12:07** (final_forcing_8z5.txt = F10; table_8z5.json: P1 humid
  max 1.48, P4 smax max 1.414, P10 min 0.005, P13 1 / 2 / 4 m, P6 272.2 / 273.15 / 274.1, seed 2026); diagnostics
  automatic, result ~13:30.

- **12:07 review read 12:12:** L test confirmed - round 8fL (F5 100 km, F6 1000 km, F8 500 km; pr 0.75, tas 2 K,
  8 members each + member 0 = 25 ranks, forcing-only, vs 8f-F2 at 300 km) is built and queued behind 8z5 with
  automatic diagnostics (spread, R_F, s_state / s_excl, shares per L); result ~15:00.

- **12:30 review read 12:33:** tas 1 K adopted by Cowork (production forcing = F11: pr lognormal 1.0, tas 1 K, L 300
  km, tau 1 d; applied to settings/demo_Global_v2/perturbation.json). 8z5 continues with F10 as launched; its report
  will list all T1-T5 numbers (incl. the snow-unit long-term snow share and the flood-unit wetland share) next to
  8z4 / 8z2. L test stays queued.

**River-storage check (Cowork 12:45, Fan), diagnostics on the 8z4 ensemble (applied table, F11; 30 members),
13:05; CSVs $W/8z4/report/river_check.csv (all units) and river_check_large_rivers.csv, copies in temp/exp_logs/job8/.**
Columns: member-0 river seasonal amplitude (half range of the 12-month climatology, mm), GRACE amplitude (same), river
spread (time-mean member std of the river anomaly, mm), spread / amplitude, river share of the SEASONAL spread
(covariance-consistent), the same for wetlands, river spread / GRACE sigma, TWS spread; "unit" = the unit holding the
largest member-0 mean streamflow within +-6 deg lat / +-8 deg lon of the river mouth (unit id = sub_basin id of the
frozen mask, name = shapefile NAME):
| river | unit | name | max Q m3/s | river amp m0 | GRACE amp | river spread | spread/amp | river seas share | wet amp m0 | wet spread | wet seas share | river spread/sigma | tws spread | flood |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Amazon | 567 | sa_6167 | 190503 | 115.3 | 199.5 | 30.6 | 0.27 | 0.53 | 76.1 | 20.5 | 0.21 | 1.88 | 50.9 | True |
| Congo | 39 | af_1321 | 38464 | 51.7 | 124.1 | 26.9 | 0.52 | 0.72 | 1.5 | 0.9 | 0.00 | 1.34 | 33.6 | False |
| Ganges-Brahmaputra | 250 | as_45311 | 35041 | 198.7 | 224.8 | 48.1 | 0.24 | 0.89 | 5.4 | 2.5 | -0.01 | 2.73 | 54.6 | False |
| Mekong | 218 | as_4422 | 15553 | 83.0 | 310.8 | 20.8 | 0.25 | 0.50 | 48.3 | 26.9 | 0.21 | 1.17 | 60.5 | True |
| Mississippi | 485 | na_7328 | 21239 | 30.9 | 105.8 | 12.2 | 0.40 | 0.26 | 24.4 | 17.5 | 0.19 | 0.67 | 41.3 | True |
| Parana | 649 | sa_64212 | 19172 | 43.8 | 55.2 | 32.8 | 0.75 | 0.75 | 6.6 | 9.4 | 0.02 | 1.61 | 42.9 | False |
| Niger | 77 | af_1338 | 6966 | 30.1 | 188.0 | 9.4 | 0.31 | 0.16 | 14.2 | 4.7 | 0.03 | 0.55 | 32.3 | True |
| Zambezi | 27 | af_12302 | 3695 | 12.1 | 85.0 | 5.3 | 0.43 | 0.13 | 12.8 | 5.7 | 0.07 | 0.28 | 19.8 | True |
| Ob | 687 | si_3123 | 14438 | 71.6 | 81.1 | 18.3 | 0.26 | 0.80 | 13.7 | 9.3 | 0.04 | 1.68 | 31.9 | True |
| Yenisei | 713 | si_321 | 20531 | 50.2 | 112.9 | 7.4 | 0.15 | 0.34 | 0.1 | 0.0 | -0.00 | 0.63 | 20.4 | False |
| Lena | 741 | si_3421 | 17276 | 125.8 | 57.0 | 21.7 | 0.17 | 0.95 | 3.1 | 0.9 | -0.01 | 2.21 | 26.3 | False |
| Danube | 351 | eu_22735 | 6908 | 30.6 | 68.3 | 13.7 | 0.45 | 0.37 | 0.0 | 0.0 | -0.00 | 0.83 | 27.2 | False |
| Yangtze | 210 | as_43504 | 30435 | 13.8 | 58.8 | 5.1 | 0.37 | 0.16 | 1.1 | 1.4 | 0.01 | 0.28 | 26.3 | False |
- **Flood units (178):** river spread median 3.8 mm (p10 1.5, p90 17.4), river seasonal share median 0.07 (p10 0.02,
  p90 0.53); wetland spread median 14.6 mm, wetland share 0.25; **river + wetlands >= 0.5 of the seasonal spread in 68
  of the 178 flood units**; river spread / river amplitude 0.51 median; river spread / GRACE sigma 0.25; member-0
  river amplitude median 7.6 mm vs GRACE TWS amplitude 98 mm. All units: river spread median 3.2 mm (p90 16.8),
  share 0.08; 162 units with a river share >= 0.3.
- **Reading:** on the big rivers the river storage already carries the seasonal spread (Amazon 0.53, Congo 0.72,
  Ganges-Brahmaputra 0.89, Lena 0.95, Ob 0.80, Parana 0.75; river spread 1.2-2.7 x the GRACE sigma there), and the
  wetlands add 0.2 in the Amazon / Mekong / Mississippi units. The weak cases are the floodplain units of smaller
  rivers (Niger 0.16, Zambezi 0.13, Mississippi 0.26, Yangtze 0.16, Danube 0.37) where member-0's river amplitude is
  only 12-31 mm against a GRACE amplitude of 60-190 mm: the river spread (5-14 mm) is 0.3-0.5 of that small
  amplitude, so widening P12 (roughness 1/3-5/3) cannot create seasonal river water the model does not store; there
  the seasonal signal sits in soil / wetlands / groundwater in WaterGAP. Median flood unit: 3.8 mm river spread,
  about half the member-0 river amplitude.

**Round 8z5 - log-symmetric table (P1 humid 1.073 / 1.26 / 1.48, P4 smax 0.707 / 1 / 1.414, P10 0.005 / 0.01 / 0.02,
P13 wetland depth 1 / 2 / 4 m; rest as applied; forcing F10 = pr 1.0, tas 2 K; runs 12:07-13:34, 31 ranks;
diagnostics 13:36; `temp/exp_logs/job8/8z5_report.md`; free disk 1870 TB).**
```
                              8z4 (applied table, F11)  8z5 (log-symmetric, F10)  10:25 rule
spread median (p10 / p90)     24.0 (9.1 / 44.3)         22.2 (8.5 / 42.6)         (ii) -7.5 % <= 15 %        PASS
s_state / s_excl              21.1 / 1.9                19.1 / 1.9
R_state median (% < 1)        0.53 (89 %)               0.49 (91 %)               (ii) -7.5 % <= 15 %        PASS
K / K_eff median              0.72 / 0.61               0.69 / 0.56
RMS ens-mean vs GRACE         41.2                      41.5   (member 0 41.0)
ens-mean - m0: TWS            +4.7 [-9.0, +25.5]        -1.2 [-16.4, +14.7]       (i) |median| <= 5, p90 <= 25  PASS
   soil / gw / wet / river    +8.3 / -5.0 / +2.3 / -0.2 +2.1 / -5.5 / +2.2 / +0.2
   snow in the snow units     -1.2 (tas 1 K)            -3.8 (tas 2 K)
amplitude ratio ens / m0      1.02                      0.98 [0.94, 1.07]
T1 gw long-term share         0.15                      0.15                      F
T2 seasonal soil+river+wet+snow 0.91                    0.87                      (iii) pass                 PASS
T3 snow units seasonal / long 0.33 / 0.26               0.37 / 0.28               F (long-term)
T4 flood units wetlands       0.25                      0.26                      (iii) pass                 PASS
T5 R_state                    0.53                      0.49                      F
flags                         0                         2 members (m10, m27: outlet flow > 50 % in 10 / 13 % of units)   (iv) FAIL
```
- **Mean shift:** the log-symmetric ranges remove the soil shift (+8.3 -> +2.1 mm) and bring the TWS level of the
  ensemble mean to -1.2 mm (p10 / p90 -16 / +15 instead of -9 / +26). The groundwater shift (-5 mm) and the
  wetland shift (+2.2 mm) do NOT change: they are not from the asymmetric ranges but from nonlinear responses
  (gw: the precipitation noise through the recharge threshold, see 8f: forcing alone gives -0.9 to -2.7; wetlands:
  the wetland storage responds asymmetrically to depth). Cost: -7.5 % spread and R_state.
- **The only failing condition is (iv), and it is a threshold artefact:** the two flagged members are NOT extreme
  (m10: PT humid 1.22, smax 0.81, gamma x1.22, recharge factor 1.69, roughness x0.79, wetland depth 1.7 m; m27: PT
  humid 1.20, max_daily_pet 10, gamma x0.75, degree-day 0.79) and the SAME member numbers in 8z4 with almost the
  same values (m10: PT 1.25, smax 0.85, otherwise identical draws) were not flagged - the flag is "> 50 % mean-flow
  change in > 10 % of the units" and these members sit at 8-13 % in both runs. The seed-2027 run (8z3) had the same
  kind of flags (12 / 15 %). So by the letter 8z5 fails on (iv) only, not on the mean shift; if Cowork treats the
  outlet-flow threshold as the soft criterion it is (e.g. > 15 %), 8z5 passes all four. **Decision is Cowork's**;
  my reading: the log-symmetric table is the better production table (unbiased mean at -7 % spread), and the
  outlet-flow metric should be reported, not used as a hard flag for joint ensembles (it was designed for OAT ends).
- 8fL (L test, 25 ranks) started 13:36 automatically; result ~15:00.

- **13:50 review read 13:55:** (1) river: noted, P12 unchanged. (2) production table = 8z5 + F11 verified in the synced
  settings/demo_Global_v2/perturbation.json (seed 2026, pr 1.0, tas 1 K, P1 humid 1.073 / 1.26 / 1.48, P4 0.707 / 1 /
  1.414, P10 0.005 / 0.01 / 0.02, P13 1 / 2 / 4, P6 272.2 / 273.15 / 274.1); the JOB 8 DONE block now shows this table
  with the updated kept / reason columns and the production line. (3) **Outlet-flow metric of the 30 8z5 members
  (share of units whose member-0 outlet changes its mean flow by > 50 %): median 5.5 %, p90 9.7 %, max 13.1 %
  (member 27); members above 10 %: 2 (m10, m27); above 15 %: none.** JOB 8 CLOSED follows after the 8fL result.

**Round 8fL - correlation-length test, forcing only (pr lognormal 0.75, tas 2 K, tau 1 d; 8 members per L + member 0;
runs 13:36-14:59, 25 ranks; diagnostics 15:00; `temp/exp_logs/job8/8fL_report.md`; F2 = 300 km from 8f for
comparison; documentation only, no decision rule).**
```
 L (km)   spread median (p10 / p90)  R_F median (<=1 / <=1.5)  s_state  s_excl  units s_excl > s_state  bias RMS / 8-member sampling -> systematic  K_eff  flags
 100 F5    9.5 (3.5 / 17.8)           0.24 (99 / 100 %)         7.9      1.1     97                      4.8 / 3.4 -> 3.4 mm                          0.20   0
 300 F2   16.4 (5.9 / 30.8)           0.41 (96 /  99 %)        14.2      1.4     71                      6.8 / 5.8 -> 3.5 mm                          0.41   0
 500 F8   18.6 (7.0 / 35.0)           0.49 (93 /  98 %)        16.1      1.6     68                      7.6 / 6.6 -> 3.8 mm                          0.47   0
1000 F6   20.6 (7.8 / 40.8)           0.55 (89 /  98 %)        17.4      1.7     68                      8.1 / 7.3 -> 3.5 mm                          0.51   0
```
- The unit-mean spread grows with L because a longer correlation length cancels less of the daily noise inside a unit
  (100 km: -42 % vs 300 km; 500 km: +13 %; 1000 km: +26 %); the state / excluded split, the storage shares (seasonal
  soil 0.29, river 0.11-0.13, gw 0.09-0.10; long-term soil 0.17, gw 0.17, lake 0.08-0.09) and the ensemble-mean
  biases (TWS -3.0 to -3.4 mm, snow in the snow units -3.4, amplitude ratio 0.98) are the same for all L; no flag at
  any L. So L is a pure spread knob with no realism signal in these diagnostics; 300 km stays unless Cowork prefers
  the larger spread of 500 km (Cowork's call). Forcing sets F5 / F6 / F8 remain under $W/8g.
**JOB 8 CLOSED 10 Oct 2026 15:05.** Production table = 8z5 (log-symmetric) + F11, applied by Cowork to
settings/demo_Global_v2/perturbation.json. Nothing is running; tmux session `job8` and
/work/PyGLDA_v2_external_data/temp_job8/ (inputs, monthly files, reports, forcing sets 8g/; daily files deleted) are
kept; all scripts and reports are in temp/exp_logs/job8/ on both machines. src_*, settings/ and exp_Ucloud.py were
never edited by Claude Code; nothing under Ensemble_input, Ensemble_Initialization, OL_output or Res was touched.

**11 Oct 11:05 - new UCloud job 12418791 (port 2540, 32 vCPU, allocation 1 h -> ends 12:01). Clean-up orders of 10 Oct
15:35 / 16:40 / 16:55 read; INVENTORY taken (du -s, GB) before any deletion - nothing deleted yet, waiting for Fan's
confirmation of the two ambiguous points below (irreversible, 3.9 TB):**
```
folder (under /work/PyGLDA_v2_external_data)   size      order            note
temp_job8                                       2183 GB   15:35 delete     Job 8 inputs / monthly / reports / forcing sets 8g (copies of reports in temp/exp_logs/job8)
OL_output                                       1004 GB   16:40 delete     global open loop 2002-2019 of demo_Danube (30 members, daily); the pilot writes its own OL_output_v2
Ensemble_input                                   536 GB   16:40 delete     30-member perturbation of 5 Oct (parameters + forcing 2000-2019)
DA_output (demo_1..3, Amazon A1/A2, Danube, exp1, exp4)   87 GB   16:40 delete
Ensemble_Initialization                           35 GB   16:40 delete     restarts of the 30 members (6 dates)
temp_job7                                          5 GB   16:40 delete     Job 7 check runs
GRACE/OLmean (11 files)                            1 GB   16:40 delete     regenerable
--- total to delete                             ~3851 GB
KEEP: Input_data 69 GB, GRACE 236 GB (output / output_TUD / SaGEA / global_mask / obs), Basin 1 GB, Res 43 GB, envs 6, pkgs 3, Extra 12, Auxiliary 1
```
"demo" items found outside Input_data / GRACE/output* / Basin (16:55 order): (a) the DA_output/demo_* and GRACE/OLmean/demo_*
files - already covered above; (b) **Res/demo_1, demo_2, demo_3, demo_Amazon_A1, demo_Amazon_A2, demo_Danube (incl. the
6 Oct product Res/demo_Danube/product), demo_Danube_exp1, demo_Danube_exp4 = 43 GB** - the 16:40 and 16:55 orders both say
KEEP Res, the 16:55 order says delete "demo case outputs, demo products" -> contradiction, NOT deleted until Fan says which;
(c) **/work/PyGLDA/settings/demo_* and src_demo/** (code, 1 MB each) - these are SYNCED to the desktop and in git: deleting
them on UCloud would propagate to the workstation; the GATE text says Fan deletes the other settings/demo_* folders himself
-> NOT touched by me. Job 9 gate: src_demo/demo_global.py md5 00886009 and settings/demo_Global_v2/Config_ReWaterGAP.json
md5 56a3dcbf match on both machines (driver ready); WaterGAPLandMask md5 9a21d88b ok; the mask move (step 0) is pending.
This 1-hour job cannot host Job 9 (>= 10 h needed) - a longer job is required for it.

**11 Oct 11:10 Fan confirmed (desktop session): delete the listed output; KEEP Res/demo_* (contradiction resolved in
favour of the KEEP-Res rule); settings/demo_* and src_demo untouched (synced code). Deleted 11:17-11:27 by
temp/exp_logs/cleanup_11oct.sh in tmux `cleanup` (log cleanup_11oct.out):**
```
folder                       size      result
temp_job8                    2183 GB   deleted   (Job 8 inputs, monthly files, report copies, forcing sets 8g)
OL_output                    1004 GB   deleted   (global open loop of demo_Danube 2002-2019)
Ensemble_input                536 GB   deleted   (30-member perturbation of 5 Oct)
DA_output                      87 GB   deleted   (demo_1/2/3, Amazon A1/A2, Danube, exp1, exp4)
Ensemble_Initialization        35 GB   deleted
temp_job7                       5 GB   deleted
GRACE/OLmean                    1 GB   deleted   (regenerable)
Res (demo_1..3, Amazon A1/A2, Danube incl. product, exp1, exp4)   43 GB   kept (Fan 11:10)
Input_data 69 GB, GRACE 236 GB, Basin 1 GB, Extra 12 GB, Auxiliary 1 GB, envs 6 GB, pkgs 3 GB, Initialization 0   kept
settings/demo_*, src_demo (code, synced)   kept - Fan deletes the obsolete settings folders on the workstation
total freed 3851 GB; remaining under /work/PyGLDA_v2_external_data: 371 GB; project quota free 1867 TB of 7.45 PB (df)
```
**JOB 8 OUTPUT DELETED 11:27. ALL OUTPUT DELETED 11 Oct 2026 11:27.** Evidence kept: temp/exp_logs/job8/ (71 MB, both
machines: all round reports .md, decision / summary / ensemble_summary CSVs, history, final tables, scripts, river
check with per-unit CSV, F5 / F6 / F8 / F10 per-unit CSVs). NOTE: the per-unit CSVs of the OAT round 8a and of the
joint ensembles 8z..8z5 / 8z-P (772-unit tables of spread, R, K, shares) were only under temp_job8/<round>/report and
are gone with it; all quoted medians / percentiles / regional values survive in the reports.
- **Job 9 step 0 done 11:30:** `Basin/mask/mask/GlobalBasins_v1.0` moved to `Basin/mask/GlobalBasins_v1.0` (the
  `mask/mask/GlobalBasins` copy kept); md5 verified against GlobalBasins_v1.0_settings.txt: _res_0.5.h5 0a57670d0e5c7ccf0f201689ca474f57 and _res_1.h5 e83cc4dafc91a8ed782ff8104125b610 both match. settings/demo_Global_v2/DA_setting.json (basin_mask -> Basin/mask/GlobalBasins_v1.0/...) now resolves on UCloud without any json edit. WaterGAPLandMask md5
  9a21d88b ok. Gate ok (demo_global.py 00886009, Config_ReWaterGAP.json 56a3dcbf on both machines). **Job 9 itself
  needs a >= 10 h job; this job (12418791) has 1 h (ends 12:01), so Job 9 starts in the next long job.**

**Job 9 - global pilot DA, case GDA_Ucloud (job 12418791 extended to 4 h -> ends 15:01; 32 vCPU).**
- 11:31 first attempt with the old case name: config_external_data, read_config_and_save and check_units passed (772
  units, IDs equal, 0 cells in two units, 0 empty, md5 True); get_GRACE_obs failed at once because the obs output
  folder did not exist (the code does not create `preprocess_res` - small gap for Cowork: `GRACE_CSR_mascon.basin_TWS`
  opens `<dir_out>/<basin>_signal.hdf5` without mkdir); nothing else ran.
- **12:00 order (Cowork / Fan) read 12:05 and applied:** no version suffixes anywhere; case = GDA_Ucloud. The empty
  folders GRACE/output_v2 and GRACE/OLmean_v2 I had created are removed; no other *_v2 folder exists. Synced md5s
  verified on both machines: src_demo/GDA_Ucloud.py 22b0fc62, settings/GDA_Ucloud/DA_setting.json 83e729c1,
  perturbation.json 24faa2cb, Config_ReWaterGAP.json 56a3dcbf. The driver is NOT edited: my runners
  temp/exp_logs/job9/run_step1.py (the demo1 steps) and run_stage9.py (<stage> of the demo2 chain, one mpiexec per
  stage so that a job end between stages loses nothing) import GDA_Ucloud.py for the case configuration and call
  the same GDA steps. Step 1 relaunched 12:07 in tmux `job9` (serial): config_external_data, read_config_and_save,
  check_units, get_GRACE_obs -> GRACE/output/GlobalBasins_v1.0_{signal,cov,gridded_signal}.hdf5 (2002-01..2006-12).

- **Step 1 done 11:51 (UCloud clock):** config_external_data, read_config_and_save, check_units (772 / 772 units,
  IDs equal, 0 cells in two units, 0 empty, md5 True), get_GRACE_obs -> GRACE/output/GlobalBasins_v1.0_signal.hdf5
  (54 epochs 2002-04-15 .. 2006-12-15, 772 units), _cov.hdf5 (54 x 772 x 772), _gridded_signal.hdf5 (54 x 55461),
  42 s. Job allocation now ends ~23:00 (670 min remaining at 11:51).
- **Stage chain started 11:51 and ABORTED in `model_perturbation` (rank 14 and all ranks):**
  `FileNotFoundError: /work/PyGLDA_v2_external_data/Ensemble_input_v2/Ens_0` (parallel_logs/perturbation/rank_14.log).
  Two causes: **(1) settings/GDA_Ucloud/perturbation.json (md5 24faa2cb) still has `"dir": {"out":
  ".../Ensemble_input_v2"}`** - the production json kept the v2 output path; DA_setting.json is right (Ensemble_input,
  Ensemble_Initialization, OL_output, DA_output). config_external_data only rewrites the data-root prefix, not the
  folder name. **Cowork / Fan: please set `dir.out` to `/work/PyGLDA_v2_external_data/Ensemble_input` in
  settings/GDA_Ucloud/perturbation.json** (I do not edit settings); I rerun the chain as soon as the synced md5 changes.
  **(2)** the standard folders Ensemble_input, Ensemble_Initialization, OL_output, DA_output and GRACE/OLmean were
  deleted yesterday and the code creates only the Ens_k subfolders (`mkdir(exist_ok=True)` without parents in
  perturbation_base._make_dir / OpenLoop.configure_Ens_output / configure_ini_for_resume) -> re-created them empty on
  the data side (12:00). Small code gap for Cowork: `parents=True` in those mkdir calls (and for `preprocess_res`).

- **12:10 review read** (no new entry after 12:00; perturbation.json still 24faa2cb). **Safe default taken (Fan's protocol:
  do not wait idle):** the perturbation stage runs with `setting_dir` pointing to a COPY of settings/GDA_Ucloud in
  temp/exp_logs/job9/settings_fix/ that differs only in perturbation.json `dir.out` = .../Ensemble_input (plain name);
  all later stages use the real settings/GDA_Ucloud (DA_setting.json already has the plain names). The original
  settings file is untouched. Chain relaunched 12:08 (tmux job9, window `chain`): perturbation -> spin_up -> OL ->
  collect_OL -> DA -> collect_DA, one mpiexec -n 31 per stage, job-time check before each stage.
  **QUESTION FOR COWORK:** please change `dir.out` in settings/GDA_Ucloud/perturbation.json to
  `/work/PyGLDA_v2_external_data/Ensemble_input` on the workstation so the committed settings match what ran (no
  rerun needed: the generated Ensemble_input is identical, same seed and table).

- **12:20 review read 12:25:** perturbation.json fixed by Cowork (md5 1421d1eb on both machines, dir.out =
  .../Ensemble_input, seed 2026 - identical to the corrected copy the running perturbation stage uses, so its output is
  the same and the stage is NOT restarted); from the spin-up on, the chain uses the real settings/GDA_Ucloud. Code
  gaps noted by Cowork. Perturbation progress 11:56: forcing month 45 / 84 on every rank (one member per rank).

- **Stage perturbation done 11:58 (4 min, 31 ranks, one member per rank):** Ensemble_input/Ens_0..30, each with the
  parameter file (production table, seed 2026) and 84 forcing months 2000-01..2006-12 (correlated mode, pr lognormal
  1.0, tas 1 K, 300 km, 1 d); 191 GB. **Stage spin_up started 11:58:30** (31 ranks, 8 cycles of 2000 + 2000-2001 = 10
  model years per member with its own parameters and forcing, restarts -> Ensemble_Initialization); expected ~70-90
  min. Remaining job time 663 min at 11:58.

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
