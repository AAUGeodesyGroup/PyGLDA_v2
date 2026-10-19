# Cowork review - paper jobs (only Cowork writes this file; newest first)

Claude Code (UCloud) reads this file before every job and every MPI step, and at least hourly while a step runs; its
instructions override the job list in doc/ucloud_notes.md §8. Answers go into §7 of doc/ucloud_notes.md (quote the
review number). Cowork never writes doc/ucloud_notes.md while the paper jobs run; Claude Code never writes this file.

- 18 Oct ~10:40 UCloud time (Cowork, review 11 of your 10:21 entry): GBM files arrived (sync delay) and Fig. 6 is
  redrawn from GBM_units_monthly.csv - thank you. Budget copy rule and job14_summary_extra.py accepted.
  **(6c) definition (read-only, after the budget chain; global case only):** purpose = check whether the 18 % switch
  to groundwater dominance follows the OPEN-LOOP ensemble spread. Inputs: the 30 OL members, monthly means of the
  storage groups gw, soil, snow, surface (+ canopy into soil) and TWS per cell, 2002-2006, GRACE-assimilated cells
  only (the mask of storage_share_*.nc); cell classes from storage_share_OL.nc / storage_share_DA.nc (dominant
  group = largest s_k): S2G = soil in OL and groundwater in DA; S2S = soil in both; ALL = all cells with a class.
  Per cell: deviation of each member from the ensemble mean, x'_k(m, t) and tws'(m, t); spread share
  q_k = sum_{m,t} x'_k tws' / sum_{m,t} tws'^2 (sums over members and months; the q_k add to 1, can be < 0); also
  the ratio of the OL ensemble standard deviation of each group to that of TWS, averaged over months.
  Output temp/exp_logs/paper/results/GDA_Ucloud/spread_share_by_switch.csv: one row per (class, region) with region
  = the 8 HydroBASINS regions (as in group_series) plus "global": n_cells, area km2, median and p25 / p75 of q_gw,
  q_soil, q_snow, q_surface. Plus a 3-line verdict in §7: is q_gw in S2G clearly larger than in S2S (yes / no)?
- 18 Oct ~10:15 UCloud time (Cowork, review 10 of your 10:13 entry): **restart and (6a) accepted - thank you.**
  (1) GBM verdict accepted: the Fig. 6 misfit is a footprint effect. Cowork will redraw the Ganges-Brahmaputra
  panels from GBM_units_monthly.csv (12 units, 84 % of the polygon) and state the polygon value once in the text.
  **Please copy GBM_units.csv, GBM_units_monthly.csv and GBM_units_report.txt into
  temp/exp_logs/paper/results/GDA_Ucloud/** (only that folder syncs to the workstation) - they are not there yet.
  (2) Budget: same rule - when each case finishes, copy results/<case>/budget/{budget_monthly.csv,
  budget_summary.csv, budget_seasonal.csv, README.md} (not the member npz files) into
  temp/exp_logs/paper/results/<case>/budget/ and write a one-line §7 entry per case.
  (3) Member spread (I 3239 - 24241 km3 cumulative): expected, fine. Please make budget_summary.csv report the
  ensemble mean AND the 10/50/90 % member percentiles of I, dS, Q, E and the shares dS/I, Q/I, E/I, so the paper can
  quote the ensemble mean with its spread; also report the clipped water (km3) if the threshold logs give it.
  (4) (6c) spread_share_by_switch.csv: still wanted after the budget chain (read-only, cheap).
  Nothing else to change; continue.
- 17 Oct ~21:00 (Cowork, review 9): **NEW JOB 14 - water budget of the assimilation increments (read-only, serial,
  no MPI, no new model runs).** Fan's decision for the paper: Fig. 13 becomes "where does the water added by the
  assimilation go?". The open loop loses too much water (global TWS trend -1.42 vs GRACE -0.95 mm/yr), the DA adds
  water on land, and the mean flow of the large rivers rises (median +4 %). We need the closed budget of DA - OL.
  Use only existing output: Res/GDA_Ucloud/{OL,DA}/Ens_k yearly files 2002-2006 (all 30 members, member 0 excluded),
  the same for Res/RDA_Danube and Res/RDA_Amazon, and the filter's increment record if one exists.
  (1) For every member and month, as DA minus OL (km3, area-weighted with the WaterGAP cell area / land fraction you
      already use for tws; same land domain as the GRACE-assimilated cells for G, the basin mask for D1 / A1):
      dS = change of total storage tws (end of month minus end of the previous month, cumulated from 2002-01-01),
      Q = river outflow of the domain to the ocean and to inland sinks (sum of `dis` at the cells whose downstream
          cell is ocean / outside the domain / an endorheic sink, from the model's drainage direction; for D1 / A1 the
          basin outlet(s) plus any flow leaving the basin mask), as a monthly volume,
      E = total evapotranspiration `evap-total` (incl. water-use consumption, as the model writes it), monthly volume,
      I = the net water added by the analysis. If the increments are recorded (DA_output, increment diagnosis),
          sum them per month directly (net, and gross positive / gross negative separately); otherwise derive I from
          the balance I = dS + Q + E (precipitation is identical in OL and DA per member). Report both if possible
          and the closure residual I - (dS + Q + E) (should be ~0 apart from the safety clip; report the clipped water).
      Cumulate all four from 2002-01 to 2006-12 and take the ensemble mean and the 10-90 % range over members.
  (2) Domains: G = global assimilated land and the 8 HydroBASINS regions of Fig. 8 (as in group_series); D1 = Danube
      basin; A1 = Amazon basin (as assimilated, without Tocantins-Araguaia).
  (3) Also per domain the split of the cumulative net increment at 2006-12: fraction retained in storage (dS / I),
      leaving as discharge (Q / I), leaving as evapotranspiration (E / I); and the same at the seasonal scale:
      mean seasonal cycle (12 months) of the four terms (monthly, not cumulated).
  (4) Outputs to temp/exp_logs/paper/results/<case>/budget/ : budget_monthly.csv (columns domain, month, term in
      {I, dS, Q, E, residual, I_gross_pos, I_gross_neg}, ens_mean, p10, p90; units km3 per month and cumulative as
      separate columns), budget_summary.csv (domain, cumulative terms 2002-2006, fractions, ens_mean, p10, p90),
      budget_seasonal.csv, and a short README with the definitions you used (outlet / sink rule, area weights).
      Script in temp/exp_logs/paper/job14_budget.py; read one member-year at a time (peak RSS < 3 GB; 10d lesson).
  (5) In §7: the method in 5 lines, the global and regional fractions, the closure residual, run time and peak RSS,
      then "JOB 14 DONE". Do not change any model or DA code; do not clean anything.
  (6) Two small additions in the same session (read-only, after the budget):
      (a) Ganges-Brahmaputra footprint check: the GBM series of Fig. 10c (GangesBrahmaputra_monthly.csv) is the
          polygon mean of the gridded output, while the filter fits GRACE on the GlobalBasins units. List the units
          that intersect the GBM polygon (ID, area inside the polygon in %), and for the union of the units that lie
          >= 50 % inside it write GBM_units_monthly.csv with the same columns as GangesBrahmaputra_monthly.csv (OL / DA
          per storage group + tws, and the unit-mean GRACE as assimilated). Report RMS(DA - GRACE) on the units vs the
          polygon (48 -> 17 mm). This tells us whether the 17 mm misfit is a footprint effect or a real under-fit.
      (b) Copy Res/RDA_Danube/{Res_OL.h5, Res_DA.h5, GRACE_Danube.h5} and Res/RDA_Danube/evaluation/paper/summary.json
          into temp/exp_logs/paper/results/RDA_Danube/viewer/ (for the Fig. 3 viewer screenshot of the paper run;
          report the sizes).
      (c) Attribution check (for the referee point "the groundwater shift follows the ensemble"): from the OL
          ensemble, per 0.5-degree cell, the covariance-consistent share of each storage group in the SEASONAL TWS
          spread (as in your earlier spread-share diagnostics: time-mean cov_members(x_k, TWS) / var_members(TWS) of the
          monthly anomalies), then medians over (i) cells that switch from soil to groundwater dominance (OL -> DA,
          storage_share_{OL,DA}.nc) and (ii) all other cells with a seasonal signal, per HydroBASINS region. Write
          spread_share_by_switch.csv to results/GDA_Ucloud/budget/.
- 17 Oct ~16:50 UCloud time (Cowork, review 8): **PAPER JOBS DONE accepted - thank you.** All results are in the
  manuscript (Figs. 8-12 and 14). Nothing more to run in this UCloud job; do not clean any output folder (Fan
  decides later). Job 13 (NGGM) stays blocked until Fan decides.
- 17 Oct ~15:05 UCloud time (Cowork, review 7 - on your entries up to 14:59):
  (1) **Job 12 (Amazon) accepted**, including the frank filter-health reading (over-dispersed ensemble, chi2 0.10,
  inflation at the cap in sub-basins 10 / 18, large river increments). No rerun - the paper reports it as is and
  discusses it in the limitations.
  (2) **Outside-mask wording - answer to your QUESTION:** one general sentence for both basins, numbers in the text:
  "the state is updated only inside the basin; outside it, only the storages of cells that receive the routed
  outflow change (rivers, lakes and wetlands, and through their exchange groundwater and soil by a few millimetres),
  e.g. Lake Titicaca and the Pantanal downstream of the Amazon boundary." No further check needed.
  (3) **10e2 verdict accepted** (good analysis). Decision: Fig. 14a uses the 7-day-mean annual maximum; the one-day
  pulse on day 2 of a window becomes a limitation in Sect. 6 (river increments are released within a day). 10e3 is
  useful - let it finish; report the 7-day and 30-day numbers and save the npz as planned.
  (4) **10d3 accepted** (MAIN_BAS 4060025450, 1.58 million km2 - correct); the GBM groundwater cycle is a strong
  result for Fig. 10b.
  (5) **Final copy before PAPER JOBS DONE:** to `temp/exp_logs/paper/results/<case>/` also the nc / shp files the
  figures need (all < 200 MB): GDA_Ucloud/paper/{storage_share_OL.nc, storage_share_DA.nc, discharge_annmax.nc,
  Harmonic_*.nc, GangesBrahmaputra.*, discharge_check_series.npz, 10e3 outputs}; then write PAPER JOBS DONE with the
  file list. Do not clean any output folders. Thank you - very solid work today.
- 17 Oct ~13:30 UCloud time (Cowork, review 6 - on your entries up to 13:27): memory action (10d killed, memguard2)
  accepted - protecting the Amazon DA_run comes first. For the 10d restart after the DA: read one member and one
  year at a time (xr.open_dataset per yearly file, only the variables of the four groups + tws, compute the basin
  series, close the file) instead of open_mfdataset over 5 years - peak RSS should then stay near 1-2 GB, and 10d can
  run next to the DA of the next job. Note: review 5 (below) was published at ~12:56 but did not reach you (the
  file was briefly reverted on the desktop side); it is here now - please answer its points (2) and (4) in §7.
- 17 Oct ~13:00 UCloud time (Cowork, review 5 - on your entries up to 12:54):
  (1) **Danube D1 accepted** (RMS 31.5 -> 5.8 mm, corr 0.997, trends followed in every sub-basin, s_k table, 2006 as
  the Fig. 12 year). The low chi2 / over-fit signature is a known consequence of the diagonal R - no action.
  (2) **Outside-mask check - answer to your QUESTION:** good diagnosis. The paper will say it slightly differently,
  because the downstream change is a feature, not a leak: "the state is updated only inside the basin; outside it,
  soil, snow, groundwater and wetlands are identical to the open loop, and only the river storage of a few cells
  downstream of the basin outlet and boundary changes (by at most 1.5 mm), because the corrected outflow is routed
  onward." No further check needed. Please do the same check for the Amazon (Job 12) and report the same numbers.
  (3) 10e2 plan (a)-(c) + day-of-month profile accepted; waiting for the verdict.
  (4) Your 12:25 note says my 12:24 save was built on an old copy - correct, and it is why Cowork now writes only
  THIS file. Review 4 (the pointer) was itself overwritten in ucloud_notes.md, so you may not have seen it: from now
  on read `doc/cowork_review_paper.md` (this file) instead of the old block; acknowledge in §7.
- 17 Oct ~12:25 UCloud time (Cowork, review 4) - **answer to your QUESTION FOR COWORK (11:55):** yes, the cause is on
  both sides: Cowork also saves this file as a whole (stage -> edit -> write back), and Syncthing delivers whole files,
  so whichever save arrives last wins and the other side's recent lines vanish. Your 11:55 save also removed review 3.
  **Fix from now on: two files, one writer each.** Cowork writes ONLY `doc/cowork_review_paper.md` (this block and all
  later reviews); you write ONLY `doc/ucloud_notes.md` (§7). Cowork will not save ucloud_notes.md again during the
  paper jobs. Re-read `doc/cowork_review_paper.md` where you used to re-read this block (before every job / MPI step,
  hourly). Review 3 (the 10e discharge plausibility check) is repeated there - please do it.
- 17 Oct ~11:50 UCloud time (Cowork, review 3 - repeated here because it was lost from ucloud_notes.md at 11:55):
  file intact - thanks; the insert-only rule works. Memory handling (10d paused, cgroup numbers, memguard.sh) accepted.
  **10e needs a plausibility check before it goes into Fig. 14a:** the largest increases (Churchill / La Grande
  +130-230 %, Tocantins +146 %, Sao Francisco +103-132 %, Parnaiba +128 %) are all rivers with large reservoirs /
  regulated outflow, and +100-230 % on the annual peak is more than a storage correction of tens of mm should give.
  Suspects: (i) river / wetland increments released as a one-day pulse on the first day of a window; (ii) water added
  upstream of a reservoir whose storage is NOT in the DA state. Please run (read-only, after the Danube DA has finished
  if memory is tight):
  (a) for the 3323 big-river cells: the share of DA annual-maximum days that fall on the FIRST day of a GRACE window
      (OL for comparison; expected ~1/30 if unrelated), and the median rel_change recomputed with the window-start
      days excluded (annual maxima of the remaining days);
  (b) for the top 3 increase cells (53.75N 64.25W, 53.75N 78.75W, 9.75S 48.25W): ensemble-mean daily dis 2002-2006
      OL vs DA (csv into paper/), and the mean increment per window of riverstor, local / global wetland storage and
      groundwater in the 3x3-cell box upstream; say whether reservoirstor (not in the state) differs between OL and DA;
  (c) the same rel_change statistics for the 7-day-mean annual maximum (smooths one-day pulses).
  Report numbers + a one-line verdict ("peaks driven by window-start pulses: yes / no"). Fig. 14a waits for this; Jobs
  11 / 12 continue unchanged. Upper Volga decreases (-17..-30 %) look consistent with the EU trend correction - fine.
- 17 Oct ~11:50 UCloud time (Cowork, review 3): file intact - thanks; the insert-only rule works. Memory handling
  (10d paused, cgroup numbers) accepted. **10e needs a plausibility check before it goes into Fig. 14a:** the largest
  increases (Churchill / La Grande +130-230 %, Tocantins +146 %, Sao Francisco +103-132 %, Parnaiba +128 %) are all
  rivers with large reservoirs / regulated outflow, and +100-230 % on the annual peak is more than a storage
  correction of tens of mm should give. Suspects: (i) river / wetland increments released as a one-day pulse on the
  first day of a window; (ii) water added upstream of a reservoir whose storage is NOT in the DA state. Please run
  (read-only, after or between the MPI steps - not before the Danube DA has finished if memory is tight):
  (a) for the 3323 big-river cells: the share of DA annual-maximum days that fall on the FIRST day of a GRACE window
      (OL for comparison; expected ~1/30 if unrelated), and the median rel_change recomputed with the window-start
      days excluded (annual maxima of the remaining days);
  (b) for the top 3 increase cells (53.75N 64.25W, 53.75N 78.75W, 9.75S 48.25W): ensemble-mean daily dis 2002-2006
      OL vs DA (csv into paper/), and the mean increment per window of riverstor, local / global wetland storage and
      groundwater in the 3x3 cells upstream box; say whether reservoirstor (not in the state) changes between OL and DA;
  (c) the same rel_change statistics for the 7-day-mean annual maximum (smooths one-day pulses).
  Report numbers + a one-line verdict ("peaks driven by window-start pulses: yes / no"). Fig. 14a waits for this; Jobs
  11 / 12 continue unchanged. Upper Volga decreases (-17..-30 %) look consistent with the EU trend correction - fine.
- 17 Oct ~11:20 UCloud time (Cowork, review 2): **YOUR LAST SAVE OF THIS FILE DELETED TEXT.** The version written
  after the 11:10 incident no longer contained (a) your own §7 entries from "PAPER JOBS (section 8 ..." to the
  Job 11 command list (10:52), and (b) Cowork review 1 below. Cowork restored both from its copy (unchanged text).
  Probably the file was saved from a copy read before Cowork's / your own later edits. **From now on:** edit this file
  only with small in-place insertions (Edit tool / append), never rewrite the whole file, and re-read it immediately
  before every edit; after each edit check that the line count did not shrink. Please confirm in §7 that review 1 and
  your 10:52 entries are back, then answer review 1 (memory point (1): report `free -g` during the Danube DA_run with
  10c / 10e / 10d running in parallel - now THREE single processes next to 31 ranks; if available memory < 12 GB,
  stop 10e / 10d until DA_run has finished).
  11:10 incident noted - the restart plan is fine; keep the PID-file rule. Danube DA ETA ~12:50 accepted.
- 17 Oct ~10:55 UCloud time (Cowork, review 1 of your §7 entries up to 10:52): **all steps so far match the spec - good.**
  Job 0 checks, 10a, 10b and the Danube prepare step are accepted; the 10a / 10b numbers go into the paper as they are.
  Points:
  (1) **MEMORY - running Job 10 alongside the MPI chain:** the global DA used 3.0-3.2 GB per rank (65-69 GB for 31
  ranks); the regional DA_run should be similar (WaterGAP runs globally on every rank). 10c / 10d / 10e read 30
  members x 5 years of global yearly files in ONE process. Watch `free -g` while RDA_Danube DA_run and 10c overlap: if
  available memory drops below 12 GB, pause the Job 10 script (kill -STOP <pid>, kill -CONT later) or let it finish
  between the MPI steps - an OOM kill of a DA rank would cost the whole Danube run. Note the peak RSS of each Job 10
  script in §7.
  (2) 10b caveat accepted: Fig. 9 stays three maps side by side (GRACE | OL | DA), no per-cell difference map. Keep
  the hot-spot table; the paper text will quote NW India, North China Plain, Caspian, Central Valley, La Plata.
  (3) 10f: the dense / sparse component sizes of Fig. 6 come from Cowork's earlier measurement - no action; the
  totals you measured stand.
  (4) Outside-mask check (Job 11 step 4): use `region_mask` of the DA yearly file AND, if the DA output is cropped to
  the basin box, say so - then the check is "every cell of the box outside the Danube mask equals the OL" (state the
  box and the number of cells checked).
  (5) Nothing else to change; continue. Next Cowork review in ~30 min.
- 17 Oct (Cowork): block opened, no instructions yet.

