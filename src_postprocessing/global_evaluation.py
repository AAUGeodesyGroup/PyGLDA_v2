"""
Evaluation report of a GLOBAL PyGLDA v2 assimilation run (GRACE TWS of the global units -> WaterGAP, EnKF).

The regional report (src_postprocessing.da_evaluation) is built around one basin and a few sub-basins (a basin
series, a panel per sub-basin); with 772 units the run is judged per unit and by groups of units instead. The window
statistics of a unit are the same as in the regional report (da_evaluation.window_statistics). Each call writes

    Res/<case>/evaluation_global/<tag>/
        unit_metrics.csv     one row per unit (ID 1..n): fit of OL and DA to GRACE (RMS, correlation, RMS
                             reduction 1 - RMS_DA / RMS_OL), filter statistics (median chi^2, gain, forecast spread,
                             GRACE sigma, innovation), trends of GRACE / OL / DA, the mean absolute increment per
                             storage, and the unit attributes of the shapefile (region, basin class, SNR, area,
                             lake / wetland / glacier fractions, endorheic, centroid)
        group_metrics.csv    the same fit numbers for area-weighted groups of units: global, HydroBASINS region,
                             basin class, SNR class, latitude band
        group_series.csv     the area-weighted window series (GRACE, OL, DA) of every group
        summary.json / summary.txt   overview, filter health, clipping, warnings
        DA_setting.json, logs/       copies of the configuration and the filter / clipping / inflation logs
The figures are drawn from these files by src_postprocessing.global_visualization.

Usage:  GDA.da_evaluation(tag='pilot1')   (src_FlowControl/Global_DA.py), or
        from src_postprocessing.global_evaluation import evaluate_global_run
It reads results only and does not touch the assimilation code.
"""
import csv, json, datetime as dt
from pathlib import Path
import numpy as np
import h5py

from src_postprocessing.da_evaluation import (COMP, _rms, _corr, _trend, _decyear, _load_obs, _read_threshold_csvs,
                                              _read_adaptive_log, _clipping_summary, _filter_log_summary,
                                              _method_string, archive_logs, observation_windows, window_statistics)

THRESHOLDS = dict(chi2_low=0.5, chi2_high=2.0,         # median chi^2 of a unit outside this range
                  worse_ratio=1.0,                     # RMS(DA - GRACE) > ratio x RMS(OL - GRACE)
                  clip_frac=0.05,                      # clipped fraction of the checked cells of one storage
                  trend_cell_mm_yr=20.0,               # |trend| of a grid cell created by the DA (Harmonic maps)
                  n_list=15)                           # entries listed per warning
SNR_CLASSES = [('SNR < 3', 0.0, 3.0), ('SNR 3-6', 3.0, 6.0), ('SNR >= 6', 6.0, np.inf)]
LAT_BANDS = [('south of 30S', -90.0, -30.0), ('30S-0', -30.0, 0.0), ('0-30N', 0.0, 30.0), ('30N-60N', 30.0, 60.0),
             ('north of 60N', 60.0, 90.0)]
UNIT_ATTRS = ('NAME', 'REGION', 'BAS_CLASS', 'SNR', 'SUB_AREA', 'LAKE_FRAC', 'WET_FRAC', 'GLAC_EFF', 'ENDO')


# ------------------------------------------------------------------------------------------------ loading
def _load_daily(path, units, variables):
    """dict[unit][var] -> (nmem, nday), members 1..N (member '0' = OL thread of the DA run, left out)"""
    out = {}
    with h5py.File(path, 'r') as f:
        for u in units:
            out[u] = {}
            for v in variables:
                key = '%s/%s' % (u, v)
                if key in f:
                    mem = sorted([m for m in f[key].keys() if m != '0'], key=int)
                    out[u][v] = np.array([f[key][m][:] for m in mem])
    return out


def _unit_table(shp_path, n_unit):
    """attributes of the units 1..n from the shapefile (centroid lat/lon added); None when it is not available"""
    try:
        import geopandas as gpd
        g = gpd.read_file(shp_path).sort_values('ID')
    except Exception as err:
        print('global evaluation: shapefile not read (%s) - no unit attributes and no groups' % err)
        return None
    if len(g) != n_unit or list(g['ID']) != list(range(1, n_unit + 1)):
        raise ValueError('global evaluation: %s has %d units with IDs %s.., the observations %d' %
                         (shp_path, len(g), list(g['ID'])[:3], n_unit))
    c = g.geometry.representative_point()
    tab = {a: g[a].values for a in UNIT_ATTRS if a in g}
    tab['lat_c'], tab['lon_c'] = c.y.values, c.x.values
    return tab


def _groups(tab, n_unit):
    """{group name: unit indices (0-based)}: global, regions, basin classes, SNR classes, latitude bands"""
    idx = np.arange(n_unit)
    grp = {'global': idx}
    if tab is None:
        return grp
    if 'REGION' in tab:                     # a unit spanning two HydroBASINS regions ('af,eu') -> its first region
        reg = np.array([str(r).split(',')[0] for r in tab['REGION']])
        for v in sorted(set(reg)):
            grp['region: %s' % v] = idx[reg == v]
    if 'BAS_CLASS' in tab:
        for v in sorted(set(tab['BAS_CLASS'])):
            grp['class: %s' % v] = idx[tab['BAS_CLASS'] == v]
    if 'SNR' in tab:
        for name, lo, hi in SNR_CLASSES:
            grp['snr: %s' % name] = idx[(tab['SNR'] >= lo) & (tab['SNR'] < hi)]
    for name, lo, hi in LAT_BANDS:
        grp['lat: %s' % name] = idx[(tab['lat_c'] >= lo) & (tab['lat_c'] < hi)]
    return {k: v for k, v in grp.items() if v.size}


# ------------------------------------------------------------------------------------------------ main entry
def evaluate_global_run(res_dir, obs_file, out_root, tag=None, basin='GlobalBasins_v1.0', shp_path=None,
                        da_output_dir=None, setting_file=None, start=dt.date(2002, 1, 1), thresholds=None,
                        extra_files=None, verbose=True):
    """
    res_dir       : Res/<case> (Res_OL.h5, Res_DA.h5, GRACE_<basin>.h5, Harmonic_*.nc)
    obs_file      : <basin>_obs_GRACE.hdf5 (windows, covariance, unit areas)
    out_root      : Res/<case>/evaluation_global; the report goes to out_root/<tag>
    tag           : name of the run; None: date and time
    shp_path      : unit shapefile (IDs 1..n): attributes and groups; None: no attributes, global group only
    da_output_dir : DA_output/<case> (filter_summary.json, adaptive_inflation_log.csv, Ens_k/threshold_monthly.csv)
    setting_file  : DA_setting.json of the run (copied into the report)
    start         : first day of the daily series in Res_*.h5
    returns       : the summary dict (as in summary.json)
    """
    TH = dict(THRESHOLDS, **(thresholds or {}))
    tag = tag or dt.datetime.now().strftime('%Y%m%d_%H%M')
    res_dir = Path(res_dir)
    out = Path(out_root) / tag
    out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- data
    durs, sig_sub, _ = _load_obs(obs_file)
    n_unit = sig_sub.shape[1]
    units = ['sub_basin_%d' % k for k in range(1, n_unit + 1)]
    storages = [c[0] for c in COMP]
    DA = _load_daily(res_dir / 'Res_DA.h5', units, storages + ['tws'])
    OL = _load_daily(res_dir / 'Res_OL.h5', units, storages + ['tws'])
    with h5py.File(res_dir / ('GRACE_%s.h5' % basin), 'r') as g:
        GR = np.array([g['original/basin_%d' % k][:] for k in range(1, n_unit + 1)])       # (unit, epoch)
    nmem, nday = DA[units[0]]['tws'].shape
    days = [start + dt.timedelta(days=i) for i in range(nday)]
    win = observation_windows(durs, start, nday)
    kk = np.array([w[0] for w in win])
    wdate = [durs[k][0] for k in kk]
    tdec = np.array([_decyear(durs[k][0] + (durs[k][1] - durs[k][0]) / 2) for k in kk])
    tab = _unit_table(shp_path, n_unit) if shp_path is not None else None
    with h5py.File(obs_file, 'r') as o:
        area = o['sub_basin_area'][:] if 'sub_basin_area' in o else np.ones(n_unit)
    if tab is not None and 'SUB_AREA' in tab:
        area = np.asarray(tab['SUB_AREA'], dtype=float)

    # ---------------------------------------------------------------- per unit
    S = [window_statistics(OL[u]['tws'], DA[u]['tws'], GR[j, kk], sig_sub[kk, j], win) for j, u in enumerate(units)]
    rows = []
    for j, u in enumerate(units):
        s = S[j]
        rms_ol, rms_da = _rms(s['ol'] - s['gr']), _rms(s['da'] - s['gr'])
        r = dict(ID=j + 1, rms_OL=rms_ol, rms_DA=rms_da,
                 rms_reduction=1.0 - rms_da / rms_ol if rms_ol > 0 else np.nan,
                 corr_OL=_corr(s['ol'], s['gr']), corr_DA=_corr(s['da'], s['gr']),
                 chi2_median=float(np.median(s['chi2'])),
                 gain=float(np.sum(s['inc'] * s['d']) / np.sum(s['d'] ** 2)) if np.sum(s['d'] ** 2) > 0 else np.nan,
                 spread_fc_median=float(np.median(s['sd_fc'])), sigma_GRACE_median=float(np.median(s['sig'])),
                 innovation_rms=_rms(s['d']), innovation_mean=float(np.mean(s['d'])),
                 trend_GRACE=_trend(tdec, s['gr']), trend_OL=_trend(tdec, s['ol']), trend_DA=_trend(tdec, s['da']))
        for v in storages:                                         # mean |increment| of the storage per window
            if v in DA[u] and v in OL[u]:
                inc = [np.mean((DA[u][v][:, i0] - DA[u][v][:, i0 - 1]) - (OL[u][v][:, i0] - OL[u][v][:, i0 - 1]))
                       for _, i0, _ in win]
                r['abs_increment_%s' % v] = float(np.mean(np.abs(inc)))
        if tab is not None:
            for a in UNIT_ATTRS + ('lat_c', 'lon_c'):
                if a in tab:
                    val = tab[a][j]
                    r[a] = val.item() if hasattr(val, 'item') else val
        rows.append(r)

    # ---------------------------------------------------------------- groups (area-weighted series)
    groups = _groups(tab, n_unit)
    G, grows = {}, []
    for name, ix in groups.items():
        w = area[ix] / area[ix].sum()
        gser = {k: np.sum(w[:, None] * np.array([S[j][k] for j in ix]), axis=0) for k in ('gr', 'ol', 'da')}
        G[name] = gser
        rms_ol, rms_da = _rms(gser['ol'] - gser['gr']), _rms(gser['da'] - gser['gr'])
        grows.append(dict(group=name, n_units=int(ix.size), area_km2=float(area[ix].sum()),
                          rms_OL=rms_ol, rms_DA=rms_da, rms_reduction=1.0 - rms_da / rms_ol if rms_ol > 0 else np.nan,
                          corr_OL=_corr(gser['ol'], gser['gr']), corr_DA=_corr(gser['da'], gser['gr']),
                          trend_GRACE=_trend(tdec, gser['gr']), trend_OL=_trend(tdec, gser['ol']),
                          trend_DA=_trend(tdec, gser['da']),
                          median_unit_rms_reduction=float(np.nanmedian([rows[j]['rms_reduction'] for j in ix])),
                          median_unit_chi2=float(np.median([rows[j]['chi2_median'] for j in ix]))))

    # ---------------------------------------------------------------- filter logs
    fsum, thr, alog = {}, [], []
    if da_output_dir is not None:
        p = Path(da_output_dir) / 'filter_summary.json'
        if p.exists():
            fsum = json.load(open(p))
        thr = _read_threshold_csvs(da_output_dir)
        alog = _read_adaptive_log(da_output_dir)

    chi2_all = np.concatenate([s['chi2'] for s in S])
    unit_chi2 = np.array([r['chi2_median'] for r in rows])
    red = np.array([r['rms_reduction'] for r in rows])
    summ = dict(tag=tag, created=dt.datetime.now().strftime('%Y-%m-%d %H:%M'), n_units=n_unit, n_members=int(nmem),
                n_windows=len(win), period='%s to %s' % (days[0], days[-1]), method=_method_string(setting_file))
    summ['fit_global'] = {k: grows[0][k] for k in ('rms_OL', 'rms_DA', 'rms_reduction', 'corr_OL', 'corr_DA',
                                                   'trend_GRACE', 'trend_OL', 'trend_DA')}
    summ['units'] = dict(rms_reduction_median=float(np.nanmedian(red)),
                         rms_reduction_p10_p90=[float(np.nanpercentile(red, 10)), float(np.nanpercentile(red, 90))],
                         frac_improved=float(np.mean(red > 0)),
                         corr_DA_median=float(np.nanmedian([r['corr_DA'] for r in rows])),
                         corr_OL_median=float(np.nanmedian([r['corr_OL'] for r in rows])))
    summ['filter'] = dict(chi2_median_all=float(np.median(chi2_all)), chi2_mean_all=float(np.mean(chi2_all)),
                          frac_unit_windows_chi2_above_high=float(np.mean(chi2_all > TH['chi2_high'])),
                          frac_unit_windows_chi2_below_low=float(np.mean(chi2_all < TH['chi2_low'])),
                          gain_median=float(np.nanmedian([r['gain'] for r in rows])),
                          spread_fc_median=float(np.median([r['spread_fc_median'] for r in rows])),
                          sigma_GRACE_median=float(np.median([r['sigma_GRACE_median'] for r in rows])))
    if alog:
        s_add = np.array([r['sigma_added_mm'] for r in alog])
        summ['adaptive_inflation'] = dict(sigma_added_median=float(np.median(s_add)), sigma_added_max=float(s_add.max()),
                                          frac_at_min=float(np.mean(s_add <= s_add.min() + 1e-6)),
                                          frac_near_upper_cap=float(np.mean(
                                              s_add >= 0.99 * 3.0 * np.array([r['obs_error_mm'] for r in alog]))))
    summ['clipping'] = _clipping_summary(thr)
    summ['filter_log'] = _filter_log_summary(fsum)
    summ['groups'] = {g['group']: {k: g[k] for k in ('n_units', 'rms_reduction', 'corr_DA', 'median_unit_chi2')}
                      for g in grows}
    summ['warnings'] = _warnings(rows, unit_chi2, summ, res_dir, TH)
    summ['archived'] = archive_logs(out, da_output_dir, setting_file, None, extra_files)

    # ---------------------------------------------------------------- output
    _write_csv(out / 'unit_metrics.csv', rows)
    _write_csv(out / 'group_metrics.csv', grows)
    with open(out / 'group_series.csv', 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['group', 'window_start', 'GRACE', 'OL', 'DA'])
        for name, gser in G.items():
            for i, d in enumerate(wdate):
                wr.writerow([name, d.isoformat(), '%.3f' % gser['gr'][i], '%.3f' % gser['ol'][i], '%.3f' % gser['da'][i]])
    json.dump(_round(summ), open(out / 'summary.json', 'w'), indent=1)
    text = _report_text(_round(summ), grows)
    open(out / 'summary.txt', 'w').write(text)
    if verbose:
        print(text)
        print('global evaluation report written to %s' % out)
    return summ


# ------------------------------------------------------------------------------------------------ warnings
def _warnings(rows, unit_chi2, summ, res_dir, TH):
    W, n = [], TH['n_list']
    bad = [(r['ID'], c) for r, c in zip(rows, unit_chi2) if not TH['chi2_low'] <= c <= TH['chi2_high']]
    if bad:
        bad.sort(key=lambda x: -abs(np.log(max(x[1], 1e-9))))
        W.append(dict(kind='chi2_units', n=len(bad), text='%d units with median chi^2 outside [%g, %g]' %
                      (len(bad), TH['chi2_low'], TH['chi2_high']), items=[dict(ID=i, chi2=c) for i, c in bad[:n]]))
    worse = [(r['ID'], r['rms_DA'] / r['rms_OL']) for r in rows
             if r['rms_OL'] > 0 and r['rms_DA'] > TH['worse_ratio'] * r['rms_OL']]
    if worse:
        worse.sort(key=lambda x: -x[1])
        W.append(dict(kind='DA_worse_than_OL', n=len(worse),
                      text='%d units where DA fits GRACE worse than the open loop' % len(worse),
                      items=[dict(ID=i, rms_ratio_DA_OL=q) for i, q in worse[:n]]))
    for v, c in summ.get('clipping', {}).items():
        f = c['frac_lower'] + c['frac_upper']
        if f > TH['clip_frac']:
            W.append(dict(kind='clipping', text='%s: %.1f %% of the checked cells clipped' % (v, 100 * f)))
    try:                                                            # grid cells whose trend the DA created
        import netCDF4
        H = {}
        for st in ('OL', 'DA'):
            with netCDF4.Dataset(Path(res_dir) / ('Harmonic_%s.nc' % st)) as nc:
                key = 'tws_trend_ensmean' if 'tws_trend_ensmean' in nc.variables else 'tws_trend'
                H[st] = np.ma.filled(nc.variables[key][:], np.nan).astype(float)
        big = (np.abs(np.nan_to_num(H['DA'])) > TH['trend_cell_mm_yr']) & \
              (np.abs(np.nan_to_num(H['OL'])) <= TH['trend_cell_mm_yr'])
        if big.any():
            W.append(dict(kind='trend_cells_created_by_DA', n=int(big.sum()),
                          text='%d grid cells with |trend| > %g mm/yr in DA but not in OL' %
                               (int(big.sum()), TH['trend_cell_mm_yr'])))
    except (OSError, KeyError, FileNotFoundError):
        pass
    return W


# ------------------------------------------------------------------------------------------------ output helpers
def _round(x, nd=3):
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v, nd) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), nd)
    if isinstance(x, np.integer):
        return int(x)
    return x


def _write_csv(path, rows):
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: ('%.4g' % v if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})


def _report_text(s, grows):
    L = ['Global evaluation  %s   (%s)' % (s['tag'], s['created']),
         '%d units, %d members, %d windows, %s' % (s['n_units'], s['n_members'], s['n_windows'], s['period']),
         'method: %s' % s['method'], '',
         'Fit to GRACE, global land (area-weighted window means, GRACE anomalies on the OL mean)',
         '  RMS  OL %(rms_OL)s mm   DA %(rms_DA)s mm   reduction %(rms_reduction)s' % s['fit_global'],
         '  corr OL %(corr_OL)s   DA %(corr_DA)s' % s['fit_global'],
         '  trend GRACE %(trend_GRACE)s  OL %(trend_OL)s  DA %(trend_DA)s mm/yr' % s['fit_global'], '',
         'Units: median RMS reduction %s (p10 %s, p90 %s), %.0f %% of the units improved; median corr OL %s -> DA %s'
         % (s['units']['rms_reduction_median'], s['units']['rms_reduction_p10_p90'][0],
            s['units']['rms_reduction_p10_p90'][1], 100 * s['units']['frac_improved'], s['units']['corr_OL_median'],
            s['units']['corr_DA_median']), '',
         'Filter: chi^2 median %(chi2_median_all)s (mean %(chi2_mean_all)s), unit-windows above the high limit '
         '%(frac_unit_windows_chi2_above_high)s, below the low limit %(frac_unit_windows_chi2_below_low)s; median gain '
         '%(gain_median)s, forecast spread %(spread_fc_median)s mm, GRACE sigma %(sigma_GRACE_median)s mm'
         % s['filter'], '', 'Groups (area-weighted series):',
         '  %-34s %6s %9s %9s %9s %7s %7s %9s' % ('group', 'units', 'RMS_OL', 'RMS_DA', 'reduct.', 'corrOL', 'corrDA',
                                                  'unit chi2')]
    for g in grows:
        L.append('  %-34s %6d %9.1f %9.1f %9.2f %7.2f %7.2f %9.2f' % (
            g['group'][:34], g['n_units'], g['rms_OL'], g['rms_DA'], g['rms_reduction'], g['corr_OL'], g['corr_DA'],
            g['median_unit_chi2']))
    if s.get('clipping'):
        L += ['', 'Clipping (fraction of checked cells, water per member):']
        for v, c in s['clipping'].items():
            L.append('  %-12s lower %s upper %s, added %s mm, removed %s mm' % (
                v, c['frac_lower'], c['frac_upper'], c['water_added_mm_per_member'], c['water_removed_mm_per_member']))
    L += ['', 'Warnings:'] + (['  none'] if not s['warnings'] else
                              ['  - %s%s' % (w['text'], (': ' + ', '.join(
                                  'ID %s (%s)' % (it['ID'], ', '.join('%s %s' % (k, v) for k, v in it.items() if k != 'ID'))
                                  for it in w['items'])) if w.get('items') else '') for w in s['warnings']])
    return '\n'.join(L) + '\n'
