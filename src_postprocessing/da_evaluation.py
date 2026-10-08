"""
Standard evaluation report of one PyGLDA v2 assimilation run (GRACE TWS -> WaterGAP, EnKF).

Run it after post_processing (it needs the collected Res_OL.h5 / Res_DA.h5 / GRACE_<basin>.h5 and, if present,
the Harmonic_*.nc maps). Each call writes one report folder

    Res/<case>/evaluation/<tag>/
        summary.txt            readable report: fit to GRACE, filter statistics, comparison with the reference, warnings
        summary.json           the same numbers (read back when this run is used as a reference later)
        monthly_series.csv     window means of GRACE / OL / DA (basin and sub-basins), kept so that later runs can be
                               compared with this one after its Res files have been overwritten
        DA_setting.json, filter_summary.json   copies of the configuration and filter log of the run
        fig1_basin_timeseries.png   GRACE (+-sigma), OL, DA (+ reference DA) and the residuals
        fig2_correction_climatology.png   mean seasonal correction DA - OL per storage, basin and sub-basins
        fig3_groundwater_drift.png  monthly DA - OL groundwater, basin and sub-basins
        fig4_sub_basins.png         GRACE / OL / DA per sub-basin
        fig5_filter_health.png      chi^2 per window, forecast spread vs innovation, added spread, clipping
and appends one line to Res/<case>/evaluation/runs_index.csv (one row per evaluated run).

Numbers in the report
  fit       RMS and correlation of the window-mean TWS (OL, DA) against GRACE, overall, by season and per
            sub-basin; the GRACE mean is aligned to the OL mean (only anomalies are compared). Key months.
  trends    linear trends of GRACE / OL / DA [mm/yr], basin and sub-basins.
  filter    innovation d = GRACE - forecast mean, increment = jump of DA - OL on the first day of the window,
            gain = sum(inc d) / sum(d^2), forecast spread, chi^2 = d^2 / (sd_fc^2 + sigma_GRACE^2)
            (expected ~1 for a consistent filter), mean ensemble spread of the analysis by calendar month.
            The forecast is reconstructed as window mean - increment (exact for the uniform increment, close
            enough otherwise).
  warnings  cells with extreme trends (Harmonic maps), months with large GRACE errors, sub-basins fitted
            more tightly than the GRACE error, chi^2 far from 1, clipping rates (Ens_k/threshold_monthly.csv),
            large single-cell increments / capped weights (filter_summary.json), the adaptive inflation at
            its limits (adaptive_inflation_log.csv).

Usage:  RDA.da_evaluation(tag='run10', reference='run9')   (src_FlowControl/Regional_DA.py), or
        from src_postprocessing.da_evaluation import evaluate_run
        evaluate_run(res_dir, obs_file, out_root, tag, da_output_dir=..., setting_file=..., reference='run9')
It reads results only and does not touch the assimilation code.
"""
import csv, json, shutil, datetime as dt
from pathlib import Path
import numpy as np, h5py
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12, 'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
                     'axes.linewidth': 1.8, 'xtick.major.width': 1.6, 'ytick.major.width': 1.6,
                     'xtick.major.size': 5, 'ytick.major.size': 5, 'legend.frameon': False, 'legend.fontsize': 11})

COMP = [('groundwstor', 'Groundwater', '#2a78d6', 'o'), ('soilmoist', 'Soil moisture', '#eb6834', 's'),
        ('swe', 'Snow', '#1baf7a', '^'), ('riverstor', 'River', '#eda100', 'D')]
C_GR, C_OL, C_DA, C_REF = 'k', '#8c8c8c', '#2a78d6', '#eb6834'
SEASONS = {'DJF': (12, 1, 2), 'MAM': (3, 4, 5), 'JJA': (6, 7, 8), 'SON': (9, 10, 11)}
KEY_MONTHS = ['2003-08', '2006-03', '2006-04', '2006-05', '2007-09', '2007-10', '2010-04', '2011-04', '2012-09',
              '2016-01']
# warning thresholds (all can be changed through evaluate_run(thresholds={...}))
THRESHOLDS = dict(trend_cell_mm_yr=20.0,          # |trend| of a grid cell (OL or DA)
                  grace_sigma_factor=3.0,         # GRACE sigma > factor x its median
                  overfit_ratio=0.7,              # sub-basin RMS(DA - GRACE) < ratio x the RMS expected for a
                                                  # consistent analysis, sqrt(mean(sigma^4 / (sd_fc^2 + sigma^2)))
                  chi2_low=0.5, chi2_high=2.0,    # median chi^2 outside this range
                  clip_frac=0.05,                 # fraction of clipped cells of one storage in one month
                  single_increment_mm=1000.0,     # largest single-cell increment
                  n_list=8)                       # rows listed per warning


# ----------------------------------------------------------------------------------------------------- helpers
def _rms(x):
    x = np.asarray(x, float)
    return float(np.sqrt(np.nanmean(x ** 2))) if x.size else float('nan')


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float('nan')


def _trend(t, x):
    return float(np.polyfit(t, x, 1)[0]) if len(x) > 2 else float('nan')


def _r(x, n=2):
    if isinstance(x, dict):
        return {k: _r(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, n) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), n)
    if isinstance(x, np.integer):
        return int(x)
    return x


def _decyear(d):
    y0 = dt.date(d.year, 1, 1)
    return d.year + (d - y0).days / ((dt.date(d.year + 1, 1, 1) - y0).days)


# ------------------------------------------------------------------------------------------------- loading
def _load_res(path, units, variables):
    """dict[unit][var] -> (nmem, ndays); member '0' (the OL thread of the DA run) is left out"""
    out = {}
    with h5py.File(path, 'r') as f:
        for u in units:
            out[u] = {}
            for v in variables:
                key = '%s/%s' % (u, v)
                if key not in f:
                    continue
                mem = sorted([m for m in f[key].keys() if m != '0'], key=int)
                out[u][v] = np.array([f[key][m][:] for m in mem])
    return out


def _load_obs(obs_file):
    with h5py.File(obs_file, 'r') as o:
        durs = [[dt.datetime.strptime(x, '%Y-%m-%d').date() for x in d.split('_')]
                for d in o['duration'][:].astype(str)]
        cov = o['cov'][:]
        area = o['sub_basin_area'][:] if 'sub_basin_area' in o else np.ones(cov.shape[1])
    w = area / area.sum()
    sig_sub = np.sqrt(np.einsum('kjj->kj', cov))
    sig_basin = np.sqrt(np.einsum('i,kij,j->k', w, cov, w))
    return durs, sig_sub, sig_basin


def _read_threshold_csvs(da_output_dir):
    rows = []
    for p in sorted(Path(da_output_dir).glob('Ens_*/threshold_monthly.csv')):
        if p.parent.name == 'Ens_0':
            continue
        with open(p) as fh:
            for r in csv.DictReader(fh):
                r['member'] = p.parent.name
                rows.append(r)
    return rows


def _read_adaptive_log(da_output_dir):
    p = Path(da_output_dir) / 'adaptive_inflation_log.csv'
    if not p.exists():
        return []
    with open(p) as fh:
        return [{k: (v if k == 'date' else float(v)) for k, v in r.items()} for r in csv.DictReader(fh)]


# ------------------------------------------------------------------------------------------------ main entry
def evaluate_run(res_dir, obs_file, out_root, tag, basin='basin', da_output_dir=None, setting_file=None,
                 reference=None, start=dt.date(2002, 1, 1), key_months=None, thresholds=None, figures_dir=None,
                 extra_files=None, verbose=True):
    """
    res_dir       : Res/<case>  (Res_OL.h5, Res_DA.h5, GRACE_<basin>.h5, Harmonic_{OL,DA,GRACE}.nc)
    obs_file      : <basin>_obs_GRACE.hdf5 written by get_GRACE_obs (windows, covariance, sub-basin areas)
    out_root      : folder that holds the reports (Res/<case>/evaluation); the report goes to out_root/<tag>
    tag           : name of this run (e.g. 'run10'); None: date and time (YYYYmmdd_HHMM). An existing report
                    with the same tag is overwritten.
    basin         : basin name, used for GRACE_<basin>.h5
    da_output_dir : DA_output/<case> (filter_summary.json, adaptive_inflation_log.csv, Ens_k/threshold_monthly.csv)
    setting_file  : DA_setting.json of the run (copied into the report)
    reference     : tag of an earlier report in out_root, the path of a report folder, or 'previous' (the last
                    run in runs_index.csv) to compare with; None: no comparison
    start         : first day of the daily series in Res_*.h5
    figures_dir   : Res/<case>/figures; the increment diagnosis and DA_eval_stats.json found there are archived
    extra_files   : further files to archive in <report>/logs (e.g. rank logs)
    returns       : the summary dict (as in summary.json)
    """
    TH = dict(THRESHOLDS, **(thresholds or {}))
    tag = tag or dt.datetime.now().strftime('%Y%m%d_%H%M')
    key_months = KEY_MONTHS if key_months is None else key_months
    res_dir, out_root = Path(res_dir), Path(out_root)
    out = out_root / tag
    out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- data
    durs, sig_sub, sig_basin = _load_obs(obs_file)
    nsub = sig_sub.shape[1]
    units = ['basin'] + ['sub_basin_%d' % k for k in range(1, nsub + 1)]
    gkeys = ['basin_%d' % k for k in range(nsub + 1)]
    names = ['Basin'] + ['Sub-basin %d' % k for k in range(1, nsub + 1)]
    variables = [c[0] for c in COMP] + ['tws']
    DA = _load_res(res_dir / 'Res_DA.h5', units, variables)
    OL = _load_res(res_dir / 'Res_OL.h5', units, variables)
    with h5py.File(res_dir / ('GRACE_%s.h5' % basin), 'r') as g:
        GR = {u: g['original/%s' % k][:] for u, k in zip(units, gkeys)}
    nday = DA['basin']['tws'].shape[1]
    days = [start + dt.timedelta(days=i) for i in range(nday)]
    nmem = DA['basin']['tws'].shape[0]

    # observation windows inside the simulation (and not starting on day 0: no increment there)
    win = []
    for k, (a, b) in enumerate(durs):
        i0, i1 = (a - start).days, (b - start).days
        if i0 >= 1 and i1 < nday:
            win.append((k, i0, i1))
    kk = np.array([w[0] for w in win])
    wdate = [durs[k][0] for k in kk]
    wmon = np.array([d.month for d in wdate])
    tdec = np.array([_decyear(durs[k][0] + (durs[k][1] - durs[k][0]) / 2) for k in kk])
    sig = {u: (sig_basin[kk] if u == 'basin' else sig_sub[kk, j - 1]) for j, u in enumerate(units)}

    def wmean(x):            # x (nmem, nday) -> (nmem, nwin)
        return np.array([x[:, i0:i1 + 1].mean(1) for _, i0, i1 in win]).T

    def jump(x, i0):         # (nmem,) jump on the first day of a window
        return x[:, i0] - x[:, i0 - 1]

    S = {}                   # per unit: window series and window statistics
    for u in units:
        ol, da = wmean(OL[u]['tws']), wmean(DA[u]['tws'])
        g = GR[u][kk]
        g = g - np.mean(g - ol.mean(0))                                   # GRACE anomalies on the OL mean
        inc = np.array([jump(DA[u]['tws'], i0) - jump(OL[u]['tws'], i0) for _, i0, _ in win]).T
        fc = da - inc
        d = g - fc.mean(0)
        sdf = fc.std(0, ddof=1)
        chi2 = d ** 2 / (sdf ** 2 + sig[u] ** 2)
        S[u] = dict(gr=g, ol=ol.mean(0), da=da.mean(0), da_sd=da.std(0, ddof=1), inc=inc.mean(0), d=d,
                    sd_fc=sdf, chi2=chi2, sig=sig[u])

    # ---------------------------------------------------------------- numbers
    def fit(u, sel=slice(None)):
        s = S[u]
        return dict(rms_OL=_rms(s['ol'][sel] - s['gr'][sel]), rms_DA=_rms(s['da'][sel] - s['gr'][sel]),
                    corr_OL=_corr(s['ol'][sel], s['gr'][sel]), corr_DA=_corr(s['da'][sel], s['gr'][sel]))

    def gain(u, sel):
        d, i = S[u]['d'][sel], S[u]['inc'][sel]
        return float(np.sum(i * d) / np.sum(d * d)) if d.size else float('nan')

    summ = dict(tag=tag, created=dt.datetime.now().strftime('%Y-%m-%d %H:%M'), n_members=int(nmem),
                n_windows=len(win), period='%s to %s' % (days[0], days[-1]))
    summ['method'] = _method_string(setting_file)
    b = 'basin'
    summ['fit_basin'] = fit(b)
    summ['fit_basin']['rms_by_season_DA'] = {s: _rms((S[b]['da'] - S[b]['gr'])[np.isin(wmon, m)])
                                              for s, m in SEASONS.items()}
    summ['fit_basin']['rms_by_season_OL'] = {s: _rms((S[b]['ol'] - S[b]['gr'])[np.isin(wmon, m)])
                                              for s, m in SEASONS.items()}
    def rms_expected(u):
        '''RMS(analysis - GRACE) of a consistent analysis: for each window E[(y - Hxa)^2] = (1 - K) sigma^2 with
        K = sd_fc^2 / (sd_fc^2 + sigma^2), i.e. sigma^4 / (sd_fc^2 + sigma^2); the analysis is expected to sit
        closer to GRACE than sigma (gain 0.7 -> ~0.55 sigma), so RMS < sigma alone is no sign of over-fitting.
        Only regular windows (sigma <= grace_sigma_factor x its median, the same rule as the degraded-month
        warning): the sigma^4 term of a few degraded months (run12 sub-basin 1: sigma up to 176 mm) otherwise
        dominates the mean (19.4 mm expected vs 11.4 mm on regular windows). The actual RMS is returned on the
        same windows. Caveat: the scalar formula ignores the information a sub-basin receives from its correlated
        neighbours in the joint update, so a modest ratio below 1 is not by itself over-fitting.
        Returns (expected RMS, actual RMS on the regular windows, number of regular windows).'''
        s = S[u]
        keep = s['sig'] <= TH['grace_sigma_factor'] * np.median(s['sig'])
        exp = _rms((s['sig'] ** 2 / np.sqrt(s['sd_fc'] ** 2 + s['sig'] ** 2))[keep])
        act = _rms((s['da'] - s['gr'])[keep])
        return exp, act, int(keep.sum())

    summ['fit_sub_basins'] = {}
    summ['grace_sigma_factor'] = TH['grace_sigma_factor']
    for u in units[1:]:
        exp, act, nreg = rms_expected(u)
        summ['fit_sub_basins'][u] = dict(fit(u), sigma_GRACE_median=float(np.median(S[u]['sig'])),
                                         rms_DA_expected=exp, rms_DA_regular=act, n_regular_windows=nreg)
    ym = ['%04d-%02d' % (d.year, d.month) for d in wdate]
    summ['key_months'] = {m: dict(GRACE=S[b]['gr'][ym.index(m)], OL_minus_GRACE=S[b]['ol'][ym.index(m)] - S[b]['gr'][ym.index(m)],
                                  DA_minus_GRACE=S[b]['da'][ym.index(m)] - S[b]['gr'][ym.index(m)],
                                  sigma_GRACE=S[b]['sig'][ym.index(m)])
                          for m in key_months if m in ym}
    summ['trend_mm_yr'] = {u: dict(GRACE=_trend(tdec, S[u]['gr']), OL=_trend(tdec, S[u]['ol']),
                                   DA=_trend(tdec, S[u]['da'])) for u in units}
    spring, summer = np.isin(wmon, (2, 3, 4)), np.isin(wmon, range(6, 11))
    summ['filter_basin'] = dict(
        gain_all=gain(b, slice(None)), gain_FebApr=gain(b, spring), gain_JunOct=gain(b, summer),
        spread_fc_median=float(np.median(S[b]['sd_fc'])), spread_fc_FebApr=float(np.median(S[b]['sd_fc'][spring])),
        spread_fc_JunOct=float(np.median(S[b]['sd_fc'][summer])), sigma_GRACE_median=float(np.median(S[b]['sig'])),
        chi2_mean=float(np.mean(S[b]['chi2'])), chi2_median=float(np.median(S[b]['chi2'])),
        innovation_rms=_rms(S[b]['d']), innovation_mean=float(np.mean(S[b]['d'])))
    summ['chi2_median_sub_basins'] = {u: float(np.median(S[u]['chi2'])) for u in units[1:]}
    mon = np.array([d.month for d in days])
    summ['analysis_spread_by_month'] = {v: [float(DA[b][v].std(0, ddof=1)[mon == m].mean()) for m in range(1, 13)]
                                        for v in ['tws', 'groundwstor', 'soilmoist', 'swe']}
    clim = {u: _climatology(DA[u], OL[u], days) for u in units}
    summ['correction_climatology_basin'] = {v: clim[b][v][0].tolist() for v in clim[b]}

    # ---------------------------------------------------------------- logs of the run
    fsum, thr, alog = {}, [], []
    if da_output_dir is not None:
        p = Path(da_output_dir) / 'filter_summary.json'
        if p.exists():
            fsum = json.load(open(p))
        thr = _read_threshold_csvs(da_output_dir)
        alog = _read_adaptive_log(da_output_dir)
    summ['archived'] = archive_logs(out, da_output_dir, setting_file, figures_dir, extra_files)
    if fsum:                                          # what the filter really used (the settings file may be newer)
        sub = fsum.get('bounds', {}).get('soil_upper_bound')
        summ['method'] += ', soil_upper_bound=%s' % bool(sub)
    summ['clipping'] = _clipping_summary(thr)
    summ['adaptive_inflation'] = _adaptive_summary(alog, sig_sub)
    summ['filter_log'] = _filter_log_summary(fsum)

    # ---------------------------------------------------------------- warnings
    summ['warnings'] = _warnings(summ, S, units, wdate, sig_sub, kk, res_dir, thr, TH)

    # ---------------------------------------------------------------- reference
    ref = _load_reference(reference, out_root, tag)
    if ref is not None:
        summ['reference'] = ref['summary'].get('tag')
        summ['comparison'] = _compare(summ, ref['summary'])

    # ---------------------------------------------------------------- output
    _write_series(out / 'monthly_series.csv', units, wdate, S)
    json.dump(_r(summ, 3), open(out / 'summary.json', 'w'), indent=1)
    text = _report_text(_r(summ, 3), units)
    open(out / 'summary.txt', 'w').write(text)
    _append_index(out_root / 'runs_index.csv', summ)

    _fig_basin(out, S, wdate, ref, tag, summ)
    _fig_climatology(out, clim, units, names, tag)
    _fig_gw_drift(out, DA, OL, units, names, days, tag)
    _fig_sub_basins(out, S, units, names, wdate, ref, tag)
    _fig_health(out, S, wdate, alog, thr, tag, nsub)
    if verbose:
        print(text)
        print('evaluation report written to %s' % out)
    return summ


# ------------------------------------------------------------------------------------------------ archive
def archive_logs(out, da_output_dir=None, setting_file=None, figures_dir=None, extra_files=None):
    """
    Copy the diagnostic logs of a run into its report folder, so that they survive clean_temp_output and the
    next run (which overwrite DA_output/<case>, Res/<case>/figures and rewrite DA_setting.json):
        <out>/DA_setting.json
        <out>/logs/filter_summary.json, adaptive_inflation_log.csv
        <out>/logs/Ens_k/threshold_log.json, threshold_monthly.csv
        <out>/logs/figures/da_increment_diagnosis_*, da_increments_*.png, DA_eval_stats.json
        <out>/logs/<extra files>
    returns the list of archived files (relative to out)
    """
    out = Path(out)
    logs = out / 'logs'
    done = []

    def cp(src, dst):
        src = Path(src)
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            done.append(str(dst.relative_to(out)))
    if setting_file is not None:
        cp(setting_file, out / 'DA_setting.json')
    if da_output_dir is not None:
        d = Path(da_output_dir)
        for name in ('filter_summary.json', 'adaptive_inflation_log.csv'):
            cp(d / name, logs / name)
        for ens in sorted(d.glob('Ens_*')):
            for name in ('threshold_log.json', 'threshold_monthly.csv'):
                cp(ens / name, logs / ens.name / name)
    if figures_dir is not None:
        f = Path(figures_dir)
        for p in list(f.glob('da_increment_diagnosis_*')) + list(f.glob('da_increments_*')) + \
                list(f.glob('DA_eval_stats.json')):
            cp(p, logs / 'figures' / p.name)
    for p in extra_files or []:
        cp(p, logs / Path(p).name)
    return done


# ------------------------------------------------------------------------------------------------ numbers
def _method_string(setting_file):
    try:
        m = json.load(open(setting_file))['method']
    except Exception:
        return ''
    parts = [str(m.get('fusion_method', ''))]
    for key in ('localization', 'inflation', 'increment_partition'):
        c = m.get(key)
        if isinstance(c, dict):
            extra = {k: v for k, v in c.items() if k in ('max_weight_ratio', 'alpha', 'factor', 'sigma', 'split',
                                                         'split_storages', 'spatial_blend', 'max_cell_factor')}
            parts.append('%s=%s%s' % (key.split('_')[-1], c.get('kind'), (' %s' % extra) if extra else ''))
        elif c is not None:
            parts.append('%s=%s' % (key, c))
    if m.get('obs_error_correlation') not in (None, 'full'):
        parts.append('R=%s' % m['obs_error_correlation'])
    if m.get('obs_perturbation_centering'):
        parts.append('centred obs perturbations')
    if isinstance(m.get('snow_upper_bound'), dict):
        parts.append('snow <= %s x forecast + %s mm' % (m['snow_upper_bound'].get('factor'),
                                                         m['snow_upper_bound'].get('offset_mm')))
    if isinstance(m.get('snow_lower_bound'), dict):
        parts.append('snow >= %s x forecast' % m['snow_lower_bound'].get('factor'))
    if m.get('obs_error_inflation'):
        parts.append('obs_error_inflation=%s' % m['obs_error_inflation'])
    return ', '.join(parts)


def _climatology(da, ol, days):
    """mean (and year-to-year std) of DA - OL per calendar month; monthly means first"""
    ym = np.array([d.year * 100 + d.month for d in days])
    yms = np.unique(ym)
    out = {}
    for v in [c[0] for c in COMP] + ['tws']:
        if v not in da:
            continue
        x = da[v].mean(0) - ol[v].mean(0)
        mm = np.array([x[ym == k].mean() for k in yms])
        out[v] = (np.array([mm[yms % 100 == m].mean() for m in range(1, 13)]),
                  np.array([mm[yms % 100 == m].std() for m in range(1, 13)]))
    return out


def _clipping_summary(rows):
    if not rows:
        return {}
    out = {}
    for v in sorted(set(r['var'] for r in rows)):
        rv = [r for r in rows if r['var'] == v]
        n = sum(float(r['n_checked']) for r in rv)
        nm = len(set(r['member'] for r in rv))
        out[v] = dict(frac_lower=sum(float(r['n_lower']) for r in rv) / max(n, 1),
                      frac_upper=sum(float(r['n_upper']) for r in rv) / max(n, 1),
                      water_added_mm_per_member=sum(float(r['water_added_mm']) for r in rv) / nm,
                      water_removed_mm_per_member=sum(float(r['water_removed_mm']) for r in rv) / nm)
    return out


def _adaptive_summary(log, sig_sub):
    if not log:
        return {}
    sb = sorted(set(int(r['sub_basin']) for r in log))
    out = {}
    for j in sb:
        rj = [r for r in log if int(r['sub_basin']) == j]
        s = np.array([r['sigma_added_mm'] for r in rj])
        cap = np.array([3.0 * r['obs_error_mm'] for r in rj])
        out['sub_basin_%d' % j] = dict(sigma_added_mean=float(s.mean()), sigma_added_max=float(s.max()),
                                       date_of_max=rj[int(np.argmax(s))]['date'],
                                       frac_at_min=float(np.mean(s <= min(s.min(), 5.0) + 1e-6)),
                                       frac_near_upper_cap=float(np.mean(s >= 0.99 * cap)),
                                       bias_last=float(rj[-1]['bias_mm']))
    return out


def _filter_log_summary(fs):
    if not fs:
        return {}
    p = fs.get('partition', {})
    keep = {k: p.get(k) for k in ('kind', 'max_weight_ratio', 'n_weights_capped', 'largest_weight_ratio',
                                  'largest_weight_ratio_where', 'weight_share_removed_max', 'max_abs_inc_new',
                                  'max_inc_where', 'n_bound_capped', 'water_redistributed_mm') if k in p}
    for k in ('obs_error_correlation', 'n_obs_decorrelated', 'obs_perturbation_centering', 'n_obs_centered',
              'mean_perturbation_removed_rms_mm'):
        if k in fs:
            keep[k] = fs[k]
    inf = fs.get('inflation', {})
    keep['inflation'] = {k: inf.get(k) for k in ('kind', 'n_applied', 'mean_obs_space_spread_added_mm') if k in inf}
    return keep


def _warnings(summ, S, units, wdate, sig_sub, kk, res_dir, thr, TH):
    W, n = [], TH['n_list']
    # 1. grid cells with extreme trends
    try:
        import netCDF4
        H, C = {}, {}
        for st in ('OL', 'DA', 'GRACE'):
            p = Path(res_dir) / ('Harmonic_%s.nc' % st)
            if p.exists():
                with netCDF4.Dataset(p) as nc:
                    key = 'tws_trend_ensmean' if 'tws_trend_ensmean' in nc.variables else 'tws_trend'
                    H[st] = np.ma.filled(nc.variables[key][:], np.nan).astype(float)
                    C[st] = (np.asarray(nc.variables['lat'][:], float), np.asarray(nc.variables['lon'][:], float))
                    if st == 'DA' and 'tws_trend_ensspread' in nc.variables:
                        H['DA_spread'] = np.ma.filled(nc.variables['tws_trend_ensspread'][:], np.nan)
        if 'DA' in H:
            lat, lon = C['DA']

            def at(st, la, lo):
                '''value of map `st` at the cell (la, lo), looked up by coordinate: the maps need not share a grid
                (Harmonic_GRACE.nc may be on another box than the model maps)'''
                if st not in H:
                    return None
                la_s, lo_s = C[st]
                i, j = np.argmin(np.abs(la_s - la)), np.argmin(np.abs(lo_s - lo))
                if abs(la_s[i] - la) > 1e-6 or abs(lo_s[j] - lo) > 1e-6:
                    return None
                return float(H[st][i, j])

            big = np.zeros_like(H['DA'], bool)
            for st in ('OL', 'DA'):
                if st in H and H[st].shape == H['DA'].shape:
                    big |= np.abs(np.nan_to_num(H[st])) > TH['trend_cell_mm_yr']
            iy, ix = np.where(big)
            order = np.argsort(-np.abs(np.nan_to_num(H['DA'][iy, ix])))
            cells = [dict(lat=float(lat[iy[o]]), lon=float(lon[ix[o]]),
                          OL=at('OL', lat[iy[o]], lon[ix[o]]) if 'OL' in H else float(H['DA'][iy[o], ix[o]]),
                          DA=float(H['DA'][iy[o], ix[o]]),
                          GRACE=at('GRACE', lat[iy[o]], lon[ix[o]]),
                          DA_member_spread=float(H['DA_spread'][iy[o], ix[o]]) if 'DA_spread' in H else None)
                     for o in order]
            if cells:
                W.append(dict(kind='extreme_trend_cells', n=len(cells),
                              text='%d grid cells with |trend| > %g mm/yr in OL or DA' %
                                   (len(cells), TH['trend_cell_mm_yr']), items=cells[:n]))
    except Exception as err:
        W.append(dict(kind='harmonic_maps_not_read', text=str(err)))
    # 2. GRACE months with a large error
    med = np.median(sig_sub, 0)
    bad = [(durs_k, j) for durs_k in range(len(kk)) for j in range(sig_sub.shape[1])
           if sig_sub[kk[durs_k], j] > TH['grace_sigma_factor'] * med[j]]
    if bad:
        months = sorted(set(b[0] for b in bad))
        W.append(dict(kind='large_GRACE_error', n=len(months),
                      text='%d windows with a sub-basin GRACE sigma > %g x its median' %
                           (len(months), TH['grace_sigma_factor']),
                      items=[dict(month=str(wdate[m])[:7], max_sigma_mm=float(sig_sub[kk[m]].max()),
                                  median_sigma_mm=float(np.median(med))) for m in months][:n]))
    # 3. sub-basins fitted more tightly than a consistent analysis would be (not: than the GRACE error - the
    #    analysis is expected to be closer to GRACE than sigma, see rms_expected)
    over = {u: v for u, v in summ['fit_sub_basins'].items()
            if v['rms_DA_regular'] < TH['overfit_ratio'] * v['rms_DA_expected']}
    if over:
        W.append(dict(kind='fit_tighter_than_expected', n=len(over),
                      text='%d sub-basins with RMS(DA - GRACE) < %g x the RMS expected for a consistent analysis, '
                           'regular windows only (possible over-fitting)' % (len(over), TH['overfit_ratio']),
                      items=[dict(unit=u, rms_DA_regular=v['rms_DA_regular'], rms_DA_expected=v['rms_DA_expected'],
                                  n_regular_windows=v['n_regular_windows'], sigma_GRACE=v['sigma_GRACE_median'])
                             for u, v in over.items()]))
    # 4. chi^2
    c = summ['filter_basin']['chi2_median']
    if not TH['chi2_low'] <= c <= TH['chi2_high']:
        W.append(dict(kind='chi2', text='median basin chi^2 = %.2f (expected ~1: %s)' %
                                         (c, 'spread too small' if c > 1 else 'spread too large')))
    bad_sb = {u: v for u, v in summ['chi2_median_sub_basins'].items() if not TH['chi2_low'] <= v <= TH['chi2_high']}
    if bad_sb:
        W.append(dict(kind='chi2_sub_basins', text='sub-basins with median chi^2 outside [%g, %g]' %
                                                   (TH['chi2_low'], TH['chi2_high']), items=_r(bad_sb)))
    # 5. clipping
    if thr:
        agg = {}
        for r in thr:
            key = (r['var'], r['month'])
            a = agg.setdefault(key, [0.0, 0.0, 0.0])
            a[0] += float(r['n_checked']); a[1] += float(r['n_lower']); a[2] += float(r['n_upper'])
        hi = sorted([(k, (a[1] + a[2]) / max(a[0], 1), a[1] / max(a[0], 1), a[2] / max(a[0], 1))
                     for k, a in agg.items() if (a[1] + a[2]) / max(a[0], 1) > TH['clip_frac']], key=lambda x: -x[1])
        if hi:
            W.append(dict(kind='clipping', n=len(hi),
                          text='%d storage-months with more than %g%% of the cells clipped (all members)' %
                               (len(hi), 100 * TH['clip_frac']),
                          items=[dict(var=k[0], month=k[1], frac=f, frac_lower=fl, frac_upper=fu)
                                 for k, f, fl, fu in hi[:n]]))
    # 6. single-cell increments, capped weights
    fl = summ.get('filter_log', {})
    mi = fl.get('max_abs_inc_new')
    if mi is not None and mi > TH['single_increment_mm']:
        W.append(dict(kind='large_single_cell_increment', text='largest single-cell increment %.0f mm' % mi,
                      items=[fl.get('max_inc_where')]))
    if fl.get('n_weights_capped'):
        W.append(dict(kind='weights_capped', text='%d partition weights capped (largest weight / median = %.1f)' %
                                                  (fl['n_weights_capped'], fl.get('largest_weight_ratio') or np.nan),
                      items=[fl.get('largest_weight_ratio_where')]))
    # 7. adaptive inflation at its limits
    for u, a in summ.get('adaptive_inflation', {}).items():
        if a['frac_near_upper_cap'] > 0.1:
            W.append(dict(kind='adaptive_inflation_at_cap',
                          text='%s: added spread at the upper cap in %.0f%% of the windows' %
                               (u, 100 * a['frac_near_upper_cap'])))
    return W


# ------------------------------------------------------------------------------------------------ reference
def _load_reference(reference, out_root, tag=None):
    if reference is None:
        return None
    if reference == 'previous':                       # last evaluated run in runs_index.csv (other than this one)
        idx = Path(out_root) / 'runs_index.csv'
        tags = [r['tag'] for r in csv.DictReader(open(idx))] if idx.exists() else []
        tags = [t for t in tags if t != tag]
        if not tags:
            return None
        reference = tags[-1]
    p = Path(out_root) / str(reference)               # a tag in out_root first, otherwise a path
    if not (p / 'summary.json').exists():
        p = Path(reference)
    if not (p / 'summary.json').exists():
        print('reference report %s not found - no comparison' % p)
        return None
    ref = dict(summary=json.load(open(p / 'summary.json')), series={})
    if (p / 'monthly_series.csv').exists():
        with open(p / 'monthly_series.csv') as fh:
            for r in csv.DictReader(fh):
                ref['series'].setdefault(r['unit'], {})[r['window_start']] = float(r['DA'])
    return ref


def _compare(a, b):
    def g(s, *keys):
        for k in keys:
            if not isinstance(s, dict) or k not in s:
                return None
            s = s[k]
        return s
    rows = [('RMS DA-GRACE basin [mm]', ('fit_basin', 'rms_DA')),
            ('corr DA-GRACE basin', ('fit_basin', 'corr_DA')),
            ('RMS MAM [mm]', ('fit_basin', 'rms_by_season_DA', 'MAM')),
            ('RMS JJA [mm]', ('fit_basin', 'rms_by_season_DA', 'JJA')),
            ('RMS SON [mm]', ('fit_basin', 'rms_by_season_DA', 'SON')),
            ('RMS DJF [mm]', ('fit_basin', 'rms_by_season_DA', 'DJF')),
            ('trend DA basin [mm/yr]', ('trend_mm_yr', 'basin', 'DA')),
            ('gain (all windows)', ('filter_basin', 'gain_all')),
            ('chi2 median', ('filter_basin', 'chi2_median')),
            ('chi2 mean', ('filter_basin', 'chi2_mean')),
            ('forecast spread median [mm]', ('filter_basin', 'spread_fc_median')),
            ('largest single-cell increment [mm]', ('filter_log', 'max_abs_inc_new'))]
    rows += [('RMS DA-GRACE %s [mm]' % u, ('fit_sub_basins', u, 'rms_DA')) for u in sorted(a['fit_sub_basins'])]
    rows += [('%s DA-GRACE [mm]' % m, ('key_months', m, 'DA_minus_GRACE')) for m in a['key_months']]
    out = []
    for lab, keys in rows:
        x, y = g(a, *keys), g(b, *keys)
        out.append(dict(item=lab, this=x, reference=y,
                        change=(x - y) if isinstance(x, (int, float)) and isinstance(y, (int, float)) else None))
    return out


# ------------------------------------------------------------------------------------------------ writing
def _write_series(path, units, wdate, S):
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['unit', 'window_start', 'GRACE', 'sigma_GRACE', 'OL', 'DA', 'DA_spread', 'innovation',
                    'increment', 'spread_forecast', 'chi2'])
        for u in units:
            s = S[u]
            for i, d in enumerate(wdate):
                w.writerow([u, str(d)] + ['%.3f' % s[k][i] for k in ('gr', 'sig', 'ol', 'da', 'da_sd', 'd', 'inc',
                                                                      'sd_fc', 'chi2')])


def _append_index(path, s):
    """one row per evaluated run; a run evaluated again replaces its row"""
    head = ['tag', 'created', 'method', 'rms_OL', 'rms_DA', 'corr_DA', 'rms_MAM', 'trend_GRACE', 'trend_DA', 'gain',
            'chi2_median', 'max_single_increment_mm', 'n_warnings']
    rows = []
    if path.exists():
        rows = [r for r in csv.DictReader(open(path)) if r.get('tag') != s['tag']]
    f, t = s['fit_basin'], s['trend_mm_yr']['basin']
    vals = [f['rms_OL'], f['rms_DA'], f['corr_DA'], f['rms_by_season_DA']['MAM'], t['GRACE'], t['DA'],
            s['filter_basin']['gain_all'], s['filter_basin']['chi2_median'], s.get('filter_log', {}).get('max_abs_inc_new')]
    rows.append(dict(zip(head, [s['tag'], s['created'], s['method']] +
                         ['%.3f' % x if x is not None else '' for x in vals] + [len(s['warnings'])])))
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=head, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def _report_text(s, units):
    L = []
    P = L.append
    P('PyGLDA v2 assimilation evaluation: %s   (%s)' % (s['tag'], s['created']))
    P('method : %s' % s['method'])
    P('period : %s, %d members, %d observation windows' % (s['period'], s['n_members'], s['n_windows']))
    if s.get('reference'):
        P('reference run : %s' % s['reference'])
    f = s['fit_basin']
    P('\n1. Fit to GRACE (window means, GRACE mean aligned to the OL mean)')
    P('   basin   RMS OL %6.1f  DA %6.1f mm   corr OL %.3f  DA %.3f' % (f['rms_OL'], f['rms_DA'], f['corr_OL'],
                                                                      f['corr_DA']))
    P('   by season (DA / OL):  ' + '   '.join('%s %.1f / %.1f' % (k, f['rms_by_season_DA'][k], f['rms_by_season_OL'][k])
                                              for k in SEASONS))
    P('   %-12s %8s %8s %8s %8s %8s %8s %10s' % ('sub-basin', 'RMS OL', 'RMS DA', 'DA reg', 'expected', 'corr DA',
                                                  'sigma_GR', 'chi2 med'))
    for u in units[1:]:
        v = s['fit_sub_basins'][u]
        P('   %-12s %8.1f %8.1f %8.1f %8.1f %8.3f %8.1f %10.2f' % (
            u.replace('sub_basin_', 'sub '), v['rms_OL'], v['rms_DA'], v.get('rms_DA_regular', float('nan')),
            v.get('rms_DA_expected', float('nan')), v['corr_DA'], v['sigma_GRACE_median'],
            s['chi2_median_sub_basins'][u]))
    P('   (DA reg / expected: RMS(DA - GRACE) and the RMS of a consistent analysis, sqrt(mean(sigma^4 / '
      '(spread_fc^2 + sigma^2))), both on windows with sigma <= %g x its median)' % s.get('grace_sigma_factor', 3.0))
    P('   key months (basin)   GRACE   OL-GR   DA-GR   sigma')
    for m, v in s['key_months'].items():
        P('     %s        %7.1f %7.1f %7.1f %7.1f' % (m, v['GRACE'], v['OL_minus_GRACE'], v['DA_minus_GRACE'],
                                                    v['sigma_GRACE']))
    P('\n2. Trends [mm/yr]           GRACE      OL      DA')
    for u in units:
        t = s['trend_mm_yr'][u]
        P('   %-22s %7.2f %7.2f %7.2f' % (u, t['GRACE'], t['OL'], t['DA']))
    fb = s['filter_basin']
    P('\n3. Filter (basin)')
    P('   gain all %.2f  Feb-Apr %.2f  Jun-Oct %.2f   (share of the innovation taken by the increment)' %
      (fb['gain_all'], fb['gain_FebApr'], fb['gain_JunOct']))
    P('   forecast spread median %.1f mm (Feb-Apr %.1f, Jun-Oct %.1f);  GRACE sigma median %.1f mm' %
      (fb['spread_fc_median'], fb['spread_fc_FebApr'], fb['spread_fc_JunOct'], fb['sigma_GRACE_median']))
    P('   chi2 mean %.2f  median %.2f (~1 expected);  innovation mean %.1f  RMS %.1f mm' %
      (fb['chi2_mean'], fb['chi2_median'], fb['innovation_mean'], fb['innovation_rms']))
    P('   analysis spread by month (J..D):')
    for v, x in s['analysis_spread_by_month'].items():
        P('     %-12s %s' % (v, ' '.join('%5.1f' % a for a in x)))
    P('   mean correction DA-OL by month, basin (J..D):')
    for v, x in s['correction_climatology_basin'].items():
        P('     %-12s %s' % (v, ' '.join('%6.1f' % a for a in x)))
    if s.get('filter_log'):
        P('\n4. Filter log (filter_summary.json)')
        for k, v in s['filter_log'].items():
            P('   %s: %s' % (k, v))
    if s.get('clipping'):
        P('\n5. Clipping (all members, whole run)   frac lower  frac upper  added / removed per member [mm, summed over cells and windows]')
        for v, c in s['clipping'].items():
            P('   %-18s %10.4f %11.4f %13.0f %15.0f' % (v, c['frac_lower'], c['frac_upper'],
                                                       c['water_added_mm_per_member'], c['water_removed_mm_per_member']))
    if s.get('adaptive_inflation'):
        P('\n6. Adaptive inflation (obs-space sigma added [mm])   mean    max  (date)        at min  at cap  bias end')
        for u, a in s['adaptive_inflation'].items():
            P('   %-14s %35.1f %6.1f  (%s)  %5.2f  %6.2f  %7.1f' % (u, a['sigma_added_mean'], a['sigma_added_max'],
                                                                 a['date_of_max'], a['frac_at_min'],
                                                                 a['frac_near_upper_cap'], a['bias_last']))
    if s.get('comparison'):
        P('\n7. Comparison with %s' % s['reference'])
        P('   %-36s %10s %10s %10s' % ('', 'this', 'reference', 'change'))
        for r in s['comparison']:
            fmt = lambda x: '%10.2f' % x if isinstance(x, (int, float)) else '%10s' % '-'
            P('   %-36s %s %s %s' % (r['item'], fmt(r['this']), fmt(r['reference']), fmt(r['change'])))
    if s.get('archived'):
        P('\n   %d log files archived in logs/ of this report' % len(s['archived']))
    P('\n8. Warnings (%d)' % len(s['warnings']))
    for w in s['warnings']:
        P('   * %s' % w['text'])
        for it in w.get('items', []) or []:
            P('       %s' % it)
    return '\n'.join(L) + '\n'


# ------------------------------------------------------------------------------------------------ figures
def _datefmt(ax):
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))


def _save(fig, out, name):
    fig.savefig(out / (name + '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _ref_series(ref, unit, wdate):
    if ref is None or unit not in ref['series']:
        return None
    s = ref['series'][unit]
    return np.array([s.get(str(d), np.nan) for d in wdate])


def _fig_basin(out, S, wdate, ref, tag, summ):
    s = S['basin']
    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.08))
    ax[0].fill_between(wdate, s['gr'] - s['sig'], s['gr'] + s['sig'], color='0.85', lw=0, label='GRACE ±σ')
    ax[0].plot(wdate, s['gr'], color=C_GR, lw=1.8, marker='o', ms=3, label='GRACE')
    ax[0].plot(wdate, s['ol'], color=C_OL, lw=1.8, label='OL (RMS %.1f)' % summ['fit_basin']['rms_OL'])
    rf = _ref_series(ref, 'basin', wdate)
    if rf is not None:
        ax[0].plot(wdate, rf, color=C_REF, lw=1.6, ls='--', label='DA %s (RMS %.1f)' %
                   (summ['reference'], ref['summary']['fit_basin']['rms_DA']))
    ax[0].plot(wdate, s['da'], color=C_DA, lw=2.2, label='DA %s (RMS %.1f)' % (tag, summ['fit_basin']['rms_DA']))
    lo, hi = min(s['gr'].min(), s['ol'].min()), max(s['gr'].max(), s['ol'].max())
    ax[0].set_ylim(lo - 0.08 * (hi - lo), hi + 0.25 * (hi - lo))
    ax[0].set_ylabel('TWS anomaly [mm]'); ax[0].legend(ncol=3, loc='upper left'); ax[0].grid(alpha=0.25)
    ax[0].set_title('Basin TWS, window means: %s' % tag)
    ax[1].fill_between(wdate, -s['sig'], s['sig'], color='0.85', lw=0)
    ax[1].axhline(0, color='0.3', lw=1)
    ax[1].plot(wdate, s['ol'] - s['gr'], color=C_OL, lw=1.6, label='OL − GRACE')
    if rf is not None:
        ax[1].plot(wdate, rf - s['gr'], color=C_REF, lw=1.4, ls='--', label='DA %s − GRACE' % summ['reference'])
    ax[1].plot(wdate, s['da'] - s['gr'], color=C_DA, lw=2, label='DA − GRACE')
    r = np.abs(np.concatenate([s['ol'] - s['gr'], s['da'] - s['gr']])).max()
    ax[1].set_ylim(-1.15 * r, 1.15 * r)
    ax[1].set_ylabel('residual [mm]'); ax[1].legend(ncol=3, loc='upper left'); ax[1].grid(alpha=0.25)
    _datefmt(ax[1])
    _save(fig, out, 'fig1_basin_timeseries')


def _fig_climatology(out, clim, units, names, tag):
    x = np.arange(1, 13)
    nsub = len(units) - 1
    ncol = 3
    nrow = 1 + int(np.ceil(nsub / ncol))
    fig = plt.figure(figsize=(15, 3.6 * nrow + 1))
    gs = fig.add_gridspec(nrow, ncol + 1, height_ratios=[1.35] + [1] * (nrow - 1), hspace=0.45, wspace=0.32)

    def draw(ax, r):
        ax.axhline(0, color='0.25', lw=1.0)
        m, sd = r['tws']
        ax.fill_between(x, m - sd, m + sd, color='0.85', lw=0, zorder=0)
        ax.plot(x, m, color='k', lw=2.4, zorder=4)
        for v, lab, c, mk in COMP:
            if v in r:
                ax.plot(x, r[v][0], color=c, lw=2, marker=mk, ms=5, mec='white', mew=1.2, zorder=5)
        ax.set_xticks(x); ax.set_xticklabels(list('JFMAMJJASOND')); ax.set_xlim(0.6, 12.4); ax.grid(alpha=0.25)
    axb = fig.add_subplot(gs[0, 1:3]); draw(axb, clim['basin']); axb.set_ylabel('mm')
    axb.set_title('Basin: mean seasonal correction DA − OL (%s)' % tag)
    axl = fig.add_subplot(gs[0, 0]); axl.axis('off')
    h = [plt.Line2D([], [], color=c, lw=2, marker=mk, ms=7, mec='white', label=lab) for _, lab, c, mk in COMP]
    h += [plt.Line2D([], [], color='k', lw=2.4, label='TWS correction'),
          plt.Rectangle((0, 0), 1, 1, color='0.85', label='TWS ±1 std (year to year)')]
    axl.legend(handles=h, loc='center left')
    for i, u in enumerate(units[1:]):
        ax = fig.add_subplot(gs[1 + i // ncol, i % ncol]); draw(ax, clim[u]); ax.set_title(names[i + 1], fontsize=12)
        if i % ncol == 0:
            ax.set_ylabel('mm')
    _save(fig, out, 'fig2_correction_climatology')


def _fig_gw_drift(out, DA, OL, units, names, days, tag):
    ym = np.array([d.year * 100 + d.month for d in days])
    yms = np.unique(ym)
    t = [dt.date(k // 100, k % 100, 15) for k in yms]
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(units) - 1))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axhline(0, color='0.3', lw=1)
    for i, u in enumerate(units):
        x = DA[u]['groundwstor'].mean(0) - OL[u]['groundwstor'].mean(0)
        mm = np.array([x[ym == k].mean() for k in yms])
        if u == 'basin':
            ax.plot(t, mm, color='k', lw=2.6, label='Basin', zorder=5)
        else:
            ax.plot(t, mm, color=cols[i - 1], lw=1.5, label=names[i])
    ax.set_ylabel('DA − OL groundwater [mm]'); ax.grid(alpha=0.25); ax.legend(ncol=4, loc='upper left')
    ax.set_title('Groundwater correction over time (monthly means): %s' % tag)
    _datefmt(ax)
    _save(fig, out, 'fig3_groundwater_drift')


def _fig_sub_basins(out, S, units, names, wdate, ref, tag):
    nsub = len(units) - 1
    ncol = 2
    nrow = int(np.ceil(nsub / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(15, 3.2 * nrow), sharex=True, squeeze=False)
    for i, u in enumerate(units[1:]):
        ax = axs[i // ncol, i % ncol]
        s = S[u]
        ax.fill_between(wdate, s['gr'] - s['sig'], s['gr'] + s['sig'], color='0.85', lw=0)
        ax.plot(wdate, s['gr'], color=C_GR, lw=1.4)
        ax.plot(wdate, s['ol'], color=C_OL, lw=1.4)
        rf = _ref_series(ref, u, wdate)
        if rf is not None:
            ax.plot(wdate, rf, color=C_REF, lw=1.3, ls='--')
        ax.plot(wdate, s['da'], color=C_DA, lw=1.8)
        lo = min(s['gr'].min(), s['ol'].min(), s['da'].min()); hi = max(s['gr'].max(), s['ol'].max(), s['da'].max())
        ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))     # a huge GRACE sigma must not set the scale
        ax.set_title('%s   RMS OL %.1f  DA %.1f   σ_GR %.1f mm' % (names[i + 1], _rms(s['ol'] - s['gr']),
                                                                 _rms(s['da'] - s['gr']), np.median(s['sig'])),
                     fontsize=11)
        ax.grid(alpha=0.25)
        if i % ncol == 0:
            ax.set_ylabel('mm')
        _datefmt(ax)
    for j in range(nsub, nrow * ncol):
        axs[j // ncol, j % ncol].axis('off')
    h = [plt.Line2D([], [], color=C_GR, lw=1.6, label='GRACE (±σ)'), plt.Line2D([], [], color=C_OL, lw=1.6, label='OL'),
         plt.Line2D([], [], color=C_DA, lw=1.8, label='DA %s' % tag)]
    if ref is not None:
        h.append(plt.Line2D([], [], color=C_REF, lw=1.4, ls='--', label='DA %s' % ref['summary'].get('tag')))
    fig.legend(handles=h, loc='upper center', ncol=len(h), bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    _save(fig, out, 'fig4_sub_basins')


def _fig_health(out, S, wdate, alog, thr, tag, nsub):
    s = S['basin']
    fig, axs = plt.subplots(2, 2, figsize=(15, 9))
    ax = axs[0, 0]
    ax.semilogy(wdate, np.maximum(s['chi2'], 1e-3), 'o', color=C_DA, ms=4)
    ax.axhline(1, color='k', lw=1.5); ax.grid(alpha=0.25)
    ax.set_title('Basin χ² per window (median %.2f, mean %.2f)' % (np.median(s['chi2']), np.mean(s['chi2']))); _datefmt(ax)
    ax = axs[0, 1]
    m = np.array([d.month for d in wdate]); x = np.arange(1, 13)
    ax.plot(x, [_rms(s['d'][m == k]) for k in x], color='k', lw=2, marker='o', label='RMS innovation')
    ax.plot(x, [np.sqrt(np.median(s['sd_fc'][m == k] ** 2 + s['sig'][m == k] ** 2)) for k in x], color=C_DA, lw=2,
            marker='s', label='√(spread² + σ_GR²), median')
    ax.plot(x, [np.median(s['sd_fc'][m == k]) for k in x], color=C_DA, lw=1.5, ls='--', label='forecast spread, median')
    ax.set_xticks(x); ax.set_xticklabels(list('JFMAMJJASOND')); ax.grid(alpha=0.25); ax.legend()
    ax.set_title('Innovation vs expected (basin)'); ax.set_ylabel('mm')
    ax = axs[1, 0]
    if alog:
        cols = plt.cm.viridis(np.linspace(0, 0.9, nsub))
        for j in range(1, nsub + 1):
            r = [a for a in alog if int(a['sub_basin']) == j]
            ax.plot([dt.datetime.strptime(a['date'][:10], '%Y-%m-%d').date() for a in r],
                    [a['sigma_added_mm'] for a in r], color=cols[j - 1], lw=1.4, label='sub %d' % j)
        ax.legend(ncol=3); ax.set_title('Adaptive inflation: added obs-space spread'); ax.set_ylabel('mm')
        _datefmt(ax)
    else:
        ax.text(0.5, 0.5, 'no adaptive inflation log', ha='center', transform=ax.transAxes)
    ax.grid(alpha=0.25)
    ax = axs[1, 1]
    if thr:
        for v, lab, c, mk in COMP:
            agg = {}
            for r in thr:
                if r['var'] == v:
                    a = agg.setdefault(r['month'], [0.0, 0.0])
                    a[0] += float(r['n_checked']); a[1] += float(r['n_lower']) + float(r['n_upper'])
            if agg:
                ks = sorted(agg)
                ax.plot([dt.date(int(k[:4]), int(k[5:7]), 15) for k in ks],
                        [100 * agg[k][1] / max(agg[k][0], 1) for k in ks], color=c, lw=1.4, label=lab)
        ax.legend(); ax.set_ylabel('% of cells clipped'); ax.set_title('Clipping per month (all members)')
        _datefmt(ax)
    else:
        ax.text(0.5, 0.5, 'no threshold_monthly.csv', ha='center', transform=ax.transAxes)
    ax.grid(alpha=0.25)
    fig.suptitle('Filter health: %s' % tag, y=0.995, fontweight='bold')
    fig.tight_layout()
    _save(fig, out, 'fig5_filter_health')


def demo1():
    """stand-alone use with the demo_2 paths"""
    root = Path('/media/user/My Book/Fan/PyGLDA_v2_external_data')
    evaluate_run(res_dir=root / 'Res/demo_2', obs_file=root / 'GRACE/obs/Danube_obs_GRACE.hdf5',
                 out_root=root / 'Res/demo_2/evaluation', tag='run', basin='Danube',
                 da_output_dir=root / 'DA_output/demo_2',
                 setting_file='/media/user/My Book/Fan/PyGLDA_v2/settings/demo_2/DA_setting.json', reference=None)


if __name__ == '__main__':
    demo1()
