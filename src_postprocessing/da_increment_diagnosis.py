"""
Diagnosis of the assimilation increments of a PyGLDA v2 run from the collected basin time series.

Reads only the small files
    <RES>/OL/Ens_k/basin_ts_OL.h5, <RES>/DA/Ens_k/basin_ts_DA.h5   (daily storages, basin and sub-basins)
    <OBS>  (GRACE observations written by get_GRACE_obs: time_epoch, duration, ens_0, cov, sub_basin_area)
and prints / saves

  A. tug-of-war table   : mean increment per window vs mean drift inside the windows, per storage
                          (increment = jump of DA-OL at the first day of a window; drift = daily change of DA-OL
                          on the other days). Increment + 4 x drift ~ 0 means the filter puts water in a store
                          and the model takes it out again before the next update.
  B. events             : windows whose basin-mean increment of one storage exceeds a threshold in any member,
                          with the sub-basin breakdown of the largest ones and the observation innovation.
  C. correlations       : cross-member correlation of every storage with TWS in the open loop and in the DA
                          (negative values in the DA = the filter has made the storages compensate each other).
  D. peaks              : year-by-year maximum / minimum of the window-mean TWS anomaly (OL, DA, GRACE).
  E. figure             : basin-mean increments per storage over time (<OUT>/da_increments_<tag>.png)

Usage:  called by RDA.increment_diagnosis() (src_FlowControl/Regional_DA.py) after post_processing, or
        from src_postprocessing.da_increment_diagnosis import run_diagnosis
        run_diagnosis(res_dir, obs_file, out_dir, tag, nens, start)
    or  python src_postprocessing/da_increment_diagnosis.py  (demo1 below; paths taken from the DA setting file)
It reads collected results only and does not touch the assimilation code.
"""
import json, datetime as dt
from pathlib import Path
import numpy as np, h5py
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

STORAGES = ['groundwstor', 'soilmoist', 'riverstor', 'swe', 'canopystor', 'locallakestor', 'globallakestor',
            'localwetlandstor', 'globalwetlandstor', 'reservoirstor']
plt.rcParams.update({'font.size': 11, 'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
                     'axes.linewidth': 1.8, 'xtick.major.width': 1.6, 'ytick.major.width': 1.6, 'legend.fontsize': 10})


def run_diagnosis(res_dir, obs_file, out_dir, tag='DA', nens=4, start=dt.date(2002, 1, 1),
                  event_threshold_mm=40.0, n_events_detail=5, verbose=True):
    """
    res_dir   : Res/<case> (contains OL/ and DA/ with Ens_k/basin_ts_*.h5)
    obs_file  : <basin>_obs_GRACE.hdf5 written by get_GRACE_obs
    out_dir   : where the text/json report and the figure are written
    start     : first day of the collected daily series
    returns   : dict with the numbers of tables A-D
    """
    RES, OBS, OUT, TAG, NENS, START = res_dir, obs_file, out_dir, tag, nens, start
    EVENT_THRESHOLD_MM, N_EVENTS_DETAIL = event_threshold_mm, n_events_detail
    Path(OUT).mkdir(parents=True, exist_ok=True)

    def load_stage(stage):
        """dict[var][key] -> array (NENS, ndays); key = 'basin' or 'sub_basin_j'"""
        out = {}
        for m in range(1, NENS + 1):
            f = h5py.File(Path(RES) / stage / ('Ens_%d' % m) / ('basin_ts_%s.h5' % stage), 'r')
            for v in list(STORAGES) + ['tws']:
                if v not in f:
                    continue
                out.setdefault(v, {})
                for k in f[v]:
                    out[v].setdefault(k, []).append(f[v][k][:])
            f.close()
        return {v: {k: np.array(a) for k, a in d.items()} for v, d in out.items()}


    OL, DA = load_stage('OL'), load_stage('DA')
    nd = min(OL['tws']['basin'].shape[1], DA['tws']['basin'].shape[1])
    for S in (OL, DA):
        for v in S:
            for k in S[v]:
                S[v][k] = S[v][k][:, :nd]
    storages = [v for v in STORAGES if v in DA]
    days = np.array([START + dt.timedelta(days=i) for i in range(nd)])
    day_index = {d: i for i, d in enumerate(days)}
    nsub = len([k for k in DA['tws'] if k.startswith('sub_basin_')])
    keys = ['basin'] + ['sub_basin_%d' % j for j in range(1, nsub + 1)]

    o = h5py.File(OBS, 'r')
    epochs = [dt.datetime.strptime(e, '%Y-%m-%d').date() for e in o['time_epoch'][:].astype(str)]
    durs = [[dt.datetime.strptime(x, '%Y-%m-%d').date() for x in d.split('_')] for d in o['duration'][:].astype(str)]
    Y = o['ens_0'][:]                                              # unperturbed observations (nobs, nsub), model datum
    area = o['sub_basin_area'][:]; w = area / area.sum()
    windows = [(i, a, b) for i, (a, b) in enumerate(durs) if a in day_index and b in day_index]
    starts = np.array([day_index[a] for _, a, _ in windows])       # first day of each window
    starts = starts[starts > 0]
    inside = np.setdiff1d(np.arange(1, nd), starts)


    def diff_da_ol(v, k):
        """daily change of (DA - OL) per member, (NENS, nd-1)"""
        return np.diff(DA[v][k] - OL[v][k], axis=1)


    report = {'tag': TAG, 'n_windows': int(len(starts)), 'n_members': NENS}
    lines = ['Increment diagnosis  [%s]   %d windows, %d members, %s .. %s' % (TAG, len(starts), NENS, days[0], days[-1])]

    # ------------------------------------------------------------------ A. tug of war
    lines += ['', 'A. Tug of war (basin mean, ensemble mean) [mm/day]: increment at window start vs drift inside windows',
              '   %-18s %10s %10s %12s %12s' % ('storage', 'increment', 'drift', 'incr+4*drift', 'OL drift')]
    report['tug_of_war'] = {}
    for v in storages + ['tws']:
        d = diff_da_ol(v, 'basin').mean(0)
        d_ol = np.diff(OL[v]['basin'], axis=1).mean(0)
        inc, drf = d[starts - 1].mean(), d[inside - 1].mean()
        report['tug_of_war'][v] = dict(increment=float(inc), drift=float(drf))
        lines.append('   %-18s %+10.3f %+10.3f %+12.3f %+12.3f' % (v, inc, drf, inc + 4 * drf, d_ol[inside - 1].mean()))
    lines += ['', '   per sub-basin, increment / drift for groundwstor and riverstor [mm/day]',
              '   %-12s %9s %9s | %9s %9s' % ('sub-basin', 'GW incr', 'GW drift', 'RIV incr', 'RIV drift')]
    for k in keys[1:]:
        row = []
        for v in ['groundwstor', 'riverstor']:
            if v in DA:
                d = diff_da_ol(v, k).mean(0); row += [d[starts - 1].mean(), d[inside - 1].mean()]
            else:
                row += [np.nan, np.nan]
        lines.append('   %-12s %+9.3f %+9.3f | %+9.3f %+9.3f' % (k.replace('sub_basin_', ''), *row))

    # ------------------------------------------------------------------ B. events
    lines += ['', 'B. Events: windows with |basin-mean increment| > %.0f mm for one storage in any member' % EVENT_THRESHOLD_MM]
    J = {v: diff_da_ol(v, 'basin') for v in storages + ['tws']}      # (NENS, nd-1)
    score = np.zeros(len(starts))
    for v in storages:
        score = np.maximum(score, np.abs(J[v][:, starts - 1]).max(0))
    ev = np.where(score > EVENT_THRESHOLD_MM)[0]
    report['events'] = []
    lines.append('   %d of %d windows;  date, then per storage the increment range over members [mm]' % (len(ev), len(starts)))
    for i in ev:
        s = starts[i]
        txt = ' '.join('%s %+.0f..%+.0f' % (v[:5], J[v][:, s - 1].min(), J[v][:, s - 1].max())
                       for v in ['groundwstor', 'soilmoist', 'riverstor', 'tws'] if v in J)
        lines.append('   %s  %s' % (days[s], txt))
        report['events'].append(dict(date=str(days[s]), score=float(score[i])))
    order = ev[np.argsort(-score[ev])][:N_EVENTS_DETAIL]
    for i in order:
        s = starts[i]
        m = int(np.argmax(np.abs(J['groundwstor'][:, s - 1]))) if 'groundwstor' in J else 0
        # innovation of that window: observation minus window mean of the previous analysis and of the open loop
        wi = [x for x in windows if day_index[x[1]] == s][0]
        a, b = day_index[wi[1]], day_index[wi[2]]
        lines += ['', '   detail %s, member %d (largest groundwater jump): sub-basin, increments GW/soil/river/TWS [mm], '
                      'obs - OL window mean, DA TWS before the window - OL' % (days[s], m + 1)]
        for j, k in enumerate(keys):
            inc = [DA[v][k][m, s] - DA[v][k][m, s - 1] for v in ['groundwstor', 'soilmoist', 'riverstor', 'tws'] if v in DA]
            if k == 'basin':
                ol_win = (OL['tws'][k][:, a:b + 1].mean(1)).mean(); obs = Y[wi[0]] @ w
            else:
                ol_win = OL['tws'][k][:, a:b + 1].mean(); obs = Y[wi[0], j - 1]
            lines.append('   %-6s %8.0f %8.0f %8.0f %8.0f | inno %8.1f | DA-OL before %8.0f'
                         % (k.replace('sub_basin_', ''), *inc, obs - ol_win, DA['tws'][k][m, s - 1] - OL['tws'][k][m, s - 1]))

    # ------------------------------------------------------------------ C. correlations
    lines += ['', 'C. Cross-member correlation of each storage with TWS (mean over days) and member spread [mm], basin',
              '   %-18s %8s %8s | %8s %8s' % ('storage', 'OL corr', 'OL sprd', 'DA corr', 'DA sprd')]
    report['correlation'] = {}


    def corr_spread(S, v, k):
        T, V = S['tws'][k], S[v][k]
        Tc, Vc = T - T.mean(0), V - V.mean(0)
        c = (Tc * Vc).sum(0) / np.sqrt((Tc ** 2).sum(0) * (Vc ** 2).sum(0) + 1e-12)
        return float(np.nanmean(c)), float(np.nanmean(V.std(0, ddof=1)))


    for v in storages:
        c0, s0 = corr_spread(OL, v, 'basin'); c1, s1 = corr_spread(DA, v, 'basin')
        report['correlation'][v] = dict(OL=c0, DA=c1, spread_OL=s0, spread_DA=s1)
        lines.append('   %-18s %+8.2f %8.1f | %+8.2f %8.1f' % (v, c0, s0, c1, s1))

    # ------------------------------------------------------------------ D. yearly peaks (window means, anomalies)
    lines += ['', 'D. Yearly maximum / minimum of the window-mean basin TWS anomaly [mm] (common mean removed)',
              '   %4s | %8s %8s %8s | %8s %8s %8s | %8s %8s %8s' % ('year', 'OL max', 'DA max', 'GRACE', 'OL min', 'DA min', 'GRACE', 'amp OL', 'amp DA', 'amp GR')]
    wm_ol, wm_da, obs_b, tmid = [], [], [], []
    for i, a, b in windows:
        ia, ib = day_index[a], day_index[b]
        wm_ol.append((OL['tws']['basin'][:, ia:ib + 1].mean(1) @ np.ones(NENS) / NENS))
        wm_da.append((DA['tws']['basin'][:, ia:ib + 1].mean(1) @ np.ones(NENS) / NENS))
        obs_b.append(Y[i] @ w); tmid.append(epochs[i])
    wm_ol, wm_da, obs_b = np.array(wm_ol), np.array(wm_da), np.array(obs_b)
    wm_ol -= wm_ol.mean(); wm_da -= wm_da.mean(); obs_b -= obs_b.mean()
    years = np.array([d.year for d in tmid])
    report['peaks'] = {}
    for y in np.unique(years):
        sel = years == y
        if sel.sum() < 40:
            continue
        r = dict(ol_max=wm_ol[sel].max(), da_max=wm_da[sel].max(), gr_max=obs_b[sel].max(),
                 ol_min=wm_ol[sel].min(), da_min=wm_da[sel].min(), gr_min=obs_b[sel].min())
        report['peaks'][int(y)] = {k: float(x) for k, x in r.items()}
        lines.append('   %4d | %8.0f %8.0f %8.0f | %8.0f %8.0f %8.0f | %8.0f %8.0f %8.0f'
                     % (y, r['ol_max'], r['da_max'], r['gr_max'], r['ol_min'], r['da_min'], r['gr_min'],
                        r['ol_max'] - r['ol_min'], r['da_max'] - r['da_min'], r['gr_max'] - r['gr_min']))
    resid_ol, resid_da = obs_b - wm_ol, obs_b - wm_da
    lines += ['', '   basin: RMS(obs-OL) %.1f mm, RMS(obs-DA) %.1f mm, r(OL) %.3f, r(DA) %.3f'
              % (np.sqrt((resid_ol ** 2).mean()), np.sqrt((resid_da ** 2).mean()),
                 np.corrcoef(obs_b, wm_ol)[0, 1], np.corrcoef(obs_b, wm_da)[0, 1])]
    report['basin'] = dict(rms_ol=float(np.sqrt((resid_ol ** 2).mean())), rms_da=float(np.sqrt((resid_da ** 2).mean())))

    # ------------------------------------------------------------------ E. figure: increments per storage
    show = [v for v in ['groundwstor', 'soilmoist', 'riverstor', 'tws'] if v in J]
    fig, axes = plt.subplots(len(show), 1, figsize=(12, 2.6 * len(show)), sharex=True)
    for ax, v in zip(np.atleast_1d(axes), show):
        inc = J[v][:, starts - 1]                                  # (NENS, nwin)
        for m in range(NENS):
            ax.plot(days[starts], inc[m], color='0.6', lw=0.7)
        ax.plot(days[starts], inc.mean(0), color='tab:red', lw=1.2, label='ensemble mean')
        ax.axhline(0, color='k', lw=0.8)
        ax.set_ylabel('%s\nincrement [mm]' % v)
        lim = np.percentile(np.abs(inc), 99.5)
        ax.set_ylim(-1.2 * lim, 1.2 * lim)
    np.atleast_1d(axes)[0].set_title('Basin-mean increments per window  [%s]  (grey: members, red: ensemble mean)' % TAG)
    np.atleast_1d(axes)[0].legend(loc='upper right', frameon=False)
    fig.tight_layout()
    fig.savefig(Path(OUT) / ('da_increments_%s.png' % TAG), dpi=150)

    # ------------------------------------------------------------------ output
    txt = '\n'.join(lines)
    if verbose:
        print(txt)
    (Path(OUT) / ('da_increment_diagnosis_%s.txt' % TAG)).write_text(txt)
    json.dump(report, open(Path(OUT) / ('da_increment_diagnosis_%s.json' % TAG), 'w'), indent=1)
    if verbose:
        print('\nwritten: %s' % (Path(OUT) / ('da_increment_diagnosis_%s.{txt,json}' % TAG)))
    plt.close(fig)
    return report


def demo1():
    from src_DA.configure_DA import config_DA
    cfg = config_DA.loadjson('/media/user/My Book/Fan/PyGLDA_v2/settings/demo_3/DA_setting.json').process()
    res_dir = Path(cfg.basic.res_permanent) / cfg.basic.case
    obs_file = Path(cfg.obs.dir) / ('%s_obs_GRACE.hdf5' % cfg.basic.basin)
    out_dir = Path(cfg.basic.res_permanent) / cfg.basic.case / 'figures'
    run_diagnosis(res_dir=res_dir, obs_file=obs_file, out_dir=out_dir, tag=cfg.basic.case, nens=cfg.basic.ensemble,
                  start=dt.date(2002, 1, 1))


if __name__ == '__main__':
    demo1()

