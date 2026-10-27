"""
Removal of co- and post-seismic gravity changes of great earthquakes from gridded GRACE / GRACE-FO mass anomalies
(CSR mascons), before the unit averaging of the observations (applied to the CSR mascons only, not to an SH product).

Method (as Tanaka & Heki 2014 GRL and GravIS, Boergens et al. 2025 ESSD): for every grid cell within `cap_km` of an
event's rupture trace, the full monthly record is fitted by least squares with
    bias + trend + acceleration + annual + semi-annual
    + sum over events e touching the cell:  S_e H(t - t_e) + sum_k P_ek H(t - t_e) (1 - exp(-(t - t_e) / tau_ek))
with the time constants tau_ek fixed per event (from GNSS / GRACE studies). The fitted event terms (step and
post-seismic relaxation) are subtracted; a cosine taper over the outer `taper_km` of the cap avoids an edge.
Hydrology is kept in the other terms and in the residual; the fit uses the WHOLE record (2002 to the end of the
file), which separates a step from interannual hydrology much better than a 5-year window. The size of spurious
corrections is measured with placebo events (fake dates / places; see `significance` in fit_and_correct).

Settings, DA_setting.json -> obs.GRACE:
    "earthquake_correction": {"cap_km": 1000, "taper_km": 200, "events": ["sumatra2004", "nias2005", "maule2010",
                                                                          "tohoku2011"]}
absent / null = off (no correction applied).
"""
import numpy as np

EARTH_R = 6371.0

'''rupture traces (approximate, from the USGS finite-fault extents), event time (decimal year), post-seismic time
constants [yr] (Tanaka & Heki 2014: Sumatra 0.25 / 0.9, Maule 0.15 / ~1.0, Tohoku 0.14 / 0.4); Nias: step only
(three months after Sumatra, inside its footprint)'''
EVENTS = {
    'sumatra2004': dict(date=(2004, 12, 26), mw=9.1, taus=(0.25, 0.9),
                        trace=[(2.5, 95.5), (5.0, 94.3), (7.0, 93.5), (10.0, 92.6), (14.0, 93.0)]),
    'nias2005': dict(date=(2005, 3, 28), mw=8.6, taus=(),
                     trace=[(0.0, 98.4), (1.2, 97.6), (2.6, 96.4)]),
    'maule2010': dict(date=(2010, 2, 27), mw=8.8, taus=(0.15, 1.0),
                      trace=[(-38.5, -74.4), (-36.0, -73.2), (-33.5, -72.2)]),
    'tohoku2011': dict(date=(2011, 3, 11), mw=9.0, taus=(0.14, 0.4),
                       trace=[(35.5, 141.2), (37.0, 142.4), (38.5, 143.2), (40.0, 143.6)]),
}


def decimal_year(y, m, d):
    import datetime as dt
    t = dt.date(y, m, d)
    a, b = dt.date(y, 1, 1), dt.date(y + 1, 1, 1)
    return y + (t - a).days / (b - a).days


def _xyz(lat, lon):
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def distance_to_trace(lat, lon, trace, n_sub=50):
    """great-circle distance [km] of points (lat, lon arrays) to a polyline (densified)"""
    pts = []
    for (a1, o1), (a2, o2) in zip(trace[:-1], trace[1:]):
        f = np.linspace(0, 1, n_sub, endpoint=False)
        pts.append(np.column_stack([a1 + f * (a2 - a1), o1 + f * (o2 - o1)]))
    pts.append(np.array([trace[-1]]))
    pts = np.vstack(pts)
    from scipy.spatial import cKDTree
    chord, _ = cKDTree(_xyz(pts[:, 0], pts[:, 1])).query(_xyz(np.ravel(lat), np.ravel(lon)))
    return (2 * EARTH_R * np.arcsin(np.clip(chord / 2, 0, 1))).reshape(np.shape(lat))


def _event_columns(t, ev, t_e=None):
    te = decimal_year(*ev['date']) if t_e is None else t_e
    H = (t > te).astype(float)
    cols = [H] + [H * (1 - np.exp(-np.clip(t - te, 0, None) / tau)) for tau in ev['taus']]
    return cols


def fit_and_correct(data, t, lat, lon, events=None, cap_km=1000.0, taper_km=200.0, event_dates=None, report=True,
                    significance=None):
    """
    data : (n_t, n_lat, n_lon) monthly mass anomalies (any unit; NaN allowed only for whole cells);
    t    : (n_t,) decimal years of the months;  lat, lon : (n_lat, n_lon) cell centres.
    events : names in EVENTS (default all); event_dates : optional {name: decimal year} (placebo tests).
    significance : {"n_placebo": 10, "z_lo": 2.0, "z_hi": 4.0} or None. Per cell and event, the step is refitted at
        n_placebo fake dates (same model, the event date replaced; dates spread over the record, at least 1.5 yr away
        from every real event of the cell); sigma_p = robust spread (1.4826 x MAD) of those placebo steps measures how
        large a "step" the cell's own hydrology produces. The event terms are applied with the weight
        clip((|S| / sigma_p - z_lo) / (z_hi - z_lo), 0, 1): nothing below z_lo, everything above z_hi.
        Needed: without it, free per-cell fits in wet regions remove real water (in a placebo test with traces
        moved into the Amazon the corrections were median 59 / p90 191 mm two years after the fake event).
    Returns (corrected copy, correction (n_t, n_lat, n_lon), info dict).
    """
    names = list(EVENTS) if events is None else list(events)
    out = np.array(data, dtype=float, copy=True)
    corr = np.zeros_like(out)
    t = np.asarray(t, dtype=float)
    tc = t - t.mean()
    base = [np.ones_like(t), tc, tc ** 2, np.cos(2 * np.pi * t), np.sin(2 * np.pi * t),
            np.cos(4 * np.pi * t), np.sin(4 * np.pi * t)]
    dist = {e: distance_to_trace(lat, lon, EVENTS[e]['trace']) for e in names}
    near = np.zeros(lat.shape, dtype=bool)
    for e in names:
        near |= dist[e] <= cap_km
    info = dict(events=names, cap_km=cap_km, taper_km=taper_km, n_cells=int(near.sum()), step_by_event={})
    ii, jj = np.nonzero(near & np.isfinite(out).all(0))
    # group cells by the set of events that touch them (same design matrix)
    keys = np.array([tuple(bool(dist[e][i, j] <= cap_km) for e in names) for i, j in zip(ii, jj)])
    for key in {tuple(k) for k in keys}:
        sel = np.all(keys == np.array(key), axis=1)
        evs = [e for e, on in zip(names, key) if on]
        ev_cols, owner = [], []
        for e in evs:
            te = None if event_dates is None else event_dates.get(e)
            c = _event_columns(t, EVENTS[e], te)
            ev_cols += c
            owner += [e] * len(c)
        X = np.column_stack(base + ev_cols)
        Y = out[:, ii[sel], jj[sel]]
        b = np.linalg.lstsq(X, Y, rcond=None)[0]
        nb = len(base)
        Xe = X[:, nb:]
        zw = {}
        if significance:
            zw = _placebo_weights(Y, t, base, evs, b, nb, owner, event_dates, significance, info)
        for k, e in enumerate(owner):
            w = np.clip((cap_km - dist[e][ii[sel], jj[sel]]) / max(taper_km, 1e-9), 0, 1)
            w = 0.5 - 0.5 * np.cos(np.pi * w)                                 # 0 at the cap edge, 1 inside
            if e in zw:
                w = w * zw[e]
            term = Xe[:, [k]] * b[nb + k][None, :] * w[None, :]
            corr[:, ii[sel], jj[sel]] += term
            if k == owner.index(e):                                           # the step column of event e
                info['step_by_event'].setdefault(e, []).append(b[nb + k] * w)
    out -= corr
    for e in list(info['step_by_event']):
        s = np.concatenate(info['step_by_event'][e])
        info['step_by_event'][e] = dict(n=int(s.size), median=float(np.median(s)), p05=float(np.percentile(s, 5)),
                                        p95=float(np.percentile(s, 95)), max_abs=float(np.abs(s).max()))
    if report:
        print('earthquake correction: %d cells within %s km of %s; steps %s' % (info['n_cells'], cap_km, names,
                                                                               info['step_by_event']))
    return out, corr, info


def _placebo_weights(Y, t, base, evs, b, nb, owner, event_dates, sig, info):
    """weights per event (cells of this group) from placebo steps at fake dates, see fit_and_correct"""
    n_p = int(sig.get('n_placebo', 10))
    z_lo, z_hi = float(sig.get('z_lo', 2.0)), float(sig.get('z_hi', 4.0))
    real = {e: (decimal_year(*EVENTS[e]['date']) if event_dates is None or e not in event_dates else event_dates[e])
            for e in evs}
    lo, hi = t.min() + 1.0, t.max() - 2.5
    cand = np.linspace(lo, hi, 4 * n_p)
    cand = [c for c in cand if all(abs(c - r) >= 1.5 for r in real.values())]
    if len(cand) > n_p:
        cand = [cand[i] for i in np.linspace(0, len(cand) - 1, n_p).astype(int)]
    out = {}
    for e in evs:
        k_step = owner.index(e)
        steps = []
        for c in cand:
            cols = []
            for e2 in evs:
                cols += _event_columns(t, EVENTS[e2], c if e2 == e else real[e2])
            Xp = np.column_stack(base + cols)
            bp = np.linalg.lstsq(Xp, Y, rcond=None)[0]
            steps.append(bp[nb + k_step])
        steps = np.array(steps)
        sp = 1.4826 * np.median(np.abs(steps - np.median(steps, 0)), 0)
        z = np.abs(b[nb + k_step]) / np.maximum(sp, 1e-6)
        out[e] = np.clip((z - z_lo) / (z_hi - z_lo), 0.0, 1.0)
        st = info.setdefault('significance', {}).setdefault(e, dict(n_cells=0, n_full=0, n_zero=0, n_placebo=len(cand)))
        st['n_cells'] += int(z.size)
        st['n_full'] += int((out[e] >= 1).sum())
        st['n_zero'] += int((out[e] <= 0).sum())
    return out
