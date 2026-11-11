"""
Reservoirs in the DA state during their filling years (DA review 280 Part B, Cowork 10 Nov 2026).

Why
    With reservoirs subtracted from GRACE (all runs so far), a reservoir that fills in reality faster or further than
    in WaterGAP leaves a positive GRACE residual that the filter can only give to the land around it; a reservoir that
    WaterGAP fills while GRACE sees less drains the land. Boguchany (unit 662, 2013-): land DA-OL trend -14.3 mm/yr
    while the OL land trend is -1.3. Review 280-C found 21 such units.

What it does
    "reservoir_state" in the method block of the run copy of DA_setting.json switches it on. The reservoir storage
    (reservoirstor) must then be in the DA state ("states" of the run copy), with its bounds 0 .. capacity (review 260,
    src_DA.bounds). For every analysis window only the reservoir cells that are ACTIVE in that window (listed in the
    schedule with start <= last day of the window <= end) are updated. Every other reservoir cell is treated exactly
    as before, i.e. as subtracted from GRACE:
        obs_n  <- obs_n - H x_n(inactive reservoir rows)       (each member its own window-mean storage)
        x_n(inactive reservoir rows) <- 0 for the analysis, pinned (LB = UB = 0), and given back unchanged afterwards
    so the run is identical to the subtraction runs wherever no reservoir is active (checked by a unit test below).
    The increment that would hit a reservoir bound goes to the other free storages of the sub-basin by the
    bound-aware partition; with "bound_overflow": {"exclude": ["groundwstor"]} (as in LT9RL0) it does not go into
    groundwater (the same rule as for lakes). WaterGAP's release logic is not touched: the daily reservoir balance
    continues from the analysed storage.

Schedule (CSV, built offline by build_schedule from the review 280 B1 inventory and the OL storage series)
    columns name, lat, lon, unit, capacity_km3, start, end, rule; one row per reservoir outflow cell
    (WaterGAP keeps the storage of a GRanD reservoir in its outflow cell). Dates YYYY-MM-DD.
    Filling rule (default): start = 1 January of the commissioning year (not before the run start); end = the
    earlier of
        (a) the first month the OL storage reaches full_frac (0.9) of capacity, and
        (b) the plateau: the first month m at least 12 months after start where the OL storage holds at least
            plateau_min_frac (0.2) of capacity and its 12-month mean grows by less than plateau_frac (0.05) of
            capacity over the following 12 months
    plus hold_months (12). (b) is needed because WaterGAP never fills Three Gorges, Bakun and Sainte-Marguerite-3 to
    90 % (review 280 B1). Without an OL series (e.g. GERD 2020-, after the OL record) end = start + max_fill_months
    (60). Reservoirs listed in whole_period are active over the whole run (Kariba / Akosombo / Nasser test).

Method keys ("reservoir_state" in the method block):
    {"schedule": "<path>.csv",             required
     "log": true}                          optional, default true: per window and active reservoir cell the ensemble-
                                           mean forecast and analysis storage, the increment, the members at a bound
                                           -> <DA_output_temp_dir>/<case>/reservoir_state_log.csv (main rank)
Absent / null = off (bit-identical to the previous code).

Hooks (three lines in src_DA.EnKF_localized): build in __init__, before_update / pin_bounds / after_update in
update(). Nothing else of the filter, the model or WaterGAP changes.
"""
import csv
from datetime import date, datetime
from pathlib import Path

import numpy as np

RES = 'reservoirstor'
GRID_RES = 0.5


# ---------------------------------------------------------------------------------------------------- grid helpers
def latlon_to_rowcol(lat, lon, res=GRID_RES):
    """WaterGAP / PyGLDA global grid: row 0 = 89.75 N, col 0 = -179.75 E (src_auxiliary.shp2mask.load_mask)"""
    row = int(round((90.0 - res / 2 - float(lat)) / res))
    col = int(round((float(lon) + 180.0 - res / 2) / res))
    return row, col


def state_cell_index(global_2d, rows, cols):
    """index of grid cells (rows, cols) in the state vector's cell order (basin cells in row-major order of the
    global grid); -1 where the cell is not in the basin mask"""
    g = np.asarray(global_2d).astype(bool)
    flat = np.flatnonzero(g.ravel())
    lin = np.asarray(rows) * g.shape[1] + np.asarray(cols)
    k = np.searchsorted(flat, lin)
    ok = (k < flat.size) & (flat[np.minimum(k, flat.size - 1)] == lin)
    return np.where(ok, k, -1)


def _d(s):
    if s is None or str(s).strip() in ('', 'nan', 'None'):
        return None
    return datetime.strptime(str(s).strip()[:10], '%Y-%m-%d').date()


def _add_months(d: date, n: int) -> date:
    y, m = divmod(d.year * 12 + d.month - 1 + n, 12)
    return date(y, m + 1, 1)


# ---------------------------------------------------------------------------------------------------- runtime part
class ReservoirSchedule:

    def __init__(self, schedule_csv, names, n_cell, local_mask, log_path=None):
        names = list(names)
        if RES not in names:
            raise ValueError('reservoir_state: %s must be in the DA state (states of the run copy), state = %s'
                             % (RES, names))
        self.nvar = len(names)
        self.i_res = names.index(RES)
        self.n_cell = int(n_cell)
        self.path = str(schedule_csv)
        rows = list(csv.DictReader(open(self.path)))
        if not rows:
            raise ValueError('reservoir_state: schedule %s is empty' % self.path)
        rc = [latlon_to_rowcol(r['lat'], r['lon']) for r in rows]
        idx = state_cell_index(local_mask['global_2d'], [a for a, _ in rc], [b for _, b in rc])
        self.entries = []
        n_out = 0
        for r, k in zip(rows, idx):
            if k < 0:
                n_out += 1                                        # outside this run's basin mask (regional runs)
                continue
            s, e = _d(r.get('start')), _d(r.get('end'))
            if s is None:
                raise ValueError('reservoir_state: %s has no start date' % r.get('name'))
            self.entries.append(dict(name=r.get('name', ''), cell=int(k), start=s, end=e, lat=float(r['lat']),
                                     lon=float(r['lon']), unit=r.get('unit', ''), rule=r.get('rule', '')))
        self.n_outside = n_out
        self._cells = np.array([e['cell'] for e in self.entries], dtype=int)
        self._res_rows_all = np.arange(self.i_res, self.n_cell * self.nvar, self.nvar)
        self.log_path = log_path
        self._log = []
        self.stats = dict(n_windows=0, n_windows_with_active=0, max_active=0, active_without_capacity=0)
        self._date = None
        print('reservoir_state: %d reservoir cells in the schedule %s inside the DA mask (%d outside); '
              'state storage %s at position %d of %d'
              % (len(self.entries), self.path, n_out, RES, self.i_res, self.nvar))

    @classmethod
    def from_setting(cls, cfg, names, n_cell, local_mask, out_dir=None):
        """None when "reservoir_state" is absent / null / false"""
        if not cfg:
            return None
        cfg = dict(cfg)
        if 'schedule' not in cfg:
            raise ValueError('reservoir_state needs "schedule" (CSV built by src_DA.reservoir_state.build_schedule)')
        log = None
        if cfg.get('log', True) and out_dir is not None:
            log = Path(out_dir) / 'reservoir_state_log.csv'
        return cls(cfg['schedule'], names, n_cell, local_mask, log_path=log)

    # ------------------------------------------------------------------ per window
    def active_cells(self, day):
        """state cell indices of the reservoirs active on `day` (the last day of the window)"""
        d = _d(day) if not isinstance(day, date) else day
        act = [e['cell'] for e in self.entries if e['start'] <= d and (e['end'] is None or d <= e['end'])]
        return np.unique(np.array(act, dtype=int))

    def inactive_rows(self, day, n_state):
        """bool (n_state,): the reservoir rows that are NOT updated in this window (all reservoir rows of the state
        except those of the active cells)"""
        if n_state != self.n_cell * self.nvar:
            raise ValueError('reservoir_state: state has %d elements, expected %d cells x %d storages'
                             % (n_state, self.n_cell, self.nvar))
        m = np.zeros(n_state, dtype=bool)
        m[self._res_rows_all] = True
        act = self.active_cells(day)
        if act.size:
            m[act * self.nvar + self.i_res] = False
        return m, act

    def before_update(self, f, obs, ens_states, day):
        """
        f: the filter (EnKF_localized; f._DM is the observation operator H applied to n_state x N).
        Returns (obs', X', ctx): the inactive reservoir storages moved from the state to the observation side, i.e.
        exactly the subtraction of the previous runs for these cells. ens_states is not modified (run_mpi takes the
        increment as update - ens_states).
        """
        X = np.asarray(ens_states, dtype=float)
        m, act = self.inactive_rows(day, X.shape[0])
        Xi = np.where(m[:, None], X, 0.0)
        obs2 = np.asarray(obs, dtype=float) - np.asarray(f._DM(states=Xi))
        X2 = np.where(m[:, None], 0.0, X)
        for nm in ('_ens_min', '_ens_max'):                       # window extremes of the moved rows: 0 as well
            a = getattr(f, nm, None)
            if a is not None and np.shape(a) == X.shape:
                setattr(f, nm, np.where(m[:, None], 0.0, a))
        self._date = str(day)[:10]
        st = self.stats
        st['n_windows'] += 1
        st['n_windows_with_active'] += int(act.size > 0)
        st['max_active'] = max(st['max_active'], int(act.size))
        return obs2, X2, dict(mask=m, active=act, X=X)

    def pin_bounds(self, bounds, ctx):
        """after bounds.set_window: the moved rows get LB = UB = 0, so neither the partition nor the inflation can
        place water there; active cells whose current capacity is 0 (not yet commissioned in WaterGAP) are counted"""
        m = ctx['mask']
        bounds.LB[m] = 0.0
        bounds.UB[m] = 0.0
        act = ctx['active']
        if act.size:
            no_cap = np.isfinite(bounds.ub[act * self.nvar + self.i_res]) & (bounds.ub[act * self.nvar + self.i_res] <= 0)
            self.stats['active_without_capacity'] += int(no_cap.sum())
            ctx['no_capacity'] = act[no_cap]

    def after_update(self, Xa, ctx, bounds=None):
        """the moved rows get their forecast back (zero increment); log the active cells"""
        X = ctx['X']
        m = ctx['mask']
        out = np.where(m[:, None], X, np.asarray(Xa, dtype=float))
        if self.log_path is not None and ctx['active'].size:
            for c in ctx['active']:
                r = c * self.nvar + self.i_res
                e = next(x for x in self.entries if x['cell'] == c)
                ub = None if bounds is None else float(bounds.ub[r])
                n_lo = int(np.sum(out[r] <= 1e-9))
                n_hi = 0 if ub is None or not np.isfinite(ub) else int(np.sum(out[r] >= ub - 1e-6))
                self._log.append((self._date, e['name'], e['unit'], int(c), float(X[r].mean()), float(out[r].mean()),
                                  float(out[r].mean() - X[r].mean()), float(out[r].std()), ub, n_lo, n_hi))
        return out

    def save_log(self):
        if self.log_path is None or not self._log:
            return
        p = Path(self.log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['window_end', 'name', 'unit', 'state_cell', 'forecast_mean_mm', 'analysis_mean_mm',
                        'increment_mm', 'analysis_spread_mm', 'capacity_mm', 'members_at_zero', 'members_at_capacity'])
            w.writerows(self._log)
        print('reservoir_state log written: %s (%d rows)' % (p, len(self._log)))

    def summary(self):
        return dict(self.stats, schedule=self.path, n_reservoir_cells=len(self.entries), n_outside_mask=self.n_outside)


# ---------------------------------------------------------------------------------------------------- offline part
def filling_end(series, months, capacity, start, full_frac=0.9, plateau_frac=0.05, plateau_min_frac=0.2,
                min_months=12):
    """
    series   : OL storage (km3) per month, NaN where missing;  months: list of date (first of month)
    Returns (end_of_filling, rule): the first month that satisfies (a) storage >= full_frac x capacity or (b) the
    plateau (see module doc), whichever comes first; (None, 'no OL series') when the series is empty after start.
    """
    s = np.asarray(series, dtype=float)
    t = [i for i, m in enumerate(months) if m >= start]
    if not t or not np.isfinite(s[t]).any():
        return None, 'no OL series'
    i0 = t[0]
    full = [i for i in t if np.isfinite(s[i]) and s[i] >= full_frac * capacity]
    i_full = full[0] if full else None
    m12 = np.full(s.size, np.nan)                       # 12-month mean ending at month i
    for i in range(11, s.size):
        w = s[i - 11:i + 1]
        if np.isfinite(w).all():
            m12[i] = w.mean()
    i_pl = None
    for i in t:
        if i - i0 < min_months or i + 12 >= s.size:
            continue
        if not (np.isfinite(m12[i]) and np.isfinite(m12[i + 12]) and np.isfinite(s[i])):
            continue
        if s[i] >= plateau_min_frac * capacity and (m12[i + 12] - m12[i]) < plateau_frac * capacity:
            i_pl = i
            break
    cand = [(i, r) for i, r in ((i_full, 'full %.0f%%' % (100 * full_frac)), (i_pl, 'plateau')) if i is not None]
    if not cand:
        return None, 'still filling at the end of the OL series'
    i, r = min(cand)
    return months[i], r


def build_schedule(inventory_csv, storage_csv, out_csv, run_start='2002-01-01', run_end='2024-12-31',
                   min_startyear=2000, full_frac=0.9, hold_months=12, plateau_frac=0.05, plateau_min_frac=0.2,
                   max_fill_months=60, whole_period=(), min_capacity_km3=0.0, overrides=None):
    """
    inventory_csv : review 280 B1 reservoir_inventory_2002_2024.csv (name_by_location, lat, lon, T9_unit, startyear,
                    capacity_km3, ...)
    storage_csv   : review 280 B1 reservoir_OL_storage_monthly_km3.csv (first column YYYY-MM, one column per
                    reservoir named <name>_<lat>_<lon>)
    whole_period  : names (or 'lat_lon' keys) active over the whole run (old reservoirs with a trend error:
                    Kariba / Akosombo / Nasser test)
    overrides     : {name: {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}} replaces the rule for these reservoirs (e.g.
                    Three Gorges, whose real staged filling to 175 m ran until 2010 while WaterGAP's OL storage
                    plateaus in 2004)
    Writes out_csv (see module doc) and returns its rows. Reservoirs commissioned before min_startyear and not in
    whole_period are not written (they stay subtracted).
    """
    rs, re_ = _d(run_start), _d(run_end)
    inv = list(csv.DictReader(open(inventory_csv)))
    with open(storage_csv) as fh:
        rd = csv.reader(fh)
        head = next(rd)
        data = [r for r in rd if r and r[0].strip()]
    months = [datetime.strptime(r[0][:7], '%Y-%m').date() for r in data]
    cols = {}
    for j, h in enumerate(head[1:], start=1):
        parts = h.rsplit('_', 2)
        if len(parts) == 3:
            try:
                cols[(round(float(parts[1]), 2), round(float(parts[2]), 2))] = j
            except ValueError:
                pass
    wp = set(whole_period)
    overrides = dict(overrides or {})
    out = []
    for r in inv:
        lat, lon = float(r['lat']), float(r['lon'])
        name = (r.get('name_by_location') or '').strip() or 'n.i._%s_%s' % (r['lat'], r['lon'])
        cap = float(r['capacity_km3'])
        sy = int(float(r['startyear']))
        key = '%s_%s' % (r['lat'], r['lon'])
        if cap < min_capacity_km3:
            continue
        if name in wp or key in wp:
            out.append(dict(name=name, lat=lat, lon=lon, unit=r.get('T9_unit', ''), capacity_km3=cap,
                            start=rs.isoformat(), end=re_.isoformat(), rule='whole period'))
            continue
        if name in overrides or key in overrides:
            o = overrides.get(name, overrides.get(key))
            st = _d(o.get('start')) or max(date(sy, 1, 1), rs)
            en = _d(o.get('end')) or re_
            out.append(dict(name=name, lat=lat, lon=lon, unit=r.get('T9_unit', ''), capacity_km3=cap,
                            start=max(st, rs).isoformat(), end=min(en, re_).isoformat(), rule='override'))
            continue
        if sy < min_startyear:
            continue
        start = max(date(sy, 1, 1), rs)
        j = cols.get((round(lat, 2), round(lon, 2)))
        ser = [float(x[j]) if j is not None and x[j] not in ('', 'nan') else np.nan for x in data]
        if date(sy, 1, 1) > months[-1]:
            ser = [np.nan] * len(months)
        end_fill, rule = filling_end(ser, months, cap, start, full_frac, plateau_frac, plateau_min_frac)
        if end_fill is None:
            end = _add_months(start, max_fill_months)
            rule = '%s: start + %d months' % (rule, max_fill_months)
        else:
            end = _add_months(end_fill, hold_months)
            rule = '%s on %s + %d months' % (rule, end_fill.isoformat()[:7], hold_months)
        end = min(end, re_)
        if end < start:
            continue
        out.append(dict(name=name, lat=lat, lon=lon, unit=r.get('T9_unit', ''), capacity_km3=cap,
                        start=start.isoformat(), end=end.isoformat(), rule=rule))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['name', 'lat', 'lon', 'unit', 'capacity_km3', 'start', 'end', 'rule'])
        w.writeheader()
        w.writerows(out)
    print('reservoir schedule: %d reservoirs -> %s' % (len(out), out_csv))
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='build the reservoir filling schedule for "reservoir_state"')
    ap.add_argument('inventory_csv')
    ap.add_argument('storage_csv')
    ap.add_argument('out_csv')
    ap.add_argument('--run-start', default='2002-01-01')
    ap.add_argument('--run-end', default='2024-12-31')
    ap.add_argument('--whole-period', nargs='*', default=[])
    ap.add_argument('--overrides', default=None, help='JSON file {name: {"start": .., "end": ..}}')
    a = ap.parse_args()
    import json
    ov = json.load(open(a.overrides)) if a.overrides else None
    build_schedule(a.inventory_csv, a.storage_csv, a.out_csv, a.run_start, a.run_end, whole_period=a.whole_period,
                   overrides=ov)
