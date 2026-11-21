"""
Optional scaling of the per-window constants of the filter by the length of the analysis window
("method": {"window_scaling": {"reference_days": 30.44}} in DA_setting.json; absent = off, every code path unchanged).

Several settings act once per analysis window and were tuned for monthly windows: the adaptive additive inflation
(weight_current, bias_memory, min_sigma) and the seasonal split of the non-negative partition (climatology counts /
memory_years, lowpass and interannual tau_months). With 5-day windows they would act about six times per month. With
window_scaling on, a window of n days counts as r = n / reference_days "reference windows":
    rate per window      a  ->  1 - (1 - a)^r         (weight_current, bias_memory, 1/tau_months, 1/memory_years)
    variance floor       min_sigma -> min_sigma * sqrt(r)   (white additive noise accumulates with the number of windows)
    climatology count    += r instead of += 1           (counts are in reference windows, i.e. months)
so that, per unit of time, the filter forgets and adds noise as with monthly windows. The window length n is the number
of days of the observation window (EnKF.run_mpi sets f._window_days).
"""


def window_ratio(f):
    """r = window days / reference_days, or None when window_scaling is off (or the window length is unknown)"""
    cfg = getattr(getattr(f, 'DA_setting', None) and f.DA_setting.method, 'window_scaling', None)
    if not cfg:
        return None
    days = getattr(f, '_window_days', None)
    if not days:
        return None
    return float(days) / float(cfg.get('reference_days', 30.44))


def per_window(rate, r):
    """rate per reference window -> rate per window of r reference windows"""
    return 1.0 - (1.0 - float(rate)) ** r


def release_days(f, days, window_days):
    """increment_release days, capped at the length of the window just analysed when window_scaling is on (the
    increment is released over the first days of the next window; with 5-day windows a 10-day release would still be
    running when the next analysis comes). Off: days unchanged."""
    cfg = getattr(getattr(f, 'DA_setting', None) and f.DA_setting.method, 'window_scaling', None)
    if not cfg or not window_days:
        return days
    return max(1, min(int(days), int(window_days)))
