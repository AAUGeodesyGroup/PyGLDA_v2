"""
Builds the localized EnKF from the "method" block of DA_setting.json (used by src_FlowControl.DA_GRACE).

    "method": {
        "fusion_method": "EnKF_localized",
        "localization": {"kind": "none" | "block" | "gaussian", "length_km": 300, "cutoff": 2.0},
        "inflation": {"kind": "none" | "multiplicative" | "rtps" | "additive" | "adaptive_additive", ...},
        "increment_partition": {"kind": "enkf" | "non_negative"},
        "obs_error_inflation": {},
        "obs_error_correlation": "full" | "diagonal",
        "obs_perturbation_centering": false | true,
        "soil_upper_bound": true | false        (soil <= smax inside the bound-aware partition; default true)
        "snow_upper_bound": {"factor": 2.0, "offset_mm": 20}  (snow <= factor x forecast + offset; false = none)
    }

Settings written before the restructuring (Oct 2026) are translated:
    "inflation": 1.6                         -> {"kind": "multiplicative", "factor": 1.6}
    "rtps_alpha": 0.7, "rtps_space": "obs"   -> {"kind": "rtps", "alpha": 0.7, "space": "obs"}
    "additive_inflation": {"groundwstor": 15, "swe": 5, "months": [..], "seed": 42}
                                             -> {"kind": "additive", "sigma": {...}, "months": [..], "seed": 42}
    "additive_inflation": {"mode": "adaptive", ...}  -> {"kind": "adaptive_additive", ...}
    "increment_partition": "non_negative"    -> {"kind": "non_negative"}
Only one inflation scheme can be active; a configuration with several stops with an error.
"""
from src_DA.localization import make_localization
from src_DA.inflation import make_inflation
from src_DA.partition import make_partition


def inflation_config(method):
    """the "inflation" block of the method settings, old flat keys translated"""
    new = getattr(method, 'inflation', None)
    if isinstance(new, dict):
        cfg = dict(new)
        legacy = []
    else:
        cfg = {'kind': 'none'}
        legacy = []
        factor = 1.0 if new is None else float(new)
        if factor != 1.0:
            legacy.append({'kind': 'multiplicative', 'factor': factor})
    alpha = float(getattr(method, 'rtps_alpha', 0.0) or 0.0)
    if alpha > 0:
        legacy.append({'kind': 'rtps', 'alpha': alpha, 'space': getattr(method, 'rtps_space', 'state')})
    add = dict(getattr(method, 'additive_inflation', None) or {})
    if add:
        mode = str(add.pop('mode', 'fixed')).lower()
        if mode == 'adaptive':
            legacy.append(dict(add, kind='adaptive_additive'))
        else:
            extra = {k: add.pop(k) for k in ('months', 'seed', 'exact_spread') if k in add}
            legacy.append(dict(extra, kind='additive', sigma=add))
    if legacy:
        if isinstance(new, dict) and str(new.get('kind', 'none')).lower() != 'none':
            raise ValueError('method: the "inflation" block and the old keys (inflation factor, rtps_alpha, '
                             'additive_inflation) are both set; keep only the "inflation" block')
        if len(legacy) > 1:
            raise ValueError('method: more than one inflation scheme is active (%s); choose one "inflation" block'
                             % [c['kind'] for c in legacy])
        cfg = legacy[0]
        print('filter_factory: old inflation keys translated to "inflation": %s' % cfg)
    return cfg


def build_filter(configDA, model, obs, sv, sv_excluded):
    from src_DA.EnKF_localized import EnKF_localized
    m = configDA.method
    return EnKF_localized(DA_setting=configDA, model=model, obs=obs, sv=sv, sv_excluded=sv_excluded,
                          localization=make_localization(getattr(m, 'localization', None)),
                          inflation=make_inflation(inflation_config(m)),
                          partition=make_partition(getattr(m, 'increment_partition', None)),
                          obs_error_inflation=getattr(m, 'obs_error_inflation', None),
                          obs_error_correlation=getattr(m, 'obs_error_correlation', 'full'),
                          obs_perturbation_centering=getattr(m, 'obs_perturbation_centering', False),
                          soil_upper_bound=getattr(m, 'soil_upper_bound', True),
                          snow_upper_bound=getattr(m, 'snow_upper_bound', None))
