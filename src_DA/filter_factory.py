"""
Builds the localized EnKF from the "method" block of DA_setting.json (used by src_FlowControl.DA_GRACE).

    "method": {
        "fusion_method": "EnKF_localized" | "LETKF",     (LETKF: src_DA.LETKF, same components, deterministic
                                                          analysis; optional "letkf": {"min_taper": 0.0})
        "localization": {"kind": "none" | "block" | "gaussian", "length_km": 300, "cutoff": 2.0},
        "inflation": {"kind": "none" | "multiplicative" | "rtps" | "additive" | "adaptive_additive", ...},
        "increment_partition": {"kind": "enkf" | "non_negative"},
        "obs_error_inflation": 0.5                       (factor on the error variance of every sub-basin; 1 = off)
        "obs_error_correlation": "full" | "diagonal",
        "obs_perturbation_centering": false | true,
        "soil_upper_bound": true | false        (soil <= smax inside the bound-aware partition; default true)
        "soil_daily_bounds": false | true       (every day of the window within the soil bounds; src_DA.EnKF)
        "increment_release": {"storages": ["riverstor"], "days": 10}   (src_DA.EnKF; default off)
        "snow_bounds": {"lower_factor": 0.5,                          snow >= f x forecast
                        "upper_factor": 2.0, "upper_offset_mm": 20,   snow <= f x forecast + o
                        "envelope_factor": 1.5, "envelope_offset_mm": 10}
                                                 snow <= f x the open loop's largest snow of the calendar month + o
                       (null / false switches a limit off; without envelope_factor the envelope cap is off;
                        read by src_DA.configure_DA.snow_bounds)
    }

Only one inflation scheme can be active ("inflation" block).
"""
from src_DA.localization import make_localization
from src_DA.inflation import make_inflation
from src_DA.partition import make_partition
from src_DA.configure_DA import snow_bounds


def inflation_config(method):
    """the "inflation" block of the method settings"""
    return dict(getattr(method, 'inflation', None) or {'kind': 'none'})


def fusion_method_name(method):
    """"EnKF_localized" | "LETKF" | ... from the method block (enum member or string)"""
    fm = getattr(method, 'fusion_method', 'EnKF_localized')
    return str(getattr(fm, 'name', fm))


def build_filter(configDA, model, obs, sv, sv_excluded):
    """the filter chosen by "fusion_method": EnKF_localized (stochastic EnKF) or LETKF (deterministic
    ensemble-transform analysis, src_DA.LETKF), both assembled from the same components"""
    m = configDA.method
    kw = dict(DA_setting=configDA, model=model, obs=obs, sv=sv, sv_excluded=sv_excluded,
              localization=make_localization(getattr(m, 'localization', None)),
              inflation=make_inflation(inflation_config(m)),
              partition=make_partition(getattr(m, 'increment_partition', None)),
              obs_error_inflation=getattr(m, 'obs_error_inflation', None),
              obs_error_correlation=getattr(m, 'obs_error_correlation', 'full'),
              obs_perturbation_centering=getattr(m, 'obs_perturbation_centering', False),
              soil_upper_bound=getattr(m, 'soil_upper_bound', True),
              snow_bounds=snow_bounds(m))
    name = fusion_method_name(m)
    if name == 'LETKF':
        from src_DA.LETKF import LETKF_localized
        return LETKF_localized(letkf=getattr(m, 'letkf', None), **kw)
    if name != 'EnKF_localized':
        raise ValueError('filter_factory: fusion_method %s is not built here (EnKF_localized | LETKF)' % name)
    from src_DA.EnKF_localized import EnKF_localized
    return EnKF_localized(**kw)
