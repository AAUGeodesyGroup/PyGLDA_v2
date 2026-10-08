"""
Per-cell envelope (minimum / maximum over time and ensemble members) of selected WaterGAP storages from the
collected open-loop results Res/<case>/OL/Ens_k/daily_output_<year>.nc. Used by src_DA.Threshold to bound
storages that have no physical capacity in the model (river storage): the analysis is kept within
[env_low * min, env_high * max] of what the model itself produced over the open-loop period.

Output: <Auxiliary_dir>/state_envelope_<basin>.nc with variables <var>_min and <var>_max on the (lat, lon) box
of the collected files (NaN outside the basin). Called from DA_GRACE.make_state_envelope() (automatically in RDA.DA_run); rerun it whenever the
open loop changes (new perturbation, new basin, new period).
"""
from pathlib import Path
import numpy as np
import warnings
import xarray as xr


def make_state_envelope(res_dir, case: str, ens: int, basin: str, out_dir,
                        variables=('riverstor', 'groundwstor'),
                        members=None, force: bool = False, verbose: bool = True):
    """
    res_dir   : configDA.basic.res_permanent
    ens       : number of perturbed members (files Ens_1 .. Ens_ens are used; Ens_0 = unperturbed is skipped
                unless members is given explicitly)
    variables : storages to process
    force     : recompute even if the output exists
    returns   : path of the written file (or the existing one)
    """
    out_fn = Path(out_dir) / ('state_envelope_%s.nc' % basin)
    if out_fn.exists() and not force:
        with xr.open_dataset(out_fn) as old:
            missing = [v for v in variables if ('%s_min' % v not in old) or ('%s_max' % v not in old)]
        if not missing:
            if verbose:
                print('state envelope exists: %s (use force=True to recompute)' % out_fn)
            return out_fn
        if verbose:
            print('state envelope %s lacks %s -> recomputed' % (out_fn, ','.join(missing)))

    members = list(range(1, ens + 1)) if members is None else list(members)
    vmin, vmax, lat, lon, n_files = {v: None for v in variables}, {v: None for v in variables}, None, None, 0

    for m in members:
        files = sorted((Path(res_dir) / case / 'OL' / ('Ens_%d' % m)).glob('daily_output_*.nc'))
        for fn in files:
            with xr.open_dataset(fn) as ds:
                if lat is None:
                    lat, lon = ds['lat'].values.copy(), ds['lon'].values.copy()
                for v in variables:
                    if v not in ds:
                        raise KeyError('%s not in %s' % (v, fn))
                    a = ds[v].values                             # (time, lat, lon), file dtype (no float64 copy)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN outside the basin
                        lo = np.nanmin(a, axis=0).astype(float)       # min / max are exact in the file dtype
                        hi = np.nanmax(a, axis=0).astype(float)
                    del a
                    vmin[v] = lo if vmin[v] is None else np.fmin(vmin[v], lo)
                    vmax[v] = hi if vmax[v] is None else np.fmax(vmax[v], hi)
            n_files += 1
        if verbose:
            print('state envelope: member %d, %d yearly files' % (m, len(files)))

    if n_files == 0:
        raise FileNotFoundError('no collected OL files under %s' % (Path(res_dir) / case / 'OL'))

    data = {}
    for v in variables:
        data['%s_min' % v] = (('lat', 'lon'), vmin[v])
        data['%s_max' % v] = (('lat', 'lon'), vmax[v])
    out = xr.Dataset(data, coords={'lat': lat, 'lon': lon},
                     attrs=dict(description='per-cell min/max over time and members of the open loop, mm',
                                case=case, members=str(members), n_files=n_files, variables=','.join(variables)))
    out_fn.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(out_fn)
    if verbose:
        for v in variables:
            print('  %-12s min %.1f .. %.1f mm, max %.1f .. %.1f mm (basin cells)' %
                  (v, np.nanmin(vmin[v]), np.nanmax(vmin[v]), np.nanmin(vmax[v]), np.nanmax(vmax[v])))
        print('written: %s' % out_fn)
    return out_fn
