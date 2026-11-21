"""
Mass-consistent observation operator (sensitivity option; not used in the default configuration).

GRACE gives water mass per WHOLE 0.5-degree cell (lwe thickness = mass / cell area). The WaterGAP daily files hold every
storage in mm per CONTINENTAL area of the cell (UnitConverter: km3 -> mm via cell_area x contfrac; the vertical storages
are written x landarea_frac / contfrac). A unit mean that compares the model with GRACE must therefore weight every cell
by its continental fraction:

    H_jc = cos(lat_c) * cf_c / sum_{c in j} cos(lat_c)        (cf = continental fraction 0..1)

i.e. the cos-lat weights multiplied by cf and NOT re-normalised (the row sums to the unit's mean cf). The default
operator, cos(lat_c) / sum cos(lat_c), counts the model water of a coastal / lake-shore cell 1 / cf times too much.

Switch (DA_setting.json, "obs" block; absent / false = the default cos(lat) operator):
    "operator": {"mass_consistent": true, "contfrac_file": "<netCDF with the WaterGAP continental fraction>",
                 "contfrac_var": "contfrac"}
contfrac_file: the static field WaterGAP itself reads as land_surface_water_fraction.contfrac (percent 0..100; a file
already in fractions 0..1 is accepted). It is used consistently by
    - the filter's design matrices (both the DA state and the excluded storages; src_FlowControl.DA_GRACE),
    - the unit time series of the collect step (basin_ts_<stage>.h5, src_postprocessing.statistical_analysis via
      src_auxiliary.shp2mask.unit_weights), hence the open-loop mean added to the GRACE anomalies (gather_OLmean) and
      Res_<stage>.h5 of the evaluation.
The GRACE side is unchanged. The gridded product keeps mm per continental area (document it).
NOTE: the OL collect (basin_ts_OL.h5) and gather_OLmean must be redone with the switch on before the obs file is built;
an OL mean from the old operator would leave a constant offset per coastal unit that the filter would "correct".
"""
import numpy as np

RES = 0.5
NLAT, NLON = 360, 720


def operator_config(configDA):
    """the "operator" block of the obs settings, or None when the mass-consistent operator is off"""
    obs = getattr(configDA, 'obs', None)
    cfg = getattr(obs, 'operator', None) if obs is not None and not isinstance(obs, dict) else \
        (obs.get('operator') if isinstance(obs, dict) else None)
    if not cfg or not cfg.get('mass_consistent', False):
        return None
    if not cfg.get('contfrac_file'):
        raise ValueError('obs.operator.mass_consistent needs "contfrac_file" (WaterGAP continental fraction, netCDF)')
    return dict(cfg)


_CACHE = {}


def load_contfrac_global(path, var='contfrac'):
    """continental fraction 0..1 on the PyGLDA global grid (row 0 = 89.75 N, col 0 = -179.75 E), placed by the file's
    own lat / lon coordinates; cells missing in the file or NaN -> 0 (no continental area)"""
    import xarray as xr
    key = (str(path), var)
    if key in _CACHE:
        return _CACHE[key]
    with xr.open_dataset(path) as ds:
        if var not in ds:
            raise KeyError('contfrac_file %s has no variable %s (variables: %s)' % (path, var, list(ds.data_vars)))
        da = ds[var].squeeze(drop=True)
        latn = next((n for n in da.dims if n.lower() in ('lat', 'latitude', 'y')), None)
        lonn = next((n for n in da.dims if n.lower() in ('lon', 'longitude', 'x')), None)
        if latn is None or lonn is None or da.ndim != 2:
            raise ValueError('contfrac_file %s: %s must be 2-D (lat, lon), got dims %s' % (path, var, da.dims))
        da = da.transpose(latn, lonn)
        lat = np.asarray(da[latn].values, dtype=float)
        lon = np.asarray(da[lonn].values, dtype=float)
        val = np.asarray(da.values, dtype=float)
    lon = np.where(lon > 180, lon - 360, lon)
    i = np.rint((90 - RES / 2 - lat) / RES).astype(int)
    j = np.rint((lon - (-180 + RES / 2)) / RES).astype(int)
    if (i.min() < 0) or (i.max() >= NLAT) or (j.min() < 0) or (j.max() >= NLON):
        raise ValueError('contfrac_file %s is not on the 0.5-degree grid' % path)
    out = np.zeros((NLAT, NLON))
    out[np.ix_(i, j)] = np.where(np.isfinite(val), val, 0.0)
    if np.nanmax(out) > 1.0 + 1e-6:
        out = out / 100.0                         # percent (WaterGAP convention) -> fraction
    out = np.clip(out, 0.0, 1.0)
    _CACHE[key] = out
    return out


def cell_factor_for_state(contfrac_global, local_mask):
    """continental fraction of every state cell (basin cells in row-major order of the global grid, the order of
    ExtractStates / ObsDesignMatrix), shape (n_cell,)"""
    g = np.asarray(local_mask['global_2d']).astype(bool)
    if g.shape != contfrac_global.shape:
        raise ValueError('mass-consistent operator: global mask %s vs contfrac %s' % (g.shape, contfrac_global.shape))
    return np.asarray(contfrac_global, dtype=float)[g]


def describe(cell_factor):
    cf = np.asarray(cell_factor, dtype=float)
    return 'mass-consistent operator: %d cells, %d with cf < 0.99 (min %.2f, mean %.3f)' % (
        cf.size, int((cf < 0.99).sum()), float(cf.min()) if cf.size else np.nan, float(cf.mean()) if cf.size else np.nan)
