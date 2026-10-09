"""
Per-cell envelope (minimum / maximum over time and ensemble members) of selected WaterGAP storages from the
collected open-loop results Res/<case>/OL/Ens_k/daily_output_<year>.nc. Used by src_DA.Threshold to bound
storages that have no physical capacity in the model (river storage): the analysis is kept within
[env_low * min, env_high * max] of what the model itself produced over the open-loop period.

Output: <Auxiliary_dir>/state_envelope_<basin>.nc with variables <var>_min and <var>_max on the (lat, lon) box
of the collected files (NaN outside the basin), and for the storages in `monthly` <var>_max_month (month, lat, lon):
the largest value of every calendar month over all years and members (snow cap of src_DA.Threshold / bounds,
"envelope_factor" of "snow_bounds", 9 Oct 2026). Called from DA_GRACE.make_state_envelope() (automatically in RDA.DA_run).

Parallel (comm given, 9 Oct 2026): every rank reads only the members assigned to it (member k on rank k % size, so
in the DA layout Ens_k on rank k) and the per-rank minima / maxima are combined with one MPI reduction; the result is
identical to the serial loop. Until 9 Oct 2026 rank 0 read all members while the other ranks waited.
Reuse: the file records a fingerprint of the open-loop files it was built from (names, sizes, modification times);
an existing file with the same variables and the same fingerprint is reused, any change of the open loop (new
perturbation, new collection, other period or members) rebuilds it. force=True always rebuilds.
"""
from pathlib import Path
import hashlib
import numpy as np
import warnings
import xarray as xr


def _member_files(res_dir, case, m):
    return sorted((Path(res_dir) / case / 'OL' / ('Ens_%d' % m)).glob('daily_output_*.nc'))


def _fingerprint(res_dir, case, members):
    """hash of the names, sizes and modification times of the open-loop files of the members"""
    h = hashlib.md5()
    for m in members:
        for fn in _member_files(res_dir, case, m):
            st = fn.stat()
            h.update(('%d|%s|%d|%d;' % (m, fn.name, st.st_size, int(st.st_mtime))).encode())
    return h.hexdigest()


def _reusable(out_fn, variables, fp, verbose, monthly=()):
    if not out_fn.exists():
        return False
    with xr.open_dataset(out_fn) as old:
        missing = [v for v in variables if ('%s_min' % v not in old) or ('%s_max' % v not in old)] + \
                  [v + '_max_month' for v in monthly if '%s_max_month' % v not in old]
        old_fp = old.attrs.get('ol_fingerprint', '')
    if missing:
        if verbose:
            print('state envelope %s lacks %s -> rebuilt' % (out_fn, ','.join(missing)))
        return False
    if old_fp != fp:
        if verbose:
            print('state envelope %s was built from another open loop -> rebuilt' % out_fn)
        return False
    if verbose:
        print('state envelope reused: %s (same open-loop files)' % out_fn)
    return True


def _member_extremes(res_dir, case, my_members, variables, verbose, monthly=()):
    """min / max over time and the given members (NaN where a cell has no value), for the storages in `monthly` also
    the max of every calendar month (12, lat, lon); lat, lon; number of files"""
    vmin, vmax, lat, lon, n_files = {v: None for v in variables}, {v: None for v in variables}, None, None, 0
    vmon = {v: None for v in monthly}
    for m in my_members:
        files = _member_files(res_dir, case, m)
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
                if monthly:
                    months = ds['time'].dt.month.values
                for v in monthly:
                    if v not in ds:
                        raise KeyError('%s not in %s' % (v, fn))
                    a = ds[v].values
                    if vmon[v] is None:
                        vmon[v] = np.full((12,) + a.shape[1:], np.nan)
                    for mo in np.unique(months):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            hi = np.nanmax(a[months == mo], axis=0).astype(float)
                        vmon[v][mo - 1] = np.fmax(vmon[v][mo - 1], hi)
                    del a
            n_files += 1
        if verbose:
            print('state envelope: member %d, %d yearly files' % (m, len(files)))
    return vmin, vmax, vmon, lat, lon, n_files


def make_state_envelope(res_dir, case: str, ens: int, basin: str, out_dir,
                        variables=('riverstor', 'groundwstor'),
                        members=None, force: bool = False, verbose: bool = True, comm=None, root: int = 0,
                        monthly=()):
    """
    res_dir   : configDA.basic.res_permanent
    ens       : number of perturbed members (files Ens_1 .. Ens_ens are used; Ens_0 = unperturbed is skipped
                unless members is given explicitly)
    variables : storages to process (min / max over the whole period)
    monthly   : storages whose maximum per calendar month is also written (<var>_max_month), e.g. ('swe',)
    force     : rebuild even if a file built from the same open loop exists
    comm      : MPI communicator: all its ranks must call; members are spread over the ranks, `root` writes.
                None = serial (one process reads everything)
    returns   : path of the written file (or the reused one)
    """
    out_fn = Path(out_dir) / ('state_envelope_%s.nc' % basin)
    members = list(range(1, ens + 1)) if members is None else list(members)
    rank, size = (0, 1) if comm is None else (comm.Get_rank(), comm.Get_size())

    '''step-1: reuse an envelope built from the same open-loop files (decided on root, shared with all ranks)'''
    decision = None
    if rank == root:
        fp = _fingerprint(res_dir, case, members)
        decision = (fp, (not force) and _reusable(out_fn, variables, fp, verbose, monthly))
    if comm is not None:
        decision = comm.bcast(decision, root=root)
    fp, reuse = decision
    if reuse:
        return out_fn

    '''step-2: min / max of the members of this rank'''
    my_members = [m for m in members if m % size == rank]
    monthly = list(monthly)
    vmin, vmax, vmon, lat, lon, n_files = _member_extremes(res_dir, case, my_members, variables,
                                                           verbose and (comm is None or rank == root), monthly)

    '''step-3: combine the ranks: NaN-aware min / max via +-inf, one reduction per variable and edge'''
    if comm is not None:
        from mpi4py import MPI
        grids = [g for g in comm.allgather(None if lat is None else (lat, lon)) if g is not None]
        lat, lon = grids[0] if grids else (None, None)          # the grid of the collected files (same on all)
        if lat is None:
            raise FileNotFoundError('no collected OL files under %s' % (Path(res_dir) / case / 'OL'))
        for v in variables:
            lo = np.full((len(lat), len(lon)), np.inf) if vmin[v] is None else np.where(np.isnan(vmin[v]), np.inf, vmin[v])
            hi = np.full((len(lat), len(lon)), -np.inf) if vmax[v] is None else np.where(np.isnan(vmax[v]), -np.inf, vmax[v])
            lo_all = np.empty_like(lo) if rank == root else None
            hi_all = np.empty_like(hi) if rank == root else None
            comm.Reduce(np.ascontiguousarray(lo), lo_all, op=MPI.MIN, root=root)
            comm.Reduce(np.ascontiguousarray(hi), hi_all, op=MPI.MAX, root=root)
            if rank == root:
                vmin[v] = np.where(np.isinf(lo_all), np.nan, lo_all)
                vmax[v] = np.where(np.isinf(hi_all), np.nan, hi_all)
        for v in monthly:
            hi = np.full((12, len(lat), len(lon)), -np.inf) if vmon[v] is None else np.where(np.isnan(vmon[v]), -np.inf, vmon[v])
            hi_all = np.empty_like(hi) if rank == root else None
            comm.Reduce(np.ascontiguousarray(hi), hi_all, op=MPI.MAX, root=root)
            if rank == root:
                vmon[v] = np.where(np.isinf(hi_all), np.nan, hi_all)
        n_files = comm.reduce(n_files, op=MPI.SUM, root=root)
        if rank != root:
            comm.barrier()                      # wait until root has written the file
            return out_fn

    '''step-4: write (root)'''
    if n_files == 0:
        raise FileNotFoundError('no collected OL files under %s' % (Path(res_dir) / case / 'OL'))
    data = {}
    for v in variables:
        data['%s_min' % v] = (('lat', 'lon'), vmin[v])
        data['%s_max' % v] = (('lat', 'lon'), vmax[v])
    for v in monthly:
        data['%s_max_month' % v] = (('month', 'lat', 'lon'), vmon[v])
    out = xr.Dataset(data, coords={'lat': lat, 'lon': lon, 'month': np.arange(1, 13)},
                     attrs=dict(description='per-cell min/max over time and members of the open loop, mm',
                                case=case, members=str(members), n_files=n_files, variables=','.join(variables),
                                monthly=','.join(monthly),
                                ol_fingerprint=fp))
    out_fn.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(out_fn)
    if verbose:
        for v in variables:
            print('  %-12s min %.1f .. %.1f mm, max %.1f .. %.1f mm (basin cells)' %
                  (v, np.nanmin(vmin[v]), np.nanmax(vmin[v]), np.nanmin(vmax[v]), np.nanmax(vmax[v])))
        for v in monthly:
            print('  %-12s monthly max: Jan %.1f, Jul %.1f mm (largest cell)' %
                  (v, np.nanmax(vmon[v][0]), np.nanmax(vmon[v][6])))
        print('written: %s (%d files%s)' % (out_fn, n_files, '' if comm is None else ', %d ranks' % size))
    if comm is not None:
        comm.barrier()
    return out_fn
