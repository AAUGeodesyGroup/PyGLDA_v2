"""
Global data assimilation: GRACE TWS of the global basin units (GlobalBasins v1.0, 772 units) into WaterGAP 2.2e.

The workflow is the regional one (src_FlowControl.Regional_DA.RDA), which GDA inherits; only the steps that differ
for the globe are overridden:

    check_units()               frozen unit set: mask, shapefile and IDs consistent (replaces config_basin_mask)
    get_GRACE_obs()             unit TWS from the CSR mascons (regional code, frozen mask) and the 772 x 772 GRACE
                                covariance per month from the DDK3 samples in one sparse matrix product
    collect_and_statistics()    MPI: unit series of every storage straight from the daily output (OL: the global
                                open loop of the source case; DA: DA_output/<case>), no yearly copy of the global
                                fields; for the OL also the open-loop envelope (riverstor, groundwstor)
    DA_run()                    as RDA.DA_run, but the envelope is the one written by collect_and_statistics(OL)

Not used: model_perturbation, spin_up, OL_run - the open loop of the source case (e.g. demo_Danube) is global and
is reused (same Ensemble_input, Ensemble_Initialization and OL_output folders in settings/<case>/DA_setting.json).

Note: RDA's methods are static and read RDA.<attribute>. Every case attribute assigned to GDA (GDA.case = ...) is
therefore also assigned to RDA (metaclass _CaseToRDA), so that the inherited steps (DA_run, post_processing, ...) run
the global case.

Order (driver: src_demo/demo_global.py)
    serial:  GDA.config_external_data(), GDA.check_units(), GDA.get_GRACE_obs()
    MPI:     GDA.collect_and_statistics(Stage.OL) -> GDA.DA_run() -> GDA.collect_and_statistics(Stage.DA)

First version (7 Oct 2026). Known limits, to be revised step by step:
  - the filter (src_DA) still holds dense n_state x n_obs matrices on every rank (design matrix, block taper):
    ~1 GB each for 772 units -> a sparse path is the next step before a 30-member run on UCloud (89 GB)
  - no harmonic maps for the global case (they need yearly files), no global evaluation yet
"""
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import h5py

from src_FlowControl.Regional_DA import RDA, _abort_with_report
from src_DA.EnumDA import Stage


class _CaseToRDA(type):
    """metaclass of GDA: a case attribute assigned to GDA (GDA.case = ...) is stored on RDA, because the inherited
    RDA methods are static and read RDA.<attribute>; GDA.case then reads it back from RDA"""
    case_attributes = ('setting_dir', 'ens', 'external_data_path', 'case', 'basin', 'shp_path', 'spin_up_start',
                       'spin_up_end', 'spin_up_years', 'sim_begin_time', 'sim_end_time')

    def __setattr__(cls, name, value):
        if name in _CaseToRDA.case_attributes:
            setattr(RDA, name, value)                   # stored once, on RDA; GDA reads it from there
        else:
            super().__setattr__(name, value)


class GDA(RDA, metaclass=_CaseToRDA):
    """
    The case (case, basin, ens, setting_dir, external_data_path, shp_path, spin-up and simulation period) is set in
    the driver (src_demo/demo_global.py), e.g. GDA.case = 'demo_Global'. GDA keeps no defaults of its own: reading
    GDA.case gives RDA.case, and assigning GDA.case assigns RDA.case, so the two can never differ.
    """

    # ------------------------------------------------------------------ units
    @staticmethod
    def check_units():
        """
        Consistency of the frozen unit set (replaces RDA.config_basin_mask: the global mask is never rebuilt here):
        sub_basin_k of the mask = ID k of the shapefile, no empty unit, no cell in two units, mask area vs SUB_AREA,
        md5 against <name>_settings.txt next to the shapefile (if present).
        """
        import hashlib
        import geopandas as gpd
        from src_DA.configure_DA import config_DA

        '''step-1: mask (the one in DA_setting.json, written by read_config_and_save)'''
        configDA = config_DA.loadjson(Path(GDA.setting_dir) / 'DA_setting.json').process()
        mp = Path(configDA.basic.basin_mask)
        with h5py.File(mp, 'r') as f:
            keys = [k for k in f.keys() if k.startswith('sub_basin_')]
            ids = sorted(int(k.split('_')[-1]) for k in keys)
            basin = f['basin'][:].astype(bool)
            count = np.zeros(basin.shape, dtype=np.int32)
            n_cell = {}
            for k in ids:
                m = f['sub_basin_%d' % k][:].astype(bool)
                count += m
                n_cell[k] = int(m.sum())

        '''step-2: shapefile'''
        units = gpd.read_file(GDA.shp_path)
        shp_ids = sorted(units.ID.astype(int))

        '''step-3: mask area vs polygon area'''
        res = 0.5
        lat = np.arange(90 - res / 2, -90, -res)
        cell_km2 = np.cos(np.deg2rad(lat))[:, None] * (111.32 * res) * (110.57 * res) * np.ones((1, 720))
        area_mask = {}
        with h5py.File(mp, 'r') as f:
            for k in ids:
                area_mask[k] = float((f['sub_basin_%d' % k][:] * cell_km2).sum())
        sub_area = units.set_index('ID').SUB_AREA
        ratio = np.array([area_mask[k] / sub_area[k] for k in ids])

        '''step-4: md5'''
        md5_ok = None
        txt = sorted(Path(GDA.shp_path).parent.glob('*_settings.txt'))
        if txt:
            ref = {}
            for line in open(txt[0]):
                parts = line.split()
                if len(parts) >= 2 and len(parts[-1]) == 32 and parts[-2].endswith(('.shp', '.h5')):
                    ref[parts[-2]] = parts[-1]
            got = {}
            for fn in (Path(GDA.shp_path), mp):
                if fn.name in ref:
                    got[fn.name] = hashlib.md5(open(fn, 'rb').read()).hexdigest()
            md5_ok = all(got[k] == ref[k] for k in got) if got else None

        print('\n=================== Global units: %s ===================' % GDA.basin)
        print('mask      : %s' % mp)
        print('shapefile : %s' % GDA.shp_path)
        print('units     : %d in the mask, %d in the shapefile, IDs equal: %s' % (len(ids), len(shp_ids), ids == shp_ids))
        print('cells     : %d in the mask, %d in two units, %d empty units' %
              (int(basin.sum()), int((count > 1).sum()), sum(v == 0 for v in n_cell.values())))
        print('mask area / SUB_AREA : median %.2f, 5-95 %% %.2f-%.2f' % (np.median(ratio), *np.percentile(ratio, [5, 95])))
        print('md5 vs %s : %s' % (txt[0].name if txt else '(no settings file)', md5_ok))
        ok = ids == shp_ids and (count > 1).sum() == 0 and all(v > 0 for v in n_cell.values()) and md5_ok is not False
        if not ok:
            raise ValueError('global unit set is inconsistent, see above')
        return dict(n_units=len(ids), n_cells=int(basin.sum()), md5_ok=md5_ok)

    # ------------------------------------------------------------------ observations
    @staticmethod
    def get_GRACE_obs(is_diagonal=False):
        """
        GRACE observations of the units -> configDA.obs.GRACE['preprocess_res']:
          <basin>_signal.hdf5, <basin>_gridded_signal.hdf5   regional code (GRACE_CSR_mascon) with the frozen mask
          <basin>_cov.hdf5                                   vectorised, fractional 1-degree weights (_basin_COV)
        Only the monthly CSR mascons ('Mascon_monthly') for now.
        """
        from src_OBS.prepare_GRACE_mascon import GRACE_CSR_mascon
        from src_DA.configure_DA import config_DA
        import geopandas as gpd

        configDA = config_DA.loadjson(Path(GDA.setting_dir) / 'DA_setting.json').process()
        kind = configDA.obs.GRACE.get('kind', 'Mascon_monthly')
        if kind != 'Mascon_monthly':
            raise NotImplementedError('GDA.get_GRACE_obs: only Mascon_monthly so far, got %s' % kind)
        t1 = datetime.strptime(GDA.sim_begin_time, '%Y-%m-%d').strftime('%Y-%m')
        t2 = datetime.strptime(GDA.sim_end_time, '%Y-%m-%d').strftime('%Y-%m')
        dir_in = configDA.obs.GRACE['EWH_grid_dir']
        dir_out = configDA.obs.GRACE['preprocess_res']
        land_fn = Path(GDA.external_data_path) / 'GRACE/global_mask/GlobalLandMaskForGRACE.hdf5'

        '''step-1: unit TWS and gridded TWS with the regional code; the frozen mask is used as it is (no
        generate_mask: it would rebuild the 0.5-degree mask without the model land and the removed units)'''
        GR = GRACE_CSR_mascon(basin_name=GDA.basin, shp_path=GDA.shp_path)
        GR.configure_global_land_ocean_mask(fn=land_fn)
        GR.save_mask_dir = str(Path(configDA.basic.basin_mask).parent)
        units = gpd.read_file(GDA.shp_path).sort_values('ID')
        GR._sub_basin_area = units.SUB_AREA.values.astype(float)              # km2, in the order of sub_basin_k
        GR.basin_TWS(month_begin=t1, month_end=t2, dir_in=dir_in, dir_out=dir_out)
        GR.grid_TWS(month_begin=t1, month_end=t2, dir_in=dir_in, dir_out=dir_out)

        '''step-2: covariance of the units from the DDK3 samples'''
        GDA._basin_COV(mask_path=configDA.basic.basin_mask, dir_in=configDA.obs.GRACE['cov_dir'], dir_out=dir_out,
                      time_epochs=GR._aux.getTimeReference()['time_epoch'], land_fn=land_fn,
                      is_diagonal=is_diagonal)
        pass

    # ------------------------------------------------------------------ collection
    @staticmethod
    def collect_and_statistics(stage, post_process=True, envelope_vars=('riverstor', 'groundwstor')):
        """
        MPI stage (mpiexec -n ens+1), replaces RDA.collect_and_statistics for the globe:
          step 1  unit series of every storage from the daily output of this member, read directly
                  (OL: <OL_output_temp_dir>/Ens_k, the global open loop of the source case; DA:
                  <DA_output_temp_dir>/<case>/Ens_k) -> Res/<case>/<stage>/Ens_k/basin_ts_<stage>.h5, same layout as
                  the regional file ('basin' + sub_basin_k per variable), same weights (shp2mask.unit_weights, as in
                  statistical_analysis.BasinAverageAnalysis). No yearly copy of the global fields.
          step 2  OL only: per-cell min/max of envelope_vars over members 1..ens and the period ->
                  <Auxiliary_dir>/state_envelope_<basin>.nc (used by src_DA.Threshold)
          step 3  (post_process) RDA.post_processing(stages=[stage], harmonic=False) on rank 1: Res_<stage>.h5 and,
                  for the DA, the GRACE files. Harmonic maps are not made (they need yearly files).
        """
        from mpi4py import MPI
        import pandas as pd
        import netCDF4 as nc
        from src_DA.configure_DA import config_DA
        from src_DA.EnumDA import WaterGap_storage_variables
        from src_auxiliary.shp2mask import unit_weights

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        assert size == (GDA.ens + 1), 'Not enough threads for parallelization! Required threads are %s' % (GDA.ens + 1)

        if rank != 1:
            log_dir = str(Path(__file__).resolve().parent.parent / 'parallel_logs' / ('collect_global_%s' % stage.name))
            os.makedirs(log_dir, exist_ok=True)
            sys.stdout = open(os.path.join(log_dir, f"rank_{rank}.log"), 'w', encoding='utf-8', buffering=1)
            sys.stderr = sys.stdout

        configDA = config_DA.loadjson(Path(GDA.setting_dir) / 'DA_setting.json').process()
        if stage == Stage.OL:
            state_dir = Path(configDA.basic.OL_output_temp_dir) / ('Ens_%d' % rank)
        else:
            state_dir = Path(configDA.basic.DA_output_temp_dir) / GDA.case / ('Ens_%d' % rank)
        out_dir = Path(configDA.basic.res_permanent) / GDA.case / stage.name / ('Ens_%d' % rank)
        days = pd.date_range(GDA.sim_begin_time, GDA.sim_end_time)
        variables = [v.name for v in WaterGap_storage_variables]
        do_env = stage == Stage.OL and rank != 0

        if rank == 1:
            print('\n=================== Global collection: %s ===================' % stage.name)
            print('case %s | units %s | %s to %s | %d members (+ unperturbed)' %
                  (GDA.case, GDA.basin, GDA.sim_begin_time, GDA.sim_end_time, GDA.ens))
            print('step 1: unit series from %s -> %s' % (state_dir.parent, out_dir.parent))

        env_min = {v: None for v in envelope_vars}
        env_max = {v: None for v in envelope_vars}
        try:
            '''step-1: unit series'''
            W, keys, grid = None, None, None
            series = {v: np.full((0, 0), np.nan) for v in variables}
            for n, day in enumerate(days):
                fn = state_dir / ('daily_output_%s.nc' % day.strftime('%Y-%m-%d'))
                with nc.Dataset(fn) as ds:
                    if W is None:
                        lat, lon = ds['lat'][:].data, ds['lon'][:].data
                        W, keys = unit_weights(configDA.basic.basin_mask, lat, lon)
                        grid = (len(lat), len(lon))
                        series = {v: np.full((len(keys), len(days)), np.nan) for v in variables}
                    for v in variables:
                        a = np.ma.filled(ds[v][:].astype(float), np.nan).reshape(-1)
                        series[v][:, n] = W @ a
                        if do_env and v in envelope_vars:
                            a2 = a.reshape(grid)
                            env_min[v] = a2.copy() if env_min[v] is None else np.fmin(env_min[v], a2)
                            env_max[v] = a2.copy() if env_max[v] is None else np.fmax(env_max[v], a2)
                if rank == 1 and (n % 365 == 0 or n == len(days) - 1):
                    print('  %s (%d/%d days)' % (day.strftime('%Y-%m-%d'), n + 1, len(days)), flush=True)

            out_dir.mkdir(parents=True, exist_ok=True)
            with h5py.File(out_dir / ('basin_ts_%s.h5' % stage.name), 'w') as hf:
                for v in variables:
                    g = hf.create_group(v)
                    for r, key in enumerate(keys):
                        g.create_dataset(key, data=series[v][r])
            print('member %d: %d units x %d days -> %s' % (rank, len(keys) - 1, len(days),
                                                          out_dir / ('basin_ts_%s.h5' % stage.name)), flush=True)
        except Exception:
            _abort_with_report(comm, rank, 'collect_global')

        '''step-2: open-loop envelope (members 1..ens)'''
        if stage == Stage.OL:
            for v in envelope_vars:
                lo = comm.gather(env_min[v], root=1)
                hi = comm.gather(env_max[v], root=1)
                if rank == 1:
                    env_min[v] = np.fmin.reduce([x for x in lo if x is not None])
                    env_max[v] = np.fmax.reduce([x for x in hi if x is not None])
            if rank == 1:
                import xarray as xr
                with h5py.File(configDA.basic.basin_mask, 'r') as f:
                    inside = f['basin'][:].astype(bool)
                res = 0.5
                lat_g = np.arange(90 - res / 2, -90, -res)
                lon_g = np.arange(-180 + res / 2, 180, res)
                data = {}
                for v in envelope_vars:
                    data['%s_min' % v] = (('lat', 'lon'), np.where(inside, env_min[v], np.nan))
                    data['%s_max' % v] = (('lat', 'lon'), np.where(inside, env_max[v], np.nan))
                env_fn = Path(configDA.basic.Auxiliary_dir) / ('state_envelope_%s.nc' % GDA.basin)
                xr.Dataset(data, coords={'lat': lat_g, 'lon': lon_g},
                           attrs=dict(description='per-cell min/max over time and members of the open loop, mm',
                                      case=GDA.case, members='1..%d' % GDA.ens, period='%s..%s' % (
                                          GDA.sim_begin_time, GDA.sim_end_time),
                                      variables=','.join(envelope_vars))).to_netcdf(env_fn)
                print('step 2: open-loop envelope (%s) -> %s' % (', '.join(envelope_vars), env_fn))

        comm.barrier()

        '''step-3: post-processing on rank 1'''
        if post_process:
            if rank == 1:
                try:
                    RDA.post_processing(stages=[stage], harmonic=False)
                    print('Post-processing of %s finished: %s' % (stage.name, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                except Exception:
                    import traceback
                    traceback.print_exc()
                    print('[WARNING] post-processing of %s failed; the unit series are complete. Rerun serially: '
                          'GDA.post_processing(stages=[Stage.%s], harmonic=False)' % (stage.name, stage.name), flush=True)
            comm.barrier()
        if rank == 1:
            print('Global collection of %s finished: %s' % (stage.name, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        pass

    # ------------------------------------------------------------------ data assimilation
    @staticmethod
    def DA_run():
        """as RDA.DA_run; the open-loop envelope comes from GDA.collect_and_statistics(Stage.OL) (the regional
        make_state_envelope needs the yearly OL files, which the global case does not write)"""
        from mpi4py import MPI
        from src_FlowControl.DA_GRACE import DA_GRACE

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        da = DA_GRACE(setting_dir=Path(GDA.setting_dir), case=GDA.case)
        da.configure_setting(ens_size=GDA.ens, case_name=GDA.case, basin_name=GDA.basin,
                             basin_dir=Path(GDA.external_data_path) / 'Basin', basin_shp=GDA.shp_path)
        da.configure_date(begin_date=GDA.sim_begin_time, end_date=GDA.sim_end_time)
        if rank == 0:
            da.save_configuration()
        comm.barrier()

        da.reload_setting()

        if rank == 0:
            da.gather_OLmean()
            da.generate_perturbed_GRACE_obs()
            env_fn = Path(da.configDA.basic.Auxiliary_dir) / ('state_envelope_%s.nc' % GDA.basin)
            if not env_fn.exists():
                print('[WARNING] %s not found: run GDA.collect_and_statistics(Stage.OL) first (riverstor is then '
                      'bounded by the positive floor only)' % env_fn)

        comm.barrier()

        da.prepare_design_matrix()

        try:
            da.run_DA(rank=rank)
            pass
        except Exception:
            _abort_with_report(comm, rank, 'DA')

        pass

    # ------------------------------------------------------------------ internal helpers
    @staticmethod
    def _basin_COV(mask_path, dir_in, dir_out, time_epochs, land_fn=None, is_diagonal=False):
        """
        internal (called by get_GRACE_obs):
        772 x 772 covariance of the unit means per month from the GRACE error samples (<dir_in>/*<YYYY-MM>*.npy,
        [n_sample, 180, 360] or [n_sample, 64800], 1 degree, latitude from -90, metres).
        Unit weights on the 1-degree grid: cos(lat) x number of the unit's 0.5-degree cells in the 1-degree cell
        (fractional, so that small and coastal units keep their cells), times the GRACE 1-degree land mask as in the
        regional basin_COV (a unit without land cells there keeps its weights without the land mask).
        One sparse product per month instead of the regional loop over units; the months are those of
        time_epochs (the GRACE months of the signal file).
        """
        from scipy import sparse
        from tqdm import tqdm

        '''step-1: weights (north-first 0.5 deg -> 1 deg -> south-first, as the samples)'''
        res = 0.5
        lat05 = np.arange(90 - res / 2, -90, -res)
        cos05 = np.cos(np.deg2rad(lat05))[:, None]
        land1 = None
        if land_fn is not None:
            land1 = np.flipud(h5py.File(land_fn, 'r')['resolution_1']['mask'][:])     # north-first, as regional
        rows, cols, vals, n_noland = [], [], [], 0
        with h5py.File(mask_path, 'r') as f:
            ids = sorted(int(k.split('_')[-1]) for k in f.keys() if k.startswith('sub_basin_'))
            for r, k in enumerate(ids):
                m = f['sub_basin_%d' % k][:] * cos05                                  # (360, 720)
                w1 = m.reshape(180, 2, 360, 2).sum(axis=(1, 3))                       # (180, 360) north-first
                if land1 is not None and (w1 * land1).sum() > 0:
                    w1 = w1 * land1
                elif land1 is not None:
                    n_noland += 1
                w1 = np.flipud(w1).ravel()                                            # south-first, as the samples
                nz = np.nonzero(w1)[0]
                rows.append(np.full(len(nz), r)); cols.append(nz); vals.append(w1[nz] / w1[nz].sum())
        W = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(len(ids), 180 * 360))
        print('\nGRACE covariance of %d units from the samples in %s (%d units without 1-degree land cells keep '
              'their weights without the land mask)' % (len(ids), dir_in, n_noland))

        '''step-2: monthly covariance, one sample file per GRACE month of the signal (time_epochs)'''
        fns = sorted(os.listdir(dir_in))
        COV = []
        for epoch in tqdm(time_epochs, desc="Processing months"):
            month = datetime.strptime(epoch, '%Y-%m-%d').strftime('%Y-%m')
            tn = [x for x in fns if month in x]
            if not tn:
                raise FileNotFoundError('GRACE month %s: no error samples in %s' % (month, dir_in))
            x = np.load(str(Path(dir_in) / tn[-1]))
            x = np.nan_to_num(x.reshape(x.shape[0], -1))                              # (n_sample, 64800)
            u = np.asarray(W @ x.T) * 1000                                            # (n_unit, n_sample), mm
            cov = np.cov(u)
            if is_diagonal:
                cov = np.diag(np.diag(cov))
            COV.append(cov)

        '''step-3: save (same layout as the regional <basin>_cov.hdf5)'''
        hm = h5py.File(Path(dir_out) / ('%s_cov.hdf5' % GDA.basin), 'w')
        hm.create_dataset('data', data=np.array(COV))
        dt = h5py.special_dtype(vlen=str)
        hm.create_dataset('time_epoch', data=time_epochs, dtype=dt)
        hm.close()
        sd = np.sqrt(np.array([np.diag(c) for c in COV]))
        print('GRACE unit sigma: median %.1f mm (5-95 %%: %.1f-%.1f mm), %d months -> %s' %
              (np.median(sd), *np.percentile(sd, [5, 95]), len(COV), Path(dir_out) / ('%s_cov.hdf5' % GDA.basin)))
        print('Finished: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        pass


def demo1():
    '''serial checks of the global case (paths of the workstation)'''
    GDA.case = 'demo_Global'
    GDA.basin = 'GlobalBasins_v1.0'
    GDA.setting_dir = str(Path(__file__).resolve().parent.parent / 'settings' / GDA.case)
    GDA.external_data_path = '/media/user/My Book/Fan/PyGLDA_v2_external_data'
    GDA.shp_path = GDA.external_data_path + '/Basin/shp/Global_v1.0/GlobalBasins_v1.0.shp'
    GDA.check_units()
    pass


if __name__ == '__main__':
    demo1()
