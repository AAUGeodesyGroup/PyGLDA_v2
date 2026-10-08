"""
Global data assimilation: GRACE TWS of the global basin units (GlobalBasins v1.0, 772 units) into WaterGAP 2.2e.

The workflow is the regional one (src_FlowControl.Regional_DA.RDA), which GDA inherits; only the steps that differ
for the globe are overridden:

    check_units()               frozen unit set: mask, shapefile and IDs consistent (replaces config_basin_mask)
    get_GRACE_obs()             unit TWS, 772 x 772 GRACE covariance and gridded TWS with the regional GRACE
                                processing on the frozen unit mask (no second rasterisation of the shapefile)

All other steps are the inherited regional ones: collect_and_statistics (the yearly merge writes the full grid for a
global unit mask, see merge_standardize.yearly_merge), DA_run with the open-loop envelope, harmonic maps,
post-processing, export. OL yearly files come from the global open loop of the source case (OL_output), DA yearly
files from DA_output/<case>.

Not used: model_perturbation, spin_up, OL_run - the open loop of the source case (e.g. demo_Danube) is global and
is reused (same Ensemble_input, Ensemble_Initialization and OL_output folders in settings/<case>/DA_setting.json).

Note: RDA's methods are static and read RDA.<attribute>. Every case attribute assigned to GDA (GDA.case = ...) is
therefore also assigned to RDA (metaclass _CaseToRDA), so that the inherited steps (DA_run, post_processing, ...) run
the global case.

Order (driver: src_demo/demo_global.py)
    serial:  GDA.config_external_data(), GDA.check_units(), GDA.get_GRACE_obs()
    MPI:     GDA.collect_and_statistics(Stage.OL) -> GDA.DA_run() -> GDA.collect_and_statistics(Stage.DA)  (inherited)

First version (7 Oct 2026). Known limits, to be revised step by step:
  - the filter (src_DA) still holds dense n_state x n_obs matrices on every rank (design matrix, block taper):
    ~1 GB each for 772 units -> a sparse path is the next step before a 30-member run on UCloud (89 GB)
  - no global evaluation yet (visualization / increment_diagnosis / da_evaluation are made for a few sub-basins)
"""
from pathlib import Path
from datetime import datetime

import numpy as np
import h5py

from src_FlowControl.Regional_DA import RDA


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
            for fn in (Path(GDA.shp_path), mp, mp.with_name(mp.name.replace('_res_0.5.h5', '_res_1.h5'))):
                if fn.name in ref and fn.exists():
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
    def get_GRACE_obs():
        """
        GRACE observations of the units -> configDA.obs.GRACE['preprocess_res'], with the regional GRACE processing
        (src_OBS.prepare_GRACE_mascon.GRACE_CSR_mascon) on the frozen unit mask:
          <basin>_signal.hdf5           unit TWS per GRACE month (0.5-degree mask, WaterGAP land)
          <basin>_cov.hdf5              772 x 772 covariance per month from the DDK3 samples (1-degree mask derived
                                        from the 0.5-degree one, shp2mask.mask_05_to_1)
          <basin>_gridded_signal.hdf5   gridded TWS of the unit cells
        The frozen mask is used as it is (GRACE_preparation.use_mask), not rasterised again from the shapefile, which
        would lose the model land and the removed units. Only the monthly CSR mascons ('Mascon_monthly') for now.
        """
        from src_OBS.prepare_GRACE_mascon import GRACE_CSR_mascon
        from src_DA.configure_DA import config_DA

        configDA = config_DA.loadjson(Path(GDA.setting_dir) / 'DA_setting.json').process()
        kind = configDA.obs.GRACE.get('kind', 'Mascon_monthly')
        if kind != 'Mascon_monthly':
            raise NotImplementedError('GDA.get_GRACE_obs: only Mascon_monthly so far, got %s' % kind)
        t1 = datetime.strptime(GDA.sim_begin_time, '%Y-%m-%d').strftime('%Y-%m')
        t2 = datetime.strptime(GDA.sim_end_time, '%Y-%m-%d').strftime('%Y-%m')
        dir_out = configDA.obs.GRACE['preprocess_res']

        GR = GRACE_CSR_mascon(basin_name=GDA.basin, shp_path=GDA.shp_path)
        GR.configure_global_land_ocean_mask(fn=RDA._land_mask())
        GR.use_mask(mask_dir=Path(configDA.basic.basin_mask).parent)
        GR.basin_TWS(month_begin=t1, month_end=t2, dir_in=configDA.obs.GRACE['EWH_grid_dir'], dir_out=dir_out)
        GR.basin_COV(month_begin=t1, month_end=t2, dir_in=configDA.obs.GRACE['cov_dir'], dir_out=dir_out)
        GR.grid_TWS(month_begin=t1, month_end=t2, dir_in=configDA.obs.GRACE['EWH_grid_dir'], dir_out=dir_out)
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
