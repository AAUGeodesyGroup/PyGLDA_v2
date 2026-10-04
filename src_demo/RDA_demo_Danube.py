"""
Danube test case (demo_2): CSR monthly mascons, 6 HydroBASINS sub-basins (Basin/shp/Danube/Danube.shp),
state = groundwater + soil + river + snow, EnKF_localized (block) + RTPS 0.7 + non-negative increment partition.

WaterGAP runs globally and demo_2 uses the same Config_ReWaterGAP.json / perturbation.json / period as demo_3,
so the Amazon perturbation, spin-up and open loop (OL_output/Ens_k, global daily files) are reused as they are.
Order of the stages (they overwrite the demo_2 results of the old 3-sub-basin run):
  serial   : config_basin_mask (new shapefile), get_GRACE_obs (mascons)
  mpiexec  : collect_and_statistics(OL)  -> crops the global OL into Res/demo_2/OL with the new mask
             DA_run, collect_and_statistics(DA)
  serial   : post_processing, visualization, increment_diagnosis, da_evaluation
(config_external_data / read_config_and_save / model_perturbation / spin_up / OL_run only if the shared
 OL_output or Ensemble_* directories were cleaned or the perturbation changes)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src_FlowControl.Regional_DA import RDA

RDA.setting_dir = '/media/user/My Book/Fan/PyGLDA_v2/settings/demo_2'
RDA.ens = 4

RDA.external_data_path = '/media/user/My Book/Fan/PyGLDA_v2_external_data'
RDA.case = 'demo_2'
RDA.basin = 'Danube'
RDA.shp_path = '/media/user/My Book/Fan/PyGLDA_v2_external_data/Basin/shp/Danube/Danube.shp'

'''for spin-up'''
RDA.spin_up_start = '2000-01-01'
RDA.spin_up_end = '2001-12-31'
RDA.spin_up_years = 5

'''for open-loop and data assimilation'''
RDA.sim_begin_time = '2002-01-01'
RDA.sim_end_time = '2016-04-30'


def demo1():
    '''before the open loop (serial) -- not needed: shared with demo_3 (global run, same perturbation)'''
    RDA.config_external_data()
    RDA.read_config_and_save()
    # RDA.model_perturbation()

    '''before the data assimilation (serial): mask from the new shapefile, mascon observations'''
    # RDA.config_basin_mask()
    # RDA.get_GRACE_obs(is_diagonal=False)

    '''after the data assimilation (serial)'''
    RDA.post_processing()
    RDA.visualization()
    RDA.increment_diagnosis()
    '''evaluation report + archive of the run logs -> Res/demo_2/evaluation/<tag>/ (set the tag for every run)'''
    RDA.da_evaluation(tag='run11a', reference='run10')

    '''house-keeping: remove the temporary daily output of a collected stage (dry run first, then dry_run=False)'''
    # from src_DA.EnumDA import Stage
    # RDA.clean_temp_output(Stage.OL, dry_run=True)
    # RDA.clean_temp_output(Stage.DA, dry_run=True)
    pass


def demo2():
    '''mpiexec -n 5 python -u RDA_demo_Danube.py Config_ReWaterGAP.json'''
    from src_DA.EnumDA import Stage
    from mpi4py import MPI

    # RDA.spin_up()          # not needed: shared with demo_3
    # RDA.OL_run()           # not needed: shared with demo_3
    # RDA.collect_and_statistics(Stage.OL, skip_collect=False)

    try:
        RDA.DA_run()
    except Exception:
        import traceback
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
    RDA.collect_and_statistics(Stage.DA, skip_collect=False)
    pass


if __name__ == '__main__':
    # demo1()
    demo2()
