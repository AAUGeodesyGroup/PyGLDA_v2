"""PyGLDA v2 driver - UCloud. Local twin: RDA_local.py"""
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent          # the PyGLDA_v2 folder
sys.path.insert(0, str(CODE_ROOT))

from src_FlowControl.Regional_DA import RDA
from src_DA.EnumDA import Stage

DATA_ROOT = '/work/PyGLDA_v2_external_data'

RDA.case = 'demo_Danube'
RDA.basin = 'Danube'
RDA.ens = 30
RDA.setting_dir = str(CODE_ROOT / 'settings' / RDA.case)
RDA.external_data_path = DATA_ROOT
RDA.shp_path = DATA_ROOT + '/Basin/shp/Danube/Danube.shp'

'''for spin-up'''
RDA.spin_up_start = '2000-01-01'
RDA.spin_up_end = '2001-12-31'
RDA.spin_up_years = 5

'''for open-loop and data assimilation'''
RDA.sim_begin_time = '2002-01-01'
RDA.sim_end_time = '2019-12-31'

def demo1():
    '''serial: python RDA_Ucloud.py Config_ReWaterGAP.json'''

    '''before the open loop'''
    # RDA.config_external_data()        # done 2026-10-05 on UCloud (paths rewritten in settings/demo_Danube)
    # RDA.read_config_and_save()        # done 2026-10-05 (ensemble 30, todate 2019-12-31)

    '''before the data assimilation'''
    # RDA.config_basin_mask()
    # RDA.get_GRACE_obs(is_diagonal=False)   # done 2026-10-05 (2002-01..2019-12, GRACE/output)

    '''after the data assimilation'''
    RDA.visualization()
    RDA.increment_diagnosis()
    RDA.da_evaluation(tag='run13', reference=DATA_ROOT + '/Res/demo_Danube/evaluation/run12')   # new tag for every run;
    #   reference = a tag in Res/demo_Danube/evaluation or the path of a report folder (the 4-member runs are in demo_2)

    '''data product for distribution (only when the data are shared): daily ensemble mean and spread (OL, DA),
    gridded GRACE TWS, basin time series -> Res/<case>/product/PyGLDA-v2_<case>_<version>/'''
    # RDA.export_product(version='v1.0')   # done 2026-10-06 for run12 -> Res/demo_Danube/product/PyGLDA-v2_demo_Danube_v1.0

    '''house-keeping: remove the temporary daily output of a collected stage (dry run first)'''
    # from src_DA.EnumDA import Stage
    # RDA.clean_temp_output(Stage.OL, dry_run=True)
    # RDA.clean_temp_output(Stage.DA, dry_run=True)
    pass


def demo2():
    '''MPI: mpiexec -n <ens+1> python -u RDA_Ucloud.py Config_ReWaterGAP.json'''
    from src_DA.EnumDA import Stage
    from mpi4py import MPI

    '''settings are synced between machines: rank 0 first writes the paths of this machine into the JSON files'''
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        RDA.config_external_data()
    comm.barrier()

    # RDA.model_perturbation()          # done 2026-10-05 (31 members, 2000-01..2019-12)
    # RDA.spin_up()                     # done 2026-10-05 (restart states 2001-12-31, all 31 members)
    # RDA.OL_run()                                            # done 2026-10-05 (2002-01..2019-12)
    # RDA.collect_and_statistics(Stage.OL, skip_collect=False)  # done 2026-10-05

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
