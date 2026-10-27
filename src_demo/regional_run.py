"""
PyGLDA v2 - regional experiments: GRACE (CSR RL06.3 mascons) assimilated into WaterGAP 2.2e for the sub-basins of the
Danube (6) or the Amazon (18), 30 members (settings/RDA_Danube, settings/RDA_Amazon). WaterGAP always runs on the
global grid, so the regional cases use the global open loop of src_demo/global_run.py (same ensemble folders and
period): only its collection for the sub-basins of the case is needed. Without a global open loop, the stage
"ensemble" builds one (it writes the shared ensemble folders).

Run from any folder; the case and the stage are chosen on the command line:
    python regional_run.py Danube prepare                       run settings, sub-basin mask, GRACE observations
    mpiexec -n 31 python -u regional_run.py Danube open_loop    collection of the (global) open loop for the case
    mpiexec -n 31 python -u regional_run.py Danube ensemble     perturbation, spin-up, open loop and its collection
    mpiexec -n 31 python -u regional_run.py Danube assimilate   data assimilation and its collection
    python regional_run.py Danube evaluate                      evaluation report and figures
The data root (inputs, GRACE, masks, outputs) is taken from the environment variable PYGLDA_DATA.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
sys.path.insert(0, str(CODE_ROOT))

from src_FlowControl.Regional_DA import RDA
from src_DA.EnumDA import Stage

DATA_ROOT = os.environ.get('PYGLDA_DATA', '/work/PyGLDA_v2_external_data')

CASES = {'Danube': dict(case='RDA_Danube', basin='Danube', shp='Basin/shp/Danube/Danube.shp'),
         'Amazon': dict(case='RDA_Amazon', basin='Amazon', shp='Basin/shp/Amazon/Amazon.shp')}
STAGES = ('prepare', 'open_loop', 'ensemble', 'assimilate', 'evaluate')


def configure(name):
    c = CASES[name]
    RDA.case = c['case']
    RDA.basin = c['basin']
    RDA.ens = 30
    RDA.external_data_path = DATA_ROOT
    RDA.shp_path = DATA_ROOT + '/' + c['shp']
    '''spin-up: 8 x the first year, then the whole spin-up period (10 model years per member)'''
    RDA.spin_up_start = '2000-01-01'
    RDA.spin_up_end = '2001-12-31'
    RDA.spin_up_years = 8
    '''open loop and data assimilation'''
    RDA.sim_begin_time = '2002-01-01'
    RDA.sim_end_time = '2006-12-31'
    return CODE_ROOT / 'settings' / RDA.case


def settings(source, mpi):
    """run copy of the settings (written by one process, read by all)"""
    if not mpi:
        RDA.prepare_settings(source)
        return
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        RDA.prepare_settings(source)
    comm.barrier()
    if comm.Get_rank() != 0:
        RDA.prepare_settings(source, write=False)


def main(name, stage, tag=None):
    os.chdir(HERE)                                  # WaterGAP reads cf_conv.json from the working directory
    settings(configure(name), mpi=stage in ('open_loop', 'ensemble', 'assimilate'))
    sys.argv = [sys.argv[0], str(Path(RDA.setting_dir) / 'Config_ReWaterGAP.json')]   # WaterGAP's own CLI
    if stage == 'prepare':
        RDA.config_basin_mask()
        RDA.get_GRACE_obs(is_diagonal=False)
    elif stage == 'open_loop':
        RDA.collect_and_statistics(Stage.OL)
    elif stage == 'ensemble':
        RDA.model_perturbation()
        RDA.spin_up()
        RDA.OL_run()
        RDA.collect_and_statistics(Stage.OL)
    elif stage == 'assimilate':
        RDA.DA_run()
        RDA.collect_and_statistics(Stage.DA)
    elif stage == 'evaluate':
        RDA.da_evaluation(tag=tag)
        RDA.visualization()


if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[1] not in CASES or sys.argv[2] not in STAGES:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], tag=sys.argv[3] if len(sys.argv) > 3 else 'run')
