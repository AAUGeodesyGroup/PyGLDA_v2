"""
PyGLDA v2 - global experiment: GRACE (CSR RL06.3 mascons) assimilated into WaterGAP 2.2e on the 737 hydrological units
of GlobalBasins v2.0, 30 members (settings/GDA_Global).

Run from any folder; the stages are chosen on the command line:
    python global_run.py prepare                       run settings, unit check, GRACE observations of the units
    mpiexec -n 31 python -u global_run.py ensemble     perturbation, spin-up, open loop and its collection
    mpiexec -n 31 python -u global_run.py assimilate   data assimilation and its collection
    python global_run.py evaluate                      evaluation report and figures (Res/GDA_Global/evaluation_global)
The data root (inputs, GRACE, masks, outputs) is taken from the environment variable PYGLDA_DATA.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
sys.path.insert(0, str(CODE_ROOT))

from src_FlowControl.Global_DA import GDA
from src_DA.EnumDA import Stage

DATA_ROOT = os.environ.get('PYGLDA_DATA', '/work/PyGLDA_v2_external_data')

GDA.case = 'GDA_Global'
GDA.basin = 'GlobalBasins_v2.0'
GDA.ens = 30
GDA.external_data_path = DATA_ROOT
GDA.shp_path = DATA_ROOT + '/Basin/shp/Global_v2.0/GlobalBasins_v2.0.shp'

'''spin-up: WaterGAP repeats the first year spin_up_years times and then runs the whole spin-up period
(8 x 2000 + 2000-2001 = 10 model years per member, with its own parameters and forcing)'''
GDA.spin_up_start = '2000-01-01'
GDA.spin_up_end = '2001-12-31'
GDA.spin_up_years = 8

'''open loop and data assimilation'''
GDA.sim_begin_time = '2002-01-01'
GDA.sim_end_time = '2012-12-31'

SETTINGS = CODE_ROOT / 'settings' / GDA.case
STAGES = ('prepare', 'ensemble', 'assimilate', 'evaluate')


def settings(mpi):
    """run copy of the settings (written by one process, read by all)"""
    if not mpi:
        GDA.prepare_settings(SETTINGS)
        return
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        GDA.prepare_settings(SETTINGS)
    comm.barrier()
    if comm.Get_rank() != 0:
        GDA.prepare_settings(SETTINGS, write=False)


def main(stage, tag=None):
    os.chdir(HERE)                                  # WaterGAP reads cf_conv.json from the working directory
    settings(mpi=stage in ('ensemble', 'assimilate'))
    sys.argv = [sys.argv[0], str(Path(GDA.setting_dir) / 'Config_ReWaterGAP.json')]   # WaterGAP's own CLI
    if stage == 'prepare':
        GDA.check_units()
        GDA.get_GRACE_obs()
    elif stage == 'ensemble':
        GDA.model_perturbation()
        GDA.spin_up()
        GDA.OL_run()
        GDA.collect_and_statistics(Stage.OL)
    elif stage == 'assimilate':
        GDA.DA_run()
        GDA.collect_and_statistics(Stage.DA)
    elif stage == 'evaluate':
        GDA.da_evaluation(tag=tag)
        GDA.visualization(tag=tag)


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        sys.exit(__doc__)
    main(sys.argv[1], tag=sys.argv[2] if len(sys.argv) > 2 else 'run')
