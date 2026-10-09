"""
PyGLDA v2 driver - UCloud experiments before the global run (doc/ucloud_notes.md, READ FIRST section 6), 10 Oct 2026.
Each experiment is its own case (own settings folder, own Res/<case>), so no earlier result is overwritten.

    EXP=exp1   Danube, snow cap + cap-aware snow noise                      case demo_Danube_exp1
    EXP=exp2a  Amazon, riverstor in the DA state, NOT in the inflation        case demo_Amazon_A1  (= global settings)
    EXP=exp2b  Amazon, riverstor in the DA state AND in the inflation         case demo_Amazon_A2
    EXP=exp4   Danube as exp1 over the full period 2002-2019 (snow ratchet)  case demo_Danube_exp4
All: 30 members, 2002-01-01..2005-12-31 (exp4: ..2019-12-31), the shared global open loop (OL_output/Ens_k) and the
restarts of 2001-12-31.

Steps, from src_demo/ (the experiment and the step are chosen with environment variables; the argument stays the
model configuration as for RDA_Ucloud.py):
    EXP=exp1 STEP=prepare  python -u exp_Ucloud.py Config_ReWaterGAP.json                  serial
    EXP=exp1 STEP=run      mpiexec -n 31 python -u exp_Ucloud.py Config_ReWaterGAP.json    MPI
    EXP=exp1 STEP=evaluate python -u exp_Ucloud.py Config_ReWaterGAP.json                  serial
Order: exp1 -> exp2a -> exp2b -> exp4, each prepare/run/evaluate. Evaluate right after
each run: parallel_logs/DA, <basin>_obs_GRACE.hdf5 and the state envelope of a basin are overwritten by the next run.
"""
import os
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent          # the PyGLDA_v2 folder
sys.path.insert(0, str(CODE_ROOT))

from src_FlowControl.Regional_DA import RDA
from src_DA.EnumDA import Stage

DATA_ROOT = '/work/PyGLDA_v2_external_data'

EXPERIMENTS = {
    'exp1': dict(case='demo_Danube_exp1', basin='Danube', shp='Danube/Danube.shp', prepare_basin=False,
                 end='2005-12-31', reference=DATA_ROOT + '/Res/demo_Danube/evaluation/run13'),
    'exp2a': dict(case='demo_Amazon_A1', basin='Amazon', shp='Amazon/Amazon.shp', prepare_basin=True,
                  end='2005-12-31', reference=None),
    'exp2b': dict(case='demo_Amazon_A2', basin='Amazon', shp='Amazon/Amazon.shp', prepare_basin=False,
                  end='2005-12-31', reference=DATA_ROOT + '/Res/demo_Amazon_A1/evaluation/exp2a'),
    'exp4': dict(case='demo_Danube_exp4', basin='Danube', shp='Danube/Danube.shp', prepare_basin=False,
                 end='2019-12-31', reference=DATA_ROOT + '/Res/demo_Danube/evaluation/run13'),
}

EXP = os.environ.get('EXP', '')
STEP = os.environ.get('STEP', '')
if EXP not in EXPERIMENTS or STEP not in ('prepare', 'run', 'evaluate'):
    raise SystemExit('exp_Ucloud.py: set EXP to one of %s and STEP to prepare | run | evaluate (got EXP=%r STEP=%r)'
                     % (list(EXPERIMENTS), EXP, STEP))
cfg = EXPERIMENTS[EXP]

RDA.case = cfg['case']
RDA.basin = cfg['basin']
RDA.ens = 30
RDA.setting_dir = str(CODE_ROOT / 'settings' / RDA.case)
RDA.external_data_path = DATA_ROOT
RDA.shp_path = DATA_ROOT + '/Basin/shp/' + cfg['shp']

'''spin-up of the shared ensemble (not run here; restarts 2001-12-31 in Ensemble_Initialization/Ens_k)'''
RDA.spin_up_start = '2000-01-01'
RDA.spin_up_end = '2001-12-31'
RDA.spin_up_years = 5

'''open loop (collected from the shared global daily OL) and data assimilation'''
RDA.sim_begin_time = '2002-01-01'
RDA.sim_end_time = cfg['end']


def prepare():
    '''serial: paths of this machine, case / basin / ensemble into DA_setting.json; for a new basin also the masks
    (WaterGAP land mask) and the GRACE observations of the period'''
    RDA.config_external_data()
    RDA.read_config_and_save()
    if cfg['prepare_basin']:
        RDA.config_basin_mask()
        RDA.get_GRACE_obs(is_diagonal=False)
    pass


def run():
    '''MPI (31 ranks): collect the shared global open loop for this case and period, data assimilation, collect DA'''
    from mpi4py import MPI

    '''settings are synced between machines: rank 0 first writes the paths of this machine into the JSON files'''
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        RDA.config_external_data()
    comm.barrier()

    '''open loop of the case: Res/<case>/OL/Ens_k/daily_output_<year>.nc, basin series, Res_OL.h5 (ol_spread split),
    harmonic maps; the global daily OL in OL_output/Ens_k is only read'''
    RDA.collect_and_statistics(Stage.OL)

    try:
        RDA.DA_run()
    except Exception:
        import traceback
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
    RDA.collect_and_statistics(Stage.DA)
    pass


def evaluate():
    '''serial, right after the run: increment diagnosis and evaluation report -> Res/<case>/evaluation/<EXP>/'''
    RDA.increment_diagnosis(tag=EXP)
    RDA.da_evaluation(tag=EXP, reference=cfg['reference'])
    pass


if __name__ == '__main__':
    dict(prepare=prepare, run=run, evaluate=evaluate)[STEP]()
