"""
run_stage.py - launch ONE MPI stage of the PyGLDA regional data assimilation.

Used by the walkthrough notebook (PyGLDA_DA_walkthrough.ipynb), which cannot run MPI stages
inside its kernel. Each stage is started as

    mpiexec -n <ens+1> python -u run_stage.py <stage> <case.json> Config_ReWaterGAP.json

    <stage>      spin_up | OL | collect_OL | DA | collect_DA
    <case.json>  the RDA settings written by the notebook (setting_dir, ens, case, basin, ...)
    Config_ReWaterGAP.json  must stay the first *remaining* argument: WaterGAP's own CLI parser
                 reads sys.argv when its controller is imported and expects the config file there.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if len(sys.argv) < 3:
    sys.exit(__doc__)
stage, case_json = sys.argv[1], sys.argv[2]
sys.argv = [sys.argv[0]] + sys.argv[3:]          # leave only what WaterGAP's parser expects

from src_FlowControl.Regional_DA import RDA      # noqa: E402  (after the argv fix, on purpose)
from src_DA.EnumDA import Stage                  # noqa: E402

for key, value in json.load(open(case_json)).items():
    setattr(RDA, key, value)

STAGES = {
    'spin_up':    RDA.spin_up,
    'OL':         RDA.OL_run,
    'collect_OL': lambda: RDA.collect_and_statistics(Stage.OL, skip_collect=False),
    'DA':         RDA.DA_run,
    'collect_DA': lambda: RDA.collect_and_statistics(Stage.DA, skip_collect=False),
}
if stage not in STAGES:
    sys.exit(f"unknown stage {stage!r}; choose from {', '.join(STAGES)}")
STAGES[stage]()
