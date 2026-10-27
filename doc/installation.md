# PyGLDA v2 — installation guide

This guide covers the Python environment for running PyGLDA v2, on a desktop and on an HPC node (the example
is UCloud, the Danish national research cloud), and how to launch the code once it is installed. Everything lives in `installation/`:

| file | purpose |
|---|---|
| `setup_env.sh` | one script for both machines: checks, creates, verifies, exports |
| `environment.yml` | the model environment `pyglda_v2` (WaterGAP pins + PyGLDA additions) |
| `environment_pygmt.yml` | the visualization environment `pygmt` (post-processing figures only) |
| `environment.lock.yml` | exact versions of the last successful desktop install |
| `environment-win.yml`, `setup_env.ps1` | Windows variants (not covered here) |

---

## 1. Why the environment is pinned

WaterGAP 2.2e (vendored in `src_GHM/ReWaterGAP/`) is locked to `numpy==1.23.5` and `numba==0.56.4`,
which are ABI-locked to each other. That forces Python 3.10 and a set of older pins (pandas 1.5.3,
xarray 2023.2.0, netcdf4 1.6.3, ...). The consequences:

- **numpy must never be upgraded** inside `pyglda_v2`. A `pip install` of anything that pulls a newer numpy
  breaks numba silently.
- **pygmt cannot live in the same env**: it needs a modern numpy, and the solver does not finish with the
  1.23.5 pin. Figures are made in a second env, `pygmt`, which does not run the model.
- **MPI is mpich**, pinned in the yml. The launcher and `mpi4py` must come from the same build, so PyGLDA is
  always started with the env's own `mpiexec` (see section 5).
- `hdf5` and `libnetcdf` are pinned to `nompi` builds so netCDF4 and h5py share one library and the solve
  stays fast.

---

## 2. Desktop install (env by name)

Requirements: Linux, Miniforge or any conda with `mamba` (mamba is strongly preferred, the conda solver can
take > 20 min on this spec), ~3 GB of disk for the env.

```bash
cd PyGLDA_v2
bash installation/setup_env.sh            # phase 1: checks and a dry-run solve, installs nothing
bash installation/setup_env.sh install    # phase 2: create, verify, export lock
conda activate pyglda_v2
```

Add `--with-pygmt` to the install line to create the visualization env as well.
Phase 2 refuses to run if the dry-run fails, and never touches `base` or any other env.
The log is `installation/env_setup.log`; the solved versions go to `installation/environment.lock.yml`.

---

## 3. HPC install (env under a persistent prefix; example: UCloud)

On UCloud only two folders survive a job: the code folder `/work/PyGLDA` and the data root `/work/PyGLDA_v2_external_data`. The job's home directory, `/usr/local`, and the default
conda location are wiped. So everything goes under a **prefix** in the data root:

```
/work/PyGLDA_v2_external_data/envs/
├── miniforge3/        private Miniforge (the UCloud image has no conda)
├── pyglda_v2/         the model env
├── pygmt/             the visualization env            (--with-pygmt)
├── activate.sh        source in every shell
├── ucloud_init.sh        select as the job "Initialization" script
├── environment.lock.yml, environment_pygmt.lock.yml
└── env_setup.log, last_job_init.log
```

One-time setup, from any UCloud job that mounts both folders:

```bash
cd /work/PyGLDA
bash installation/setup_env.sh install --prefix /work/PyGLDA_v2_external_data/envs --with-pygmt
```

The script bootstraps Miniforge if no conda is found, creates the envs, runs the verifications of section 4,
and writes `activate.sh` and `ucloud_init.sh`. Re-running it is safe: existing envs are verified, not rebuilt.
Keep large files out of the code folder; the envs folder is deliberately in the data root.

### Every new UCloud job

1. In the submission form mount both folders, enable ssh and tmux, and select
   `/work/PyGLDA_v2_external_data/envs/ucloud_init.sh` in the **Initialization** field.
2. Open a terminal. Every shell now starts with `pyglda_v2` on PATH, and `conda`, `mamba` and
   `conda activate pygmt` work. If the init script was not selected, run
   `source /work/PyGLDA_v2_external_data/envs/activate.sh` yourself.
3. Check `cat /work/PyGLDA_v2_external_data/envs/last_job_init.log`: a fresh timestamp and both
   `rank 0 of 2` and `rank 1 of 2`.
4. Start long stages inside `tmux` (`tmux new -s <name>`, detach with Ctrl+B then D, `tmux attach -t <name>`).
   The init script also switches on mouse-wheel scrolling and a 50 000-line scrollback for tmux; without the
   mouse, scroll with Ctrl+B then `[`, leave with `q`.

Only the ssh port changes between jobs (it is shown on the job page). The host key entry in
`~/.ssh/known_hosts` on the desktop is per port, so a new port adds a new entry.

### Job size

`nproc` on UCloud reports the host's cores, not the job's. The real limits are in
`/sys/fs/cgroup/cpu.max` and `/sys/fs/cgroup/memory.max`. Each rank runs WaterGAP globally and needs
about **2.2 GB**; a 31-rank run (30 members + control) therefore needs ~70 GB and 31 cores. A 4-vCPU /
12 GB job runs the 5-rank smoke test at the edge of its memory and nothing larger.

---

## 4. What the script verifies

All four must pass before the install is declared good (they are the ones that fail silently otherwise):

1. every package imports inside the env with the pinned version;
2. a numba `@njit` function runs on a numpy array (proves the numpy/numba ABI lock holds);
3. `mpirun -n 2` **with the env's launcher** prints `rank 0 of 2` and `rank 1 of 2`;
   two lines of rank 0 mean launcher and mpi4py come from different MPI builds;
4. with `--with-pygmt`: pygmt writes a PNG, which proves GMT and Ghostscript are both found.

Then the exact solved versions are exported to the lock file.

---

## 5. Running PyGLDA

Set the data root and start the drivers of `src_demo/` (see `doc/configuration.md` for the stages):

```bash
export PYGLDA_DATA=/work/PyGLDA_v2_external_data
cd /work/PyGLDA/src_demo
python global_run.py prepare                        # serial stages
mpiexec -n 31 python -u global_run.py ensemble      # MPI stages: ensemble size + 1 ranks
mpiexec -n 31 python -u global_run.py assimilate
python global_run.py evaluate
```

- **Use the env's `mpiexec`.** Inside an activated env it is first on PATH. On UCloud the system
  `/usr/local/bin/mpirun` is Open MPI and starts every rank as rank 0 with no error message.
- WaterGAP parses `sys.argv` at import time and reads `cf_conv.json` from the working directory; the drivers take
  care of both (they pass the run copy of `Config_ReWaterGAP.json` and change to `src_demo/`).

Rank 0 is the unperturbed control member and is never updated; one rank prints to the terminal, every other rank
logs to `parallel_logs/<stage>/rank_k.log` at the repository root.

---

## 6. Data

The data root holds the inputs and receives all outputs:

| folder | content |
|---|---|
| `Input_data/` | WaterGAP 2.2e input: monthly climate forcing (GSWP3-W5E5, converted with `src_auxiliary/ForcingFieldConversion.py`), water use, static land data, global parameter file |
| `Basin/shp/`, `Basin/mask/` | unit shapefiles (GlobalBasins v2.0, Danube, Amazon) and their 0.5° / 1° masks |
| `GRACE/` | CSR RL06.3 mascons (`SaGEA/signal_Mascon`), DDK3 Monte-Carlo error samples (`SaGEA/sample_DDK3`, produced with SaGEA), the WaterGAP land mask (`global_mask/`), and the prepared observations |
| `Auxiliary/` | `smax.nc` (maximum soil water content of WaterGAP) |
| `Ensemble_input/`, `Ensemble_Initialization/`, `OL_output/`, `DA_output/`, `Res/`, `settings/` | written by the runs |

---

## 7. The visualization env `pygmt`

Used only by `src_postprocessing/Visualization.py`, `src_postprocessing/global_visualization.py` and
`src_auxiliary/HydroShed.py`, which import pygmt lazily inside the plotting functions. It uses Python 3.13,
pygmt 0.16, GMT 6.6, plus Ghostscript (GMT cannot write PNG/PDF without it) and `pandas<3`.

```bash
conda activate pygmt
cd /work/PyGLDA
python -c "from src_postprocessing.Visualization import *"
```

Activate it rather than calling `.../pygmt/bin/python` by full path: GMT finds `gs` through PATH and
otherwise fails with `psconvert [ERROR]: Cannot execute Ghostscript (gs)`.
Figure pop-ups are off by default in the pipeline (`allow_pop_up=False`); figures are written to the
`fig_path` of each call. There is no display on UCloud, so leave it that way there.

---

## 8. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `mpirun -n 2` prints `rank 0` twice | system launcher (Open MPI) with mpich mpi4py | use the env's `mpiexec`; source `activate.sh` |
| `conda: command not found` on UCloud | shell hooks not loaded | `source .../envs/activate.sh` (it sources Miniforge's `conda.sh`/`mamba.sh`) |
| `psconvert [ERROR]: Cannot execute Ghostscript (gs)` | `gs` not on PATH | `conda activate pygmt`, do not call its python by full path |
| numba import error or `SystemError` at JIT | numpy was upgraded | rebuild the env from `environment.yml`; never `pip install` into it |
| solver runs for > 20 min | hdf5/libnetcdf anchors removed from the yml | restore the `nompi_*` pins |
| run killed near the end of a stage, no traceback | cgroup memory limit | check `/sys/fs/cgroup/memory.max`; ~2.2 GB per rank |
| config changes do not reach WaterGAP | a `from src_GHM.ReWaterGAP...` import somewhere | bare imports only; `grep -rn "from src_GHM.ReWaterGAP" --include=*.py .` must be empty |
| init script did not run on UCloud | not selected in the form | select `envs/ucloud_init.sh` under Initialization, or source `activate.sh` manually |
