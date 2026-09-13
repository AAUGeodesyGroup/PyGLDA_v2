#!/usr/bin/env bash
# PyGLDA_v2 environment setup.
#   ./setup_env.sh           -> phase 1: system checks + mamba dry-run only (installs nothing)
#   ./setup_env.sh install   -> phase 2: create env + verify imports + 2-rank MPI test
# Everything is logged to installation/env_setup.log

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YML="$HERE/environment.yml"
LOG="$HERE/env_setup.log"
ENV_NAME="pyglda_v2"
PHASE="${1:-check}"

exec > >(tee "$LOG") 2>&1
echo "=== PyGLDA_v2 env setup | phase=$PHASE | $(date) ==="
echo

section () { echo; echo "----- $1 -----"; }

section "solver"
if command -v mamba >/dev/null 2>&1; then SOLVER=mamba; else SOLVER=conda; fi
echo "using: $SOLVER"
$SOLVER --version 2>&1 | head -3
echo "existing envs:"
$SOLVER env list 2>&1

section "system MPI"
which mpirun mpiexec 2>&1
mpirun --version 2>&1 | head -3
case "$(mpirun --version 2>&1 | head -1)" in
  *"Open MPI"*|*"OpenRTE"*) MPI_VARIANT=openmpi ;;
  *MPICH*|*HYDRA*)          MPI_VARIANT=mpich ;;
  *"Intel"*)                MPI_VARIANT=intel-unknown ;;
  *)                        MPI_VARIANT=none ;;
esac
echo "detected system MPI variant: $MPI_VARIANT"

section "disk space (conda pkgs dir + envs dir)"
$SOLVER info 2>/dev/null | grep -iE "envs directories|package cache" -A2
df -h "$($SOLVER info --base 2>/dev/null || echo "$HOME")" 2>&1

section "GMT (needed by pygmt)"
which gmt 2>&1 && gmt --version 2>&1 || echo "no system gmt (conda-forge will provide one)"

section "environment.yml as it will be used"
if [ ! -f "$YML" ]; then echo "ERROR: $YML not found"; exit 1; fi
# align the MPI pin in the yml with the system launcher so mpirun and mpi4py agree
if [ "$MPI_VARIANT" = mpich ]; then
  sed -i 's/^  - openmpi\b.*/  - mpich/' "$YML"
  echo "(patched yml: openmpi -> mpich to match system mpirun)"
fi
cat "$YML"

section "dry-run solve"
$SOLVER env create -f "$YML" --dry-run 2>&1
DRY_RC=$?
echo "dry-run exit code: $DRY_RC"

if [ "$PHASE" != install ]; then
  echo
  echo "=== check phase done. Nothing was installed. Rerun with 'install' to proceed. ==="
  exit $DRY_RC
fi

if [ $DRY_RC -ne 0 ]; then
  echo "dry-run failed; refusing to install."; exit $DRY_RC
fi

section "create environment $ENV_NAME"
$SOLVER env create -f "$YML" -y 2>&1
CREATE_RC=$?
echo "create exit code: $CREATE_RC"
[ $CREATE_RC -ne 0 ] && { echo "create failed"; exit $CREATE_RC; }

section "verify: versions and imports"
$SOLVER run -n "$ENV_NAME" python - <<'PY'
import importlib, sys
print("python", sys.version.split()[0])
pkgs = ["numpy","numba","pandas","xarray","netCDF4","dask","tqdm","termcolor",
        "matplotlib","sklearn","mpi4py","h5py","scipy","shapely","geopandas"]
bad = []
for p in pkgs:
    try:
        m = importlib.import_module(p)
        print(f"  OK   {p:12s} {getattr(m,'__version__','?')}")
    except Exception as e:
        bad.append(p); print(f"  FAIL {p:12s} {type(e).__name__}: {e}")
# the one that matters most: numba must JIT against this numpy
import numba, numpy as np
@numba.njit
def f(a): return a.sum()
print("  numba JIT on numpy array ->", f(np.arange(5.0)))
print("IMPORT SUMMARY:", "all OK" if not bad else f"FAILED: {bad}")
PY

section "verify: 2-rank MPI (must print rank 0 and rank 1, not 0 and 0)"
$SOLVER run -n "$ENV_NAME" mpirun -n 2 python -c \
  "from mpi4py import MPI; c=MPI.COMM_WORLD; print(f'rank {c.Get_rank()} of {c.Get_size()}  |  {MPI.Get_library_version().splitlines()[0]}')" 2>&1
echo
echo "which mpirun inside env: $($SOLVER run -n "$ENV_NAME" which mpirun 2>/dev/null)"
echo "which mpirun on system : $(which mpirun 2>/dev/null)"

section "export exact solved versions"
$SOLVER env export -n "$ENV_NAME" --no-builds > "$HERE/environment.lock.yml" 2>&1 && \
  echo "wrote installation/environment.lock.yml"

echo
echo "=== install phase done | $(date) ==="
