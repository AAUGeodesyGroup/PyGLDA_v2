#!/usr/bin/env bash
# PyGLDA_v2 environment setup — desktop and UCloud (or any cluster with persistent storage).
#
#   bash setup_env.sh                        phase 1: system checks + dry-run solve. Installs nothing.
#   bash setup_env.sh install                phase 2: create env "pyglda_v2" in conda's default envs dir,
#                                            verify imports / numba / 2-rank MPI, export the lock file.
#   bash setup_env.sh install --prefix DIR   same, but everything lives under DIR (persistent storage,
#                                            e.g. /work/PyGLDA_v2_external_data/envs on UCloud):
#                                              DIR/miniforge3   private Miniforge, bootstrapped if no conda is found
#                                              DIR/pyglda_v2    the model env
#                                              DIR/pygmt        the visualization env   (--with-pygmt)
#                                              DIR/claude_home  Claude Code             (--with-claude)
#                                              DIR/activate.sh  source this in every shell
#                                              DIR/ucloud_init.sh  select as the UCloud job "Initialization" script
#   options:  --with-pygmt    also create the visualization env from environment_pygmt.yml
#             --with-claude   also install Claude Code (prefix mode only)
#
# Re-running is safe: an env that already exists is verified, not recreated.
# Log: installation/env_setup.log (default mode) or DIR/env_setup.log (prefix mode).
# Full guide: doc/installation.md

set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YML="$HERE/environment.yml"
PYGMT_YML="$HERE/environment_pygmt.yml"
ENV_NAME="pyglda_v2"
PYGMT_NAME="pygmt"

PHASE=check; PREFIX=""; WITH_PYGMT=0; WITH_CLAUDE=0
while [ $# -gt 0 ]; do
  case "$1" in
    install)        PHASE=install ;;
    check)          PHASE=check ;;
    --prefix)       PREFIX="$2"; shift ;;
    --prefix=*)     PREFIX="${1#--prefix=}" ;;
    --with-pygmt)   WITH_PYGMT=1 ;;
    --with-claude)  WITH_CLAUDE=1 ;;
    -h|--help)      sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1"; exit 2 ;;
  esac
  shift
done
if [ $WITH_CLAUDE -eq 1 ] && [ -z "$PREFIX" ]; then
  echo "--with-claude needs --prefix DIR (Claude Code is installed under DIR/claude_home)"; exit 2
fi

if [ -n "$PREFIX" ]; then
  mkdir -p "$PREFIX" || { echo "cannot create $PREFIX"; exit 1; }
  PREFIX="$(cd "$PREFIX" && pwd)"
  LOG="$PREFIX/env_setup.log"
else
  LOG="$HERE/env_setup.log"
fi
exec > >(tee "$LOG") 2>&1
echo "=== PyGLDA_v2 env setup | phase=$PHASE | prefix=${PREFIX:-<conda default>} | pygmt=$WITH_PYGMT claude=$WITH_CLAUDE | $(date) ==="
section () { echo; echo "----- $1 -----"; }
die ()     { echo "ERROR: $*"; exit 1; }

# ── solver ────────────────────────────────────────────────────────────────────
section "solver"
SOLVER=""
if [ -n "$PREFIX" ] && [ -x "$PREFIX/miniforge3/bin/mamba" ]; then
  SOLVER="$PREFIX/miniforge3/bin/mamba"; echo "using private Miniforge: $SOLVER"
elif command -v mamba >/dev/null 2>&1; then SOLVER=mamba
elif command -v conda >/dev/null 2>&1; then SOLVER=conda
fi
if [ -z "$SOLVER" ]; then
  [ -z "$PREFIX" ] && die "no conda/mamba on PATH. Install Miniforge, or use --prefix DIR to bootstrap a private one."
  echo "no conda/mamba found; a private Miniforge will be bootstrapped under $PREFIX/miniforge3"
  if [ "$PHASE" = install ]; then
    INST="$PREFIX/Miniforge3-Linux-x86_64.sh"
    [ -f "$INST" ] || curl -fsSL -o "$INST" \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh || die "download failed"
    bash "$INST" -b -p "$PREFIX/miniforge3" || die "Miniforge install failed"
    SOLVER="$PREFIX/miniforge3/bin/mamba"
  else
    echo "(check phase: nothing downloaded; the dry-run below is skipped)"
  fi
fi
if [ -n "$SOLVER" ]; then
  echo "using: $SOLVER"; $SOLVER --version 2>&1 | head -3
  export MAMBA_ROOT_PREFIX="$(dirname "$(dirname "$(command -v "$SOLVER")")")"
  echo "existing envs:"; $SOLVER env list 2>&1
fi

# ── system MPI ─────────────────────────────────────────────────────────────────
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
echo "NOTE: PyGLDA is always launched with the env's own mpirun/mpiexec (mpich). A different system launcher"
echo "      (e.g. Open MPI on the UCloud image) starts every rank as rank 0 without any error message."

# ── resources ──────────────────────────────────────────────────────────────────
section "resources"
echo "cores visible: $(nproc)  | cgroup cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)"
echo "cgroup memory.max: $( m=$(cat /sys/fs/cgroup/memory.max 2>/dev/null); [ -n "$m" ] && [ "$m" != max ] && echo $((m/1024/1024)) MB || echo "${m:-n/a}")"
echo "(each global WaterGAP rank needs ~2.2 GB; a 31-rank run needs ~70 GB and 31 cores)"
df -h "${PREFIX:-$HERE}" 2>&1 | tail -2

# ── yml ────────────────────────────────────────────────────────────────────────
section "environment.yml as it will be used"
[ -f "$YML" ] || die "$YML not found"
if [ "$MPI_VARIANT" = mpich ]; then
  sed -i 's/^  - openmpi\b.*/  - mpich/' "$YML"
  echo "(patched yml: openmpi -> mpich to match system mpirun)"
fi
cat "$YML"
if [ $WITH_PYGMT -eq 1 ]; then
  [ -f "$PYGMT_YML" ] || die "$PYGMT_YML not found"
  echo; echo "----- environment_pygmt.yml -----"; cat "$PYGMT_YML"
fi

# env target: name or prefix
if [ -n "$PREFIX" ]; then
  ENV_DIR="$PREFIX/$ENV_NAME";   ENV_TARGET=(-p "$ENV_DIR")
  PYGMT_DIR="$PREFIX/$PYGMT_NAME"; PYGMT_TARGET=(-p "$PYGMT_DIR")
else
  ENV_TARGET=(-n "$ENV_NAME");     PYGMT_TARGET=(-n "$PYGMT_NAME")
  ENV_DIR="$($SOLVER env list 2>/dev/null | awk -v n="$ENV_NAME" '$1==n{print $NF}')"
  PYGMT_DIR="$($SOLVER env list 2>/dev/null | awk -v n="$PYGMT_NAME" '$1==n{print $NF}')"
fi

# ── dry run ────────────────────────────────────────────────────────────────────
section "dry-run solve"
DRY_RC=0
if [ -n "$SOLVER" ]; then
  if [ -n "$ENV_DIR" ] && [ -x "$ENV_DIR/bin/python" ]; then
    echo "env already exists at $ENV_DIR — dry-run skipped"
  else
    $SOLVER env create -f "$YML" "${ENV_TARGET[@]}" --dry-run 2>&1; DRY_RC=$?
    echo "dry-run exit code: $DRY_RC"
  fi
fi
if [ "$PHASE" != install ]; then
  echo; echo "=== check phase done. Nothing was installed. Rerun with 'install' to proceed. ==="; exit $DRY_RC
fi
[ $DRY_RC -ne 0 ] && die "dry-run failed; refusing to install."

# ── create pyglda_v2 ───────────────────────────────────────────────────────────
section "create environment $ENV_NAME"
if [ -n "$ENV_DIR" ] && [ -x "$ENV_DIR/bin/python" ]; then
  echo "already exists at $ENV_DIR — not recreated (delete it to rebuild)"
else
  $SOLVER env create -f "$YML" "${ENV_TARGET[@]}" -y 2>&1 || die "create failed"
  [ -z "$PREFIX" ] && ENV_DIR="$($SOLVER env list 2>/dev/null | awk -v n="$ENV_NAME" '$1==n{print $NF}')"
fi
[ -x "$ENV_DIR/bin/python" ] || die "cannot locate the env's python under $ENV_DIR"
echo "env dir: $ENV_DIR"
ENV_PY="$ENV_DIR/bin/python"; ENV_MPIRUN="$ENV_DIR/bin/mpirun"

# ── verify ─────────────────────────────────────────────────────────────────────
section "verify: versions and imports"
"$ENV_PY" - <<'PY' || die "import verification failed"
import importlib, sys
print("python", sys.version.split()[0], sys.executable)
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
sys.exit(1 if bad else 0)
PY

section "verify: 2-rank MPI with the env's launcher (must print rank 0 and rank 1)"
[ -x "$ENV_MPIRUN" ] || die "no mpirun inside the env"
MPI_OUT="$("$ENV_MPIRUN" -n 2 "$ENV_PY" -c \
  "from mpi4py import MPI; c=MPI.COMM_WORLD; print(f'rank {c.Get_rank()} of {c.Get_size()}  |  {MPI.Get_library_version().splitlines()[0]}')" 2>&1)"
echo "$MPI_OUT"
if echo "$MPI_OUT" | grep -q "rank 0 of 2" && echo "$MPI_OUT" | grep -q "rank 1 of 2"; then
  echo "MPI OK"
else
  die "MPI test failed: launcher and mpi4py disagree (every rank reports 0) or mpirun crashed"
fi
echo "env mpirun   : $ENV_MPIRUN"
echo "system mpirun: $(which mpirun 2>/dev/null || echo none)  <- do NOT use this one"

section "export exact solved versions"
if [ -n "$PREFIX" ]; then LOCK="$PREFIX/environment.lock.yml"; else LOCK="$HERE/environment.lock.yml"; fi
$SOLVER env export -p "$ENV_DIR" --no-builds > "$LOCK" 2>&1 && echo "wrote $LOCK"

# ── optional: pygmt env ────────────────────────────────────────────────────────
if [ $WITH_PYGMT -eq 1 ]; then
  section "create visualization environment $PYGMT_NAME"
  if [ -n "$PYGMT_DIR" ] && [ -x "$PYGMT_DIR/bin/python" ]; then
    echo "already exists at $PYGMT_DIR — not recreated"
  else
    $SOLVER env create -f "$PYGMT_YML" "${PYGMT_TARGET[@]}" -y 2>&1 || die "pygmt env create failed"
    [ -z "$PREFIX" ] && PYGMT_DIR="$($SOLVER env list 2>/dev/null | awk -v n="$PYGMT_NAME" '$1==n{print $NF}')"
  fi
  echo "env dir: $PYGMT_DIR"
  section "verify: pygmt writes a png (needs gs on PATH)"
  PATH="$PYGMT_DIR/bin:$PATH" "$PYGMT_DIR/bin/python" - <<'PY' || die "pygmt verification failed"
import shutil, tempfile, os, pygmt, pandas, numpy
print("pygmt", pygmt.__version__, "| pandas", pandas.__version__, "| numpy", numpy.__version__, "| gs:", shutil.which("gs"))
with pygmt.clib.Session() as s: print("GMT", s.info["version"])
out = os.path.join(tempfile.mkdtemp(), "t.png")
fig = pygmt.Figure(); fig.basemap(region=[0,10,0,10], projection="X5c", frame=True); fig.savefig(out)
print("png written:", os.path.getsize(out), "bytes"); shutil.rmtree(os.path.dirname(out))
PY
  if [ -n "$PREFIX" ]; then
    $SOLVER env export -p "$PYGMT_DIR" --no-builds > "$PREFIX/environment_pygmt.lock.yml" 2>&1 && echo "wrote $PREFIX/environment_pygmt.lock.yml"
  fi
fi

# ── optional: Claude Code ──────────────────────────────────────────────────────
if [ $WITH_CLAUDE -eq 1 ]; then
  section "Claude Code (native installer, HOME redirected so the binary lands under the prefix)"
  CLAUDE_HOME="$PREFIX/claude_home"; mkdir -p "$CLAUDE_HOME" "$PREFIX/claude_config"
  if [ -x "$CLAUDE_HOME/.local/bin/claude" ]; then
    echo "already installed: $("$CLAUDE_HOME/.local/bin/claude" --version 2>&1 | head -1)"
  else
    curl -fsSL https://claude.ai/install.sh -o "$PREFIX/claude_install.sh" || die "download failed"
    HOME="$CLAUDE_HOME" bash "$PREFIX/claude_install.sh" 2>&1 | tail -3
    [ -x "$CLAUDE_HOME/.local/bin/claude" ] || die "Claude Code install failed"
    echo "installed: $("$CLAUDE_HOME/.local/bin/claude" --version 2>&1 | head -1)"
  fi
  echo "first use: source $PREFIX/activate.sh && claude   (interactive login once; it is kept in $PREFIX/claude_config)"
fi

# ── prefix mode: activation + job init scripts ─────────────────────────────────
if [ -n "$PREFIX" ]; then
  section "write $PREFIX/activate.sh and $PREFIX/ucloud_init.sh"
  cat > "$PREFIX/activate.sh" <<EOF
# PyGLDA_v2 — source this in every shell:   source $PREFIX/activate.sh
# (generated by installation/setup_env.sh on $(date +%F); ucloud_init.sh adds this line to ~/.bashrc automatically)
ENVS="$PREFIX"
export MAMBA_ROOT_PREFIX="\$ENVS/miniforge3"
[ -f "\$ENVS/miniforge3/etc/profile.d/conda.sh" ] && source "\$ENVS/miniforge3/etc/profile.d/conda.sh"
[ -f "\$ENVS/miniforge3/etc/profile.d/mamba.sh" ] && source "\$ENVS/miniforge3/etc/profile.d/mamba.sh"
export CONDA_ENVS_PATH="\$ENVS"                                            # "conda activate pyglda_v2 | pygmt" by name
export PATH="\$ENVS/$ENV_NAME/bin:\$ENVS/claude_home/.local/bin:\$PATH"    # env mpiexec must shadow the system launcher
export CLAUDE_CONFIG_DIR="\$ENVS/claude_config"                            # Claude Code login/settings survive job restarts
export DISABLE_AUTOUPDATER=1                                               # updates would land in the ephemeral ~/.local
mkdir -p "\$CLAUDE_CONFIG_DIR"
echo "$ENV_NAME activated: \$(which python) | \$(which mpiexec) | claude: \$(which claude 2>/dev/null || echo not installed)"
EOF
  cat > "$PREFIX/ucloud_init.sh" <<EOF
#!/usr/bin/env bash
# UCloud job "Initialization" script (select it in the job submission form). Generated by setup_env.sh.
# Runs once at job start in its own process, so it cannot "source" into your terminals; instead it makes
# every new shell source activate.sh, and writes a 2-rank MPI check to $PREFIX/last_job_init.log.
LINE="source $PREFIX/activate.sh"
for rc in "\$HOME/.bashrc" "\$HOME/.zshrc"; do
  grep -qF "\$LINE" "\$rc" 2>/dev/null || printf "\\n# PyGLDA env (added by ucloud_init.sh)\\n%s\\n" "\$LINE" >> "\$rc"
done
# tmux: mouse-wheel scrolling and a long scrollback for long runs (the job home is fresh every job;
# the UCloud image ships "Oh my tmux", whose ~/.tmux.conf.local is sourced last)
TC="\$HOME/.tmux.conf.local"; [ -f "\$TC" ] || TC="\$HOME/.tmux.conf"
grep -q "^set -g mouse on" "\$TC" 2>/dev/null || printf "\\n# PyGLDA (added by ucloud_init.sh)\\nset -g mouse on\\nset -g history-limit 50000\\n" >> "\$TC"
source "$PREFIX/activate.sh" >/dev/null
{ date; mpiexec -n 2 python -c "from mpi4py import MPI; print('rank', MPI.COMM_WORLD.Get_rank(), 'of', MPI.COMM_WORLD.Get_size())"; } > "$PREFIX/last_job_init.log" 2>&1
EOF
  chmod +x "$PREFIX/ucloud_init.sh"
  echo "ok"
fi

echo
echo "=== install phase done | $(date) ==="
if [ -n "$PREFIX" ]; then
  echo "next:  source $PREFIX/activate.sh"
else
  echo "next:  conda activate $ENV_NAME"
fi
