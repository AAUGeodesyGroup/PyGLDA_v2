<#
PyGLDA_v2 environment setup — Windows / PowerShell port of setup_env.sh.

  .\setup_env.ps1            -> phase 1: system checks + solver dry-run only (installs nothing)
  .\setup_env.ps1 install    -> phase 2: create env + verify imports + 2-rank MPI test

Run it from an "Anaconda/Miniforge PowerShell Prompt" (or any shell where conda/mamba is on PATH).
If scripts are blocked:  powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
Everything is logged to installation\env_setup_win.log

MPI on Windows: the env gets the MS-MPI libraries (msmpi package) for mpi4py, but the LAUNCHER
mpiexec.exe must come from Microsoft's MS-MPI runtime (msmpisetup.exe). Install that once.
#>
param(
    [ValidateSet('check', 'install')]
    [string]$Phase = 'check'
)

$Here    = Split-Path -Parent $MyInvocation.MyCommand.Path
$Yml     = Join-Path $Here 'environment-win.yml'
$Log     = Join-Path $Here 'env_setup_win.log'
$Lock    = Join-Path $Here 'environment-win.lock.yml'
$EnvName = 'pyglda_v2'

Start-Transcript -Path $Log -Force | Out-Null
Write-Host "=== PyGLDA_v2 env setup (Windows) | phase=$Phase | $(Get-Date) ==="

function Section([string]$Title) { Write-Host "`n----- $Title -----" }

function Finish([int]$Code) { Stop-Transcript | Out-Null; exit $Code }

# ---------------------------------------------------------------- solver
Section 'solver'
if (Get-Command mamba -ErrorAction SilentlyContinue)      { $Solver = 'mamba' }
elseif (Get-Command conda -ErrorAction SilentlyContinue)  { $Solver = 'conda' }
else {
    Write-Host 'ERROR: neither mamba nor conda is on PATH. Open an Anaconda/Miniforge PowerShell prompt.'
    Finish 1
}
Write-Host "using: $Solver"
& $Solver --version 2>&1 | Select-Object -First 3
Write-Host 'existing envs:'
& $Solver env list 2>&1

# ---------------------------------------------------------------- MS-MPI runtime
Section 'MS-MPI runtime (mpiexec.exe)'
$MpiExec = $null
$cmd = Get-Command mpiexec.exe -ErrorAction SilentlyContinue
if ($cmd) { $MpiExec = $cmd.Source }
elseif ($env:MSMPI_BIN -and (Test-Path (Join-Path $env:MSMPI_BIN 'mpiexec.exe'))) {
    $MpiExec = Join-Path $env:MSMPI_BIN 'mpiexec.exe'
}
elseif (Test-Path 'C:\Program Files\Microsoft MPI\Bin\mpiexec.exe') {
    $MpiExec = 'C:\Program Files\Microsoft MPI\Bin\mpiexec.exe'
}
if ($MpiExec) {
    Write-Host "found: $MpiExec"
    if ($MpiExec -notmatch 'Microsoft MPI') {
        Write-Host 'WARNING: this mpiexec is not from the MS-MPI runtime (Intel MPI or MPICH?).'
        Write-Host '         mpi4py in this env is built against MS-MPI; use the MS-MPI mpiexec to launch.'
    }
} else {
    Write-Host 'NOT FOUND. Install the MS-MPI runtime (msmpisetup.exe) from Microsoft:'
    Write-Host '  https://learn.microsoft.com/message-passing-interface/microsoft-mpi'
    Write-Host 'The env can still be created, but the 2-rank test and launching PyGLDA need it.'
}

# ---------------------------------------------------------------- disk space
Section 'disk space (conda base drive)'
$Base = (& $Solver info --base 2>$null | Select-Object -Last 1)
if ($Base) {
    Write-Host "conda base: $Base"
    $Drive = (Get-Item $Base).PSDrive
    Get-PSDrive $Drive.Name | Select-Object Name, @{n = 'FreeGB'; e = { [math]::Round($_.Free / 1GB, 1) } },
                                                    @{n = 'UsedGB'; e = { [math]::Round($_.Used / 1GB, 1) } } | Format-Table -AutoSize
}

# ---------------------------------------------------------------- yml
Section 'environment-win.yml as it will be used'
if (-not (Test-Path $Yml)) { Write-Host "ERROR: $Yml not found"; Finish 1 }
Get-Content $Yml

# ---------------------------------------------------------------- dry run
Section 'dry-run solve'
& $Solver env create -f $Yml --dry-run 2>&1
$DryRc = $LASTEXITCODE
Write-Host "dry-run exit code: $DryRc"

if ($Phase -ne 'install') {
    Write-Host "`n=== check phase done. Nothing was installed. Rerun with 'install' to proceed. ==="
    Finish $DryRc
}
if ($DryRc -ne 0) { Write-Host 'dry-run failed; refusing to install.'; Finish $DryRc }

# ---------------------------------------------------------------- create
Section "create environment $EnvName"
& $Solver env create -f $Yml -y 2>&1
$CreateRc = $LASTEXITCODE
Write-Host "create exit code: $CreateRc"
if ($CreateRc -ne 0) { Write-Host 'create failed'; Finish $CreateRc }

# ---------------------------------------------------------------- verify imports + numba JIT
Section 'verify: versions and imports'
$VerifyPy = Join-Path $env:TEMP 'pyglda_verify_env.py'
@'
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
'@ | Set-Content -Path $VerifyPy -Encoding UTF8
& $Solver run -n $EnvName python $VerifyPy 2>&1

# ---------------------------------------------------------------- verify MPI (2 ranks)
Section 'verify: 2-rank MPI (must print rank 0 and rank 1, not 0 and 0)'
$MpiPy = Join-Path $env:TEMP 'pyglda_verify_mpi.py'
@'
from mpi4py import MPI
c = MPI.COMM_WORLD
print(f"rank {c.Get_rank()} of {c.Get_size()}  |  {MPI.Get_library_version().splitlines()[0]}")
'@ | Set-Content -Path $MpiPy -Encoding UTF8
if ($MpiExec) {
    # python resolved inside the env; mpiexec is the system MS-MPI launcher
    $EnvPython = (& $Solver run -n $EnvName python -c "import sys; print(sys.executable)" 2>$null | Select-Object -Last 1)
    Write-Host "env python : $EnvPython"
    Write-Host "launcher   : $MpiExec"
    & $MpiExec -n 2 $EnvPython $MpiPy 2>&1
} else {
    Write-Host 'SKIPPED: no MS-MPI mpiexec.exe found (see section above).'
}

# ---------------------------------------------------------------- lock file
Section 'export exact solved versions'
& $Solver env export -n $EnvName --no-builds 2>&1 | Set-Content -Path $Lock -Encoding UTF8
if (Test-Path $Lock) { Write-Host "wrote installation\environment-win.lock.yml" }

Write-Host "`n=== install phase done | $(Get-Date) ==="
Finish 0
