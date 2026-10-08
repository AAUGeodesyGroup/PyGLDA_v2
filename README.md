# PyGLDA v2

**PyGLDA** (Python-based Global Land Data Assimilation system) is a framework for integrating GRACE/GRACE-FO terrestrial water storage (TWS) anomalies into the WaterGAP 2.2e global hydrological model using an Ensemble Kalman Filter (EnKF).

## Overview

PyGLDA v2 extends the original PyGLDA system by coupling it with the [WaterGAP 2.2e](https://github.com/HydrologyFrankfurt/ReWaterGAP) model. The assimilation updates the model state variables (groundwater, soil moisture, surface water storage, etc.) using GRACE satellite gravity observations, improving the accuracy of global water storage estimates.

### Key features

- Ensemble Kalman Filter (EnKF) data assimilation
- Integration with WaterGAP 2.2e hydrological model
- Support for GRACE mascon and spherical harmonic solutions
- MPI-based parallelism for ensemble runs
- Basin-averaged time series extraction and post-processing

## Repository structure

```
PyGLDA_v2/
├── src_GHM/
│   ├── ReWaterGAP/        # WaterGAP 2.2e (see modifications note below)
│   ├── Extension/         # PyGLDA extensions to WaterGAP
│   └── Interface/         # Bridge between PyGLDA and WaterGAP
├── src_DA/                # Data assimilation algorithms (EnKF, EnSQRA, etc.)
├── src_OBS/               # GRACE observation preparation
├── src_auxiliary/         # Shared utilities (GeoMathKit, shp2mask, etc.)
├── src_FlowControl/       # Experiment controllers (DA_GRACE, OpenLoop, etc.)
├── src_postprocessing/    # Post-processing and analysis
├── src_demo/              # Entry-point scripts
├── settings/              # Configuration files
├── installation/          # Installation notes and environment setup
└── parallel_logs/         # MPI run logs (populated at runtime)
```

## Modifications to WaterGAP 2.2e

PyGLDA v2 makes three minimal modifications to the WaterGAP 2.2e source, each confined to a single file. All other WaterGAP 2.2e files remain unmodified.

1. **`src_GHM/ReWaterGAP/model/land_surfacewater_fraction.py`** — A new `update_setting()` function was added to allow PyGLDA to refresh WaterGAP's run-mode globals (`anthropogenic`, `reservoir_operation`) between ensemble iterations without reloading the module.

2. **`src_GHM/ReWaterGAP/controller/configuration_module.py`** — Rewritten to expose WaterGAP's configuration parameters to PyGLDA's assimilation interface. The original file is preserved in the git history.

3. **`src_GHM/ReWaterGAP/model/verticalwaterbalance/waterbalance_vertical.py`** (performance only, 2026-10-05) — In `vert_water_balance`, the line `snow_water_storage_subgrid_out = snow_water_storage_subgrid.copy()` was replaced by `snow_water_storage_subgrid_out = snow_water_storage_subgrid`. `snow_water_balance` receives a view of each cell's 100 elevation-band storages and already updates them in place, so the copy (100 x 360 x 720 float64, 207 MB per model day and per ensemble member) only duplicated values that are overwritten cell by cell. Results are expected to be bit-identical; `temp/env_test/check_snow_copy.py` runs the original and the modified routine on the same input and compares the restart states and daily output. To revert, restore the `.copy()` on that line.

## Requirements

- Python >= 3.8
- numpy, scipy, xarray, h5py, pandas
- mpi4py (for parallel ensemble runs)
- tqdm, termcolor

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/AAUGeodesyGroup/PyGLDA_v2.git
cd PyGLDA_v2
```

### 2. Download WaterGAP data files

The `src_GHM/ReWaterGAP/` directory in this repository contains only the **source code** of WaterGAP 2.2e. Large input data files (`.nc`, `.h5`, etc.) required to run the model are not included due to their size.

You must download the complete WaterGAP 2.2e package separately and copy the missing data files into the corresponding directories under `src_GHM/ReWaterGAP/`. Do **not** overwrite the Python source files, as they contain the PyGLDA-specific modifications described above.

- ReWaterGAP releases: https://github.com/HydrologyFrankfurt/ReWaterGAP/releases

### 3. Install Python dependencies

```bash
# see doc/installation.md — the pinned conda environment (installation/environment.yml) is required;
# a plain pip install does not give the numpy/numba versions WaterGAP 2.2e is locked to.
bash installation/setup_env.sh install
```

## Usage

*To be added.*

## Acknowledgements

PyGLDA v2 is built upon the **WaterGAP 2.2e** global hydrological model, developed by the Hydrology Frankfurt group at Goethe University Frankfurt. WaterGAP 2.2e is used as the forward model in the data assimilation system.

- WaterGAP project webpage: https://www.watergap.de
- ReWaterGAP (Python implementation): https://github.com/HydrologyFrankfurt/ReWaterGAP

We thank the WaterGAP development team for making their model openly available under the LGPL-3.0 license.

## Citation

If you use PyGLDA in your research, please cite:

> Yang, F., et al. (2025). PyGLDA: A Python-based global land data assimilation system.
> *Geoscientific Model Development*, 18, 6195–6217.
> https://doi.org/10.5194/gmd-18-6195-2025

## License

PyGLDA v2 is licensed under the **GNU Lesser General Public License v3.0** (LGPL-3.0).
This is consistent with the license of the WaterGAP 2.2e model included in this repository.
See the [LICENSE](LICENSE) file for details.

## Contact

Fan Yang — [fany@plan.aau.dk](mailto:fany@plan.aau.dk)
Geodesy Group, Aalborg University (AAU)
