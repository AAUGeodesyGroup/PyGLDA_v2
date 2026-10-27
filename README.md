# PyGLDA v2

**PyGLDA** (Python-based Global Land Data Assimilation system) is a framework for integrating GRACE/GRACE-FO terrestrial water storage (TWS) anomalies into the WaterGAP 2.2e global hydrological model using an Ensemble Kalman Filter (EnKF).

## Overview

PyGLDA v2 couples the PyGLDA assimilation system in memory with the [WaterGAP 2.2e](https://github.com/HydrologyFrankfurt/ReWaterGAP) model. GRACE / GRACE-FO observations, averaged over hydrological units, update the model's groundwater, soil water, snow, wetland and river storages, and the river routing carries the correction downstream into discharge.

### Key features

- Ensemble Kalman filter (stochastic EnKF or LETKF) assembled from exchangeable components chosen in a settings file: localization (none, block, Gaussian), inflation (multiplicative, RTPS, additive, adaptive additive), increment partition (Kalman or sign-consistent, bound-aware), physical bounds valid on every day of an observation window
- Observations of any averaging period (monthly CSR mascons, 5-daily TU Delft product) through one interface; earthquake correction of the mascons
- Regional and global experiments from one global open loop; sparse and MPI-parallel, a 30-member global run fits on one 32-core node
- Collection, evaluation and export of the assimilated storages and discharge

## Repository structure

```
PyGLDA_v2/
├── src_GHM/
│   ├── ReWaterGAP/        # WaterGAP 2.2e (see modifications note below)
│   ├── Extension/         # PyGLDA versions of WaterGAP modules (fan_*), adapted for daily coupling
│   └── Interface/         # Bridge between PyGLDA and WaterGAP
├── src_DA/                # filter (EnKF_localized, LETKF) and its components, ensemble perturbation
├── src_OBS/               # GRACE observation preparation, earthquake correction
├── src_auxiliary/         # unit builder (HydroBASINS), masks, utilities
├── src_FlowControl/       # experiment stages (Regional_DA, Global_DA, DA_GRACE, OpenLoop)
├── src_postprocessing/    # collection, evaluation, figures, product export
├── src_GUI/               # results viewer (streamlit)
├── src_demo/              # drivers: global_run.py, regional_run.py
├── settings/              # one folder per case: DA_setting.json, perturbation.json, Config_ReWaterGAP.json
├── doc/                   # installation and configuration guides
├── installation/          # environment files and setup script
└── parallel_logs/         # MPI run logs (written at run time)
```

## Modifications to WaterGAP 2.2e

PyGLDA v2 makes the following minimal modifications to the WaterGAP 2.2e source, each confined to a single file. All other WaterGAP 2.2e files remain unmodified.

1. **`src_GHM/ReWaterGAP/model/land_surfacewater_fraction.py`** — A new `update_setting()` function was added to allow PyGLDA to refresh WaterGAP's run-mode globals (`anthropogenic`, `reservoir_operation`) between ensemble iterations without reloading the module.

2. **`src_GHM/ReWaterGAP/controller/configuration_module.py`** — Rewritten to expose WaterGAP's configuration parameters to PyGLDA's assimilation interface. The original file is preserved in the git history.

3. **`src_GHM/ReWaterGAP/model/verticalwaterbalance/waterbalance_vertical.py`** (performance only) — In `vert_water_balance`, the line `snow_water_storage_subgrid_out = snow_water_storage_subgrid.copy()` was replaced by `snow_water_storage_subgrid_out = snow_water_storage_subgrid`. `snow_water_balance` receives a view of each cell's 100 elevation-band storages and already updates them in place, so the copy (100 x 360 x 720 float64, 207 MB per model day and per ensemble member) only duplicated values that are overwritten cell by cell. Results are bit-identical (the original and the modified routine were compared on the same input, restart states and daily output). To revert, restore the `.copy()` on that line.

4. **`src_GHM/ReWaterGAP/model/parameters.py`** (ensemble perturbation) — After the parameter file is opened, the multipliers listed in `MULTIPLIERS` (`lai_multiplier`, `degreeday_multiplier`, `smax_multiplier`, `gw_recharge_factor_multiplier`, `max_gw_recharge_multiplier`, `netabs_gw_multiplier`, `netabs_sw_multiplier`) are added as fields of 1.0 when the file does not contain them. `src_DA/perturbation_par.py` writes them into the parameter file of each ensemble member (`perturbation.json`, `"type": "multiplier"`). They are applied once where WaterGAP reads the quantity: the leaf area index (min and max), the degree-day factor, the maximum soil water content, the groundwater recharge factor and the maximum groundwater recharge in PyGLDA's `src_GHM/Extension/fan_waterbalance_vertical_init.py` (the vertical-balance initialisation PyGLDA uses), and the potential net abstraction from groundwater and surface water in item 5. With the original parameter file (all multipliers 1.0) the model is bit-identical.

5. **`src_GHM/ReWaterGAP/model/lateralwaterbalance/waterbalance_lateral.py`** (ensemble perturbation) — In `calculate`, the monthly potential net abstraction from groundwater (`pnag`) and surface water (`pnas`) is multiplied by `netabs_gw_multiplier` / `netabs_sw_multiplier` right after it is read, before the unit conversion and the riparian aggregation.

## Requirements

- Linux, conda / mamba; Python 3.10 with the versions pinned in `installation/environment.yml` (WaterGAP 2.2e is
  locked to numpy 1.23.5 / numba 0.56.4)
- mpich and mpi4py from the same environment
- about 2.2 GB of memory per ensemble member (each member runs WaterGAP on the global grid)

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

Experiments are run by the drivers in `src_demo/`, one stage at a time; the settings of each case are in
`settings/<case>/DA_setting.json` (see [doc/configuration.md](doc/configuration.md)).

```bash
export PYGLDA_DATA=/path/to/PyGLDA_v2_external_data     # inputs, GRACE, masks; outputs are written here
cd src_demo
python global_run.py prepare                             # run settings, unit check, GRACE observations
mpiexec -n 31 python -u global_run.py ensemble           # perturbation, spin-up, open loop (30 members + control)
mpiexec -n 31 python -u global_run.py assimilate         # data assimilation
python global_run.py evaluate                            # evaluation report and figures
```

The regional experiments (Danube, Amazon) use the same stages, e.g.
`mpiexec -n 31 python -u regional_run.py Danube assimilate`; they reuse the global open loop.

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
