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
├── settings/              # templates (GDA_Global, RDA_Regional) and the GMD paper cases: DA_setting.json,
│                          #   perturbation.json, Config_ReWaterGAP.json
├── doc/                   # installation and configuration guides, settings reference
├── installation/          # environment files and setup script
└── parallel_logs/         # MPI run logs (written at run time)
```

## Modifications to WaterGAP 2.2e

WaterGAP 2.2e (ReWaterGAP, upstream commit 2ebcdd1) is included in `src_GHM/ReWaterGAP/`. PyGLDA v2 changes a few
of its files in place, none of which changes a standard (non-assimilation) WaterGAP run:

- `model/land_surfacewater_fraction.py`: new `update_setting()` to refresh the run-mode globals between members.
- `controller/configuration_module.py`: the configuration reading is wrapped in `init_config()`, so that PyGLDA can
  configure WaterGAP from Python.
- `model/verticalwaterbalance/waterbalance_vertical.py`: a redundant 207 MB copy of the sub-grid snow storage per
  model day is avoided (bit-identical results).
- `model/parameters.py`, `model/lateralwaterbalance/waterbalance_lateral.py`: the ensemble parameter multipliers
  (1.0 when absent from the parameter file).
- `model/lateralwaterbalance/lakes_wetlands.py`: a guard against 0 / 0 for lakes and wetlands written back at their
  floor by the assimilation.

Modified copies of further WaterGAP modules (ensemble multipliers, monthly forcing files, daily output, restart
paths) are in `src_GHM/Extension/`; the originals are unchanged. Every change, its reason and its test are listed in
`src_GHM/WATERGAP_CHANGES.md`, with unified diffs against the upstream version in `doc/watergap_changes/diffs/`.

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

Experiments are run by the drivers in `src_demo/`, one stage at a time; the settings are in
`settings/<folder>/DA_setting.json` (see [doc/configuration.md](doc/configuration.md) and the one-page
[doc/settings_reference.md](doc/settings_reference.md)).

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
