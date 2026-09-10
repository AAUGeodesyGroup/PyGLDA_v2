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
└── settings/              # Configuration files
```

## Modifications to WaterGAP 2.2e

PyGLDA v2 requires two minimal modifications to the WaterGAP 2.2e source, both confined to a single file each. All other WaterGAP 2.2e files remain unmodified.

1. **`src_GHM/ReWaterGAP/model/land_surfacewater_fraction.py`** — A new `update_setting()` function was added to allow PyGLDA to refresh WaterGAP's configuration with new setting file.

2. **`src_GHM/ReWaterGAP/controller/configuration_module.py`** — Rewritten to expose WaterGAP's configuration parameters to PyGLDA's assimilation interface. The original file is preserved in the git history.

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
pip install numpy scipy xarray h5py pandas tqdm termcolor mpi4py
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
