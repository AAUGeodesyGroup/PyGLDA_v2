# PyGLDA v2

**PyGLDA** (Python-based Global Land Data Assimilation system) is a framework for integrating GRACE/GRACE-FO terrestrial water storage (TWS) anomalies into the WaterGAP 2.2e global hydrological model using an Ensemble Kalman Filter (EnKF).

## Overview

PyGLDA v2 extends the original PyGLDA system by coupling it with the [WaterGAP 2.2e](https://github.com/HydrologyFrankfurt/ReWaterGAP) model. The assimilation updates the model state variables (groundwater, soil moisture, surface water storage, etc.) using GRACE satellite gravity observations, improving the accuracy of global water storage estimates.

### Key features

- Ensemble Kalman Filter (EnKF) data assimilation
- Integration with WaterGAP 2.2e hydrological model (kept as an untouched submodule)
- Support for GRACE mascon and spherical harmonic solutions
- MPI-based parallelism for ensemble runs
- Basin-averaged time series extraction and post-processing

## Repository structure

```
PyGLDA_v2/
├── src_GHM/
│   ├── ReWaterGAP/        # Original WaterGAP 2.2e (untouched)
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

## Requirements

- Python >= 3.8
- numpy, scipy, xarray, h5py, pandas
- mpi4py (for parallel ensemble runs)
- tqdm, termcolor

## Installation

*To be added.*

## Usage

*To be added.*

## Acknowledgements

PyGLDA v2 is built upon the **WaterGAP 2.2e** global hydrological model, developed by the Hydrology Frankfurt group at Goethe University Frankfurt. WaterGAP 2.2e is used as the forward model in the data assimilation system and is included in this repository in its original, unmodified form.

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
