# src_GUI - results dashboard

A browser dashboard for a finished PyGLDA v2 run. It reads the files the pipeline writes and changes nothing:
`Res/<case>/evaluation_global/<tag>/*.csv|json` (from `GDA.da_evaluation`), `Res/<case>/Res_{OL,DA}.h5`,
`GRACE_<basin>.h5`, `Harmonic_*.nc`, `GRACE/obs/<basin>_obs_GRACE.hdf5` and the unit shapefile.

## Install (once, into the pyglda_v2 env)

    pip install "streamlit>=1.36" plotly

(not added to installation/environment.yml: the pinned WaterGAP stack solves slowly with extra conda packages;
the two pip wheels install in seconds and touch nothing else.)

## Run

    cd PyGLDA_v2
    streamlit run src_GUI/dashboard.py -- --data <external_data> --case GDA_Global

then open http://localhost:8501. The data root can also come from the environment variable `PYGLDA_DATA`;
case, report tag, basin and shapefile are chosen in the sidebar. On a remote HPC job (e.g. UCloud) start it inside the
job and open the port through the job's web interface; on a workstation it runs locally.

## Tabs

- **Overview** - global fit (RMS, correlation, trends), unit statistics, filter health (chi2, gain, spread vs sigma,
  adaptive inflation), warnings, group table, summary.txt.
- **Map** - one metric per unit on the unit polygons (RMS reduction, chi2, gain, trends, mean |increment| per
  storage, ...); click a unit to open it.
- **Unit** - window means of GRACE / OL / DA with the DA spread, increments per storage and window, chi2 with
  forecast spread and GRACE sigma, daily storages (ensemble mean) OL vs DA.
- **Groups** - area-weighted series of the groups in the report (global, regions, basin classes, SNR, latitude bands).
- **Grid maps** - trend / annual amplitude / annual peak of OL, DA and GRACE from the harmonic maps, and DA - OL.

Large files are cached per session (`st.cache_data`); "reload files" in the sidebar clears the cache.
