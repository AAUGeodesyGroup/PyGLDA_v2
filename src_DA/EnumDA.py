from enum import Enum


class Stage(Enum):
    SR=0
    OL=1
    DA=2

class HydroModel(Enum):
    w3 = 0
    w3ra_v0 = 1
    w3ra_v1 = 2
    WaterGap = 3


class FusionMethod(Enum):
    EnKF_v0 = 0
    EnKF_v1 = 1
    EnKF_v2 = 2
    EnKF_localized = 3      # state-observation localization (src_DA.EnKF_localized)


class WaterGap_storage_variables(Enum):
    groundwstor = 0
    soilmoist = 1
    locallakestor = 2
    localwetlandstor = 3
    globallakestor = 4
    globalwetlandstor = 5
    riverstor = 6
    reservoirstor = 7
    swe = 8
    canopystor = 9
    tws = 10

class WaterGap_flux_variables(Enum):
    evap = 0
    runoff = 1
    baseflow = 2
    subsurfaceflow = 3
    totalflow = 4





