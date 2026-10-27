from enum import Enum


class Stage(Enum):
    OL = 1
    DA = 2


class HydroModel(Enum):
    WaterGap = 3


class FusionMethod(Enum):
    EnKF_localized = 3      # state-observation localization (src_DA.EnKF_localized)
    LETKF = 4               # deterministic ensemble-transform analysis with the same components (src_DA.LETKF)


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
