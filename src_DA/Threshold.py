from src_DA.configure_DA import config_DA
import numpy as np
from pathlib import Path
import xarray as xr
from src_auxiliary.shp2mask import load_mask


class model_state_threshold:
    """
    Apply physical boundary clipping to WaterGAP hydrological variables
    after the EnKF analysis step.
    """
    def __init__(self, configDA: config_DA):
        ds = xr.open_dataset(Path(configDA.basic.Auxiliary_dir)/'smax.nc')
        box_crop, _= load_mask(mask_path=configDA.basic.basin_mask)

        lat_slice = slice(box_crop['lat_max'], box_crop['lat_min'])  # Descending order
        lon_slice = slice(box_crop['lon_min'], box_crop['lon_max'])
        self.smax = ds.sel(lon=lon_slice, lat=lat_slice)['smax'].values # maximum soil water
        self.layers = [key for key, value in configDA.model.layer.items() if value is True]

        # Variables that represent absolute storage (must be strictly non-negative)
        self.non_negative_vars = [
            'swe', 'canopystor', 'riverstor', 'reservoirstor',
            'localwetlandstor', 'globalwetlandstor'
        ]

        # Variables that represent relative storage (allowed to be negative for water deficits)
        self.allow_negative_vars = [
            'groundwstor', 'locallakestor', 'globallakestor'
        ]

        pass

    def threshold(self, state: dict):

        # Define realistic physical upper and lower bounds (units typically in mm)
        # Adjust these limits based on your specific model domain and WaterGAP configuration.
        bounds = {
            'swe': {
                'min': 0.0,  # Snow water equivalent lower bound cannot be negative
                'max': 5000.0  # Upper limit to handle extreme accumulation zones
            },
            'groundwstor': {
                'min': -50000.0,  # Minimum threshold allowing for groundwater deficit/overdraft
                'max': 100000.0  # Maximum capacity limit for deep groundwater storage
            }
        }

        for var in self.layers:
            if var in self.allow_negative_vars:
                # Group 1: Relative storage.
                # Allow negative values (water deficits), keep the array exactly as it is. do nothing
                pass

            elif var in self.non_negative_vars:
                # Group 2: Absolute storage.
                # Cannot drop below zero. Force all negative values to 0.0.
                EPSILON = 0.0
                state[var] = np.maximum(state[var], EPSILON)
                # state[var] = np.maximum(state[var], 0.0)

            elif var == 'soilmoist':
                # Group 3: Soil moisture.
                # Must be strictly bounded between 0.0 and the maximum capacity (smax).
                state[var] = np.clip(state[var], 0.0, self.smax)

            else:
                # Fallback for any other variables not explicitly defined in the rules
                pass

            if var in bounds:
                # Force values to stay strictly within the physically plausible boundaries
                state[var] = np.clip(
                    state[var],
                    bounds[var]['min'],
                    bounds[var]['max']
                )

        return state



def demo1():
    configDA = config_DA.loadjson('/media/user/My Book/Fan/WaterGap/Extensions/DA_settings/DA_setting.json').process()
    mst = model_state_threshold(configDA=configDA)
    pass

if __name__ == '__main__':
    demo1()
