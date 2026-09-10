import sys

sys.path.append('../')

import logging
import json
from pathlib import Path
import glob
import os
import sys
import xarray as xr
import numpy as np
from src_GHM.ReWaterGAP.controller.climateforcing_handler import ClimateForcing, cm, log, cli, modname, args


from datetime import datetime


def load_nc_range(data_dir, start_str, end_str):
    """
    Load multiple NetCDF files between given dates.

    Parameters
    ----------
    data_dir : str
        Directory containing NetCDF files
    start_str : str
        Start date (e.g., "1989-01-01")
    end_str : str
        End date (e.g., "1999-12-31")
    """

    # string to datetime
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end   = datetime.strptime(end_str, "%Y-%m-%d")

    files = []

    # loop over year and month
    for year in range(start.year, end.year + 1):
        for month in range(1, 13):

            date = datetime(year, month, 1)

            # if start <= date <= end:
            filename = f"{data_dir}/{year}-{month:02d}.nc"

            if os.path.exists(filename):
                files.append(filename)

    files = sorted(files)

    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        parallel=False,
        chunks = {'time': 365}
    )

    return ds

class fan_ClimateForcing(ClimateForcing):

    def __init__(self, run_calib):
        """
        Get file path.

        Return
        ------
        climate forcings

        """
        # ==============================================================
        # path to climate forcing netcdf data
        # ==============================================================

        # forcing_path = str(Path(cm.climate_forcing_path +
        #                         r'/*.nc'))

        # forcing_path = str(Path(cm.climate_forcing_path +
        #                         r'/1981-01.nc'))

        # forcing_path = str(Path('/media/user/My Book/Fan/WaterGap/monthly_climate_forcing' +
        #                         r'/1981-01.nc'))

        # ==============================================================
        # Loading in climate forcing
        # ==============================================================
        try:

            # fd = xr.open_mfdataset(glob.glob(forcing_path), chunks={'time': 365})

            fd = load_nc_range(data_dir=cm.climate_forcing_path, start_str=cm.start, end_str=cm.end)

            #  Actual name: Precipitation, Unit:  kg m-2 s-1
            self.precipitation = fd['pr']

            #  Actual name: Downward longwave radiation  Unit: Wm−2
            self.down_longwave_radiation = fd['rlds']

            #  Actual name: Downward shortwave radiation  Unit: Wm−2
            self.down_shortwave_radiation = fd['rsds']

            #  Actual name: Air temperature, Unit: K
            self.temperature = fd['tas']

        except FileNotFoundError as error:
            log.config_logger(logging.ERROR, modname, f'Climate forcing'
                                                      f' not found. \n{error}', args.debug)
            sys.exit()  # don't run code if file does not exist
        except ValueError:
            log.config_logger(logging.ERROR, modname, 'File(s) extension '
                                                      'should be NETCDF(.nc or .nc4)', args.debug)
            sys.exit()  # don't run code if file does not exist
        else:
            if run_calib is False:
                print('Climate forcing loaded successfully')

            self.var_name = list(fd.data_vars)

            self.units = [self.precipitation.units,

                          self.down_longwave_radiation.units,

                          self.down_shortwave_radiation.units,

                          self.temperature.units]

        self.date = np.datetime64("1000-01-01").astype('datetime64[M]')
        self.fd = fd

    def check_unitandvarname(self):
        """
        Check data units and variable name.

        Returns
        -------
        None.

        """
        # ==============================================================
        # Opening cf convention file for units and variable name check
        # ==============================================================

        try:
            with open('cf_conv.json', encoding="utf-8") as cf_info:
                cf_info = json.load(cf_info)
        except FileNotFoundError:
            log.config_logger(logging.ERROR, modname, 'Cf convention file for'
                                                      ' variable and unit check not found', args.debug)

        # print('++++++++++++++++++++++' + '\n' + 'Checking variable name'
        #       + '\n' + '++++++++++++++++++++++')

        # Note!!! underscore means I am not interested in the index(numbers)
        for _, var_name in enumerate(self.var_name):
            if var_name in cf_info['variables']['shortname']:
                # print(var_name + ' follows cf convention')
                pass
            else:
                log.config_logger(logging.WARNING, modname, var_name +
                                  ' does not follow cf convention', args.debug)

        # print('\n' + '+++++++++++++++' + '\n' + 'Checking units' + '\n' +
        #       '+++++++++++++++')

        extra_units = ["mm/day", " mm day-1", "°C", "C", "degree celcius",
                       "celcius"]
        for index, units in enumerate(self.units):
            if units in cf_info['variables']['units'] or units in extra_units:
                # print('*' + self.var_name[index] + '*' + ' required in ' +
                #       units + ' found')
                pass
            else:
                log.config_logger(logging.ERROR, modname, units +
                                  ' is not a known cf convention unit.'
                                  ' Please check data units', args.debug)
                sys.exit()  # don't run code if units does not exist

    def update(self, date):
        forcing_path = str(Path(cm.climate_forcing_path + '/%s.nc' % (date.astype('datetime64[M]'))))

        if self.date == date.astype('datetime64[M]'):
            '''This avoids reopening a monthly dataset'''
            pass
        else:
            '''load file'''
            # print(self.date, date.astype('datetime64[M]'))
            self.fd.close()
            self.fd = xr.open_mfdataset(glob.glob(forcing_path))
            self.date = date.astype('datetime64[M]')

            #  Actual name: Precipitation, Unit:  kg m-2 s-1
            self.precipitation = self.fd['pr']

            #  Actual name: Downward longwave radiation  Unit: Wm−2
            self.down_longwave_radiation = self.fd['rlds']

            #  Actual name: Downward shortwave radiation  Unit: Wm−2
            self.down_shortwave_radiation = self.fd['rsds']

            #  Actual name: Air temperature, Unit: K
            self.temperature = self.fd['tas']

        pass
        # return self.fd.sel(time=str(date))


if __name__ == '__main__':
    ff = fan_ClimateForcing(run_calib=False)
    ff.check_unitandvarname()
    ff.update(np.datetime64("1995-06-01"))
    ff.update(np.datetime64("1995-02-02"))
