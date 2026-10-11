# -*- coding: utf-8 -*-
# =============================================================================
# This file is part of WaterGAP.

# WaterGAP is an opensource software which computes water flows and storages as
# well as water withdrawals and consumptive uses on all continents.

# You should have received a copy of the LGPLv3 License along with WaterGAP.
# if not see <https://www.gnu.org/licenses/lgpl-3.0>
# =============================================================================

"""Model parameter values."""

# =============================================================================
# This module contains all paameter values to run  WaterGAP
# =============================================================================
import logging
import os
import sys
from pathlib import Path
from controller import configuration_module as cm
import xarray as xr
import misc.cli_args as cli
import watergap_logger as log


# ===============================================================
# Get module name and remove the .py extension
# Module name is passed to logger
# ===============================================================
modname = os.path.basename(__file__)
modname = modname.split('.')[0]

# ++++++++++++++++++++++++++++++++++++++++++++++++
# Parsing  Argguments for CLI from cli_args module
# +++++++++++++++++++++++++++++++++++++++++++++++++
args = cli.parse_cli()

# PyGLDA (10 Oct 2026): multipliers of model quantities that are not in the parameter file, written per ensemble
# member by src_DA/perturbation_par.py and applied where WaterGAP reads the quantity (fan_waterbalance_vertical_init.py,
# waterbalance_lateral.py). A parameter file without them (the original WaterGAP file) gets 1.0 everywhere, i.e.
# the standard model, bit-identical.
MULTIPLIERS = ('lai_multiplier', 'degreeday_multiplier', 'smax_multiplier', 'gw_recharge_factor_multiplier',
               'max_gw_recharge_multiplier', 'netabs_gw_multiplier', 'netabs_sw_multiplier')


class Parameters:
    """WaterGAP parameters."""

    def __init__(self, run_calib, basin_id):
        # read in global parameters
        try:
            if run_calib is True:
                # calibration
                param_path = str(Path(cm.global_parameter_path))
            else:
                # Default
                param_path = str(Path(cm.global_parameter_path))
            
            
            self.global_params = xr.open_dataset(param_path, decode_times=False)

            # PyGLDA: missing ensemble multipliers -> 1.0 everywhere (see MULTIPLIERS)
            template = self.global_params['max_daily_pet']
            for name in MULTIPLIERS:
                if name not in self.global_params:
                    self.global_params[name] = xr.ones_like(template, dtype=float)

        except FileNotFoundError:
            log.config_logger(logging.ERROR, modname, 'Global parameter data  '
                              f'{param_path} not found', args.debug)
            sys.exit()  # dont run code if file does not exist
