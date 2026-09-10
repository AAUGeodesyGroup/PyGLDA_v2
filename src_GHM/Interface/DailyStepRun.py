# -*- coding: utf-8 -*-
# =============================================================================
# This file is part of WaterGAP.

# WaterGAP is an opensource software which computes water flows and storages as
# well as water withdrawals and consumptive uses on all continents.

# You should have received a copy of the LGPLv3 License along with WaterGAP.
# if not see <https://www.gnu.org/licenses/lgpl-3.0>
# =============================================================================
"""Run WaterGAP."""
# =============================================================================
# This module runs WaterGAP for a user selected period by calling both
# vertical and lateral waterbalance functions
# =============================================================================

import numpy as np
from tqdm import tqdm
import pandas as pd
from termcolor import colored
# from misc.time_checker_and_ascii_image import check_time
from src_GHM.ReWaterGAP.controller import configuration_module as cm
# from controller import read_forcings_and_static as rd
from src_GHM.Extension import fan_read_forcings_and_static as rd
from src_GHM.ReWaterGAP.controller import wateruse_handler as wateruse
from src_GHM.ReWaterGAP.model import parameters as pm
from src_GHM.ReWaterGAP.model import land_surfacewater_fraction_init as lwf
from src_GHM.ReWaterGAP.model.lateralwaterbalance import waterbalance_lateral as lb
# from model.utility import restart_watergap as restartwatergap
from src_GHM.ReWaterGAP.model.utility import get_upstream_basin as get_basin
# from model.verticalwaterbalance import waterbalance_vertical_init as vb
from src_GHM.Extension import fan_waterbalance_vertical_init as vb
from src_GHM.Extension import fan_createandwrite as cw
from src_GHM.Extension import fan_restart_watergap as restartwatergap
from .UnitConverter import UnitConverter

import os
os.environ["FORCE_COLOR"] = "1"

class DailyModelRun:
    def __init__(self, is_crop_save = False, **kwargs):
        """
        Initialize the model with required components.
        """
        watergap_basin = None
        basin_id = None

        if cm.ant:
            print('\n' + colored('+++ Antropogenic Run +++', 'cyan'))
            if cm.RESERVOIR_OPT is False:
                print(colored('Use only: Human water use without '
                              'global man-made reservoirs/regulated lakes',
                              'blue'))
            elif cm.SUBTRACT_USE is False:
                print(colored('Reservoirs only: Exclude human water use'
                              ' but include global man-made'
                              ' reservoirs/regulated lakes', 'blue'))
            else:
                print(colored('Standard (ant) run: Include human water'
                              ' use and include global man-made'
                              ' reservoirs/regulated lakes', 'blue'))
            # demand satisfaction option
            if cm.DELAYED_USE and cm.NEIGHBOURING_CELL:
                msg = 'Delayed water supply & Neighboring cell water supply'
            elif cm.DELAYED_USE:
                msg = 'Delayed water supply'
            elif cm.NEIGHBOURING_CELL:
                msg = 'Neighboring cell water supply'
            else:
                msg = 'none (Delayed water supply & Neighboring cell water supply)'
            satisfaction_option = \
                f"Riparian water supply (default) & {msg} option activated."
            print(colored('Demand satisfaction option: ' + satisfaction_option,
                          'blue'))

            print('\nPeriod:' + colored(f'{cm.start} to {cm.end}', 'green'))
            print('Temporal resolution:' +
                  colored(f'{cm.TEMPORAL_RES}', 'green'))
            print('Run basin:' +
                  colored(f'{cm.run_basin}', 'green'))
        else:
            print('\n' + colored('+++ Naturalised Run +++', 'cyan'))
            print('Note:' + colored(' 1. Reserviors, abstraction from surface and'
                                    ' groundwater are not considered.' + '\n'
                                    + '      2. Regulated lakes are treated as'
                                      '  global lakes ', 'blue'))
            print('\nPeriod:' + colored(f'{cm.start} to {cm.end}', 'green'))
            print('Temporal resolution:' +
                  colored(f'{cm.TEMPORAL_RES}', 'green'))
            print('Run basin:' +
                  colored(f'{cm.run_basin}', 'green'))

        # Flag to run WaterGAP calibration
        run_calib = cm.run_calib

        # =====================================================================
        # Initialize Restart module for possible restart of WaterGAP
        # =====================================================================
        self.restart_model = restartwatergap.RestartState(read_states_path=cm.read_state_path, save_states_path=cm.save_state_path)
        self.savestate_for_restart = True  # This must be true for hot-run
        self.restart = True  # This must be true for hot-run
        # =====================================================================
        # Initialize static data, climate forcings , wateruse data
        # and get data dimensions
        # =====================================================================
        self.initialize_forcings_static = rd.InitializeForcingsandStaticdata(run_calib)
        self.grid_coords = self.initialize_forcings_static.grid_coords
        self.potential_net_abstraction = wateruse.Wateruse(cm.SUBTRACT_USE, self.grid_coords, run_calib)
        self.parameters = pm.Parameters(run_calib, basin_id)

        # initialize Land surface water Fraction
        self.land_water_frac = \
            lwf.LandsurfacewaterFraction(self.initialize_forcings_static.static_data,
                                         cm.RESERVOIR_OPT)

        # =====================================================================
        #  Create and write to ouput variable if selected by user
        # =====================================================================
        if is_crop_save:
            self.create_out_var = cw.CreateandWritetoVariables(self.grid_coords,**kwargs)
        else:
            self.create_out_var = cw.CreateandWritetoVariables(self.grid_coords)

        # =====================================================================
        # Initialize Vertical Water Balance
        # =====================================================================
        self.vertical_waterbalance = \
            vb.VerticalWaterBalance(self.initialize_forcings_static, self.parameters)

        # =====================================================================
        # Initialize Lateral Water Balance
        # =====================================================================
        self.lateral_waterbalance = \
            lb.LateralWaterBalance(self.initialize_forcings_static,
                                   self.potential_net_abstraction, self.parameters,
                                   self.land_water_frac.global_lake_area,
                                   self.land_water_frac.glolake_frac,
                                   self.land_water_frac.loclake_frac)

        # =====================================================================
        # Initialize selected basin or region
        # =====================================================================
        if run_calib and cm.run_basin:
            # calibration run
            pass
        else:
            # run WaterGAP is basin of choice
            streamflow_station = self.initialize_forcings_static.static_data.stations
            self.watergap_basin = get_basin. \
                SelectUpstreamBasin(cm.run_basin,
                                    self.initialize_forcings_static.static_data.arc_id,
                                    streamflow_station,
                                    self.initialize_forcings_static.static_data.lat_lon_arcid,
                                    self.initialize_forcings_static.static_data.upstream_cells)

        # ====================================================================
        # Get time range for Loop
        # ====================================================================
        self.start_date = np.datetime64(cm.start)
        self.end_date = np.datetime64(cm.end)

        # getting time range from time input (including the first day).
        timerange_main = round((self.end_date - self.start_date + 1) / np.timedelta64(1, 'D'))

        # format main date for simulation.
        date_main = self.grid_coords['time'].values.astype('datetime64[D]')

        # Get the first available day for each month (required to load in wateruse
        # data once each month)
        date_df = pd.DataFrame({'dates': pd.to_datetime(date_main)})

        # Extract the year and month to group by month
        date_df['year'] = date_df['dates'].dt.year
        date_df['month'] = date_df['dates'].dt.month
        first_day_of_month = date_df.groupby(['year', 'month']).min()
        first_day_of_month = first_day_of_month.reset_index(drop=True)
        first_day_of_month.rename(columns={'dates': 'First Day'}, inplace=True)
        self.first_day_of_month = \
            first_day_of_month['First Day'].values.astype('datetime64[D]')

        # *********************************************************************
        print('\n' + '++++++++++++++++++++++++++++++++++++++++++++' + '\n' +
              colored('Calculating vertical & lateral water balance', 'cyan') +
              '\n' + '++++++++++++++++++++++++++++++++++++++++++++')
        # *********************************************************************

        # =================================================================
        # Update model paramters for restart if option is selected
        # =================================================================
        if self.restart:
            date_before_restart = str(self.start_date - np.timedelta64(1, 'D'))
            restart_data = self.restart_model.load_restart_info(date_before_restart)
            print(colored('Date of previous WaterGAP run: ' +
                          str(restart_data["last_date"]), 'blue'))
            print(colored('Restart date: ' +
                          str(self.start_date) + '\n', 'green'))

            self.land_water_frac. \
                update_landfrac_for_restart(restart_data["landfrac_state"])
            self.vertical_waterbalance. \
                update_vertbal_for_restart(restart_data["vert_bal_states"])
            self.lateral_waterbalance. \
                update_latbal_for_restart(restart_data["lat_bal_states"])

        #                  ====================================
        #                  ||   Main Loop for all processes  ||
        #                  ====================================
        get_annual_streamflow = []  # for calibration purpose only
        get_annual_pot_cell_runoff = []  # for calibration purpose only
        end_main_loop = False

        print(colored('Starting simulation from ' +
                      cm.start + ':' + cm.end + '\n', 'cyan'))

        self.time_range = timerange_main
        self.simulation_date = date_main

        self.unit_converter = UnitConverter(self.initialize_forcings_static.static_data.cell_area,
                                            self.initialize_forcings_static.static_data.land_surface_water_fraction.contfrac)

    def hot_run(self):
        """
        Perform the daily model run using initialized variables.
        """

        time_range, simulation_date, run_calib = self.time_range, self.simulation_date, cm.run_calib

        restart = self.restart

        for time_step, date in tqdm(zip(range(time_range), simulation_date),
                                    total=(time_range - 1), desc="Processing",
                                    disable=run_calib):

            self.initialize_forcings_static.climate_forcing.update(date)

            # =================================================================
            #  Get Land area fraction and reservoirs respective years
            # =================================================================
            # Activate reservoirs for current year
            self.lateral_waterbalance. \
                activate_res_area_storage_capacity(date, cm.RESERVOIR_OPT_YEARS,
                                                   restart)

            # Get Land area fraction
            self.land_water_frac. \
                landareafrac_with_reservior(date, cm.RESERVOIR_OPT_YEARS)

            # Get land and water fractions (used to calculate total PET)
            self.land_water_frac.get_land_and_water_freq(date)

            # Adapt global reservoir storage and land area fraction
            # due to net change in land fraction
            self.lateral_waterbalance.glores_storage = self.land_water_frac. \
                adapt_glores_storage(self.vertical_waterbalance.canopy_storage,
                                     self.vertical_waterbalance.snow_water_storage,
                                     self.vertical_waterbalance.soil_water_content,
                                     self.lateral_waterbalance.glores_area,
                                     self.lateral_waterbalance.glores_storage)

            # =================================================================
            #  Computing vertical water balance
            # =================================================================
            self.vertical_waterbalance. \
                calculate(date, self.land_water_frac.current_landareafrac,
                          self.land_water_frac.landareafrac_ratio,
                          self.watergap_basin.upstream_basin,
                          self.land_water_frac.water_freq,
                          self.land_water_frac.land_freq)

            # =================================================================
            #  Computing lateral water balance
            # =================================================================
            self.lateral_waterbalance. \
                calculate(self.vertical_waterbalance.fluxes['groundwater_recharge'],
                          self.vertical_waterbalance.fluxes['openwater_PET'],
                          self.vertical_waterbalance.fluxes['daily_precipitation'],
                          self.vertical_waterbalance.fluxes['surface_runoff'],
                          self.vertical_waterbalance.fluxes['daily_storage_transfer'],
                          self.vertical_waterbalance.fluxes['land_aet_corr'],
                          self.land_water_frac.current_landareafrac,
                          self.land_water_frac.previous_landareafrac,
                          self.land_water_frac.landwaterfrac_excl_glolake_res,
                          date, self.first_day_of_month, self.watergap_basin.upstream_basin,
                          self.vertical_waterbalance.fluxes['sum_canopy_snow_soil_storage'],
                          run_calib)

            # =================================================================
            #  Update Land Area Fraction
            # =================================================================
            land_swb_fraction = self.lateral_waterbalance.get_new_swb_fraction()
            self.land_water_frac.update_landareafrac(land_swb_fraction)

            # =============================================================
            # Write vertical and lateralbalacne variables to file
            # =============================================================
            sim_month = pd.to_datetime(date).month
            sim_day = pd.to_datetime(date).day
            sim_year = pd.to_datetime(date).year
            # Getting daily storages and fluxes and writing to variables
            vb_storages_and_fluxes = \
                self.vertical_waterbalance.get_storages_and_fluxes()

            self.create_out_var. \
                verticalbalance_write_daily_var(vb_storages_and_fluxes,
                                                sim_year, sim_month, sim_day)

            # Getting daily storages and fluxes and writing to variables
            lb_storages_and_fluxes = \
                self.lateral_waterbalance.get_storages_and_fluxes()

            self.create_out_var. \
                lateralbalance_write_daily_var(lb_storages_and_fluxes,
                                               sim_year, sim_month, sim_day)

            #  store data on daily basis
            self.create_out_var.base_units(self.initialize_forcings_static.static_data.cell_area,
                                           self.initialize_forcings_static.static_data.
                                           land_surface_water_fraction.contfrac)
            # create_out_var.save_netcdf_parallel(str(date))
            self.create_out_var.save_netcdf_daily_single_file(str(date))

            # =============================================================
            #  Get restart information if restart is needed.
            # =============================================================
            if self.end_date == date.astype('datetime64[D]'):
                if self.savestate_for_restart:
                    self.restart_model. \
                        savestate(date,
                                  self.land_water_frac.current_landareafrac,
                                  self.land_water_frac.previous_landareafrac,
                                  self.land_water_frac.landareafrac_ratio,
                                  self.land_water_frac.previous_swb_frac,
                                  self.land_water_frac.glores_frac_prevyear,
                                  self.land_water_frac.gloresfrac_change,
                                  self.land_water_frac.init_landfrac_res_flag,
                                  self.land_water_frac.landwaterfrac_excl_glolake_res,
                                  self.land_water_frac.land_and_water_freq_flag,
                                  self.land_water_frac.water_freq,
                                  self.land_water_frac.land_freq,
                                  self.land_water_frac.updated_loclake_frac,

                                  self.vertical_waterbalance.lai_days,
                                  self.vertical_waterbalance.cum_precipitation,
                                  self.vertical_waterbalance.growth_status,
                                  self.vertical_waterbalance.canopy_storage,
                                  self.vertical_waterbalance.snow_water_storage,
                                  self.vertical_waterbalance.snow_water_storage_subgrid,
                                  self.vertical_waterbalance.soil_water_content,
                                  self.vertical_waterbalance.daily_storage_transfer,

                                  self.lateral_waterbalance.groundwater_storage,
                                  self.lateral_waterbalance.loclake_storage,
                                  self.lateral_waterbalance.locwet_storage,
                                  self.lateral_waterbalance.glolake_storage,
                                  self.lateral_waterbalance.glowet_storage,
                                  self.lateral_waterbalance.river_storage,

                                  self.lateral_waterbalance.glores_storage,
                                  self.lateral_waterbalance.k_release,
                                  self.lateral_waterbalance.unsatisfied_potential_netabs_riparian,
                                  self.lateral_waterbalance.unsat_potnetabs_sw_from_demandcell,
                                  self.lateral_waterbalance.unsat_potnetabs_sw_to_supplycell,
                                  self.lateral_waterbalance.get_neighbouring_cells_map,
                                  self.lateral_waterbalance. \
                                  accumulated_unsatisfied_potential_netabs_sw,

                                  self.lateral_waterbalance.daily_unsatisfied_pot_nas,
                                  self.lateral_waterbalance. \
                                  prev_accumulated_unsatisfied_potential_netabs_sw,
                                  self.lateral_waterbalance.prev_potential_water_withdrawal_sw_irri,
                                  self.lateral_waterbalance.prev_potential_consumptive_use_sw_irri,
                                  self.lateral_waterbalance.set_res_storage_flag
                                  )

        print('Status:' + colored(' complete', 'cyan'))

        # if run_calib:
        #     sim_data_calib = {"sim_dis": get_annual_streamflow,
        #                       "pot_cell_runoff": get_annual_pot_cell_runoff}
        #     return sim_data_calib

        pass

    def update(self, day, previous_states, **kwargs):
        """

        Parameters
        ----------
        day
        previous_states

        Returns
        -------

        """
        if previous_states is not None:
            self.passState(previous_states)

        restart = self.restart
        run_calib = cm.run_calib

        self.initialize_forcings_static.climate_forcing.update(day)

        # =================================================================
        #  Get Land area fraction and reservoirs respective years
        # =================================================================
        # Activate reservoirs for current year
        self.lateral_waterbalance. \
            activate_res_area_storage_capacity(day, cm.RESERVOIR_OPT_YEARS,
                                               restart)

        # Get Land area fraction
        self.land_water_frac. \
            landareafrac_with_reservior(day, cm.RESERVOIR_OPT_YEARS)

        # Get land and water fractions (used to calculate total PET)
        self.land_water_frac.get_land_and_water_freq(day)

        # Adapt global reservoir storage and land area fraction
        # due to net change in land fraction
        self.lateral_waterbalance.glores_storage = self.land_water_frac. \
            adapt_glores_storage(self.vertical_waterbalance.canopy_storage,
                                 self.vertical_waterbalance.snow_water_storage,
                                 self.vertical_waterbalance.soil_water_content,
                                 self.lateral_waterbalance.glores_area,
                                 self.lateral_waterbalance.glores_storage)

        # =================================================================
        #  Computing vertical water balance
        # =================================================================
        self.vertical_waterbalance. \
            calculate(day, self.land_water_frac.current_landareafrac,
                      self.land_water_frac.landareafrac_ratio,
                      self.watergap_basin.upstream_basin,
                      self.land_water_frac.water_freq,
                      self.land_water_frac.land_freq)

        # =================================================================
        #  Computing lateral water balance
        # =================================================================
        self.lateral_waterbalance. \
            calculate(self.vertical_waterbalance.fluxes['groundwater_recharge'],
                      self.vertical_waterbalance.fluxes['openwater_PET'],
                      self.vertical_waterbalance.fluxes['daily_precipitation'],
                      self.vertical_waterbalance.fluxes['surface_runoff'],
                      self.vertical_waterbalance.fluxes['daily_storage_transfer'],
                      self.vertical_waterbalance.fluxes['land_aet_corr'],
                      self.land_water_frac.current_landareafrac,
                      self.land_water_frac.previous_landareafrac,
                      self.land_water_frac.landwaterfrac_excl_glolake_res,
                      day, self.first_day_of_month, self.watergap_basin.upstream_basin,
                      self.vertical_waterbalance.fluxes['sum_canopy_snow_soil_storage'],
                      run_calib)

        # =================================================================
        #  Update Land Area Fraction
        # =================================================================
        land_swb_fraction = self.lateral_waterbalance.get_new_swb_fraction()
        self.land_water_frac.update_landareafrac(land_swb_fraction)

        # =============================================================
        # Write vertical and lateralbalacne variables to file
        # =============================================================
        sim_month = pd.to_datetime(day).month
        sim_day = pd.to_datetime(day).day
        sim_year = pd.to_datetime(day).year
        # Getting daily storages and fluxes and writing to variables
        vb_storages_and_fluxes = \
            self.vertical_waterbalance.get_storages_and_fluxes()

        self.create_out_var. \
            verticalbalance_write_daily_var(vb_storages_and_fluxes,
                                            sim_year, sim_month, sim_day)

        # Getting daily storages and fluxes and writing to variables
        lb_storages_and_fluxes = \
            self.lateral_waterbalance.get_storages_and_fluxes()

        self.create_out_var. \
            lateralbalance_write_daily_var(lb_storages_and_fluxes,
                                           sim_year, sim_month, sim_day)

        #  store data on daily basis
        self.create_out_var.base_units(self.initialize_forcings_static.static_data.cell_area,
                                       self.initialize_forcings_static.static_data.
                                       land_surface_water_fraction.contfrac)
        # create_out_var.save_netcdf_parallel(str(date))
        self.create_out_var.save_netcdf_daily_single_file(str(day))

        # =============================================================
        #  Get restart information if restart is needed.
        # =============================================================
        if self.end_date == day.astype('datetime64[D]'):
            if self.savestate_for_restart:
                self.restart_model. \
                    savestate(day,
                              self.land_water_frac.current_landareafrac,
                              self.land_water_frac.previous_landareafrac,
                              self.land_water_frac.landareafrac_ratio,
                              self.land_water_frac.previous_swb_frac,
                              self.land_water_frac.glores_frac_prevyear,
                              self.land_water_frac.gloresfrac_change,
                              self.land_water_frac.init_landfrac_res_flag,
                              self.land_water_frac.landwaterfrac_excl_glolake_res,
                              self.land_water_frac.land_and_water_freq_flag,
                              self.land_water_frac.water_freq,
                              self.land_water_frac.land_freq,
                              self.land_water_frac.updated_loclake_frac,

                              self.vertical_waterbalance.lai_days,
                              self.vertical_waterbalance.cum_precipitation,
                              self.vertical_waterbalance.growth_status,
                              self.vertical_waterbalance.canopy_storage,
                              self.vertical_waterbalance.snow_water_storage,
                              self.vertical_waterbalance.snow_water_storage_subgrid,
                              self.vertical_waterbalance.soil_water_content,
                              self.vertical_waterbalance.daily_storage_transfer,

                              self.lateral_waterbalance.groundwater_storage,
                              self.lateral_waterbalance.loclake_storage,
                              self.lateral_waterbalance.locwet_storage,
                              self.lateral_waterbalance.glolake_storage,
                              self.lateral_waterbalance.glowet_storage,
                              self.lateral_waterbalance.river_storage,

                              self.lateral_waterbalance.glores_storage,
                              self.lateral_waterbalance.k_release,
                              self.lateral_waterbalance.unsatisfied_potential_netabs_riparian,
                              self.lateral_waterbalance.unsat_potnetabs_sw_from_demandcell,
                              self.lateral_waterbalance.unsat_potnetabs_sw_to_supplycell,
                              self.lateral_waterbalance.get_neighbouring_cells_map,
                              self.lateral_waterbalance. \
                              accumulated_unsatisfied_potential_netabs_sw,

                              self.lateral_waterbalance.daily_unsatisfied_pot_nas,
                              self.lateral_waterbalance. \
                              prev_accumulated_unsatisfied_potential_netabs_sw,
                              self.lateral_waterbalance.prev_potential_water_withdrawal_sw_irri,
                              self.lateral_waterbalance.prev_potential_consumptive_use_sw_irri,
                              self.lateral_waterbalance.set_res_storage_flag
                              )

        return self.getState()

    def hot_run_with_daily_timestep(self):
        """
        Perform the daily model run using initialized variables and return the state at each timestep.
        """
        time_range, simulation_date, run_calib = self.time_range, self.simulation_date, cm.run_calib

        # restart = self.restart

        states_over_time = []

        state = None
        for time_step, date in tqdm(zip(range(1, time_range+1), simulation_date),
                                    total=(time_range), desc="Processing",
                                    disable=run_calib):

            state = self.update(date, state)
            # states_over_time.append(state)

        print('Status:' + colored(' complete', 'cyan'))

        pass

    def getState(self):
        state = {
            'canopy_storage': self.vertical_waterbalance.canopy_storage.copy(),
            'snow_water_storage': self.vertical_waterbalance.snow_water_storage.copy(),
            'soil_water_content': self.vertical_waterbalance.soil_water_content.copy(),
            'groundwater_storage': self.lateral_waterbalance.groundwater_storage.copy(),
            'loclake_storage': self.lateral_waterbalance.loclake_storage.copy(),
            'locwet_storage': self.lateral_waterbalance.locwet_storage.copy(),
            'glolake_storage': self.lateral_waterbalance.glolake_storage.copy(),
            'glowet_storage': self.lateral_waterbalance.glowet_storage.copy(),
            'river_storage': self.lateral_waterbalance.river_storage.copy(),
            'glores_storage': self.lateral_waterbalance.glores_storage.copy(),
        }
        return state

    def passState(self, state:dict):

        self.vertical_waterbalance.canopy_storage = state['canopy_storage'].copy()
        self.vertical_waterbalance.snow_water_storage = state['snow_water_storage'].copy()
        self.vertical_waterbalance.soil_water_content = state['soil_water_content'].copy()
        self.lateral_waterbalance.groundwater_storage = state['groundwater_storage'].copy()
        self.lateral_waterbalance.loclake_storage = state['loclake_storage'].copy()
        self.lateral_waterbalance.locwet_storage = state['locwet_storage'].copy()
        self.lateral_waterbalance.glolake_storage = state['glolake_storage'].copy()
        self.lateral_waterbalance.glowet_storage = state['glowet_storage'].copy()
        self.lateral_waterbalance.river_storage = state['river_storage'].copy()
        self.lateral_waterbalance.glores_storage = state['glores_storage'].copy()

        pass


# if __name__ == "__main__":
#     run_with_time_check = check_time(run)
#     run_with_time_check()
