import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from src_GHM.Interface.Spinup import run as sp_run
from controller import configuration_module as cm
from src_GHM.Interface.DailyStepRun import DailyModelRun
from termcolor import colored


class SingleModel:

    def __init__(self, setting_dir=None):
        """setting_dir: directory that holds Config_ReWaterGAP.json (must be given)."""
        self.setting_json = Path(setting_dir)/'Config_ReWaterGAP.json'
        cm.init_config(config_input=self.setting_json)
        from model import land_surfacewater_fraction as lsf
        lsf.update_setting()

        pass

    def configure_time(self, begin_time='2000-01-01', end_time='2000-01-31'):
        self.period = [datetime.strptime(begin_time, '%Y-%m-%d'), datetime.strptime(end_time, '%Y-%m-%d')]
        cm.config_file['RuntimeOptions'][2]['SimulationPeriod']['start'] = begin_time
        cm.config_file['RuntimeOptions'][2]['SimulationPeriod']['end'] = end_time

        '''also make sure that the reservoir start and end year are set to the simulation period as well'''
        cm.config_file['RuntimeOptions'][2]['SimulationPeriod']['reservoir_start_year'] = self.period[0].year
        cm.config_file['RuntimeOptions'][2]['SimulationPeriod']['reservoir_end_year'] = self.period[1].year

        cm.init_config(config_input=cm.config_file)
        return self


    def model_spinup(self, spinup_years=5):
        """
        It spins up the model for the year of start date
        Returns
        -------

        """
        cm.config_file['RuntimeOptions'][2]['SimulationPeriod']['spinup_years']= spinup_years
        cm.init_config(config_input=cm.config_file)
        import sys
        sp_run()
        pass


    def configure_ini_for_resume(self, read_init_dir=None, save_init_dir=None):
        """
        Parameters
        ----------
        read_init_dir
        save_init_dir

        Returns
        -------

        """
        if read_init_dir is None:
            cm.save_state_path = cm.save_and_read_states_path
        else:
            cm.save_state_path = read_init_dir

        if save_init_dir is None:
            cm.read_state_path = cm.save_and_read_states_path
        else:
            cm.read_state_path = save_init_dir

        return self

    def model_resume(self):
        """
        It resumes the model from the last day of the spinup period to the end of the simulation period.
        Returns
        -------

        """
        cm.start = self.period[0].strftime('%Y-%m-%d')
        cm.end = self.period[1].strftime('%Y-%m-%d')
        daily_run = DailyModelRun()
        daily_run.hot_run_with_daily_timestep()
        pass
