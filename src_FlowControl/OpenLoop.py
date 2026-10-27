from src_FlowControl.SingleModel import SingleModel
from pathlib import Path
import numpy as np
from controller import configuration_module as cm
from mpi4py import MPI


class OpenLoop(SingleModel):
    def __init__(self, case='test', ensemble_id:int=3, setting_dir=None):
        """
        Openloop is ensemble run of the model, which runs the model for each ensemble member with each's parameter
        field and forcing field.
        Parameters
        ----------
        case: the name of this case study
        ensemble_id: index of this ensemble member, default is 3.
        setting_dir: directory that holds Config_ReWaterGAP.json (must be given).
        """
        super().__init__(setting_dir=setting_dir)

        self.case = case
        self.ensemble_id = ensemble_id

        if self.ensemble_id < 0:
            raise ValueError("Ensemble ID must be a non-negative integer for OpenLoop runs.")

        pass

        self.name = 'OpenLoop'

    def configure_Ens_output(self, output_dir=None):
        """
        this function is used to configure the output directory of the model, including the output directory of each ensemble member.
        Parameters
        ----------
        output_dir

        Returns
        -------

        """
        self.output_dir = Path(output_dir) / ('Ens_%s' % self.ensemble_id)
        self.output_dir.mkdir(parents=False, exist_ok=True)
        cm.config_file['FilePath']['outputDir'] = str(self.output_dir).rstrip('/') + '/'
        cm.init_config(config_input=cm.config_file)

        return self

    def configure_ini_for_resume(self, read_init_dir=None, save_init_dir=None):
        """
        this function is used to configure the initialization of the model, including reading and saving the initialization file.
        Parameters
        ----------
        read_init_dir
        save_init_dir

        Returns
        -------

        """
        if read_init_dir is None:
            cm.read_state_path = Path(cm.save_and_read_states_path)
        else:
            cm.read_state_path = Path(read_init_dir) / ('Ens_%s' % self.ensemble_id)

        if save_init_dir is None:
            cm.save_state_path = Path(cm.save_and_read_states_path)
        else:
            save_init_dir = Path(save_init_dir) / ('Ens_%s' % self.ensemble_id)
            save_init_dir.mkdir(parents=False, exist_ok=True)
            cm.save_state_path = save_init_dir

        return self

    def configure_Ens_input(self, input_dir=None):
        """
        this function is used to configure the input directory of the model, including the input directory of each ensemble member.
        Parameters
        ----------
        input_dir

        Returns
        -------

        """
        input_dir = Path(input_dir) / ('Ens_%s' % self.ensemble_id)
        cm.config_file['FilePath']['inputDir']['climate_forcing'] = \
            str(input_dir / 'monthly_climate_forcing').rstrip('/') + '/'
        cm.config_file['FilePath']['inputDir']['parameter_path'] = \
            str(input_dir / 'parameters' / 'WaterGAP_2.2e_global_parameters_gswp3_w5e5.nc')

        cm.init_config(config_input=cm.config_file)
        return self
