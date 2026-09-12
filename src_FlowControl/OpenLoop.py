from src_FlowControl.SingleModel import SingleModel
from pathlib import Path
import numpy as np
from controller import configuration_module as cm
from mpi4py import MPI


class OpenLoop(SingleModel):
    def __init__(self, case='test', ensemble_id:int=3, setting_dir='/media/user/My Book/Fan/WaterGap/Extensions/'):
        """
        Openloop is ensemble run of the model, which runs the model for each ensemble member with each's parameter
        field and forcing field.
        Parameters
        ----------
        case: the name of this case study
        ensemble_id: number of the ensemble members, default is 3.
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


class TqdmLogFilter:
    """
    A custom stream wrapper designed to intercept output streams containing tqdm control characters,
    carriage returns (\r), and excessive refresh traces, ensuring only clean text is saved to the log file.
    """

    def __init__(self, filename):
        self.terminal = sys.stdout  # Retain the original stdout just in case
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        # 1. Replace carriage returns (\r) commonly used by tqdm with newlines (\n)
        # to prevent log lines from overwriting each other or creating clutter.
        cleaned_message = message.replace('\r', '\n')

        # 2. Optional: Strip ANSI color escape codes if needed for an even cleaner log.
        # cleaned_message = re.sub(r'\x1b\[[0-9;]*m', '', cleaned_message)

        self.log.write(cleaned_message)
        self.log.flush()  # Ensure real-time writing to the file

    def flush(self):
        self.log.flush()


if __name__ == '__main__':
    import os
    import sys

    """Parallel execution using MPI. Each rank will have its own log file."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank != 0:
        log_dir = str(Path(__file__).resolve().parent.parent / 'parallel_logs' / 'OL')
        os.makedirs(log_dir, exist_ok=True)

        log_path = os.path.join(log_dir, f"rank_{rank}.log")

        sys.stdout = open(log_path, 'w', encoding='utf-8')
        # sys.stdout = TqdmLogFilter(log_path)
        sys.stderr = sys.stdout

    from misc.time_checker_and_ascii_image import check_time

    # OL = OpenLoop(ensemble_id=rank).configure_time(begin_time='2000-01-01', end_time='2000-01-31')
    # OL = OpenLoop(ensemble_id=rank).configure_time(begin_time='2002-01-01', end_time='2005-04-30')
    OL = OpenLoop(ensemble_id=rank,setting_dir='/media/user/My Book/Fan/WaterGap/Extensions/DA_settings')
    OL.configure_time(begin_time='2002-01-01', end_time='2002-01-31')
    OL.configure_Ens_output(output_dir='/media/user/My Book/Fan/WaterGap/OL_output')
    OL.configure_Ens_input(input_dir='/media/user/My Book/Fan/WaterGap/Ensemble_input')
    OL.configure_ini_for_resume(save_init_dir='/media/user/My Book/Fan/WaterGap/Ensemble_Initialization',
                                read_init_dir='/media/user/My Book/Fan/WaterGap/Ensemble_Initialization')
    # OL.configure_ini_for_resume(save_init_dir='/media/user/My Book/Fan/WaterGap/Ensemble_Initialization')
    OL.model_spinup(spinup_years=1)
    # OL.model_resume()

    """Single execution for testing"""
    # OL = SingleModel(ensemble_id=3).configure_time()
    # OL.configure_output(output_dir='/media/user/My Book/Fan/WaterGap/Ensemble_output')
    # OL.configure_input(input_dir='/media/user/My Book/Fan/WaterGap/Ensemble_input')
    # OL.model_resume()
