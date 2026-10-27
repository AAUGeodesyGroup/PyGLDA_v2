from src_DA.EnumDA import WaterGap_storage_variables
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime
from src_auxiliary.GeoMathKit import GeoMathKit
from src_DA.ObsDesignMatrix import DM_basin_average
from src_DA.configure_DA import config_DA
from src_auxiliary.shp2mask import load_mask
import xarray as xr
import netCDF4 as nc


class EnsStates:
    """
    to extract the states of the ensembles. Only works for DA process
    """

    def __init__(self, DM: DM_basin_average, configDA: config_DA, ensemble_id=0):
        self.DM = DM
        _, local_mask = load_mask(mask_path=configDA.basic.basin_mask)
        self.__local_mask = local_mask
        self.__ensemble_id = ensemble_id
        self.__states_dir = Path(configDA.basic.DA_output_temp_dir) / configDA.basic.case / ('Ens_%s' % ensemble_id)
        pass

    def load_state_dict(self, date='2002-04-01'):
        """
        load the states from a specific date and save all into a dictionary
        The state is global for OL and regional for DA
        """

        nc_file_path = Path(self.__states_dir) / ('daily_output_%s.nc' % date)

        result_dict = {}

        with xr.open_dataset(nc_file_path) as ds:

            for var in self.DM.statesnn:
                # Safety check: verify if the target variable exists in the dataset
                if var in ds.data_vars:
                    # Store as pure NumPy array
                    result_dict[var] = ds[var].values.copy()

                else:
                    raise KeyError(f"Target variable '{var}' is not found in the dataset '{nc_file_path}'!")
        return result_dict

    def get_states_by_transfer_single(self, states: dict):
        """
        This is just for one ensemble member. The states obtained are restricted within the area of study basin.
        The input state has to be regional as well and its dimension is 2D. the output is 1D
        Parameters
        ----------
        states

        Returns
        -------

        """
        mk = self.__local_mask['basin_2d'].astype(bool)
        ss = None
        for key in self.DM.statesnn:
            vv = states[key][mk]  # it is converted to a 1d vector
            if ss is None:
                ss = vv
            else:
                ss = np.vstack((ss, vv))  # 2D matrix

        if ss is not None:
            return ss.T.flatten()  # 1D vector
        else:
            return 0

    def restore_states(self, old_states: dict, new_states, isdelta=False):
        """
        update the old states: for only one state/ensemble member
        isdelta: if this is an incremental.
        the old states is a 2D matrix for a specific region
        the new state is regional and 1D
        The output is regional and 2D
        """
        mk = self.__local_mask['basin_2d'].astype(bool)

        new_states = new_states.reshape((-1, self.DM.vertical_dim))
        new_states = new_states.T

        m = 0
        for key in self.DM.statesnn:
            if not isdelta:
                old_states[key][mk] = new_states[m]  # project backwards to 2D matrix
            else:
                old_states[key][mk] += new_states[m]

            m += 1

        return old_states

    def save_to_nc(self, state: dict, date='2002-04-01'):
        """
        Load the states updated by EnKF, save them to the original NC file,
        and dynamically recalculate TWS while keeping original NaNs.
        """
        nc_file_path = Path(self.__states_dir) / ('daily_output_%s.nc' % date)

        # 1. Dynamically extract all variable names from the Enum, excluding 'tws'
        tws_components = [
            var.name for var in WaterGap_storage_variables
            if var.name != WaterGap_storage_variables.tws.name
        ]

        tws_var_name = WaterGap_storage_variables.tws.name

        # 2. Open the NetCDF file in 'a' (append/modify) mode
        with nc.Dataset(nc_file_path, mode='a') as dataset:

            # ==========================================
            # STEP 1: Overwrite assimilated variables
            # ==========================================
            for var in self.DM.statesnn:
                if var in dataset.variables:
                    dataset.variables[var][:] = state[var]
                else:
                    raise KeyError(f"ERROR: Assimilated variable '{var}' not found in {nc_file_path}!")

            # ==========================================
            # STEP 2: Recalculate and update TWS (Keeping NaNs)
            # ==========================================
            if tws_var_name in dataset.variables:

                new_tws = None

                # Sum up all components directly.
                # NumPy will automatically propagate NaNs (NaN + value = NaN).
                for comp in tws_components:
                    if comp in dataset.variables:
                        comp_data = dataset.variables[comp][:]

                        if new_tws is None:
                            # Initialize new_tws with the first valid component array
                            new_tws = comp_data.copy()
                        else:
                            # Direct matrix addition
                            new_tws += comp_data
                    else:
                        raise KeyError(
                            f"CRITICAL ERROR: Required TWS component '{comp}' is missing from {nc_file_path}! Cannot calculate valid TWS.")

                # Overwrite the TWS variable in the NetCDF file
                if new_tws is not None:
                    dataset.variables[tws_var_name][:] = new_tws

            else:
                print(f"Warning: TWS variable '{tws_var_name}' not found. Cannot update TWS.")

    def map_state_enum_to_internal(self, enum_state_dict: dict) -> dict:
        """
        Map the standardized Enum-based keys back to the internal model storage keys.

        Parameters:
        -----------
        enum_state_dict : dict
            A dictionary with keys as WaterGap_storage_variables Enum names
            (e.g., 'canopystor', 'groundwstor').

        Returns:
        -----------
        dict
            A new dictionary with keys converted back to internal model names
            (e.g., 'canopy_storage', 'groundwater_storage').
        """
        # Define the reverse mapping bridge (Enum names -> Internal model names)
        reverse_mapping = {
            WaterGap_storage_variables.canopystor.name: 'canopy_storage',
            WaterGap_storage_variables.swe.name: 'snow_water_storage',
            WaterGap_storage_variables.soilmoist.name: 'soil_water_content',
            WaterGap_storage_variables.groundwstor.name: 'groundwater_storage',
            WaterGap_storage_variables.locallakestor.name: 'loclake_storage',
            WaterGap_storage_variables.localwetlandstor.name: 'locwet_storage',
            WaterGap_storage_variables.globallakestor.name: 'glolake_storage',
            WaterGap_storage_variables.globalwetlandstor.name: 'glowet_storage',
            WaterGap_storage_variables.riverstor.name: 'river_storage',
            WaterGap_storage_variables.reservoirstor.name: 'glores_storage',
        }

        # Convert keys back using a loop with a safety check
        internal_state = {}
        for enum_key, value in enum_state_dict.items():
            if enum_key in reverse_mapping:
                internal_key = reverse_mapping[enum_key]
                internal_state[internal_key] = value.copy()
            else:
                raise KeyError(f"ERROR: Internal state key '{enum_key}' has no matching Enum mapping and was skipped.")

        return internal_state

    def box2global_transfer(self, box_data, global_data):

        global_data[self.__local_mask['global_2d'].astype(bool)] = box_data[self.__local_mask['basin_2d'].astype(bool)]

        pass

    def regional2Dstate_transfer_global2Dstate(self, rs:dict, gs:dict):
        """

        Parameters
        ----------
        rs: regional state, enum_state_for_saving
        gs: global state, model_internal_state

        Returns
        -------

        """
        rs_internal=self.map_state_enum_to_internal(enum_state_dict=rs)
        for key, vv in rs_internal.items():
            self.box2global_transfer(box_data=vv, global_data=gs[key])
        pass
