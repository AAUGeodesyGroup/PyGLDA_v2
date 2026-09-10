
import numpy as np
from pathlib import Path
import h5py
from datetime import datetime, timedelta
from src_auxiliary.GeoMathKit import GeoMathKit
from src_DA.EnumDA import Stage, WaterGap_storage_variables
import xarray as xr


class BasinAverageAnalysis:
    """
    This script is for real time analysis to extract basin time series signal for each ensemble member
    """

    def __init__(self, basin_mask_path, state_dir='', stage=Stage.OL, ens_id=0, case_name = ''):
        self.ds = None
        self.__variable_list = None
        self.bm = h5py.File(name=basin_mask_path, mode='r')
        self.__state_dir = Path(state_dir) / case_name / stage.name / f'Ens_{ens_id}'
        self.__stage = stage

    def select_variables(self, variable_list=None):
        if variable_list is None:
            self.__variable_list = [var.name for var in WaterGap_storage_variables]
        else:
            self.__variable_list = variable_list

        return self

    def get_basin_average(self):

        mask_global = self.bm['basin'][:]  # this has to be 0.5 degree
        res = 0.5
        err = res / 10
        lat_coords = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
        lon_coords = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
        mask_global = xr.DataArray(
            mask_global.astype(bool),
            coords={"lat": lat_coords, "lon": lon_coords},
            dims=["lat", "lon"],
            name="region_mask"
        )

        lon_mesh_global, lat_mesh_global = np.meshgrid(lon_coords, lat_coords)
        mask_whole_basin = mask_global.where(mask_global, drop=True)
        mask_whole_basin = mask_whole_basin.notnull().values

        res = {}
        for key in self.__variable_list:
            res[key] ={}
            for basin in self.bm.keys():
                '''produce the 2D mask for each basin'''
                mask_sub_basin =  self.bm[basin][:].astype(bool)  # this has to be 0.5 degree
                lat_local = lat_mesh_global[mask_sub_basin]
                mask_global = self.bm['basin'][:].astype(bool)
                mask_bbox = mask_whole_basin.copy()
                mask_bbox[mask_whole_basin] = mask_sub_basin[mask_global]

                '''load nc file and calculate the basin average'''
                a = self.ds[key].values[:, mask_bbox]
                b = np.cos(np.deg2rad(lat_local))
                c = np.sum(a*b,1)/np.sum(b)
                res[key][basin] = c.copy()
                pass

        hf = h5py.File(str(Path(self.__state_dir) / ('basin_ts_%s.h5'%self.__stage.name)), 'w')
        # print(str(Path(self.__state_dir) / ('basin_ts_%s.h5'%self.__stage.name)))
        for gr, bb in res.items():
            dict_group = hf.create_group(gr)
            for key, vv in bb.items():
                dict_group.create_dataset(key, data=np.array(vv))
        hf.close()

        del self.ds
        pass

    def load_nc_files(self, start_date="2002-01-01", end_date="2005-04-30"):
        """
        Load the NetCDF files of given date range
        """
        import os
        import pandas as pd
        import xarray as xr

        # 2. Generate the date range using pandas
        date_range = pd.date_range(start=start_date, end=end_date)
        years_to_process = date_range.year.unique()

        # 3. Build the list of file paths based on the date sequence
        file_paths = []

        for year in years_to_process:
            # Construct the filename matching your naming convention: daily_output_YYYY-MM-DD.nc
            filename = f"daily_output_{year}.nc"
            full_path = os.path.join(self.__state_dir, filename)

            # Check if the file exists to avoid errors
            if os.path.exists(full_path):
                file_paths.append(full_path)
            else:
                print(f"Warning: File not found and skipped -> {filename}")

        # 4. Use xarray's open_mfdataset to read and merge multiple netcdf files at once
        if file_paths:
            # combine='by_coords' automatically concatenates the datasets along the coordinate dimensions (like time)
            ds = xr.open_mfdataset(file_paths, combine="by_coords", parallel=True, chunks={"time": 365})

            # print("Successfully loaded and merged dataset:")

            self.ds = ds

            # print(ds)

            # Example: Access a specific variable for further processing
            # data_var = ds['your_variable_name']
        else:
            raise FileNotFoundError("No matching files found for the specified date range.")

        return self


class BasinAverageAnalysis_post:
    """
    This is also for the basin signal analysis. However, it is different from the above one at:
    1. this is a post-processing step to collect all the ensemble results together for further analysis
    2. in addition to the states, this script also deals with GRACE data for a collective analysis.
    """

    def __init__(self, ens, case: str, basin: str, date_begin='2002-04-01', date_end='2002-04-02'):
        self.ens = ens
        self.case = case
        self.basin = basin

        """deal with the time tag"""
        daylist = GeoMathKit.dayListByDay(begin=date_begin, end=date_end)
        day_first = daylist[0].year + (daylist[0].month - 1) / 12 + daylist[0].day / 365.25
        self.time_tag = day_first + np.arange(len(daylist)) / 365.25

        self.__states = None
        self.__GRACE = None
        pass

    def get_states(self, dir: str, stage: Stage):

        post_fix = stage.name

        temp_dict = {}
        for ens in range(self.ens + 1):
            fn = h5py.File(Path(dir) / self.case/ post_fix/ ('Ens_%s' % ens) / ('basin_ts_%s.h5' % post_fix), 'r')
            ens_dict = {}
            for basin, item in fn.items():
                state_dict = {}
                for state, item2 in item.items():
                    state_dict[state] = item2[:]

                # state_dict['TWS'] = np.sum(np.array(list(state_dict.values())), axis=0)

                ens_dict[basin] = state_dict

            temp_dict[ens] = ens_dict

        '''reformulate for easier analysis'''
        state = fn.keys()
        basins = list(item.keys())

        ff = {}
        for basin in basins:
            dd = {}
            for st in state:
                ss = {}
                for ens in range(self.ens + 1):
                    ss[ens] = temp_dict[ens][st][basin]
                dd[st] = ss
            ff[basin] = dd

        assert len(self.time_tag) == len(ss[ens]), 'Time tag is incorrect!'

        ff['time'] = self.time_tag
        self.__states = ff
        return ff

    def get_GRACE(self, obs_dir: str):
        kk = {}

        gr = h5py.File(Path(obs_dir) / ('%s_obs_GRACE.hdf5' % self.basin), 'r')

        str_time = list(gr['time_epoch'][:].astype(str))
        fraction_time = []
        for tt in str_time:
            if len(tt.split('-')) == 2:
                tt = tt + '-15'
            da = datetime.strptime(tt, '%Y-%m-%d')
            fraction_time.append(da.year + (da.month - 1) / 12 + da.day / 365.25)

        kk['time'] = np.array(fraction_time)
        kk['original'] = {}
        kk['ens_mean'] = {}

        a = gr['ens_1'][:]

        '''calculate how many ensemble member it has'''
        ens_num = -1  # because ens= 0 is included
        for dd in gr.keys():
            if 'ens_' in dd:
                ens_num += 1
        # print(ens_num)
        try:
            basin_num = np.shape(gr['cov'][0])[0]
        except Exception:
            basin_num = 1

        for ens_id in range(2, ens_num + 1):
            a += gr['ens_%s' % ens_id]

        ens_mean = a / ens_num
        original = gr['ens_0'][:]

        for basin in range(1, 1 + basin_num):
            kk['original']['basin_%s' % basin] = original[:, basin - 1]
            kk['ens_mean']['basin_%s' % basin] = ens_mean[:, basin - 1]

        # kk['original']['basin_0'] = np.mean(original, axis=1)
        # kk['ens_mean']['basin_0'] = np.mean(ens_mean, axis=1)

        '''Area-weighted'''
        sub_basin_area = gr['sub_basin_area'][:]
        kk['original']['basin_0'] = np.sum(original * sub_basin_area[None, :], axis=1) / np.sum(sub_basin_area)
        kk['ens_mean']['basin_0'] = np.sum(ens_mean * sub_basin_area[None, :], axis=1) / np.sum(sub_basin_area)

        self.__GRACE = kk

        return kk

    def save_states(self, save_dir='../temp', prefix='1'):
        ff = self.__states

        hf = h5py.File(Path(save_dir) / ('Res_%s.h5df' % prefix), 'w')

        for ii, jj in ff.items():
            if ii == 'time':
                hf.create_dataset(name=ii, data=jj)
                continue
            dict_group1 = hf.create_group(ii)
            for kk, ll in jj.items():
                dict_group2 = dict_group1.create_group(kk)
                for mm, nn in ll.items():
                    dict_group2.create_dataset(name=str(mm), data=nn)
                    pass

        hf.close()

        pass

    def save_GRACE(self, save_dir='../temp', prefix='1'):

        ff = self.__GRACE

        hf = h5py.File(Path(save_dir) / ('GRACE_%s.h5df' % prefix), 'w')

        for ii, jj in ff.items():
            if ii == 'time':
                hf.create_dataset(name=ii, data=jj)
                continue
            dict_group1 = hf.create_group(ii)
            for kk, ll in jj.items():
                dict_group1.create_dataset(name=kk, data=ll)
                pass

        hf.close()
        pass

    def load_states(self, load_dir='../temp', prefix='1'):
        """
        This is simply for loading states from pre-saved results. No calculation has been done.
        It is mostly used for plotting a figure
        """
        ff = {}

        hf = h5py.File(Path(load_dir) / ('Res_%s.h5df' % prefix), 'r')

        for ii, jj in hf.items():
            if ii == 'time':
                ff[ii] = jj[:]
                continue
            dict_group1 = {}
            for kk, ll in jj.items():
                dict_group2 = {}
                for mm, nn in ll.items():
                    dict_group2[int(mm)] = nn[:]
                    pass
                dict_group1[kk] = dict_group2
            ff[ii] = dict_group1
        return ff

    def load_GRACE(self, load_dir='../temp', prefix='1'):
        """
        This is simply for loading GRACE from pre-saved results. No calculation has been done.
        It is mostly used for plotting a figure
        """
        ff = {}

        hf = h5py.File(Path(load_dir) / ('GRACE_%s.h5df' % prefix), 'r')

        for ii, jj in hf.items():
            if ii == 'time':
                ff[ii] = jj[:]
                continue
            dict_group1 = {}
            for kk, ll in jj.items():
                dict_group1[kk] = ll[:]

            ff[ii] = dict_group1

        return ff



def demo1():
    # basin_shp = basin_shp_process(save_dir='/media/user/My Book/Fan/WaterGap/Basin/mask',
    #                               basin_name='Brahmaputra', res=0.5)
    # basin_shp.shp_to_mask(shp_path='/media/user/My Book/Fan/WaterGap/Basin/shp/Brahmaputra/Brahmaputra.shp', issave=True)

    # analysis = BasinAverageAnalysis_realtime(basin_mask_path='/media/user/My Book/Fan/WaterGap/Basin/mask/Brahmaputra_res_0.5.h5',
    #                                          state_dir='/media/user/My Book/Fan/WaterGap/Res/test', stage=Stage.OL, ens_id=0)
    #
    # analysis.load_nc_files(start_date="2002-01-01", end_date="2005-04-30").select_variables(variable_list=['tws']).get_basin_average()
    pass


def demo2():
    from mpi4py import MPI
    """Parallel execution using MPI. Each rank will have its own log file."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    analysis = BasinAverageAnalysis(
        basin_mask_path='/media/user/My Book/Fan/WaterGap/Basin/mask/Brahmaputra_res_0.5.h5',
        state_dir='/media/user/My Book/Fan/WaterGap/Res/test', stage=Stage.OL, ens_id=rank)

    analysis.load_nc_files(start_date="2002-01-01", end_date="2005-04-30").select_variables(
        variable_list=['tws']).get_basin_average()


if __name__ == '__main__':
    demo2()
