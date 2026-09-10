
from src_auxiliary.GeoMathKit import GeoMathKit


import numpy as np
from enum import Enum
import json
from pathlib import Path
from datetime import datetime, timedelta
import xarray as xr
from tqdm import tqdm


class perturb_method(Enum):
    additive = 1
    multiplicative = 2
    pass


class error_distribution(Enum):
    triangle = 1
    gaussian = 2
    pass


class perturbation:
    """
    #TODO: A sensitivity test is necessary for further analysis!
    Notice: Take care of the unit of the perturbation! see ext_forcing.py
    """

    def __init__(self, dp='./DA_settings/perturbation.json', ens_size=None):
        self.setting = json.load(open(dp, 'r'))
        if ens_size is None:
            self.ens = self.setting['ensemble']
        else:
            self.ens = ens_size
        self._format_check()
        self._make_dir()
        pass

    def _format_check(self):
        forcing = self.setting['forcing']
        par = self.setting['par']

        for key, value in forcing.items():
            assert error_distribution[value['error_distribution']] in error_distribution
            assert perturb_method[value['perturb_method']] in perturb_method
            pass

        pass

    @staticmethod
    def save_default_json():

        forcing = ('pr', 'tas', 'rsds', 'rlds')

        par = ('openwater_albedo', 'snow_albedo_thresh', 'pt_coeff_humid_arid', 'snow_freeze_temp', 'snow_melt_temp',
               'adiabatic_lapse_rate', 'runoff_frac_builtup', 'gamma', 'areal_corr_factor', 'max_daily_pet',
               'critcal_gw_precipitation', 'gw_dis_coeff', 'reduction_exponent_lakewet', 'gw_recharge_constant',
               'swb_outflow_coeff', 'lake_out_exp', 'activelake_depth', 'wetland_out_exp', 'activewetland_depth',
               'swb_drainage_area_factor', 'reduction_exponent_res', 'stat_corr_fact', 'river_roughness_coeff_mult',
               'max_canopy_storage_coefficient')

        perturbation_dict = {
            'ensemble': 30,
            'dir': {
                'in': '/media/user/My Book/Fan/WaterGap/Input_data',
                'out': '/media/user/My Book/Fan/WaterGap/Ensemble_input'
            },
            "correlation": {
                "forcing": {
                    "isSpatialCorrelated": True,
                    "isTemporalCorrelated": False,
                    "customize": False
                },
                "par": {
                    "isSpatialCorrelated": True,
                    "customize": False
                }
            },
            'forcing': {},
            'par': {}
        }

        template1 = {
            'is_perturbed': False,
            'perturb_method': perturb_method.additive.name,
            'error_distribution': error_distribution.triangle.name,
            'coefficients': [0.3]
        }

        template2 = {
            'is_perturbed': False,
            'error_distribution': error_distribution.triangle.name,
            'coefficients': [0, 1, 2]
        }

        template3 = {
            'is_perturbed': False,
            'error_distribution': error_distribution.triangle.name,
            "coefficients": {
                "humid": [0.885, 1.26, 1.65],
                "arid": [1.365, 1.74, 2.115]
            }
        }

        for key in forcing:
            perturbation_dict['forcing'][key] = template1

        for key in par:
            perturbation_dict['par'][key] = template2

        perturbation_dict['par']['pt_coeff_humid_arid'] = template3

        default_path = Path.cwd() / 'DA_settings' / 'perturbation.json'
        with open(default_path, 'w') as f:
            json.dump(perturbation_dict, f, indent=4)

        return str(default_path)

    def setDate(self, month_begin='2000-01', month_end='2000-02'):
        self.month_begin = datetime.strptime(month_begin, '%Y-%m')
        self.month_end = datetime.strptime(month_end, '%Y-%m')
        self.monthlist = GeoMathKit.monthListByMonth(begin=month_begin, end=month_end)
        return self

    def perturb_par(self):
        par = self.setting['correlation']['par']

        if par['customize']:
            self._perturb_par_customize()
            return

        """using the built-in functions"""
        if par["isSpatialCorrelated"]:
            self._par_spatial_correlated()
        else:
            assert False, 'This function is not applicable for WaterGap yet!'
        pass

    def perturb_forcing(self):
        forcing = self.setting['correlation']['forcing']

        if forcing['customize']:
            self._perturb_forcing_customize()
            return

        """using the built-in functions"""
        if forcing["isSpatialCorrelated"] and forcing["isTemporalCorrelated"]:
            self._forcing_spatiotemporal_correlated()
        elif forcing["isSpatialCorrelated"] and (not forcing["isTemporalCorrelated"]):
            self._forcing_spatial_correlated_temporal_noncorrelated()
        elif (not forcing["isSpatialCorrelated"]) and forcing["isTemporalCorrelated"]:
            self._forcing_spatial_noncorrelated_temporal_correlated()
        else:
            self._forcing_spatiotemporal_noncorrelated()

        pass

    def triangle_perturbation(self, left, mean, right, size=None):
        return np.random.triangular(left=left, mode=mean,
                                    right=right, size=size)

    def Gaussian_perturbation(self, mean, std, size=None):
        return np.random.normal(loc=mean, scale=std, size=size)

    def _perturb_forcing_customize(self):
        """
        This has to be customized by user
        """
        # todo: considering partial spatial-temporal correlation
        pass

    def _perturb_par_customize(self):
        """
        This has to be customized by user
        """
        # todo: considering partial spatial-temporal correlation
        pass

    def _make_dir(self):
        dir_out_parent = Path(self.setting['dir']['out'])

        ens_size = self.ens

        for ens_id in range(ens_size + 1):

            dir_out = dir_out_parent / ('Ens_%s' % ens_id)

            if dir_out.exists():
                pass
            else:
                dir_out.mkdir()
                pass

            for keyw in ['parameters', 'monthly_climate_forcing']:
                dir_out_par = dir_out / keyw
                if dir_out_par.exists():
                    pass
                else:
                    dir_out_par.mkdir()
                    pass

        pass

    def _par_spatial_correlated(self):

        dir_out_parent = Path(self.setting['dir']['out'])
        ens_size = self.ens

        '''produce the coefficients for each parameter based on the error distribution and perturbation method'''
        perturbed_ens = {}
        for key, value in self.setting['par'].items():
            if value['is_perturbed']:
                if value['error_distribution'] == error_distribution.triangle.name:
                    if key == 'pt_coeff_humid_arid':
                        coeff_humid = value['coefficients']['humid']
                        coeff_arid = value['coefficients']['arid']
                        perturbed_ens[key + '_humid'] = self.triangle_perturbation(left=coeff_humid[0],
                                                                                   mean=coeff_humid[1],
                                                                                   right=coeff_humid[2], size=ens_size)
                        perturbed_ens[key + '_arid'] = self.triangle_perturbation(left=coeff_arid[0],
                                                                                  mean=coeff_arid[1],
                                                                                  right=coeff_arid[2], size=ens_size)
                        continue
                    coeff = value['coefficients']
                    perturbed_ens[key] = self.triangle_perturbation(left=coeff[0], mean=coeff[1], right=coeff[2],
                                                                    size=ens_size)
                elif value['error_distribution'] == error_distribution.gaussian.name:
                    coeff = value['coefficients']
                    perturbed_ens[key] = self.Gaussian_perturbation(mean=coeff[0], std=coeff[1], size=ens_size)
                else:
                    raise ValueError(f"Unknown error distribution: {value['error_distribution']}")

        for ens_id in range(ens_size + 1):
            print('Perturbing parameters for ensemble member %s' % ens_id)
            dir_in = Path(self.setting['dir']['in']) / 'Parameters' / 'WaterGAP_2.2e_global_parameters_gswp3_w5e5.nc'
            # Open the dataset with decode_times=False to avoid decoding the problematic time units
            ds = xr.open_dataset(dir_in, decode_times=False)

            '''for each ensemble member, create a folder to store the perturbed parameters'''
            dir_out = dir_out_parent / ('Ens_%s' % ens_id) / 'parameters'

            if ens_id == 0:
                '''do not perturb the parameters for the first ensemble member, just copy the original parameters'''
                ds.to_netcdf(dir_out / 'WaterGAP_2.2e_global_parameters_gswp3_w5e5.nc')
                continue

            for key, value in self.setting['par'].items():
                if value['is_perturbed']:
                    if key == 'pt_coeff_humid_arid':
                        ds[key].values = np.where(ds[key].values == self.setting['par'][key]['coefficients']['humid'][1],
                                                  perturbed_ens[key + '_humid'][ens_id - 1],
                                                  ds[key].values)
                        ds[key].values = np.where(
                            ds[key].values == self.setting['par'][key]['coefficients']['arid'][1],
                            perturbed_ens[key + '_arid'][ens_id - 1],
                            ds[key].values)
                    else:
                        # Log mismatched values for debugging
                        non_nan_mask = ~np.isnan(ds[key].values)
                        mismatched_values = (
                                    ds[key].values[non_nan_mask] != self.setting['par'][key]['coefficients'][1])
                        if np.any(mismatched_values):
                            print(f"Key: {key}, Mismatched values: {ds[key].values[non_nan_mask][mismatched_values]}")
                            print(f"Expected value: {self.setting['par'][key]['coefficients'][1]}")

                        # Perform the assertion
                        assert np.all(
                            ds[key].values[~np.isnan(ds[key].values)] == self.setting['par'][key]['coefficients'][
                                1]), key

                        # Replace non-NaN values with perturbed coefficients
                        ds[key].values = np.where(~np.isnan(ds[key].values), perturbed_ens[key][ens_id - 1],
                                                  ds[key].values)

                    pass
            ds.to_netcdf(dir_out / 'WaterGAP_2.2e_global_parameters_gswp3_w5e5.nc')
        pass


    def _forcing_spatiotemporal_noncorrelated(self):
        """
        Todo: implement the function for spatiotemporal non-correlated perturbation
        Returns
        -------

        """
        pass

    def _forcing_spatiotemporal_correlated(self):
        """
        Todo: implement the function for spatiotemporal correlated perturbation
        """
        pass

    def _forcing_spatial_correlated_temporal_noncorrelated(self):
        dir_in = Path(self.setting['dir']['in'])
        dir_out = Path(self.setting['dir']['out'])

        forcing = self.setting['forcing']

        for month in self.monthlist:
            # print(month.strftime('%Y-%m'))

            '''load data'''
            fn_1 = dir_in / 'monthly_climate_forcing' / ('%s.nc' % (month.strftime('%Y-%m')))
            ds_1 = xr.open_dataset(fn_1, decode_times=False)

            outdata = {}
            for key in forcing:
                dict_group_load = ds_1[key].values

                if not forcing[key]['is_perturbed']:
                    outdata[key] = dict_group_load
                    continue

                dd = np.ones(tuple([self.ens] + list(np.shape(dict_group_load))))
                s_shape = tuple([np.shape(dict_group_load)[0]])
                if forcing[key]['perturb_method'] == perturb_method.multiplicative.name:
                    samples = self.__generate_perturbation(config=forcing[key], mean=np.ones(s_shape))
                    hh = samples[:, :, None, None] * dd * dict_group_load[None, :, :]
                    outdata[key] = hh

                    pass
                else:
                    samples = self.__generate_perturbation(config=forcing[key], mean=np.zeros(s_shape))
                    hh = dd * dict_group_load[None, :, :] + samples[:, :, None, None] * dd
                    outdata[key] = hh

                '''make sure that the perturbed data is not negative'''
                outdata[key][outdata[key] < 0] = 0

            '''write the perturbed data to the new file'''
            for ens in tqdm(np.arange(self.ens+1), desc='Month: %s, loop over ensemble members'%month.strftime('%Y-%m')):

                '''save data'''
                fn_2 = dir_out /('Ens_%s'%ens) /'monthly_climate_forcing' / ('%s.nc' % (month.strftime('%Y-%m')))

                if ens == 0:
                    '''save the unperturbed data'''
                    ds_1.to_netcdf(fn_2)
                    continue

                for key in outdata.keys():
                    if forcing[key]['is_perturbed']:
                        ds_1[key].values = outdata[key][ens-1]

                        if ds_1[key].dtype != 'float32':
                            ds_1[key] = ds_1[key].astype('float32')

                '''# monthly chunking for each variable'''
                ds_1.to_netcdf(fn_2, format="NETCDF4", engine="netcdf4", encoding={
                    var: {"chunksizes": ds_1[var].shape, "zlib": True,
                          "complevel": 4}
                    for var in ds_1.data_vars
                })


        pass

    def _forcing_spatial_noncorrelated_temporal_correlated(self):
        """
        TOdo: implement the function for spatial non-correlated and temporal correlated perturbation
        Returns
        -------

        """
        pass


    def __generate_perturbation(self, config: dict, mean=None):

        co = config['coefficients']

        dd = np.ones(tuple([self.ens] + list(np.shape(mean))))

        if config['perturb_method'] == perturb_method.additive.name:
            if config['error_distribution'] == error_distribution.triangle.name:
                """ co[0]: limit"""
                pp = self.triangle_perturbation(mean=0 * dd, left=0 - dd * co[0], right=dd * co[0])
            else:
                """ co[0]: std"""
                pp = self.Gaussian_perturbation(mean=0 * dd, std=co[0] * dd)
                pass

            mean = np.expand_dims(mean, axis=0) + pp

        else:
            """multiplicative error"""
            if config['error_distribution'] == error_distribution.triangle.name:
                """co[0]: error percentage"""
                pp = self.triangle_perturbation(mean=dd, left=0 - dd * co[0], right=dd * co[0])
                pass
            else:
                """co[0]: error percentage"""
                pp = self.Gaussian_perturbation(mean=dd, std=dd * co[0])

            mean = np.expand_dims(mean, axis=0) * pp

            pass

        return mean

def demo1():
    # dp = perturbation.save_default_json()
    dp = './DA_settings/perturbation.json'
    pp = perturbation(dp=dp).setDate(month_begin='2000-01', month_end='2005-04')
    # pp.perturb_par()
    pp.perturb_forcing()

    pass


if __name__ == '__main__':
    demo1()
