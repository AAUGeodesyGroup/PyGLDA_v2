"""
Ensemble perturbation, part 1: the shared base class and the model PARAMETERS (one parameter file per member).

    perturbation_base   settings (perturbation.json), ensemble size, seed / random generator, months, members per
                        MPI rank, output folders Ensemble_input/Ens_k/{parameters, monthly_climate_forcing}
    perturbation_par    parameter perturbation -> Ens_k/parameters/WaterGAP_2.2e_global_parameters_gswp3_w5e5.nc
The forcing fields are perturbed by src_DA.perturbation_forcing (perturbation_forcing, the same base class).
Both read the same perturbation.json: "par" / "correlation.par" here, "forcing" / "correlation.forcing" there.
"""
from src_auxiliary.GeoMathKit import GeoMathKit


import numpy as np
from enum import Enum
import json
from pathlib import Path
from datetime import datetime
import xarray as xr


class error_distribution(Enum):
    triangle = 1
    gaussian = 2
    uniform = 3
    lognormal = 4       # forcing only (perturbation_forcing, correlated mode): multiplicative factor, mean 1
    pass



class par_type(Enum):
    value = 1           # one global value in the parameter file, replaced by the member's value
    multiplier = 2      # factor: multiplies a field of the parameter file (gamma, river roughness), or is written as
                        # a new variable that WaterGAP applies to a quantity outside the file (see MULTIPLIERS)
    pass



'''multipliers WaterGAP applies to quantities that are not in the parameter file (the same names as in
src_GHM/ReWaterGAP/model/parameters.py MULTIPLIERS, which fills them with 1.0 when missing)'''
MULTIPLIERS = ('lai_multiplier', 'degreeday_multiplier', 'smax_multiplier', 'gw_recharge_factor_multiplier',
               'max_gw_recharge_multiplier', 'netabs_gw_multiplier', 'netabs_sw_multiplier')

'''parameters of Döll et al. (2024, HESS 28, 2259, Table 1; uniform a priori distributions) available in PyGLDA,
as [min, standard, max] in the units of the parameter file (temperatures in K). The precipitation and net radiation
multipliers are covered by the forcing perturbation; the evaporation reduction factor multiplier is not implemented.
gamma (Döll: runoff coefficient 0.3-3) and river roughness (Döll: 1-5 around 3) vary by cell and are perturbed as
multipliers of their fields.'''
DOELL2024 = {
    'pt_coeff_humid_arid': ('value', {'humid': [0.885, 1.26, 1.65], 'arid': [1.365, 1.74, 2.115]}),
    'max_canopy_storage_coefficient': ('value', [0.1, 0.3, 1.4]),
    'lai_multiplier': ('multiplier', [0.2, 1.0, 2.5]),
    'snow_freeze_temp': ('value', [272.15, 273.15, 276.15]),
    'snow_melt_temp': ('value', [269.4, 273.15, 276.9]),
    'degreeday_multiplier': ('multiplier', [0.5, 1.0, 2.0]),
    'adiabatic_lapse_rate': ('value', [0.001, 0.006, 0.01]),
    'smax_multiplier': ('multiplier', [0.5, 1.0, 3.0]),
    'gamma': ('multiplier', [0.3, 1.0, 3.0]),
    'max_daily_pet': ('value', [6.0, 15.0, 22.0]),
    'gw_recharge_factor_multiplier': ('multiplier', [0.3, 1.0, 3.0]),
    'max_gw_recharge_multiplier': ('multiplier', [0.3, 1.0, 3.0]),
    'critcal_gw_precipitation': ('value', [2.5, 12.5, 20.0]),
    'gw_dis_coeff': ('value', [0.001, 0.01, 0.02]),
    'river_roughness_coeff_mult': ('multiplier', [1 / 3, 1.0, 5 / 3]),
    'activelake_depth': ('value', [1.0, 5.0, 20.0]),
    'activewetland_depth': ('value', [1.0, 2.0, 20.0]),
    'swb_outflow_coeff': ('value', [0.001, 0.01, 0.1]),
    'netabs_gw_multiplier': ('multiplier', [-2.0, 1.0, 2.0]),
    'netabs_sw_multiplier': ('multiplier', [-2.0, 1.0, 2.0]),
}


class perturbation_base:
    """
    Settings and bookkeeping shared by the parameter and the forcing perturbation.

    Parallel: with an MPI communicator the ensemble members are split over the ranks (rank r writes Ens_r,
    Ens_r+size, ...). Rank 0 draws all random numbers and broadcasts them, so the result does not depend on the
    number of ranks. Without comm everything runs serially.
    Random numbers: a fixed "seed" in the json gives each of the two classes its own generator
    (np.random.default_rng(seed)); null / missing = the global numpy generator, shared by both in the order they are
    called (parameters first, then forcing), which reproduces the ensembles made with the single Perturbation.py.
    """

    def __init__(self, dp='./DA_settings/perturbation.json', ens_size=None):
        self.setting = json.load(open(dp, 'r'))
        if ens_size is None:
            self.ens = self.setting['ensemble']
        else:
            self.ens = ens_size
        seed = self.setting.get('seed', None)
        self.rng = np.random if seed is None else np.random.default_rng(int(seed))
        self._format_check()
        self._make_dir()
        pass

    def _format_check(self):
        """checks of the settings this class uses (overridden)"""
        pass

    def setDate(self, month_begin='2000-01', month_end='2000-02'):
        self.month_begin = datetime.strptime(month_begin, '%Y-%m')
        self.month_end = datetime.strptime(month_end, '%Y-%m')
        self.monthlist = GeoMathKit.monthListByMonth(begin=month_begin, end=month_end)
        return self

    def _members(self, comm=None):
        """ensemble members handled by this rank (all members without comm)"""
        if comm is None:
            return list(range(self.ens + 1))
        return list(range(comm.Get_rank(), self.ens + 1, comm.Get_size()))

    def triangle_perturbation(self, left, mean, right, size=None):
        return self.rng.triangular(left=left, mode=mean,
                                   right=right, size=size)

    def Gaussian_perturbation(self, mean, std, size=None):
        return self.rng.normal(loc=mean, scale=std, size=size)

    def uniform_perturbation(self, low, high, size=None):
        return self.rng.uniform(low=low, high=high, size=size)

    def _make_dir(self):
        dir_out_parent = Path(self.setting['dir']['out'])

        ens_size = self.ens

        for ens_id in range(ens_size + 1):

            dir_out = dir_out_parent / ('Ens_%s' % ens_id)

            dir_out.mkdir(exist_ok=True)                        # exist_ok: safe when all MPI ranks call it

            for keyw in ['parameters', 'monthly_climate_forcing']:
                (dir_out / keyw).mkdir(exist_ok=True)

        pass


class perturbation_par(perturbation_base):
    """
    Parameter perturbation: one value per member and parameter (spatially uniform; fields of the parameter file
    that vary by cell are multiplied), written into a copy of the original parameter file per member
    (dir.in/Parameters -> dir.out/Ens_k/parameters, Ens_0 = unperturbed copy).
    """

    def _format_check(self):
        par = self.setting['par']

        for key, value in par.items():
            assert error_distribution[value['error_distribution']] in error_distribution, key
            ptype = par_type[value.get('type', par_type.value.name)]
            if ptype == par_type.multiplier and key != 'pt_coeff_humid_arid':
                coeff = value['coefficients']
                if value['error_distribution'] != error_distribution.gaussian.name and coeff[1] != 1:
                    raise ValueError('perturbation.json: multiplier %s needs 1 as its middle coefficient, got %s'
                                     % (key, coeff))
        pass

    def perturb_par(self, comm=None):
        par = self.setting['correlation']['par']

        if par['customize']:
            self._perturb_par_customize()
            return

        """using the built-in functions"""
        if par["isSpatialCorrelated"]:
            self._par_spatial_correlated(comm=comm)
        else:
            assert False, 'This function is not applicable for WaterGap yet!'
        pass

    def _draw_par(self, value, coeff, size):
        """member values of one parameter entry: triangle / uniform from [min, middle, max], gaussian (mean, std)"""
        dist = value['error_distribution']
        if dist in (error_distribution.triangle.name, error_distribution.uniform.name) and coeff[0] == coeff[2]:
            return np.full(size, float(coeff[0]))           # fixed value (e.g. arid PT: [1.74, 1.74, 1.74])
        if dist == error_distribution.triangle.name:
            return self.triangle_perturbation(left=coeff[0], mean=coeff[1], right=coeff[2], size=size)
        if dist == error_distribution.uniform.name:
            return self.uniform_perturbation(low=coeff[0], high=coeff[2], size=size)
        if dist == error_distribution.gaussian.name:
            return self.Gaussian_perturbation(mean=coeff[0], std=coeff[1], size=size)
        raise ValueError(f"Unknown error distribution: {dist}")

    def _perturb_par_customize(self):
        """
        This has to be customized by user
        """
        # todo: considering partial spatial-temporal correlation
        pass

    def _par_spatial_correlated(self, comm=None):

        dir_out_parent = Path(self.setting['dir']['out'])
        ens_size = self.ens

        '''produce the coefficients for each parameter based on the error distribution and perturbation method
        (rank 0 draws them and broadcasts, so all ranks use the same values)'''
        perturbed_ens = {}
        for key, value in (self.setting['par'].items() if comm is None or comm.Get_rank() == 0 else []):
            if not value['is_perturbed']:
                continue
            if key == 'pt_coeff_humid_arid':
                for zone in ('humid', 'arid'):
                    perturbed_ens[key + '_' + zone] = self._draw_par(value, value['coefficients'][zone], ens_size)
                continue
            perturbed_ens[key] = self._draw_par(value, value['coefficients'], ens_size)
        if comm is not None:
            perturbed_ens = comm.bcast(perturbed_ens, root=0)

        for ens_id in self._members(comm):
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
                if not value['is_perturbed']:
                    continue
                ptype = par_type[value.get('type', par_type.value.name)]
                if key == 'pt_coeff_humid_arid':
                    '''two zones, identified by their standard values'''
                    coeff = value['coefficients']
                    for zone in ('humid', 'arid'):
                        ds[key].values = np.where(ds[key].values == coeff[zone][1],
                                                  perturbed_ens[key + '_' + zone][ens_id - 1], ds[key].values)
                elif ptype == par_type.multiplier and key in ds:
                    '''field of the parameter file (gamma, river roughness): multiplied, NaN stays NaN'''
                    ds[key].values = ds[key].values * perturbed_ens[key][ens_id - 1]
                elif ptype == par_type.multiplier:
                    '''quantity outside the parameter file: new variable, applied by WaterGAP'''
                    if key not in MULTIPLIERS:
                        raise ValueError('perturbation.json: %s is neither in the parameter file nor a known '
                                         'multiplier %s' % (key, MULTIPLIERS))
                    template = ds['max_daily_pet']
                    ds[key] = xr.full_like(template, perturbed_ens[key][ens_id - 1], dtype=float)
                else:
                    '''one global value: the file must hold the standard value given as the middle coefficient'''
                    standard = value['coefficients'][1]
                    vals = ds[key].values
                    mismatched = vals[~np.isnan(vals)] != standard
                    if np.any(mismatched):
                        print(f"Key: {key}, Mismatched values: {vals[~np.isnan(vals)][mismatched]}")
                        print(f"Expected value: {standard}")
                    assert not np.any(mismatched), key
                    ds[key].values = np.where(~np.isnan(vals), perturbed_ens[key][ens_id - 1], vals)
            ds.to_netcdf(dir_out / 'WaterGAP_2.2e_global_parameters_gswp3_w5e5.nc')
        pass
