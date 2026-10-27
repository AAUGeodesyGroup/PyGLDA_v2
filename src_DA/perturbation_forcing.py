"""
Ensemble perturbation, part 2: the FORCING fields (one set of monthly forcing files per member).

    perturbation_forcing   forcing perturbation -> Ensemble_input/Ens_k/monthly_climate_forcing/<yyyy-mm>.nc
Shared settings / members / folders: src_DA.perturbation_par.perturbation_base. The parameters are perturbed by
src_DA.perturbation_par.perturbation_par; both read the same perturbation.json ("forcing" / "correlation.forcing"
here).
"""
import numpy as np
from enum import Enum
import json
from pathlib import Path
import xarray as xr

from src_DA.perturbation_par import perturbation_base, error_distribution, DOELL2024


class perturb_method(Enum):
    additive = 1
    multiplicative = 2
    pass




class perturbation_forcing(perturbation_base):
    """
    Forcing perturbation of the monthly climate forcing files (dir.in/monthly_climate_forcing ->
    dir.out/Ens_k/monthly_climate_forcing, Ens_0 = unperturbed copy).
    Notice: additive perturbations are in the units of the forcing files.
    """

    '''zlib level of the perturbed forcing files: 1 is ~2.5x faster to write than 4 for ~8% larger files
    (a global month: ~2.7 s vs ~6.7 s per file); WaterGAP reads any level'''
    FORCING_COMPLEVEL = 1

    def _format_check(self):
        forcing = self.setting['forcing']

        for key, value in forcing.items():
            assert error_distribution[value['error_distribution']] in error_distribution
            assert perturb_method[value['perturb_method']] in perturb_method
            pass
        pass


    def perturb_forcing(self, comm=None):
        forcing = self.setting['correlation']['forcing']

        if forcing['customize']:
            self._perturb_forcing_customize()
            return

        """using the built-in functions"""
        if forcing["isSpatialCorrelated"] and forcing["isTemporalCorrelated"]:
            self._forcing_spatiotemporal_correlated(comm=comm)
        elif forcing["isSpatialCorrelated"] and (not forcing["isTemporalCorrelated"]):
            self._forcing_spatial_correlated_temporal_noncorrelated(comm=comm)
        elif (not forcing["isSpatialCorrelated"]) and forcing["isTemporalCorrelated"]:
            self._forcing_spatial_noncorrelated_temporal_correlated()
        else:
            self._forcing_spatiotemporal_noncorrelated()

        pass

    def _perturb_forcing_customize(self):
        """
        This has to be customized by user
        """
        pass

    def _forcing_spatial_correlated_temporal_noncorrelated(self, comm=None):
        """
        One random factor (multiplicative) or offset (additive) per member and per day, the same for every grid
        cell. Per month: rank 0 draws the samples of all members and broadcasts them (small: ens x days); each rank
        then builds and writes only its own members, so the expensive part (compressing the global files) runs in
        parallel and no rank holds the whole ensemble in memory.
        """
        dir_in = Path(self.setting['dir']['in'])
        dir_out = Path(self.setting['dir']['out'])

        forcing = self.setting['forcing']
        perturbed = [key for key in forcing if forcing[key]['is_perturbed']]
        root = comm is None or comm.Get_rank() == 0
        members = self._members(comm)
        verbose = comm is None or comm.Get_rank() == min(1, comm.Get_size() - 1)

        for im, month in enumerate(self.monthlist):

            '''load data'''
            fn_1 = dir_in / 'monthly_climate_forcing' / ('%s.nc' % (month.strftime('%Y-%m')))
            with xr.open_dataset(fn_1, decode_times=False) as ds:
                ds_1 = ds.load()

            '''random samples of all members (ens, days), drawn on rank 0'''
            samples = {}
            if root:
                for key in perturbed:
                    s_shape = tuple([np.shape(ds_1[key].values)[0]])
                    if forcing[key]['perturb_method'] == perturb_method.multiplicative.name:
                        samples[key] = self.__generate_perturbation(config=forcing[key], mean=np.ones(s_shape))
                    else:
                        samples[key] = self.__generate_perturbation(config=forcing[key], mean=np.zeros(s_shape))
            if comm is not None:
                samples = comm.bcast(samples, root=0)

            '''write the members of this rank'''
            for ens in members:
                fn_2 = dir_out / ('Ens_%s' % ens) / 'monthly_climate_forcing' / ('%s.nc' % (month.strftime('%Y-%m')))

                if ens == 0:
                    '''save the unperturbed data'''
                    ds_1.to_netcdf(fn_2)
                    continue

                ds_2 = ds_1.copy(deep=True)
                for key in perturbed:
                    data = ds_1[key].values
                    if forcing[key]['perturb_method'] == perturb_method.multiplicative.name:
                        hh = samples[key][ens - 1][:, None, None] * data
                    else:
                        hh = data + samples[key][ens - 1][:, None, None]
                    '''make sure that the perturbed data is not negative'''
                    hh[hh < 0] = 0
                    ds_2[key].values = hh                       # assigning .values keeps attrs/encoding
                    if ds_2[key].dtype != 'float32':
                        ds_2[key] = ds_2[key].astype('float32')

                '''one chunk per variable and month'''
                ds_2.to_netcdf(fn_2, format="NETCDF4", engine="netcdf4", encoding={
                    var: {"chunksizes": ds_2[var].shape, "zlib": True,
                          "complevel": self.FORCING_COMPLEVEL}
                    for var in ds_2.data_vars
                })

            if verbose:
                print('Forcing perturbation: %s done (%d/%d months), %d member(s) on this rank' %
                      (month.strftime('%Y-%m'), im + 1, len(self.monthlist), len(members)), flush=True)

        pass

    def _forcing_spatial_noncorrelated_temporal_correlated(self):
        """
        Spatially non-correlated, temporally correlated perturbation: not implemented.
        """
        pass

    def _forcing_spatiotemporal_correlated(self, comm=None):
        """
        Forcing errors correlated in space and time ("isSpatialCorrelated" and "isTemporalCorrelated" true), one random
        field per member, variable and day:
          - space: Gaussian correlation exp(-(r / L)^2), L = "length_km" of correlation.forcing (e^-1 at L km); the
            smoothing width in longitude grows with 1 / cos(latitude), so L is a distance everywhere; periodic in
            longitude;
          - time: AR(1) per cell, z_t = phi z_(t-1) + sqrt(1 - phi^2) eta_t, phi = exp(-1 / "time_scale_days"),
            continuous over all months of one call of perturb_forcing;
          - z is standard normal in every cell and day; per variable ("coefficients": [std]):
              multiplicative + lognormal : factor = exp(s z - s^2 / 2), s^2 = ln(1 + std^2)  (mean 1, std = std, > 0)
              multiplicative + gaussian  : factor = 1 + std z (negative values set to 0)
              additive + gaussian        : value + std z
        Precipitation lognormal 0.5 and AR(1) 24 h as in Kumar et al. (2016, JHM 17, 1951).
        Random numbers: each member and variable has its own generator from (base seed, member, variable), so the
        result does not depend on the number of MPI ranks; base seed = "seed" of the json, or (seed null) one number
        drawn on rank 0 from numpy's global generator and broadcast. Each rank writes only its own members.
        """
        dir_in = Path(self.setting['dir']['in'])
        dir_out = Path(self.setting['dir']['out'])
        corr = self.setting['correlation']['forcing']
        for key in ('length_km', 'time_scale_days'):
            if key not in corr:
                raise ValueError('perturbation.json: correlation.forcing needs "%s" for the correlated forcing mode'
                                 % key)
        length_km = float(corr['length_km'])
        tau = float(corr['time_scale_days'])
        phi = float(np.exp(-1.0 / tau)) if tau > 0 else 0.0

        forcing = self.setting['forcing']
        perturbed = [key for key in forcing if forcing[key]['is_perturbed']]
        for key in perturbed:
            method, dist = forcing[key]['perturb_method'], forcing[key]['error_distribution']
            if (method, dist) not in (('multiplicative', 'lognormal'), ('multiplicative', 'gaussian'),
                                      ('additive', 'gaussian')):
                raise ValueError('correlated forcing mode: %s %s of %s is not supported (multiplicative lognormal / '
                                 'gaussian, additive gaussian)' % (method, dist, key))

        '''base seed, the same on all ranks'''
        seed = self.setting.get('seed', None)
        if seed is None:
            base = int(np.random.randint(0, 2 ** 31 - 1)) if comm is None or comm.Get_rank() == 0 else None
            if comm is not None:
                base = comm.bcast(base, root=0)
        else:
            base = int(seed)

        members = self._members(comm)
        verbose = comm is None or comm.Get_rank() == min(1, comm.Get_size() - 1)
        rngs = {(ens, k): np.random.default_rng([base, ens, k]) for ens in members if ens > 0
                for k in range(len(perturbed))}
        z = {}                                          # AR(1) state per (member, variable)
        field = None

        for im, month in enumerate(self.monthlist):

            fn_1 = dir_in / 'monthly_climate_forcing' / ('%s.nc' % (month.strftime('%Y-%m')))
            with xr.open_dataset(fn_1, decode_times=False) as ds:
                ds_1 = ds.load()
            if field is None:
                field = correlated_field(ds_1['lat'].values, ds_1['lon'].values, length_km)

            for ens in members:
                fn_2 = dir_out / ('Ens_%s' % ens) / 'monthly_climate_forcing' / ('%s.nc' % (month.strftime('%Y-%m')))

                if ens == 0:
                    '''save the unperturbed data'''
                    ds_1.to_netcdf(fn_2)
                    continue

                ds_2 = ds_1.copy(deep=True)
                for k, key in enumerate(perturbed):
                    data = ds_1[key].values
                    rng = rngs[(ens, k)]
                    zz = np.empty(data.shape, dtype=float)
                    for d in range(data.shape[0]):
                        eta = field.draw(rng)
                        if (ens, k) not in z:
                            z[(ens, k)] = eta                                   # stationary start
                        else:
                            z[(ens, k)] = phi * z[(ens, k)] + np.sqrt(1.0 - phi ** 2) * eta
                        zz[d] = z[(ens, k)]
                    std = float(forcing[key]['coefficients'][0])
                    if forcing[key]['perturb_method'] == perturb_method.additive.name:
                        hh = data + std * zz
                    elif forcing[key]['error_distribution'] == error_distribution.lognormal.name:
                        s2 = np.log(1.0 + std ** 2)
                        hh = data * np.exp(np.sqrt(s2) * zz - 0.5 * s2)
                    else:
                        hh = data * (1.0 + std * zz)
                    '''make sure that the perturbed data is not negative'''
                    hh[hh < 0] = 0
                    ds_2[key].values = hh.astype(ds_1[key].dtype)
                    if ds_2[key].dtype != 'float32':
                        ds_2[key] = ds_2[key].astype('float32')

                ds_2.to_netcdf(fn_2, format="NETCDF4", engine="netcdf4", encoding={
                    var: {"chunksizes": ds_2[var].shape, "zlib": True,
                          "complevel": self.FORCING_COMPLEVEL}
                    for var in ds_2.data_vars
                })

            if verbose:
                print('Forcing perturbation (correlated, L = %.0f km, tau = %.1f d): %s done (%d/%d months), '
                      '%d member(s) on this rank' % (length_km, tau, month.strftime('%Y-%m'), im + 1,
                                                     len(self.monthlist), len(members)), flush=True)

        pass

    def _forcing_spatiotemporal_noncorrelated(self):
        raise NotImplementedError('forcing perturbation without spatial and temporal correlation is not implemented')

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
                """co[0]: error percentage (factor between 1 - co[0] and 1 + co[0], mode 1)"""
                pp = self.triangle_perturbation(mean=dd, left=dd * (1 - co[0]), right=dd * (1 + co[0]))
                pass
            else:
                """co[0]: error percentage"""
                pp = self.Gaussian_perturbation(mean=dd, std=dd * co[0])

            mean = np.expand_dims(mean, axis=0) * pp

            pass

        return mean


class correlated_field:
    """
    Standard-normal random fields on a regular lat / lon grid (degrees) with the Gaussian correlation
    exp(-(r / L)^2): white noise smoothed by a Gaussian kernel of std L / 2 (km) in latitude (reflected at the grid
    edges) and in longitude (periodic; width in cells grows with 1 / cos(latitude)), then scaled to unit variance in
    every cell.
    """
    KM_PER_DEG = 111.195

    def __init__(self, lat, lon, length_km):
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        self.shape = (lat.size, lon.size)
        dlat = abs(float(lat[1] - lat[0])) if lat.size > 1 else 1.0
        dlon = abs(float(lon[1] - lon[0])) if lon.size > 1 else 1.0
        s_km = 0.5 * float(length_km)                       # kernel std: correlation exp(-r^2 / (4 s^2))

        '''latitude: explicit kernel, truncated at 4 std'''
        s_lat = max(s_km / (dlat * self.KM_PER_DEG), 1e-6)
        self.half = int(np.ceil(4 * s_lat))
        w = np.exp(-0.5 * (np.arange(-self.half, self.half + 1) / s_lat) ** 2)
        self.w_lat = w / w.sum()
        self.idx_lat = self._pad_index(lat.size)

        '''longitude: periodic Gaussian in the Fourier domain, width per row'''
        coslat = np.clip(np.cos(np.deg2rad(lat)), 1e-3, None)
        s_lon = s_km / (dlon * self.KM_PER_DEG * coslat)    # cells, per row
        n = lon.size
        f = np.fft.rfftfreq(n)                              # cycles per cell
        self.h_lon = np.exp(-2.0 * (np.pi * f[None, :] * s_lon[:, None]) ** 2)

        '''variance of the smoothed unit white noise per row: latitude part exact (with the reflection), longitude
        part from the transfer function (Parseval)'''
        h2 = self.h_lon ** 2
        full = h2[:, 0] + 2 * h2[:, 1:(n + 1) // 2].sum(1) + (h2[:, n // 2] if n % 2 == 0 else 0.0)
        var_lon = full / n
        var_lat = np.zeros(lat.size)
        for r in range(lat.size):
            wsum = np.zeros(lat.size)
            np.add.at(wsum, self.idx_lat[r:r + 2 * self.half + 1], self.w_lat)
            var_lat[r] = (wsum ** 2).sum()
        self.scale = (1.0 / np.sqrt(var_lat * var_lon))[:, None]

    def _pad_index(self, nlat):
        """row index of the padded field: reflected at the edges (wrapped if the kernel is wider than the grid)"""
        mode = 'reflect' if self.half < nlat else 'wrap'
        return np.pad(np.arange(nlat), (self.half, self.half), mode=mode)

    def draw(self, rng):
        """one field, standard normal in every cell"""
        x = rng.standard_normal(self.shape)
        xp = x[self.idx_lat]
        y = np.zeros(self.shape)
        for i, wi in enumerate(self.w_lat):
            y += wi * xp[i:i + self.shape[0]]
        y = np.fft.irfft(np.fft.rfft(y, axis=1) * self.h_lon, n=self.shape[1], axis=1)
        return y * self.scale
