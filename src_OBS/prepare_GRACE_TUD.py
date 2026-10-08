"""
GRACE observations from the TU Delft 5-daily (to weekly) hybrid L3 product
(provided by Miguel, TU Delft; files 'TUD-L3-5dayEWH-GRACEv2-Hybrid-*.nc' and
'TUD-L3Uncorr-5dayEWH_UNC-GRACEv2-Hybrid-*.nc').

Product layout (both files): dims (time, lat, lon) = (910, 360, 720); lat -89.75 -> 89.75 (south up),
lon -179.75 -> 179.75; time = integer days since 2002-04-07T12:00 at irregular 4-7 day spacing;
'ewh_hf_hyb' = equivalent water height [assumed cm, checked at run time], 'ewh_gia' = static GIA field;
uncertainty file: 'ewh_hf_unc' [cm] = per-cell total standard deviation (RSS of KBR, regression,
AOD1B and ensemble terms), WITHOUT spatial error correlations.

What this module produces (same files/keys as the CSR-mascon path, so nothing downstream changes):
    <basin>_signal.hdf5          sub_basin_k (mm), time_epoch, sub_basin_area
    <basin>_gridded_signal.hdf5  tws (n_epoch, n_cell) mm, time_epoch, sub_basin_area
    <basin>_cov.hdf5             data (n_epoch, n_sub, n_sub) mm^2, time_epoch
The sub-basin error covariance is propagated from the per-cell standard deviations under an assumed
exponential spatial error correlation  rho_ij = exp(-d_ij / L)  (L = corr_length_km; L = 0 -> uncorrelated
cells, i.e. a lower bound; L -> inf -> fully correlated, i.e. an upper bound). L should be agreed with the
data provider; 300 km is a plausible default for a ~0.5-deg, ~300-km-resolution 5-daily product.

Developer: Fan Yang (fany@plan.aau.dk), Geodesy Group, Aalborg University. 2026-09.
"""
import numpy as np
import h5py
import netCDF4 as nc
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from src_OBS.prepare_GRACE import GRACE_preparation
from src_OBS.obs_auxiliary import aux_TUD_5daily

EARTH_RADIUS_KM = 6371.0


class GRACE_TUD_5daily(GRACE_preparation):

    def __init__(self, basin_name='Amazon', shp_path='../data/basin/shp/Amazon/Amazon.shp'):
        super().__init__(basin_name=basin_name, shp_path=shp_path)
        self._ewh_file = None
        self._unc_file = None
        self._ewh_var = 'ewh_hf_hyb'
        self._unc_var = 'ewh_hf_unc'
        self._unit_factor = 10.0          # cm -> mm
        self._corr_length_km = 300.0
        self._add_gia = False
        pass

    def set_extra_info(self, dir_in, ewh_file='TUD-L3-5dayEWH-GRACEv2-Hybrid-2002_2017-0.5x0.5.nc',
                       unc_file='TUD-L3Uncorr-5dayEWH_UNC-GRACEv2-Hybrid-2002_2017-0.5x0.5.nc',
                       ewh_var='ewh_hf_hyb', unc_var='ewh_hf_unc', unit='cm', corr_length_km=300.0,
                       add_gia=False, day_begin='2002-01-01', day_end='2030-01-01', max_half_window_days=3):
        """
        unit: unit of the EWH / uncertainty variables in the files ('cm', 'm' or 'mm'); converted to mm.
        corr_length_km: e-folding length of the assumed spatial error correlation used in basin_COV.
        add_gia: add the static 'ewh_gia' field to the EWH (the product is delivered GIA-corrected or not -
                 confirm with the provider; default False).
        """
        self._ewh_file = Path(dir_in) / ewh_file
        self._unc_file = Path(dir_in) / unc_file
        self._ewh_var, self._unc_var = ewh_var, unc_var
        self._unit_factor = {'cm': 10.0, 'm': 1000.0, 'mm': 1.0}[unit]
        self._corr_length_km = float(corr_length_km)
        self._add_gia = add_gia
        self._aux = aux_TUD_5daily().setTimeReference(day_begin=day_begin, day_end=day_end,
                                                      dir_in=str(dir_in), filename=ewh_file,
                                                      max_half_window_days=max_half_window_days)
        return self

    # ------------------------------------------------------------------ helpers
    def _load_mask(self, res=0.5):
        mf = h5py.File(Path(self.save_mask_dir) / ('%s_res_%s.h5' % (self.basin_name, res)), 'r')
        basins_num = len([k for k in mf.keys() if k.startswith('sub_basin')])
        land = self._05deg_mask if res == 0.5 else self._1deg_mask
        sub = {i: (mf['sub_basin_%d' % i][:] * land).astype(bool) for i in range(1, basins_num + 1)}
        basin = (mf['basin'][:] * land).astype(bool)
        return sub, basin, basins_num

    @staticmethod
    def _box(mask_2d):
        """row/col bounds (mask orientation: lat 90 -> -90) and the corresponding rows of the data (lat -90 -> 90)"""
        ii = np.where(mask_2d.any(1))[0]
        jj = np.where(mask_2d.any(0))[0]
        r0, r1, c0, c1 = ii.min(), ii.max() + 1, jj.min(), jj.max() + 1
        n_lat = mask_2d.shape[0]
        d0, d1 = n_lat - r1, n_lat - r0          # data rows (south-up) covering mask rows r0:r1
        return (r0, r1, c0, c1), (d0, d1)

    def _read_box(self, ds, var, t_slice, rows, cols):
        """read a (time, lat, lon) box from the netCDF and return it in mask orientation (lat 90 -> -90), mm"""
        (d0, d1), (c0, c1) = rows, cols
        v = ds.variables[var]
        data = np.ma.filled(v[t_slice, d0:d1, c0:c1].astype(np.float64), np.nan)[:, ::-1, :]   # masked -> NaN
        data[~np.isfinite(data)] = np.nan
        return data * self._unit_factor

    def _time_indices(self, day_begin, day_end):
        aux, a, b = self._aux.selectSubTimePeriod(day_begin=day_begin, day_end=day_end)
        idx = self._aux.getDataIndex()[a:b]          # indices into the netCDF time axis
        return aux, idx

    def inspect(self, day_begin='2002-04-01', day_end='2002-06-30'):
        """print value ranges over the basin to confirm units / sign conventions before running anything else"""
        sub, basin, nb = self._load_mask()
        (r0, r1, c0, c1), (d0, d1) = self._box(basin)
        aux, idx = self._time_indices(day_begin, day_end)
        with nc.Dataset(self._ewh_file) as ds:
            raw = np.asarray(ds.variables[self._ewh_var][idx[0]:idx[-1] + 1, d0:d1, c0:c1])[:, ::-1, :]
            gia = np.asarray(ds.variables['ewh_gia'][d0:d1, c0:c1])[::-1, :]
        with nc.Dataset(self._unc_file) as ds:
            unc = np.asarray(ds.variables[self._unc_var][idx[0]:idx[-1] + 1, d0:d1, c0:c1])[:, ::-1, :]
        bm = basin[r0:r1, c0:c1]
        print('TUD 5-daily product, %s, %d epochs %s .. %s' % (self.basin_name, len(idx), aux['time_epoch'][0], aux['time_epoch'][-1]))
        print('  windows (first 5):', aux['duration'][:5])
        print('  %s over basin: min %.3f  max %.3f  mean %.3f  std %.3f  (raw file units)' % (
            self._ewh_var, np.nanmin(raw[:, bm]), np.nanmax(raw[:, bm]), np.nanmean(raw[:, bm]), np.nanstd(raw[:, bm])))
        print('  ewh_gia over basin: min %.4f max %.4f (raw units)' % (np.nanmin(gia[bm]), np.nanmax(gia[bm])))
        print('  %s over basin: min %.3f  max %.3f  median %.3f  (raw units)' % (
            self._unc_var, np.nanmin(unc[:, bm]), np.nanmax(unc[:, bm]), np.nanmedian(unc[:, bm])))
        print('  -> with unit factor %.0f the basin EWH std is %.1f mm and the median cell sigma %.1f mm' % (
            self._unit_factor, np.nanstd(raw[:, bm]) * self._unit_factor, np.nanmedian(unc[:, bm]) * self._unit_factor))
        pass

    # ------------------------------------------------------------------ products
    def basin_TWS(self, day_begin='2002-01-01', day_end='2016-12-31', dir_in=None, dir_out='.'):
        """cosine-weighted sub-basin means [mm] of the 5-daily EWH (dir_in unused: files set in set_extra_info)"""
        print('\nStart to pre-process TUD 5-daily GRACE to obtain basin-wise TWS over places of interest...')
        sub, basin, nb = self._load_mask()
        (r0, r1, c0, c1), (d0, d1) = self._box(basin)
        aux, idx = self._time_indices(day_begin, day_end)
        lat = np.arange(90 - 0.25, -90, -0.5)[r0:r1]
        w_lat = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, c1 - c0))
        TWS = {i: np.full(len(idx), np.nan) for i in range(1, nb + 1)}
        with nc.Dataset(self._ewh_file) as ds:
            gia = np.asarray(ds.variables['ewh_gia'][d0:d1, c0:c1])[::-1, :] * self._unit_factor if self._add_gia else 0.0
            step = 50
            for s in tqdm(range(0, len(idx), step), desc='Processing epochs'):
                sel = idx[s:s + step]
                data = self._read_box(ds, self._ewh_var, slice(sel[0], sel[-1] + 1), (d0, d1), (c0, c1))
                data = data[np.asarray(sel) - sel[0]] + gia          # idx is monotonic; handle gaps in idx
                for i in range(1, nb + 1):
                    m = sub[i][r0:r1, c0:c1]
                    a = data[:, m]; b = w_lat[m]
                    TWS[i][s:s + len(sel)] = np.nansum(a * b, 1) / np.sum(b * np.isfinite(a), 1)
        out_dir = Path(dir_out); out_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(out_dir / ('%s_signal.hdf5' % self.basin_name), 'w') as hm:
            for i in range(1, nb + 1):
                hm.create_dataset('sub_basin_%d' % i, data=TWS[i])
            dt = h5py.special_dtype(vlen=str)
            hm.create_dataset('time_epoch', data=aux['time_epoch'], dtype=dt)
            hm.create_dataset('duration', data=aux['duration'], dtype=dt)
            hm.create_dataset('sub_basin_area', data=self._sub_basin_area)
        print('Finished: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        pass

    def grid_TWS(self, day_begin='2002-01-01', day_end='2016-12-31', dir_in=None, dir_out='.'):
        """gridded EWH [mm] for the basin cells (cells = basin & GRACE land mask, boolean-index order)"""
        print('\nStart to pre-process TUD 5-daily GRACE to obtain grid-wise TWS over places of interest...')
        sub, basin, nb = self._load_mask()
        (r0, r1, c0, c1), (d0, d1) = self._box(basin)
        aux, idx = self._time_indices(day_begin, day_end)
        bm = basin[r0:r1, c0:c1]
        TWS = np.full((len(idx), int(bm.sum())), np.nan)
        with nc.Dataset(self._ewh_file) as ds:
            gia = np.asarray(ds.variables['ewh_gia'][d0:d1, c0:c1])[::-1, :] * self._unit_factor if self._add_gia else 0.0
            step = 50
            for s in tqdm(range(0, len(idx), step), desc='Processing epochs'):
                sel = idx[s:s + step]
                data = self._read_box(ds, self._ewh_var, slice(sel[0], sel[-1] + 1), (d0, d1), (c0, c1))
                data = data[np.asarray(sel) - sel[0]] + gia
                TWS[s:s + len(sel)] = data[:, bm]
        out_dir = Path(dir_out)
        with h5py.File(out_dir / ('%s_gridded_signal.hdf5' % self.basin_name), 'w') as hm:
            hm.create_dataset('tws', data=TWS)
            dt = h5py.special_dtype(vlen=str)
            hm.create_dataset('time_epoch', data=aux['time_epoch'], dtype=dt)
            hm.create_dataset('duration', data=aux['duration'], dtype=dt)
            hm.create_dataset('sub_basin_area', data=self._sub_basin_area)
        print('Finished: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        pass

    def basin_COV(self, day_begin='2002-01-01', day_end='2016-12-31', dir_in=None, dir_out='.',
                  is_diagonal=False, corr_length_km=None):
        """
        Sub-basin error covariance [mm^2] per epoch, propagated from the per-cell standard deviations
        sigma_i(t) with an assumed spatial error correlation rho_ij = exp(-d_ij / L):
            R_kl(t) = sum_i sum_j  w_ki w_lj  sigma_i(t) sigma_j(t) rho_ij ,   w_ki = cos(phi_i) / sum_{i in k} cos(phi_i)
        i.e. R = (W S) rho (W S)^T with W the (n_sub x n_cell) averaging operator and S = diag(sigma).
        """
        L = self._corr_length_km if corr_length_km is None else float(corr_length_km)
        print('\nStart to pre-process TUD 5-daily GRACE to obtain COV over places of interest (L = %.0f km)...' % L)
        sub, basin, nb = self._load_mask()
        (r0, r1, c0, c1), (d0, d1) = self._box(basin)
        aux, idx = self._time_indices(day_begin, day_end)
        bm = basin[r0:r1, c0:c1]
        lat = np.arange(90 - 0.25, -90, -0.5)[r0:r1]; lon = np.arange(-180 + 0.25, 180, 0.5)[c0:c1]
        LON, LAT = np.meshgrid(lon, lat)
        cl, cn = np.deg2rad(LAT[bm]), np.deg2rad(LON[bm])
        # averaging operator (n_sub x n_cell) on the basin cells
        W = np.zeros((nb, int(bm.sum())))
        wl = np.cos(cl)
        for i in range(1, nb + 1):
            mi = sub[i][r0:r1, c0:c1][bm]
            W[i - 1, mi] = wl[mi] / wl[mi].sum()
        # spatial correlation of the cell errors
        if L > 0:
            xyz = np.stack([np.cos(cl) * np.cos(cn), np.cos(cl) * np.sin(cn), np.sin(cl)], 1)
            chord = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
            d_km = 2.0 * np.arcsin(np.clip(chord / 2.0, 0, 1)) * EARTH_RADIUS_KM
            rho = np.exp(-d_km / L)
        else:
            rho = np.eye(W.shape[1])
        COV = np.zeros((len(idx), nb, nb))
        with nc.Dataset(self._unc_file) as ds:
            step = 50
            for s in tqdm(range(0, len(idx), step), desc='Processing epochs'):
                sel = idx[s:s + step]
                sig = self._read_box(ds, self._unc_var, slice(sel[0], sel[-1] + 1), (d0, d1), (c0, c1))
                sig = sig[np.asarray(sel) - sel[0]][:, bm]              # (n_t, n_cell) mm
                sig = np.nan_to_num(sig, nan=np.nanmedian(sig))
                for t in range(sig.shape[0]):
                    WS = W * sig[t][None, :]
                    cov = WS @ rho @ WS.T
                    COV[s + t] = np.diag(np.diag(cov)) if is_diagonal else cov
        out_dir = Path(dir_out)
        with h5py.File(out_dir / ('%s_cov.hdf5' % self.basin_name), 'w') as hm:
            hm.create_dataset('data', data=COV)
            dt = h5py.special_dtype(vlen=str)
            hm.create_dataset('time_epoch', data=aux['time_epoch'], dtype=dt)
            hm.attrs['corr_length_km'] = L
            hm.attrs['note'] = 'propagated from per-cell sigma (%s) with rho_ij = exp(-d/L)' % self._unc_var
        print('Finished: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        pass


def demo_Amazon():
    """stand-alone check: units, then the three products for the Amazon"""
    base = Path('/media/user/My Book/Fan/PyGLDA_v2_external_data')
    GR = GRACE_TUD_5daily(basin_name='Amazon', shp_path=str(base / 'Basin/shp/Amazon/Amazon.shp'))
    GR.configure_global_land_ocean_mask(fn=str(base / 'GRACE/global_mask/WaterGAPLandMask.hdf5'))
    GR.generate_mask(save_dir=str(base / 'Basin/mask'))
    GR.set_extra_info(dir_in=str(base / 'GRACE/Miguel_Tudelft'), unit='cm', corr_length_km=300.0)
    GR.inspect(day_begin='2002-04-01', day_end='2002-12-31')
    out = str(base / 'GRACE/output_TUD')
    GR.basin_TWS(day_begin='2002-01-01', day_end='2016-04-30', dir_out=out)
    GR.grid_TWS(day_begin='2002-01-01', day_end='2016-04-30', dir_out=out)
    GR.basin_COV(day_begin='2002-01-01', day_end='2016-04-30', dir_out=out, is_diagonal=False)
    pass


if __name__ == '__main__':
    demo_Amazon()
