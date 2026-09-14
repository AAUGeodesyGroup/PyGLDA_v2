"""
ts.py - harmonic decomposition of (hydrological) time series by least squares
============================================================================

Developer : Fan Yang (fany@plan.aau.dk), Geodesy Group, Aalborg University
Part of   : PyGLDA v2 - Python Global Land Data Assimilation system

Fits one or many time series with a deterministic model made of

    y(t) = bias + trend * t
           + a1 cos(2 pi t)      + b1 sin(2 pi t)        (annual,         period 1 yr)
           + a2 cos(2 pi t/0.5)  + b2 sin(2 pi t/0.5)    (semi-annual,    period 0.5 yr)
           + a4 cos(2 pi t/0.25) + b4 sin(2 pi t/0.25)   (quarter-annual, period 0.25 yr)

where t is time in YEARS measured from the first epoch of the series (see `set_period`).
Which terms are included is chosen with `setDecomposition`; every cosine/sine pair is
reported back as an amplitude and a phase (see `getSignal` for the phase convention).

Typical use (as in `demo2`): monthly basin-mean TWS from an OL or DA run, one column per
(sub-)basin, to extract the linear trend and the annual amplitude/phase of each basin.

Usage
-----
    tsd = ts().set_period(time=year_fractions).setDecomposition([decomposition.trend,
                                                                  decomposition.annual])
    res = tsd.getSignal(obs)        # obs: (n_time, n_series)
    res['bias']    -> (n_series,)   fitted value at the FIRST epoch (t = 0)
    res['trend']   -> (n_series,)   slope, in data-units per year
    res['annual']  -> (2, n_series) row 0 = amplitude A, row 1 = phase phi in degrees
    res['annual_calendar'] -> phase referenced to 1 January (comparable between runs)
    res['annual_peak_doy'] -> day of year of the annual maximum
    err = tsd.getError()            # formal 1-sigma errors of the above (separate call)
"""

from enum import Enum
import numpy as np


class decomposition(Enum):
    """
    The deterministic signal components that can be fitted. The integer values fix the
    ORDER in which the corresponding columns are appended to the design matrix (after the
    leading bias column), and `getSignal` relies on the same order to read the solution.
    """
    trend = 0            # linear term, 1 column  : t
    annual = 1           # period 1.00 yr, 2 cols : cos(2 pi t), sin(2 pi t)
    semi_annual = 2      # period 0.50 yr, 2 cols : cos(4 pi t), sin(4 pi t)
    quarter_annual = 3   # period 0.25 yr, 2 cols : cos(8 pi t), sin(8 pi t)


class ts:
    """
    Least-squares harmonic decomposition of one or many time series sharing one time axis.

    Workflow: set_period()  ->  setDecomposition()  ->  getSignal()
    The design matrix is built once in `setDecomposition` and reused for every call to
    `getSignal`, so many series can be analysed with the same epochs cheaply.
    """

    def __init__(self):
        self.time = None      # time axis in years, relative to the first epoch (set by set_period)
        self.time0 = None     # absolute year fraction of the first epoch (for calendar phases)
        pass

    def set_period(self, time: np.ndarray):
        """
        Time is year fraction, e.g., 2005.22

        The axis is shifted so that the FIRST epoch becomes t = 0 (good numerical conditioning
        of the trend column). Consequences for the interpretation of the results:
          * 'bias'  is the fitted value at the first epoch, not at year 0 or at 1 January
          * the RAW phases ('annual', ...) are referenced to the first epoch (phi = 0 means a
            maximum at the first epoch). They are not comparable between series that start
            in different months. Use the '<harmonic>_calendar' entries returned by
            `getSignal` for phases referenced to 1 January.
        The trend and the amplitudes are unaffected by the shift.
        """
        time = np.asarray(time, dtype=float)
        self.time0 = float(time[0])
        self.time = time - time[0]
        return self

    def setDecomposition(self, signal=None):
        """
        put whatever you would like to analyze into the list

        signal : list of `decomposition` members to include. Default: trend + annual.
        Builds the design matrix DM of shape (n_time, n_par) with columns, in this order:
            [ 1 | t | cos, sin (annual) | cos, sin (semi-annual) | cos, sin (quarter-annual) ]
        The bias column is always present; the others only if requested. Columns are
        appended in enum order regardless of the order given in `signal`.
        """

        if signal is None:
            # signal = [decomposition.trend, decomposition.annual, decomposition.semi_annual,
            #           decomposition.quarter_annual]
            signal = [decomposition.trend, decomposition.annual]

        # kept (with possible duplicates) for the membership tests in getSignal
        self.__signal = signal

        '''remove possible duplicate element'''
        signal = list(dict.fromkeys(signal))

        # bias column: a column vector of ones, shape (n_time, 1)
        DM = np.ones(len(self.time)).T[:, None]
        # iterate over the ENUM (not over `signal`) so the column order is fixed by the enum
        for ss in decomposition:
            if ss not in signal:
                continue

            dm = self.__get_sub_dm(ss)          # (1, n_time) for trend, (2, n_time) for harmonics

            DM = np.hstack((DM, dm.T))          # append as columns -> (n_time, n_par)

        self.__DM = DM
        return self

    def __get_sub_dm(self, signal=decomposition.trend):
        """
        Design-matrix rows for one component, shape (n_terms, n_time):
          trend          -> [t]
          harmonic (P)   -> [cos(2 pi t / P), sin(2 pi t / P)]   with P = 1, 0.5, 0.25 yr
        Returns None for an unknown component.
        """

        if signal == decomposition.trend:
            dm = self.time.copy()
            return dm[None, :]

        if signal == decomposition.annual:
            dm = [np.cos((2 * np.pi) * self.time), np.sin((2 * np.pi) * self.time)]
            return np.array(dm)

        if signal == decomposition.semi_annual:
            dm = [np.cos((2 * np.pi / 0.5) * self.time), np.sin((2 * np.pi / 0.5) * self.time)]
            return np.array(dm)

        if signal == decomposition.quarter_annual:
            dm = [np.cos((2 * np.pi / 0.25) * self.time), np.sin((2 * np.pi / 0.25) * self.time)]
            return np.array(dm)

        return None

    # ------------------------------------------------------------------------------------
    def _solve(self, obs):
        """
        Least-squares core shared by getSignal and getError.

        obs : (n_time, n_series) or 1-D (n_time,). NaNs are handled PER EPOCH AND PER SERIES:
              every series is fitted on its own valid epochs, so a missing month (a GRACE gap,
              typically missing in every basin at once) only removes that epoch from the fit
              instead of discarding whole series. Series sharing the same gap pattern are
              solved together in one lstsq call. A series with fewer valid epochs than
              parameters is returned as NaN.

        Returns x (n_par, n_series), cov (n_par, n_par, n_series), sigma0 (n_series,),
        n_valid (n_series,). The result is cached so getError() can be called without
        re-solving after getSignal().
        """
        obs_m = np.array(obs, dtype=float)
        if obs_m.ndim == 1:                      # a single series -> one column
            obs_m = obs_m[:, None]
        n_time, n_series = obs_m.shape
        DM = self.__DM
        n_par = DM.shape[1]
        assert n_time == DM.shape[0], "obs has %d epochs but the time axis has %d" % (n_time, DM.shape[0])

        x = np.full((n_par, n_series), np.nan)          # solution
        cov = np.full((n_par, n_par, n_series), np.nan)  # covariance of the solution
        sigma0 = np.full(n_series, np.nan)              # a-posteriori std of one observation
        n_valid = np.zeros(n_series, dtype=int)

        valid = ~np.isnan(obs_m)                                        # (n_time, n_series)
        patterns, which = np.unique(valid, axis=1, return_inverse=True) # distinct gap patterns
        which = np.asarray(which).ravel()
        for g in range(patterns.shape[1]):
            rows = patterns[:, g]                # epochs available for this group of series
            cols = np.where(which == g)[0]       # the series sharing this pattern
            n_ok = int(rows.sum())
            n_valid[cols] = n_ok
            if n_ok < n_par:                     # under-determined: leave NaN
                continue
            A = DM[rows]                         # (n_ok, n_par)
            B = obs_m[rows][:, cols]             # (n_ok, n_cols)
            xg = np.linalg.lstsq(a=A, b=B, rcond=-1)[0]
            x[:, cols] = xg
            dof = n_ok - n_par
            if dof > 0:
                resid = B - A @ xg
                s2 = np.sum(resid ** 2, axis=0) / dof                  # per series
                N_inv = np.linalg.pinv(A.T @ A)                        # (DM^T DM)^-1 on valid rows
                cov[:, :, cols] = N_inv[:, :, None] * s2[None, None, :]
                sigma0[cols] = np.sqrt(s2)

        self.__last = (x, cov, sigma0, n_valid)
        return self.__last

    @staticmethod
    def _harmonic_k(ss):
        """cycles per year of a harmonic component: 1 (annual), 2 (semi-annual), 4 (quarter-annual)"""
        return {decomposition.annual: 1, decomposition.semi_annual: 2, decomposition.quarter_annual: 4}[ss]

    def getSignal(self, obs):
        """
        obs is the time series of data to be analyzed.

        obs : array-like of shape (n_time, n_series) - rows are epochs (same order as the
              time axis given to set_period), columns are independent series (e.g. one per
              sub-basin or grid cell). A single series may also be given as a 1-D array.
              NaNs (missing epochs) are allowed - see `_solve`.

        Returns a dict. Keys and shapes present since the first version (unchanged):
            'bias'        : (n_series,)    fitted value at the first epoch (t = 0)
            'trend'       : (n_series,)    slope in data-units per year        [if requested]
            '<harmonic>'  : (2, n_series)  row 0 amplitude A, row 1 phase phi [deg, 0..360)
                            phase referenced to the FIRST EPOCH, convention
                            a cos(w t) + b sin(w t) == A cos(w t + phi),  phi = atan2(-b, a)
        Added:
            '<harmonic>_calendar' : (2, n_series)  same amplitude, phase referenced to
                            1 JANUARY (calendar), i.e. y = A cos(2 pi k (T - year) + phi_cal)
                            with T the absolute year fraction and k = 1, 2, 4. Comparable
                            between series/runs starting in different months.
            '<harmonic>_peak_doy' : (n_series,)   day of year (1 Jan = 1, fractional) of the
                            first maximum of that harmonic, derived from the calendar phase:
                            DOY = 1 + 365.25 * ((-phi_cal/360) mod 1) / k. For the annual term
                            this is THE peak day; for the semi-annual / quarter-annual terms
                            further maxima follow every 365.25/k days (182.6 / 91.3 days).
        Formal errors are NOT part of this result: call `getError()` afterwards.
        """
        x, _, _, _ = self._solve(obs)

        # unpack in the same column order the design matrix was built in
        res = {'bias': x[0]}
        i = 1                                    # next unread row of x
        for ss in decomposition:                 # enum order == design-matrix order
            if ss not in self.__signal:
                continue
            if ss == decomposition.trend:
                res[ss.name] = x[i]              # slope per year, one value per series
                i += 1
                continue

            # harmonic: rows i (cosine coefficient a) and i+1 (sine coefficient b)
            a, b = x[i], x[i + 1]
            A = np.sqrt(a ** 2 + b ** 2)                              # amplitude
            with np.errstate(invalid='ignore', divide='ignore'):
                phi_1 = np.arccos(a / A)                              # candidate in [0, pi]   (cos phi = a/A)
                phi_2 = 2 * np.pi - phi_1                             # candidate in [pi, 2pi]
                phi = phi_1.copy()
                # choose the quadrant from the sign of sin(phi) = -b/A: negative -> phi in (pi, 2pi)
                sel = (-b / A) < 0
                phi[sel] = phi_2[sel]
            res[ss.name] = np.array([A, np.rad2deg(phi)])            # (2, n_series): amplitude, phase [deg]

            # calendar-referenced phase: y = A cos(w (T - T0) + phi) = A cos(w T + (phi - w T0)),
            # w = 2 pi k with k = 1, 2, 4 for periods 1, 1/2, 1/4 yr -> phi_cal = phi - 2 pi k T0
            k = self._harmonic_k(ss)
            phi_cal_deg = np.mod(np.rad2deg(phi) - 360.0 * k * self.time0, 360.0)
            res[ss.name + '_calendar'] = np.array([A, phi_cal_deg])

            # day of year of the (first) maximum: A cos(2 pi k f + phi_cal) peaks where
            # 2 pi k f + phi_cal = 0 (mod 2 pi)  ->  f = (-phi_cal/360 mod 1) / k  [fraction of year]
            # 1-based DOY (1 Jan = 1). Further maxima follow every 365.25/k days.
            f_peak = np.mod(-phi_cal_deg / 360.0, 1.0) / k
            res[ss.name + '_peak_doy'] = 1.0 + f_peak * 365.25

            i += 2

        return res

    def getError(self, obs=None):
        """
        Formal (1-sigma) errors of the quantities returned by getSignal.

        obs : same array as passed to getSignal, or None to reuse the solution of the most
              recent getSignal / getError call without solving again.

        Returns a dict:
            'bias_std', 'trend_std' : (n_series,)   formal 1-sigma errors
            '<harmonic>_std'        : (2, n_series) formal 1-sigma errors of amplitude A and
                                      of the phase phi [deg]; the phase error applies equally
                                      to the raw and the calendar-referenced phase
            'residual_std'          : (n_series,)   a-posteriori std of one observation,
                                      sqrt(RSS / (n_valid - n_par))
            'n_valid'               : (n_series,)   number of epochs used per series
        Method: sigma0^2 = RSS/dof per series, Cov = sigma0^2 (DM^T DM)^-1 on the valid
        epochs, then linear propagation to (A, phi):
            A   = sqrt(a^2 + b^2)  ->  dA/da = a/A,      dA/db = b/A
            phi = atan2(-b, a)     ->  dphi/da = b/A^2,  dphi/db = -a/A^2
        Assumes uncorrelated, equal-variance residuals. Monthly TWS residuals are usually
        autocorrelated, so these errors are optimistic (lower bounds); an effective-sample-
        size correction is needed before quoting them as confidence intervals.
        """
        if obs is None:
            assert getattr(self, '_ts__last', None) is not None, "call getSignal(obs) first or pass obs"
            x, cov, sigma0, n_valid = self.__last
        else:
            x, cov, sigma0, n_valid = self._solve(obs)

        err = {'bias_std': np.sqrt(cov[0, 0]), 'residual_std': sigma0, 'n_valid': n_valid}
        i = 1
        for ss in decomposition:
            if ss not in self.__signal:
                continue
            if ss == decomposition.trend:
                err[ss.name + '_std'] = np.sqrt(cov[i, i])
                i += 1
                continue
            a, b = x[i], x[i + 1]
            A2 = a ** 2 + b ** 2
            saa, sbb, sab = cov[i, i], cov[i + 1, i + 1], cov[i, i + 1]
            with np.errstate(invalid='ignore', divide='ignore'):
                var_A = (a ** 2 * saa + b ** 2 * sbb + 2 * a * b * sab) / A2
                var_phi = (b ** 2 * saa + a ** 2 * sbb - 2 * a * b * sab) / A2 ** 2
            err[ss.name + '_std'] = np.array([np.sqrt(var_A), np.rad2deg(np.sqrt(var_phi))])
            i += 2

        return err


def demo1():
    """minimal example: build the design matrix for 5 epochs (nothing is fitted)"""
    tsd = ts().set_period(time=np.array([2002.1, 2002.2, 2002.3, 2002.4, 2002.5]))

    tsd.setDecomposition()

    pass


def demo2():
    """
    real use: monthly basin-mean TWS of a DA run stored in an HDF5 file with one dataset per
    month ('YYYY-MM' -> values per sub-basin). Each month is placed at its 15th day, turned
    into a year fraction, and trend + annual cycle are fitted for every sub-basin at once.
    """
    from datetime import datetime
    from src_auxiliary.GeoMathKit import GeoMathKit
    import h5py
    hf = h5py.File('/home/user/Desktop/res/monthly_mean_TWS_DRB_DA.h5', 'r')

    time = []
    vv = []

    for x, v in hf.items():
        m = x.split('-')
        n = datetime(year=int(m[0]), month=int(m[1]), day=15)    # mid-month epoch
        time.append(GeoMathKit.year_fraction(n))
        vv.append(v[:])                                          # -> obs rows = months, cols = sub-basins
        pass

    tsd = ts().set_period(time=np.array(time)).setDecomposition()
    res = tsd.getSignal(obs=vv)
    pass


if __name__ == '__main__':
    # demo1()
    demo2()
