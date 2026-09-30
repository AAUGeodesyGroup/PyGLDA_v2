from src_DA.configure_DA import config_DA
from src_postprocessing.statistical_analysis import BasinAverageAnalysis_post
from src_DA.EnumDA import WaterGap_storage_variables, Stage
from pathlib import Path
import numpy as np
import xarray as xr



class visualization:

    def __init__(self, configDA:config_DA):
        self._config = configDA
        pass

    @staticmethod
    def _nice_step(span: float) -> float:
        """1-2-5 rounded tick step for an axis span"""
        if span <= 0:
            return 1.0
        e = 10 ** np.floor(np.log10(span))
        m = span / e
        return float(e * (1 if m < 1.5 else 2 if m < 3.5 else 5 if m < 7.5 else 10))

    @staticmethod
    def _shared_legend(fig, cur_x: float, cur_y: float, width: float, y_offset: float = 1.3, box: str = '+gwhite+p0.5p'):
        """
        Draw the legend collected from the labelled plot calls once, above the top-left panel (origin 0/0),
        spanning `width` cm, and return the origin to where it was. The number of legend columns is set by the
        '+N<n>' modifier attached to the first label.
        """
        if (cur_x, cur_y) != (0.0, 0.0):
            fig.shift_origin(xshift='%gc' % (-cur_x), yshift='%gc' % (-cur_y))
        fig.legend(position='JTL+jBL+o0c/%gc+w%gc' % (y_offset, width), box=box)
        if (cur_x, cur_y) != (0.0, 0.0):
            fig.shift_origin(xshift='%gc' % cur_x, yshift='%gc' % cur_y)

    def basin_ensemble(self, allow_pop_up: bool = True, fig_path=None, stage: Stage = Stage.DA, variables=None,
                       basin_id: str = 'basin', ncol: int = 2, min_range: float = 1.0, show_unperturbed: bool = False,
                       skip_start_days: int = 20, panel_width: float = 12, panel_height: float = 3,
                       row_gap: float = 1.4, col_gap: float = 2.5, fig_name=None):
        """
        Ensemble time series of the water storage compartments (basin average) for one stage.

        stage            : Stage.OL or Stage.DA (results read from Res/<case>/Res_<stage>.h5)
        variables        : list of storage names to plot (default: every WaterGap_storage_variables member)
        basin_id         : 'basin' (whole basin) or 'sub_basin_k'
        ncol             : number of panel columns; panels are filled column by column, balanced automatically
        min_range        : a variable whose plotted range (max-min over all drawn series) is below this value [mm]
                           is skipped (e.g. constant global lakes); set to 0 to draw everything
        show_unperturbed : also draw member 0 (unperturbed run) as a dashed black line
        skip_start_days  : days ignored at the beginning when computing the y-range (initial transient)
        The y-range is derived from all drawn series (members, mean, optional unperturbed), not from member 0.
        """
        import pygmt

        cg = self._config
        ens_size = cg.basic.ensemble

        bp = BasinAverageAnalysis_post(ens=ens_size, case=cg.basic.case, basin=cg.basic.basin,
                                       date_begin='2000-01-01', date_end='2000-01-31')
        states = bp.load_states(load_dir=Path(cg.basic.res_permanent) / cg.basic.case, prefix=stage.name)
        fan_time = states['time']
        series_of = states[basin_id]

        if variables is None:
            variables = [v.name for v in WaterGap_storage_variables]
        variables = [v for v in variables if v in series_of]

        """decide which panels are drawn and their y-ranges before drawing, so that the layout is balanced"""
        panels = []
        for var in variables:
            members = [np.asarray(series_of[var][e], dtype=float) for e in range(1, ens_size + 1)]
            mean = np.mean(members, axis=0)
            drawn = members + [mean] + ([np.asarray(series_of[var][0], dtype=float)] if show_unperturbed else [])
            k0 = min(skip_start_days, len(fan_time) - 1)
            vmin = min(np.nanmin(d[k0:]) for d in drawn)
            vmax = max(np.nanmax(d[k0:]) for d in drawn)
            if (vmax - vmin) < min_range:
                print('basin_ensemble: %s skipped (range %.2f mm < %.2f mm)' % (var, vmax - vmin, min_range))
                continue
            panels.append(dict(var=var, members=members, mean=mean, vmin=vmin, vmax=vmax))

        if not panels:
            print('basin_ensemble: nothing to plot')
            return

        n = len(panels)
        nrow = int(np.ceil(n / ncol))
        x_step, y_step = panel_width + col_gap, panel_height + row_gap
        cur_x, cur_y = 0.0, 0.0

        fig = pygmt.Figure()
        pygmt.config(FONT_TITLE="19p,5", MAP_TITLE_OFFSET="-0.2p", MAP_FRAME_TYPE="plain",
                     FONT_ANNOT_PRIMARY='11p,5', FONT_LABEL='11p,5', MAP_TICK_LENGTH='7p')
        x_frame = "xa2f1" if len(fan_time) > 720 else "xa1f0.5"

        for j, p in enumerate(panels):
            col, row = j // nrow, j % nrow
            tx, ty = col * x_step, -row * y_step
            if (tx, ty) != (cur_x, cur_y):
                fig.shift_origin(xshift='%gc' % (tx - cur_x), yshift='%gc' % (ty - cur_y))
                cur_x, cur_y = tx, ty

            span = p['vmax'] - p['vmin']
            dmin, dmax = p['vmin'] - 0.10 * span, p['vmax'] + 0.10 * span
            sp_2 = self._nice_step((dmax - dmin) / 4.0)
            sp_1 = sp_2 / 2.0
            fig.basemap(region=[fan_time[0] - 0.2, fan_time[-1] + 0.2, dmin, dmax],
                        projection='X%gc/%gc' % (panel_width, panel_height),
                        frame=["WSne+t%s" % p['var'], x_frame, 'ya%gf%g+lwater [mm]' % (sp_2, sp_1)])
            first = (j == 0)
            n_items = 3 if show_unperturbed else 2
            for k, vv in enumerate(p['members']):
                fig.plot(x=fan_time, y=vv, pen="0.8p,grey", label='members+N%d' % n_items if (first and k == 0) else None)
            if show_unperturbed:
                fig.plot(x=fan_time, y=series_of[p['var']][0], pen="1p,black,-", label='unperturbed' if first else None)
            fig.plot(x=fan_time, y=p['mean'], pen="1.5p,blue", label='ensemble mean' if first else None, transparency=30)

        '''one shared legend above the whole figure (anchored to the top-left panel, spanning all columns)'''
        self._shared_legend(fig, cur_x, cur_y, width=ncol * x_step - col_gap)

        fig_path = Path(fig_path) / 'figures'
        fig_path.mkdir(parents=True, exist_ok=True)
        name = fig_name or ('Components_%s' % stage.name if basin_id == 'basin' else 'Components_%s_%s' % (stage.name, basin_id))
        fig.savefig(str(fig_path / (name + '.pdf')))
        fig.savefig(str(fig_path / (name + '.png')))
        if allow_pop_up:
            fig.show()
        pass

    def DA_evaluation(self, fig_path=None, basin_id: int = 0, zoom=None, allow_pop_up: bool = False,
                      signal=WaterGap_storage_variables.tws.name, save_stats: bool = True):
        """
        Evaluation of the assimilation against the (unperturbed) GRACE observations, on the observation windows:
            panel 1: open loop vs analysis (ensemble means of the perturbed members) vs GRACE
            panel 2: innovation (obs - open loop) and residual (obs - analysis) with the observation +/- sigma
            panel 3: ensemble spread of open loop and analysis vs observation sigma
        The model is averaged over each observation window ('duration' in <basin>_obs_GRACE.hdf5), so the
        comparison is exactly what the filter sees. Works for monthly and 5-daily products alike.

        basin_id : 0 = whole basin (area-weighted sub-basins), k = sub-basin k
        zoom     : optional (date_begin, date_end) strings -> additional daily zoom figure with all members
        Statistics (RMS, correlation, spread, sigma, chi-square) for all sub-basins are written to
        figures/DA_eval_stats.json when save_stats is True.
        """
        import json
        import h5py
        import datetime as dt
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        cg = self._config
        ens_size = cg.basic.ensemble
        style = {'font.size': 11, 'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
                 'axes.linewidth': 1.8, 'xtick.major.width': 1.6, 'ytick.major.width': 1.6, 'legend.fontsize': 10}
        C_OL, C_DA, C_OB = '#1f77b4', '#d62728', '#222222'

        """model: daily basin time series of both stages"""
        bp = BasinAverageAnalysis_post(ens=ens_size, case=cg.basic.case, basin=cg.basic.basin,
                                       date_begin='2000-01-01', date_end='2000-01-31')
        res_dir = Path(cg.basic.res_permanent) / cg.basic.case
        st = {s: bp.load_states(load_dir=res_dir, prefix=s.name) for s in (Stage.OL, Stage.DA)}
        nday = min(len(st[Stage.OL]['time']), len(st[Stage.DA]['time']))
        t0 = dt.datetime.strptime(cg.basic.fromdate, '%Y-%m-%d').date()
        days = np.array([t0 + dt.timedelta(days=i) for i in range(nday)])
        day_index = {d: i for i, d in enumerate(days)}
        sub_keys = sorted([k for k in st[Stage.OL].keys() if k.startswith('sub_basin_')], key=lambda k: int(k.split('_')[-1]))
        nsub = len(sub_keys)

        def members(stage, key):                                   # (ens, nday) of the perturbed members
            return np.array([st[stage][key][signal][e][:nday] for e in range(1, ens_size + 1)], dtype=float)

        """observations: unperturbed values, windows and covariance"""
        gr = h5py.File(Path(cg.obs.dir) / ('%s_obs_GRACE.hdf5' % cg.basic.basin), 'r')
        epochs = [dt.datetime.strptime(e, '%Y-%m-%d').date() for e in gr['time_epoch'][:].astype(str)]
        Y, R, area = gr['ens_0'][:], gr['cov'][:], gr['sub_basin_area'][:]
        if 'duration' in gr:
            windows = [tuple(dt.datetime.strptime(x, '%Y-%m-%d').date() for x in d.split('_'))
                       for d in gr['duration'][:].astype(str)]
        else:                                                       # monthly product without explicit windows
            import calendar
            windows = [(dt.date(e.year, e.month, 1), dt.date(e.year, e.month, calendar.monthrange(e.year, e.month)[1]))
                       for e in epochs]
        gr.close()
        w = area / area.sum()

        keep = [i for i, (a, b) in enumerate(windows) if a in day_index and b in day_index]
        tk = np.array([epochs[i] for i in keep])

        def window_mean(X):                                         # X (ens, nday) -> (nobs, ens)
            return np.array([X[:, day_index[windows[i][0]]:day_index[windows[i][1]] + 1].mean(1) for i in keep])

        M = {s: {k: window_mean(members(s, k)) for k in sub_keys} for s in (Stage.OL, Stage.DA)}
        Yk, Rk = Y[keep], R[keep]

        def series(stage, bid):
            """ensemble-mean window means, spread, observation and sigma for basin id (0 = whole basin)"""
            if bid == 0:
                mm = sum(M[stage][k] * w[j] for j, k in enumerate(sub_keys))
                obs = Yk @ w
                sig = np.sqrt(np.einsum('i,tij,j->t', w, Rk, w))
            else:
                mm = M[stage][sub_keys[bid - 1]]
                obs = Yk[:, bid - 1]
                sig = np.sqrt(Rk[:, bid - 1, bid - 1])
            return mm.mean(1), mm.std(1, ddof=1), obs, sig

        """statistics for all basins"""
        rows = []
        for bid in range(0, nsub + 1):
            ol, s_ol, obs, sig = series(Stage.OL, bid)
            da, s_da, _, _ = series(Stage.DA, bid)
            d_ol, d_da = obs - ol, obs - da
            rows.append(dict(basin_id=bid, rms_inno_OL=float(np.sqrt(np.mean(d_ol ** 2))),
                             rms_resid_DA=float(np.sqrt(np.mean(d_da ** 2))),
                             corr_OL=float(np.corrcoef(obs, ol)[0, 1]), corr_DA=float(np.corrcoef(obs, da)[0, 1]),
                             spread_OL=float(s_ol.mean()), spread_DA=float(s_da.mean()), sigma=float(sig.mean()),
                             chi2_DA=float(np.mean(d_da ** 2 / (s_da ** 2 + sig ** 2)))))
        fig_dir = Path(fig_path) / 'figures'
        fig_dir.mkdir(parents=True, exist_ok=True)
        if save_stats:
            json.dump(rows, open(fig_dir / 'DA_eval_stats.json', 'w'), indent=1)
        r = rows[basin_id]
        label = 'entire %s' % cg.basic.basin if basin_id == 0 else '%s sub-basin %d' % (cg.basic.basin, basin_id)

        """figure"""
        ol, s_ol, obs, sig = series(Stage.OL, basin_id)
        da, s_da, _, _ = series(Stage.DA, basin_id)
        datum = ol.mean()
        x = np.array(tk, dtype='datetime64[D]')
        with plt.rc_context(style):
            fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            ax = axs[0]
            ax.plot(x, ol - datum, color=C_OL, lw=1.3, label='OL, ensemble mean (window means)')
            ax.plot(x, da - datum, color=C_DA, lw=1.3, label='DA, ensemble mean (window means)')
            ax.plot(x, obs - datum, '.', color=C_OB, ms=4, label='GRACE observation')
            ax.set_ylabel('TWS anomaly [mm]')
            ax.set_title('%s: OL vs DA vs GRACE (%d members)' % (label, ens_size), loc='left', pad=28)
            ax = axs[1]
            ax.plot(x, obs - ol, '.', color=C_OL, ms=4, label='obs - OL (RMS %.0f mm, r = %.2f)' % (r['rms_inno_OL'], r['corr_OL']))
            ax.plot(x, obs - da, '.', color=C_DA, ms=4, label='obs - DA (RMS %.0f mm, r = %.2f)' % (r['rms_resid_DA'], r['corr_DA']))
            ax.fill_between(x, -sig, sig, color='0.3', alpha=.4, lw=0, label='+/- observation sigma')
            ax.axhline(0, color='0.4', lw=.8); ax.set_ylabel('mm')
            ax.set_title('Innovation / residual', loc='left', pad=28)
            ax = axs[2]
            ax.plot(x, s_ol, color=C_OL, lw=1.2, label='ensemble spread, OL')
            ax.plot(x, s_da, color=C_DA, lw=1.2, label='ensemble spread, DA')
            ax.plot(x, sig, color='0.3', lw=1.2, label='observation sigma')
            ax.set_yscale('log'); ax.set_ylabel('mm')
            ax.set_title('Spread (chi-square of the DA residuals: %.2f)' % r['chi2_DA'], loc='left', pad=28)
            for ax in axs:
                ax.grid(lw=.4, alpha=.5)
                ax.legend(frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0))
            axs[-1].xaxis.set_major_locator(mdates.YearLocator(1 if nday < 6000 else 2))
            axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            axs[-1].set_xlim(x[0], x[-1])
            fig.tight_layout()
            name = 'DA_eval_%s' % ('basin' if basin_id == 0 else 'sub_basin_%d' % basin_id)
            for ext in ('png', 'pdf'):
                fig.savefig(fig_dir / ('%s.%s' % (name, ext)), dpi=200)
            if allow_pop_up:
                plt.show()
            plt.close(fig)

            if zoom is not None:
                za, zb = (dt.datetime.strptime(z, '%Y-%m-%d').date() for z in zoom)
                key = 'basin' if basin_id == 0 else sub_keys[basin_id - 1]
                OLd, DAd = members(Stage.OL, key), members(Stage.DA, key)
                zd = (days >= za) & (days <= zb); zo = (tk >= za) & (tk <= zb)
                xd = np.array(days[zd], dtype='datetime64[D]')
                fig, ax = plt.subplots(figsize=(12, 4))
                for m in range(ens_size):
                    ax.plot(xd, DAd[m, zd] - datum, color=C_DA, lw=.6, alpha=.5)
                ax.plot(xd, OLd[:, zd].mean(0) - datum, color=C_OL, lw=1.5, label='OL mean')
                ax.plot(xd, DAd[:, zd].mean(0) - datum, color=C_DA, lw=1.5, label='DA mean (thin: members)')
                ax.plot(x[zo], obs[zo] - datum, '.', color=C_OB, ms=6, label='GRACE')
                ax.set_ylabel('TWS anomaly [mm]'); ax.grid(lw=.4, alpha=.5)
                ax.legend(frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0))
                ax.set_title('%s, %s to %s: daily DA and updates' % (label, zoom[0], zoom[1]), loc='left', pad=28)
                ax.xaxis.set_major_locator(mdates.MonthLocator([1, 4, 7, 10]))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                fig.tight_layout()
                for ext in ('png', 'pdf'):
                    fig.savefig(fig_dir / ('%s_zoom.%s' % (name, ext)), dpi=200)
                plt.close(fig)
        return rows

    def GRACE_OL_DA(self, allow_pop_up: bool = True, fig_path=None, signal=WaterGap_storage_variables.tws.name,
                    ncol: int = 2, basins=None, show_unperturbed: bool = False, obs_marker_size=None,
                    skip_start_days: int = 5, panel_width: float = 12, panel_height: float = 3,
                    row_gap: float = 1.6, col_gap: float = 2.5, fig_name=None):
        """
        Basin-average time series of one storage (default TWS) for the whole basin and every sub-basin:
        open loop (ensemble mean, blue), analysis (ensemble mean, green) and the unperturbed GRACE observation
        (black dots). One panel per basin, filled column by column into `ncol` balanced columns.

        basins           : list of basin ids to plot (0 = whole basin, k = sub-basin k); default: all
        show_unperturbed : also draw the unperturbed open-loop member (member 0) as a dashed grey line
        obs_marker_size  : GMT size of the GRACE symbols, e.g. '0.2c'; default chooses 0.2c for monthly products
                           and 0.08c for products with more than 300 epochs, so that dense 5-daily data do not
                           hide the model curves
        The y-range is derived from all drawn series (after skip_start_days), with head room for the legend.
        """
        import pygmt

        cg = self._config
        basin = cg.basic.basin

        bp = BasinAverageAnalysis_post(ens=cg.basic.ensemble, case=cg.basic.case, basin=cg.basic.basin,
                                       date_begin='2000-01-01', date_end='2000-01-31')
        res_dir = Path(cg.basic.res_permanent) / cg.basic.case
        states_OL = bp.load_states(load_dir=res_dir, prefix=Stage.OL.name)
        states_DA = bp.load_states(load_dir=res_dir, prefix=Stage.DA.name)
        GRACE = bp.load_GRACE(prefix=cg.basic.basin, load_dir=res_dir)

        OL_time, DA_time, GR_time = states_OL['time'], states_DA['time'], GRACE['time']
        n_obs = len(GR_time)
        if obs_marker_size is None:
            obs_marker_size = '0.2c' if n_obs <= 300 else '0.08c'

        n_basins = len(GRACE['original'])                       # basin_0 .. basin_{n-1}
        if basins is None:
            basins = list(range(n_basins))

        def model_key(bid):
            return 'basin' if bid == 0 else 'sub_basin_%d' % bid

        def ens_mean(states, key):
            return np.mean(np.array([states[key][signal][e] for e in range(1, cg.basic.ensemble + 1)]), axis=0)

        k0 = min(skip_start_days, len(OL_time) - 1)
        panels = []
        for bid in basins:
            key = model_key(bid)
            OL_m, DA_m = ens_mean(states_OL, key), ens_mean(states_DA, key)
            obs = GRACE['original']['basin_%d' % bid]
            drawn = [OL_m[k0:], DA_m[k0:], obs]
            if show_unperturbed:
                drawn.append(np.asarray(states_OL[key][signal][0])[k0:])
            vmin = min(np.nanmin(d) for d in drawn)
            vmax = max(np.nanmax(d) for d in drawn)                # (the previous version used min() here)
            panels.append(dict(bid=bid, key=key, OL=OL_m, DA=DA_m, obs=obs, vmin=vmin, vmax=vmax))

        n = len(panels)
        nrow = int(np.ceil(n / ncol))
        x_step, y_step = panel_width + col_gap, panel_height + row_gap
        cur_x, cur_y = 0.0, 0.0
        x_frame = "xa2f1g" if len(OL_time) > 365 * 4 else "xa1f0.5g"

        fig = pygmt.Figure()
        pygmt.config(FONT_TITLE="17p,5", MAP_TITLE_OFFSET="0p", MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY='10p,5',
                     FONT_LABEL='10p,5', MAP_TICK_LENGTH='7p')

        for j, p in enumerate(panels):
            col, row = j // nrow, j % nrow
            tx, ty = col * x_step, -row * y_step
            if (tx, ty) != (cur_x, cur_y):
                fig.shift_origin(xshift='%gc' % (tx - cur_x), yshift='%gc' % (ty - cur_y))
                cur_x, cur_y = tx, ty

            span = max(p['vmax'] - p['vmin'], 1e-6)
            dmin, dmax = p['vmin'] - 0.10 * span, p['vmax'] + 0.10 * span
            sp_2 = self._nice_step((dmax - dmin) / 4.0)
            sp_1 = sp_2 / 2.0
            title = '%s: %s (%s)' % (basin, 'whole basin' if p['bid'] == 0 else 'sub-basin %d' % p['bid'], signal)
            fig.basemap(region=[OL_time[0] - 0.2, OL_time[-1] + 0.2, dmin, dmax],
                        projection='X%gc/%gc' % (panel_width, panel_height),
                        frame=["WSne+t%s" % title, x_frame, 'ya%gf%gg+lwater [mm]' % (sp_2, sp_1)])

            # observations first (behind), model curves on top; legend entries only from the first panel
            first = (j == 0)
            n_items = 4 if show_unperturbed else 3
            fig.plot(x=GR_time, y=p['obs'], style="c%s" % obs_marker_size, fill="black",
                     label='GRACE+N%d' % n_items if first else None)
            if show_unperturbed:
                fig.plot(x=OL_time, y=states_OL[p['key']][signal][0], pen="0.8p,grey,-",
                         label='OL unperturbed' if first else None)
            fig.plot(x=OL_time, y=p['OL'], pen="1.0p,blue", label='OL (ensemble mean)' if first else None)
            fig.plot(x=DA_time, y=p['DA'], pen="1.0p,green", label='DA (ensemble mean)' if first else None)

        '''one shared legend above the whole figure (anchored to the top-left panel, spanning all columns)'''
        self._shared_legend(fig, cur_x, cur_y, width=ncol * x_step - col_gap)

        fig_path = Path(fig_path) / 'figures'
        fig_path.mkdir(parents=True, exist_ok=True)
        name = fig_name or ('DA_%s' % signal)
        fig.savefig(str(fig_path / (name + '.pdf')))
        fig.savefig(str(fig_path / (name + '.png')))
        if allow_pop_up:
            fig.show()
        pass

    def harmonic_maps(self, allow_pop_up: bool = True, fig_path=None, variable=WaterGap_storage_variables.tws.name,
                      quantities=('trend', 'annual_amp', 'annual_peak_doy'), style: str = 'smooth', fine_res: float = 0.05):
        """
        2-D maps of the per-grid-cell harmonic analysis (statistical_analysis.HarmonicMapAnalysis,
        files Res/<case>/Harmonic_{OL,DA,GRACE}.nc): one row per quantity, three columns
            OL ensemble mean | DA ensemble mean | GRACE
        Default rows: linear trend [mm/yr], annual amplitude [mm], day of year of the annual maximum.
        The three panels of a row share one colour scale and one colour bar (2nd-98th percentile over
        all three, symmetric for diverging quantities such as the trend). Other rows that are
        available: 'annual_phase', 'semi_annual_amp', 'semi_annual_peak_doy', 'residual_std'.
        Note: GRACE is only available for TWS ('bias' is not comparable, GRACE being an anomaly).

        style    : 'smooth' (default) -> the 0.5-degree field is interpolated (bicubic) to `fine_res`
                   degrees and clipped to the basin polygon of configDA.basic.basin_shp - a continuous
                   field with crisp basin edges, the PyGLDA-v1 look. Grid work is done by GMT: grdfill
                   (nearest-neighbour fill of the NaNs outside the basin) -> grdsample -nc (bicubic) ->
                   grdmask with the basin polygons.
                   'pixel' -> one coloured cell per 0.5-degree grid node, as computed.
                   In both styles the (sub-)basin outlines are drawn. The shapefile is read with
                   geopandas (re-projected to lon/lat if needed) and its rings are handed to GMT as a
                   plain multi-segment text file, so GMT never needs GDAL/PROJ for it. Rendering only;
                   the netCDF files are untouched. Output: Harmonic_maps_<variable>_<style>.pdf/png
        fine_res : grid spacing of the interpolated field, degrees (style='smooth' only).
        """
        import pygmt
        from src_postprocessing.statistical_analysis import HarmonicMapAnalysis

        if style not in ('smooth', 'pixel'):
            raise ValueError(f"style must be 'smooth' or 'pixel', got {style!r}")
        smooth = style == 'smooth'

        cg = self._config
        res_dir, case = Path(cg.basic.res_permanent), cg.basic.case
        sources = [('OL', HarmonicMapAnalysis.load(res_dir, case, Stage.OL)),
                   ('DA', HarmonicMapAnalysis.load(res_dir, case, Stage.DA))]
        grace_fn = HarmonicMapAnalysis.output_path(res_dir, case, 'GRACE')
        if grace_fn.exists() and variable == WaterGap_storage_variables.tws.name:
            sources.append(('GRACE', HarmonicMapAnalysis.load(res_dir, case, 'GRACE')))
        else:
            print(f"  [harmonic_maps] no GRACE maps for '{variable}' ({grace_fn.name} missing or variable != tws) - 2 columns")

        # map extent from the OL data (cell centres -> cell edges); all sources share the basin box
        ref = sources[0][1]
        lat, lon = ref['lat'].values, ref['lon'].values
        dlat, dlon = abs(float(lat[1] - lat[0])), abs(float(lon[1] - lon[0]))
        region = [float(lon.min()) - dlon / 2, float(lon.max()) + dlon / 2,
                  float(lat.min()) - dlat / 2, float(lat.max()) + dlat / 2]
        width = 7.0                                                       # cm per panel
        # exact panel height under the Mercator projection used below: y = ln(tan(pi/4 + lat/2))
        merc = lambda phi_deg: np.log(np.tan(np.pi / 4 + np.deg2rad(phi_deg) / 2))
        height = float(width * (merc(region[3]) - merc(region[2])) / np.deg2rad(region[1] - region[0]))
        # vertical layout per row (cm): map | x-annotations | colour bar + annotations + label | next title
        cbar_offset = 1.1                                                 # bar top below the map frame
        cbar_block = 1.6                                                  # bar (0.3) + annotations + label
        row_step = height + cbar_offset + cbar_block + 1.0                # + room for the next row's title

        def as_latlon(da):
            """grids coming back from GMT may be named x/y (or longitude/latitude): rename to lon/lat"""
            ren = {}
            for d in da.dims:
                dl = str(d).lower()
                if dl in ('x', 'longitude') and 'lon' not in da.dims:
                    ren[d] = 'lon'
                elif dl in ('y', 'latitude') and 'lat' not in da.dims:
                    ren[d] = 'lat'
            return da.rename(ren) if ren else da

        # ---- basin polygons: shapefile -> geopandas -> GMT multi-segment text file -----------------
        # (GMT reads that natively; opening the .shp inside GMT would go through GDAL/PROJ)
        import tempfile, os
        basin_shp = Path(cg.basic.basin_shp)
        poly_file = None
        if basin_shp.exists():
            import geopandas as gpd
            basins = gpd.read_file(basin_shp)
            if basins.crs is not None and not basins.crs.is_geographic:
                basins = basins.to_crs(epsg=4326)
            rings = []
            for geom in basins.geometry:
                parts = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
                for part in parts:                              # exterior rings (basins have no holes)
                    xy = np.asarray(part.exterior.coords)
                    rings.append('>\n' + '\n'.join(f'{x:.6f} {y:.6f}' for x, y in xy[:, :2]))
            fd, poly_file = tempfile.mkstemp(suffix='.txt', prefix='basin_polygons_')
            with os.fdopen(fd, 'w') as f:
                f.write('\n'.join(rings) + '\n')
        elif smooth:
            raise FileNotFoundError(f"style='smooth' needs the basin shapefile for clipping: {basin_shp}")
        else:
            print(f"  [harmonic_maps] basin shapefile not found ({basin_shp}); outlines are not drawn")

        # ---- fine clipping mask (smooth mode): GMT grdmask on the polygon file --------------------
        fine_mask = None
        if smooth:
            from pygmt.clib import Session
            from pygmt.helpers import GMTTempFile
            reg = '/'.join(f'{v:.6f}' for v in region)
            with GMTTempFile(suffix='.nc') as tmp, Session() as lib:
                # 1 inside the polygons (edges included), NaN outside; pixel registration (-r) so the
                # mask nodes coincide with the grdsample output below. Arguments as a LIST so paths
                # with spaces survive; older pygmt takes a string, then quoted.
                args = [poly_file, f'-R{reg}', f'-I{fine_res}', '-r', '-NNaN/1/1', f'-G{tmp.name}']
                try:
                    lib.call_module('grdmask', args)
                except TypeError:
                    lib.call_module('grdmask', ' '.join(f'"{a}"' if ' ' in a else a for a in args))
                fine_mask = as_latlon(xr.load_dataarray(tmp.name)).sortby('lat')

        def grdfill_nearest(da):
            """GMT grdfill, nearest-neighbour mode; keyword changed across pygmt versions
            (neighborfill since v0.15, mode='n' before)"""
            try:
                out = pygmt.grdfill(grid=da, neighborfill=True)
            except TypeError:
                out = pygmt.grdfill(grid=da, mode='n')
            return as_latlon(out)

        def smooth_field(da, period=None):
            """
            0.5-degree field (lat ascending) -> GMT grdfill (nearest-neighbour fill of the NaNs outside
            the basin, so the interpolation is not distorted at the edge) -> grdsample bicubic to the
            fine grid -> clipped with the grdmask of the basin polygon. Circular quantities (phase,
            day of year) are interpolated through sin/cos so that e.g. 360 and 5 degrees average
            correctly.
            """
            if not np.isfinite(da.values).any():
                return fine_mask * np.nan

            def to_fine(field_da):
                # Tell GMT the 0.5-degree grid is PIXEL-registered: its nodes are cell centres and its
                # domain is the cell edges (= `region`). As the default gridline registration the
                # domain would end at the outer nodes, and grdsample - which never extrapolates -
                # would shrink `region` by half a cell on each side and return a smaller grid than
                # grdmask.
                field_da = field_da.copy()
                field_da.gmt.registration = 1          # 1 = pixel
                field_da.gmt.gtype = 1                 # 1 = geographic (lon/lat)
                filled = grdfill_nearest(field_da)
                filled.gmt.registration = 1
                filled.gmt.gtype = 1
                return as_latlon(pygmt.grdsample(grid=filled, region=region, spacing=fine_res, registration='p',
                                                 interpolation='c'))

            if period is None:
                fine = to_fine(da)
            else:
                ang = 2 * np.pi * da / period
                s, c = to_fine(np.sin(ang)), to_fine(np.cos(ang))
                fine = np.mod(np.arctan2(s, c), 2 * np.pi) * period / (2 * np.pi)
                if period != 360.0:                                                # day of year is 1-based
                    fine = np.mod(fine - 1.0, period) + 1.0
            # put the interpolated field on the mask's nodes by coordinate (nearest node, tolerance of
            # half a fine cell) rather than trusting the shapes: robust to a registration mismatch
            fine = xr.DataArray(np.asarray(fine.values), dims=('lat', 'lon'),
                                coords={'lat': np.asarray(fine['lat'].values), 'lon': np.asarray(fine['lon'].values)})
            fine = fine.reindex(lat=fine_mask['lat'].values, lon=fine_mask['lon'].values,
                                method='nearest', tolerance=fine_res * 0.51)
            return fine.copy(data=fine.values * np.asarray(fine_mask.values))

        def period_of(q):
            k = 1 if q.startswith('annual') else 2 if q.startswith('semi_annual') else 4 if q.startswith('quarter_annual') else 1
            return 360.0 if q.endswith('phase') else 365.25 / k if q.endswith('peak_doy') else None

        def pick(ds, q):
            """ensemble-mean map for OL/DA, the single map for GRACE; pygmt wants ascending lat"""
            name = f'{variable}_{q}_ensmean'
            if name not in ds:
                name = f'{variable}_{q}'
            g = ds[name].sortby('lat')
            return smooth_field(g, period_of(q)) if smooth else g

        presentation = {                  # per-quantity: label, colour map, diverging (symmetric scale), fixed range
            'trend':                ('trend [mm/yr]',                'vik',    True,  None),
            'annual_amp':           ('annual amplitude [mm]',        'batlow', False, None),
            'annual_phase':         ('annual phase [deg]',           'romaO',  False, (0.0, 360.0)),
            'annual_peak_doy':      ('annual peak [day of year]',    'romaO',  False, (1.0, 365.25)),
            'semi_annual_amp':      ('semi-annual amplitude [mm]',   'batlow', False, None),
            'semi_annual_peak_doy': ('semi-annual peak [day of year]', 'romaO', False, (1.0, 183.6)),
            'residual_std':         ('residual std [mm]',            'batlow', False, None),
        }

        def limits(arrays, symmetric):
            vals = np.concatenate([np.asarray(a).ravel() for a in arrays]); vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return -1.0, 1.0
            lo, hi = np.nanpercentile(vals, [2, 98])
            if symmetric:
                m = max(abs(lo), abs(hi)); lo, hi = -m, m
            if hi <= lo:
                hi = lo + 1e-6
            return float(lo), float(hi)

        fig = pygmt.Figure()
        pygmt.config(FONT_TITLE="12p,5", MAP_TITLE_OFFSET="2p", MAP_FRAME_TYPE="plain",
                     FONT_ANNOT_PRIMARY='9p,5', FONT_LABEL='10p,5', MAP_TICK_LENGTH='4p')

        n_col = len(sources)
        gap = 1.2                                                          # cm between panels
        for r, q in enumerate(quantities):
            label, cmap, diverging, fixed = presentation.get(q, (q, 'batlow', False, None))
            grids = [(name, pick(ds, q)) for name, ds in sources]
            vmin, vmax = fixed if fixed is not None else limits([g.values for _, g in grids], diverging)
            # 100 slices between vmin and vmax; the increment makes 'continuous' meaningful
            pygmt.makecpt(cmap=cmap, series=[vmin, vmax, (vmax - vmin) / 100.0], continuous=True, background=True)

            for c, (name, g) in enumerate(grids):
                if r > 0 or c > 0:
                    fig.shift_origin(xshift=f'{width + gap}c' if c > 0 else f'-{(n_col - 1) * (width + gap)}c',
                                     yshift='0c' if c > 0 else f'-{row_step}c')
                fig.basemap(region=region, projection=f'M{width}c', frame=['WSne+t' + name, 'xa5f1', 'ya2f1'])
                fig.grdimage(grid=g, cmap=True, nan_transparent=True)
                fig.coast(shorelines='0.4p,gray30', borders='1/0.25p,gray60', resolution='i')
                if poly_file is not None:
                    fig.plot(data=poly_file, pen='0.8p,black')              # (sub-)basin outlines

            # one colour bar per row, centred under the row (origin is at the last panel of the row)
            x_centre = -(n_col - 1) * (width + gap) / 2 + width / 2
            fig.colorbar(position=f'x{x_centre}c/-{cbar_offset}c+w{min(n_col, 2) * width}c/0.3c+h+jTC',
                         frame=[f'xaf+l{label}'])

        fig_path = Path(fig_path) / 'figures'
        fig_path.mkdir(parents=False, exist_ok=True)
        fig.savefig(str(fig_path / f'Harmonic_maps_{variable}_{style}.pdf'))
        fig.savefig(str(fig_path / f'Harmonic_maps_{variable}_{style}.png'))
        if poly_file is not None and os.path.exists(poly_file):
            os.remove(poly_file)
        if allow_pop_up:
            fig.show()
        pass
