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

    def basin_ensemble(self, allow_pop_up: bool = True, fig_path=None):
        # Implement visualization logic here
        import pygmt

        cg = self._config

        ens_size = cg.basic.ensemble

        bp = BasinAverageAnalysis_post(ens=cg.basic.ensemble, case=cg.basic.case, basin=cg.basic.basin,
                                       date_begin='2000-01-01',
                                       date_end='2000-01-31')

        stage = Stage.DA
        states = bp.load_states(load_dir=Path(cg.basic.res_permanent)/cg.basic.case, prefix=stage.name)

        fan_time = states['time']

        '''plot figure'''
        fig = pygmt.Figure()
        i = 0
        for state in WaterGap_storage_variables:
            i += 1

            vv = states['basin'][state.name][0]

            vmin, vmax = np.min(vv[20:]), np.max(vv[20:])
            dmin = vmin - (vmax - vmin) * 0.1
            dmax = vmax + (vmax - vmin) * 0.1
            if dmax <10:
                continue
            sp_1 = int(np.round((vmax - vmin) / 10))
            if sp_1 == 0:
                sp_1 = 0.5
            sp_2 = sp_1 * 2

            if i == 6:
                fig.shift_origin(yshift='22c', xshift='14c')
                pass
            pygmt.config(FONT_TITLE="19p,5", MAP_TITLE_OFFSET="-0.2p", MAP_FRAME_TYPE="plain",
                         FONT_ANNOT_PRIMARY='11p,5', FONT_LABEL='11p,5', MAP_TICK_LENGTH='7p')

            if len(fan_time) > 720:
                fig.basemap(region=[fan_time[0] - 0.2, fan_time[-1] + 0.2, dmin, dmax], projection='X12c/3c',
                            frame=["WSne+t%s" % state.name, "xa2f1", 'ya%df%d+lwater [mm]' % (sp_2, sp_1)])
            else:
                fig.basemap(region=[fan_time[0] - 0.2, fan_time[-1] + 0.2, dmin, dmax], projection='X12c/3c',
                            frame=["WSne+t%s" % state.name, "xa1f0.5", 'ya%df%d+lwater [mm]' % (sp_2, sp_1)])

            mean = None
            for ens in reversed(range(ens_size + 1)):
                vv = states['basin'][state.name][ens]

                if ens == 0:
                    # fig.plot(x=fan_time, y=vv, pen="1p,blue", label='%s' % (state), transparency=30)
                    pass
                else:
                    fig.plot(x=fan_time, y=vv, pen="1p,grey")
                    if mean is None:
                        mean = vv
                    else:
                        mean += vv

            fig.plot(x=fan_time, y=mean / ens_size, pen="1.5p,blue", label='Mean', transparency=30)
            fig.legend(position='jTR', box='+gwhite+p0.5p')
            fig.shift_origin(yshift='-4.4c')

        fig_path = Path(fig_path)/'figures'
        Path(fig_path).mkdir(parents=False, exist_ok=True)
        fig.savefig(str(Path(fig_path) / 'Components.pdf'))
        fig.savefig(str(Path(fig_path) / 'Components.png'))
        if allow_pop_up:
            fig.show()
        pass


    def GRACE_OL_DA(self, allow_pop_up: bool = True, fig_path=None, signal=WaterGap_storage_variables.tws.name):
        # Implement visualization logic here
        import pygmt


        cg = self._config
        basin = cg.basic.basin

        bp = BasinAverageAnalysis_post(ens=cg.basic.ensemble, case=cg.basic.case, basin=cg.basic.basin,
                                       date_begin='2000-01-01',
                                       date_end='2000-01-31')

        stage = Stage.OL
        states_OL = bp.load_states(load_dir=Path(cg.basic.res_permanent)/cg.basic.case, prefix=stage.name)

        stage = Stage.DA
        states_DA = bp.load_states(load_dir=Path(cg.basic.res_permanent) / cg.basic.case, prefix=stage.name)

        '''load GRACE'''
        # dp_dir = Path(setting_dir) / 'DA_setting.json'
        # dp4 = json.load(open(dp_dir, 'r'))
        # GRACE = pp.get_GRACE(obs_dir=dp4['obs']['dir'])
        GRACE = bp.load_GRACE(prefix=cg.basic.basin, load_dir=Path(cg.basic.res_permanent) / cg.basic.case)

        OL_time = states_OL['time']
        DA_time = states_DA['time']
        GR_time = GRACE['time']

        '''plot figure'''
        fig = pygmt.Figure()

        '''plot figure'''
        fig = pygmt.Figure()
        pygmt.config(FONT_TITLE="17p,5", MAP_TITLE_OFFSET="0p", MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY='10p,5',
                     FONT_LABEL='10p,5', MAP_TICK_LENGTH='7p')

        i = 0
        offset = 12
        height = 4.6

        keys = list(GRACE['ens_mean'].keys())

        for j in range(len(keys)):
            i += 1
            basin_id = 'basin_%s' % j
            GRACE_ens_mean = GRACE['ens_mean'][basin_id]
            GRACE_original = GRACE['original'][basin_id]

            if basin_id == 'basin_0':
                basin_id='basin'
            else:
                basin_id = 'sub_'+basin_id
            OL = states_OL[basin_id][signal][0]
            OL_ens_mean = np.mean(np.array(list(states_OL[basin_id][signal].values()))[1:, ], axis=0)
            DA_ens_mean = np.mean(np.array(list(states_DA[basin_id][signal].values()))[1:, ], axis=0)

            values = [GRACE_ens_mean, OL, OL_ens_mean, DA_ens_mean]
            vvmin = []
            vvmax = []
            for vv in values:
                vvmin.append(np.min(vv[5:]))
                vvmax.append(np.max(vv[5:]))

            vmin, vmax = min(vvmin), min(vvmax)
            dmin = vmin - (vmax - vmin) * 0.1
            dmax = vmax + (vmax - vmin) * 0.15
            sp_1 = int(np.round((vmax - vmin) / 10))
            if sp_1 == 0:
                sp_1 = 0.5
            sp_2 = sp_1 * 2

            if len(OL_time) > 365 * 4:
                fig.basemap(region=[OL_time[0] - 0.2, OL_time[-1] + 0.2, dmin, dmax], projection='X12c/3c',
                            frame=["WSne+t%s" % (basin + '_' + signal + '_' + basin_id), "xa2f1g",
                                   'ya%df%dg+lwater [mm]' % (sp_2, sp_1)])
            else:
                fig.basemap(region=[OL_time[0] - 0.2, OL_time[-1] + 0.2, dmin, dmax], projection='X12c/3c',
                            frame=["WSne+t%s" % (basin + '_' + signal + '_' + basin_id), "xa1f0.5g",
                                   'ya%df%dg+lwater [mm]' % (sp_2, sp_1)])

            # fig.plot(x=OL_time, y=OL, pen="0.5p,blue,-", label='%s' % ('OL_unperturbed'), transparency=30)
            fig.plot(x=OL_time, y=OL_ens_mean, pen="1.0p,blue,-", label='%s' % ('OL'), transparency=30)
            # fig.plot(x=OL_time, y=OL, pen="0.5p,grey,-", label='%s' % ('OL'), transparency=30)
            # fig.plot(x=OL_time, y=OL_ens_mean, pen="0.5p,red", label='%s' % ('OL_ens_mean'), transparency=30)
            fig.plot(x=DA_time, y=DA_ens_mean, pen="1.0p,green", label='%s' % ('DA'), transparency=30)
            # fig.plot(x=GR_time, y=GRACE_ens_mean, pen="0.5p,black", label='%s' % ('GRACE_ens_mean'), transparency=30)
            # fig.plot(x=GR_time, y=GRACE_original, pen="0.5p,purple,--.", label='%s' % ('GRACE_original'),
            #          transparency=30)
            fig.plot(x=GR_time, y=GRACE_original, style="c.2c", fill="black", label='%s' % ('GRACE'), transparency=30)

            # fig.legend(position='jTR', box='+gwhite+p0.5p')
            fig.legend(position='jBL')

            sf = '%sc' % ((offset - 1) * height)
            if i % offset == 0:
                fig.shift_origin(yshift=sf, xshift='14c')
                continue

            fig.shift_origin(yshift='-%sc' % height)

            pass

        # fig_postfix = '0'
        fig_path = Path(fig_path) / 'figures'
        fig.savefig(str(fig_path / ('DA_%s.pdf' % signal)))
        fig.savefig(str(fig_path / ('DA_%s.png' % signal)))

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
