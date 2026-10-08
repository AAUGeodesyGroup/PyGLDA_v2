"""
Figures of a GLOBAL PyGLDA v2 assimilation run (PyGMT), drawn from the report of
src_postprocessing.global_evaluation (Res/<case>/evaluation_global/<tag>/), the unit shapefile and the harmonic
maps Res/<case>/Harmonic_{OL,DA,GRACE}.nc. Written into the same report folder:

    gfig1_unit_maps.png        unit maps (each unit filled with its value): RMS reduction 1 - RMS_DA / RMS_OL,
                               correlation DA - GRACE, median chi^2, trend DA - OL
    gfig2_group_series.png     area-weighted window series GRACE / OL / DA: global land and the HydroBASINS regions
    gfig3_harmonic_maps.png    per 0.5-degree cell (pixel mode): trend, annual amplitude and day of the annual peak
                               of TWS for OL | DA | GRACE (ensemble means), as Visualization.harmonic_maps
    gfig3b_harmonic_DA-OL.png  the same three quantities as DA - OL (circular difference for the day of the peak)
    gfig4_filter_health.png    distribution of the unit chi^2, innovation vs. its expected size, added inflation
                               spread by region
    gfig5_rms_vs_snr_area.png  RMS reduction of the units against their GRACE SNR and their area

Usage:  GDA.visualization(tag='pilot1')   (src_FlowControl/Global_DA.py), or
        from src_postprocessing.global_visualization import plot_global_run
Rendering only: no result file is changed. The unit polygons are handed to GMT as a multi-segment text file
(one segment per ring, -Z<value>), so GMT does not need GDAL for the shapefile.
"""
import csv, os, tempfile
from pathlib import Path
import numpy as np
import xarray as xr

PROJ = 'N0/{w}c'                   # Robinson, centred on Greenwich
REGION = [-180, 180, -60, 90]       # no Antarctica in the units
GREY_LAND = 'gray90'
REGION_NAMES = {'af': 'Africa', 'ar': 'Arctic', 'as': 'Asia', 'au': 'Australasia', 'eu': 'Europe',
                'na': 'North America', 'sa': 'South America', 'si': 'Siberia'}


# ------------------------------------------------------------------------------------------------ inputs
def _read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _num(rows, key):
    return np.array([float(r[key]) if r.get(key) not in (None, '', 'nan') else np.nan for r in rows])


def _unit_polygons(shp_path):
    """{ID: [ring (n, 2) lon/lat, ...]} of the exterior rings of the unit polygons"""
    import geopandas as gpd
    g = gpd.read_file(shp_path)
    if g.crs is not None and g.crs.to_epsg() not in (4326, None):
        g = g.to_crs(4326)
    out = {}
    for uid, geom in zip(g['ID'], g.geometry):
        parts = list(geom.geoms) if geom.geom_type == 'MultiPolygon' else [geom]
        out[int(uid)] = [np.asarray(p.exterior.coords)[:, :2] for p in parts]
    return out


def _segments_file(polys, values, tmpdir):
    """GMT multi-segment file, one segment per ring with -Z<value of its unit>; units without a value are left out"""
    fn = os.path.join(tmpdir, 'units_%d.txt' % len(os.listdir(tmpdir)))
    with open(fn, 'w') as fh:
        for uid, rings in polys.items():
            v = values.get(uid, np.nan)
            if not np.isfinite(v):
                continue
            for r in rings:
                fh.write('> -Z%.6g\n' % v)
                np.savetxt(fh, r, fmt='%.4f')
    return fn


def _nice(v):
    """1-2-5 rounded value >= v"""
    if v <= 0:
        return 1.0
    e = 10 ** np.floor(np.log10(v))
    for m in (1, 2, 2.5, 5, 10):
        if m * e >= v - 1e-12:
            return float(m * e)
    return float(10 * e)


# ------------------------------------------------------------------------------------------------ main entry
def plot_global_run(report_dir, shp_path, res_dir=None, width=11.0, dpi=200, verbose=True):
    """
    report_dir : Res/<case>/evaluation_global/<tag> (written by global_evaluation.evaluate_global_run)
    shp_path   : unit shapefile (IDs 1..n)
    res_dir    : Res/<case> with Harmonic_{OL,DA,GRACE}.nc (None: report_dir/../..)
    returns    : list of the written figures (each as .png and .pdf)
    Style as src_postprocessing.Visualization (harmonic_maps, GRACE_OL_DA).
    """
    import pygmt
    report_dir = Path(report_dir)
    res_dir = Path(res_dir) if res_dir is not None else report_dir.parent.parent
    units = _read_csv(report_dir / 'unit_metrics.csv')
    polys = _unit_polygons(shp_path)
    written = []
    with tempfile.TemporaryDirectory() as tmp, pygmt.config(FONT_TITLE='12p,5', MAP_TITLE_OFFSET='6p',
                                                             MAP_FRAME_TYPE='plain', MAP_FRAME_PEN='1.5p',
                                                             MAP_TICK_PEN_PRIMARY='1p', FONT_ANNOT_PRIMARY='9p,5',
                                                             FONT_LABEL='10p,5', MAP_TICK_LENGTH='4p',
                                                             FORMAT_GEO_MAP='ddd'):
        written.append(_fig_unit_maps(pygmt, units, polys, tmp, report_dir, width, dpi))
        written.append(_fig_group_series(pygmt, report_dir, dpi))
        written += _fig_harmonic_maps(pygmt, res_dir, report_dir, width * 0.8, dpi)
        written.append(_fig_filter_health(pygmt, units, report_dir, dpi))
        written.append(_fig_rms_vs_snr_area(pygmt, units, report_dir, dpi))
    if verbose:
        for w in written:
            print('  -> %s (and .pdf)' % w)
    return written


def _save(fig, out, name, dpi):
    fig.savefig(str(out / (name + '.pdf')))
    fn = out / (name + '.png')
    fig.savefig(str(fn), dpi=dpi)
    return fn


def _cpt(pygmt, cmap, lo, hi, reverse=False):
    """100 slices between lo and hi (as Visualization.harmonic_maps)"""
    pygmt.makecpt(cmap=cmap, series=[lo, hi, (hi - lo) / 100.0], continuous=True, background=True, reverse=reverse)


def _limits(arrays, symmetric):
    """2nd-98th percentile over all arrays, symmetric for diverging quantities (as Visualization.harmonic_maps)"""
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -1.0, 1.0
    lo, hi = np.nanpercentile(vals, [2, 98])
    if symmetric:
        m = max(abs(lo), abs(hi))
        lo, hi = -m, m
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


# ------------------------------------------------------------------------------------------------ figure 1
def _fig_unit_maps(pygmt, units, polys, tmp, out, width, dpi):
    ids = _num(units, 'ID').astype(int)
    trend_d = _num(units, 'trend_DA') - _num(units, 'trend_OL')
    chi2 = _num(units, 'chi2_median')
    panels = [                      # title, values, cmap, fixed range (None: 2-98 %, symmetric), reverse, label
        ('RMS reduction vs. GRACE', _num(units, 'rms_reduction'), 'vik', (-0.6, 0.6), True,
         '1 - RMS@-DA@- / RMS@-OL@- [-]'),
        ('Correlation DA - GRACE', _num(units, 'corr_DA'), 'batlow', (0.0, 1.0), False, 'correlation [-]'),
        ('Median @~c@~@+2@+ of the innovations', np.log10(chi2), 'vik', (-1.0, 1.0), False,
         'log@-10@- @~c@~@+2@+ [-]'),
        ('Trend DA - OL', trend_d, 'vik', None, False, 'trend [mm/yr]')]
    height = width * 0.47
    fig = pygmt.Figure()
    for k, (title, val, cmap, rng, reverse, label) in enumerate(panels):
        if k:
            fig.shift_origin(xshift='%gc' % (width + 1.2) if k % 2 else '%gc' % -(width + 1.2),
                             yshift='0c' if k % 2 else '-%gc' % (height + 4.0))
        lo, hi = rng if rng is not None else _limits([val], True)
        _cpt(pygmt, cmap, lo, hi, reverse)
        fig.basemap(region=REGION, projection=PROJ.format(w=width), frame=['WSne+t%s' % title, 'xa60f30', 'ya30f15'])
        fig.coast(land=GREY_LAND, resolution='c')
        seg = _segments_file(polys, dict(zip(ids, val)), tmp)
        fig.plot(data=seg, fill='+z', cmap=True, close=True, pen='0.1p,gray40')
        fig.coast(shorelines='0.4p,gray30', borders='1/0.25p,gray60', resolution='c')
        fig.colorbar(position='JBC+w%gc/0.3c+o0c/1.1c+h' % width, frame=['xaf+l%s' % label])
    return _save(fig, out, 'gfig1_unit_maps', dpi)


# ------------------------------------------------------------------------------------------------ figure 2
def _fig_group_series(pygmt, out, dpi):
    """as Visualization.GRACE_OL_DA: GRACE black dots, OL blue, DA green, one shared legend above the figure"""
    rows = _read_csv(out / 'group_series.csv')
    names = ['global'] + sorted({r['group'] for r in rows if r['group'].startswith('region: ')})
    pw, ph, ncol, xgap, ygap = 8.0, 4.0, 3, 2.0, 1.8
    fig = pygmt.Figure()
    t_all = np.array([np.datetime64(r['window_start']) for r in rows if r['group'] == 'global'])
    t0, t1 = t_all.min() - np.timedelta64(20, 'D'), t_all.max() + np.timedelta64(45, 'D')
    for k, name in enumerate(names):
        rr = [r for r in rows if r['group'] == name]
        t = np.array([np.datetime64(r['window_start']) + np.timedelta64(14, 'D') for r in rr])
        ser = {s: np.array([float(r[s]) for r in rr]) for s in ('GRACE', 'OL', 'DA')}
        lo = min(np.nanmin(v) for v in ser.values())
        hi = max(np.nanmax(v) for v in ser.values())
        span = max(hi - lo, 1e-6)
        col = k % ncol
        if k:
            fig.shift_origin(xshift='%gc' % (pw + xgap) if col else '%gc' % -((ncol - 1) * (pw + xgap)),
                             yshift='0c' if col else '-%gc' % (ph + ygap))
        label = 'Global land' if name == 'global' else REGION_NAMES.get(name.split(': ')[1], name.split(': ')[1])
        fig.basemap(region=[t0, t1, lo - 0.10 * span, hi + 0.10 * span], projection='X%gc/%gc' % (pw, ph),
                    frame=['WSne+t%s (TWS)' % label, 'xa1Yf3og', 'yafg+lwater [mm]'])
        first = k == 0
        fig.plot(x=t, y=ser['GRACE'], style='c0.16c', fill='black', label='GRACE+N3' if first else None)
        fig.plot(x=t, y=ser['OL'], pen='2p,blue', label='OL (ensemble mean)' if first else None)
        fig.plot(x=t, y=ser['DA'], pen='2p,green', label='DA (ensemble mean)' if first else None)
    # one shared legend above the top-left panel, spanning all columns (as Visualization._shared_legend)
    nrow = (len(names) + ncol - 1) // ncol
    cx, cy = ((len(names) - 1) % ncol) * (pw + xgap), -(nrow - 1) * (ph + ygap)
    fig.shift_origin(xshift='%gc' % -cx, yshift='%gc' % -cy)
    fig.legend(position='JTL+jBL+o0c/1.3c+w%gc' % (ncol * (pw + xgap) - xgap), box='+gwhite+p0.5p')
    return _save(fig, out, 'gfig2_group_series', dpi)


# ------------------------------------------------------------------------------------------------ figure 3
def _harmonic(res_dir, stage, q):
    fn = Path(res_dir) / ('Harmonic_%s.nc' % stage)
    if not fn.exists():
        return None
    with xr.open_dataset(fn) as ds:
        for key in ('tws_%s_ensmean' % q, 'tws_%s' % q):
            if key in ds and ds[key].ndim == 2:
                return ds[key].load()
    return None


def _pixel_grid(g):
    """0.5-degree cell values as a pixel-registered geographic grid (style='pixel' of Visualization.harmonic_maps)"""
    g = g.rename({'lon': 'x', 'lat': 'y'}) if 'lon' in g.dims else g
    g = g.sortby('y').astype('float32')
    g.gmt.registration = 1                      # pixel registration: values belong to cells, not to grid nodes
    g.gmt.gtype = 1                             # geographic
    return g


def _fig_harmonic_maps(pygmt, res_dir, out, width, dpi):
    """
    as Visualization.harmonic_maps(style='pixel'): one row per quantity, columns OL | DA | GRACE with one shared
    colour scale (2-98 %, symmetric for the trend) and one colour bar centred under the row.
    gfig3b: DA - OL of the same quantities, one row, own colour bar per panel.
    """
    presentation = {'trend': ('trend [mm/yr]', 'vik', True, None),
                    'annual_amp': ('annual amplitude [mm]', 'batlow', False, None),
                    'annual_peak_doy': ('annual peak [day of year]', 'romaO', False, (1.0, 365.25))}
    diff_label = {'trend': 'trend [mm/yr]', 'annual_amp': 'annual amplitude [mm]',
                  'annual_peak_doy': 'annual peak [days]'}
    diff_title = {'trend': 'DA - OL: trend', 'annual_amp': 'DA - OL: annual amplitude',
                  'annual_peak_doy': 'DA - OL: annual peak'}
    if _harmonic(res_dir, 'OL', 'trend') is None or _harmonic(res_dir, 'DA', 'trend') is None:
        print('  [global visualization] no Harmonic_OL/DA.nc in %s - harmonic maps skipped' % res_dir)
        return []
    height = width * 0.47
    gap, cbar_offset, cbar_block = 1.5, 1.1, 1.6
    row_step = height + cbar_offset + cbar_block + 1.0

    def panel(fig, g, title):
        fig.basemap(region=REGION, projection=PROJ.format(w=width), frame=['WSne+t%s' % title, 'xa60f30', 'ya30f15'])
        fig.grdimage(grid=_pixel_grid(g), cmap=True, nan_transparent=True, interpolation='n')
        fig.coast(shorelines='0.4p,gray30', borders='1/0.25p,gray60', resolution='c')

    '''step-1: OL | DA | GRACE'''
    fig, diffs = pygmt.Figure(), {}
    for i, q in enumerate(presentation):
        label, cmap, diverging, fixed = presentation[q]
        grids = [(s, _harmonic(res_dir, s, q)) for s in ('OL', 'DA', 'GRACE')]
        grids = [(s, g) for s, g in grids if g is not None]
        n_col = len(grids)
        vmin, vmax = fixed if fixed is not None else _limits([g.values for _, g in grids], diverging)
        ol, da = grids[0][1], grids[1][1]
        diffs[q] = (da - ol + 182.625) % 365.25 - 182.625 if q == 'annual_peak_doy' else da - ol
        _cpt(pygmt, cmap, vmin, vmax)
        for j, (name, g) in enumerate(grids):
            if i or j:
                fig.shift_origin(xshift='%gc' % (width + gap) if j else '%gc' % -((n_col - 1) * (width + gap)),
                                 yshift='0c' if j else '-%gc' % row_step)
            panel(fig, g, name)
        x_centre = -(n_col - 1) * (width + gap) / 2 + width / 2        # origin is at the last panel of the row
        fig.colorbar(position='x%gc/-%gc+w%gc/0.3c+h+jTC' % (x_centre, cbar_offset, min(n_col, 2) * width),
                     frame=['xaf+l%s' % label])
    written = [_save(fig, out, 'gfig3_harmonic_maps', dpi)]

    '''step-2: DA - OL (wider panels, so that the colour bars keep the font size of step-1)'''
    width = 1.25 * width
    fig = pygmt.Figure()
    for j, q in enumerate(presentation):
        if j:
            fig.shift_origin(xshift='%gc' % (width + gap))
        lo, hi = _limits([diffs[q].values], True)
        _cpt(pygmt, 'vik', lo, hi)
        panel(fig, diffs[q], diff_title[q])
        fig.colorbar(position='x%gc/-%gc+w%gc/0.3c+h+jTC' % (width / 2, cbar_offset, width),
                     frame=['xaf+l%s' % diff_label[q]])
    written.append(_save(fig, out, 'gfig3b_harmonic_DA-OL', dpi))
    return written


# ------------------------------------------------------------------------------------------------ figure 4
def _fig_filter_health(pygmt, units, out, dpi):
    chi2 = _num(units, 'chi2_median')
    innov = _num(units, 'innovation_rms')
    expect = np.sqrt(_num(units, 'spread_fc_median') ** 2 + _num(units, 'sigma_GRACE_median') ** 2)
    pw, ph = 7.5, 6.0
    fig = pygmt.Figure()
    lc = np.clip(np.log10(chi2[np.isfinite(chi2) & (chi2 > 0)]), -1.49, 1.49)
    counts = np.histogram(lc, bins=np.arange(-1.5, 1.51, 0.1))[0]
    fig.histogram(data=lc, region=[-1.5, 1.5, 0, _nice(counts.max() * 1.1)], projection='X%gc/%gc' % (pw, ph),
                  series=0.1,
                  fill='30/120/214', pen='0.8p,white',
                  frame=['WSne+tMedian @~c@~@+2@+ per unit', 'xa0.5f0.1+llog@-10@- @~c@~@+2@+ (0 = consistent)',
                         'yaf+lunits'])
    fig.plot(x=[0, 0], y=[0, 1e6], pen='1.5p,black,--')
    fig.shift_origin(xshift='%gc' % (pw + 2.0))
    hi = _nice(np.nanpercentile(np.concatenate([innov, expect]), 99))
    fig.basemap(region=[0, hi, 0, hi], projection='X%gc/%gc' % (pw, ph),
                frame=['WSne+tInnovation vs. its expected size',
                       'xaf+l(@~s@~@-fc@-@+2@+ + @~s@~@-GRACE@-@+2@+)@+1/2@+ [mm]', 'yaf+lRMS innovation [mm]'])
    fig.plot(x=[0, hi], y=[0, hi], pen='1.5p,black,--')
    fig.plot(x=expect, y=innov, style='c0.14c', fill='30/120/214@40')
    fig.shift_origin(xshift='%gc' % (pw + 2.0))
    alog_fn = out / 'logs' / 'adaptive_inflation_log.csv'
    if alog_fn.exists():
        reg = {int(r['ID']): (r.get('REGION') or '?').split(',')[0] for r in units}
        log = _read_csv(alog_fn)
        by = {}
        for r in log:
            by.setdefault(reg.get(int(r['sub_basin']), '?'), []).append(float(r['sigma_added_mm']))
        names = sorted(by)
        med = [float(np.median(by[n])) for n in names]
        p90 = [float(np.percentile(by[n], 90)) for n in names]
        fig.basemap(region=[-0.5, len(names) - 0.5, 0, _nice(max(p90) * 1.1)], projection='X%gc/%gc' % (pw, ph),
                    frame=['WSne+tAdded inflation spread', 'xf1', 'yaf+l@~s@~ added [mm]: median, 90th pct.'])
        for i, n in enumerate(names):
            fig.plot(x=[i, i], y=[med[i], p90[i]], pen='2p,gray50')
            fig.plot(x=[i], y=[med[i]], style='s0.3c', fill='30/120/214')
            fig.text(x=i, y=0, text=n, justify='TC', offset='0c/-0.15c', font='8p', no_clip=True)
    else:
        fig.basemap(region=[0, 1, 0, 1], projection='X%gc/%gc' % (pw, ph), frame=['+tAdded inflation spread', 'f'])
        fig.text(position='MC', text='no adaptive_inflation_log.csv', font='9p')
    return _save(fig, out, 'gfig4_filter_health', dpi)


# ------------------------------------------------------------------------------------------------ figure 5
def _fig_rms_vs_snr_area(pygmt, units, out, dpi):
    red = _num(units, 'rms_reduction')
    snr = _num(units, 'SNR')
    area = _num(units, 'SUB_AREA')
    reg = np.array([(r.get('REGION') or '?').split(',')[0] for r in units])
    colors = ['30/120/214', '235/104/52', '27/175/122', '237/161/0', '140/80/170', '200/60/80', '90/90/90',
              '0/160/200']
    lo = min(-0.5, _nice(abs(np.nanpercentile(red, 1))) * -1)
    pw, ph = 9.0, 6.5
    fig = pygmt.Figure()
    for k, (x, xl) in enumerate([(snr, 'GRACE signal-to-noise ratio of the unit'),
                                 (area / 1e3, 'unit area [10@+3@+ km@+2@+]')]):
        ok = np.isfinite(x) & (x > 0) & np.isfinite(red)
        if k:
            fig.shift_origin(xshift='%gc' % (pw + 1.8))
        xmin, xmax = np.nanmin(x[ok]) * 0.8, np.nanmax(x[ok]) * 1.25
        fig.basemap(region=[xmin, xmax, lo, 1.0], projection='X%gcl/%gc' % (pw, ph),
                    frame=['WSne+tRMS reduction vs. %s' % ('SNR' if k == 0 else 'area'),
                           'xa2f3+l%s' % xl, 'yaf+l1 - RMS@-DA@- / RMS@-OL@-'])
        fig.plot(x=[xmin, xmax], y=[0, 0], pen='1.5p,black,--')
        for i, rname in enumerate(sorted(set(reg))):
            m = ok & (reg == rname)
            fig.plot(x=x[m], y=red[m], style='c0.16c', fill=colors[i % len(colors)] + '@20',
                     label=REGION_NAMES.get(rname, rname) if k == 1 else None)
        if k == 1:
            fig.legend(position='JMR+o0.3c/0c', box='+gwhite')
    return _save(fig, out, 'gfig5_rms_vs_snr_area', dpi)
