"""
PyGLDA v2 results dashboard (Streamlit + Plotly): browse a finished global (or regional) assimilation run in the
browser. It READS the files the pipeline already writes and never touches the assimilation code or the data:

    Res/<case>/evaluation_global/<tag>/   unit_metrics.csv, group_metrics.csv, group_series.csv, summary.json
                                          (written by GDA.da_evaluation / src_postprocessing.global_evaluation)
    Res/<case>/                           Res_OL.h5, Res_DA.h5 (daily unit series of the members),
                                          GRACE_<basin>.h5 (unit GRACE series), Harmonic_{OL,DA,GRACE}.nc (grid maps)
    GRACE/obs/<basin>_obs_GRACE.hdf5      observation windows and covariance
    Basin/shp/<...>.shp                   unit polygons (map), optional

Start (pyglda_v2 env, from the PyGLDA_v2 folder or anywhere):
    streamlit run src_GUI/dashboard.py -- --data <external_data> --case GDA_Global
then open http://localhost:8501 (on a remote HPC job, e.g. UCloud: open the port through the job's web interface). All arguments are
optional and can be changed in the sidebar; the environment variable PYGLDA_DATA is used as the default data root.

Tabs: Overview (fit, filter health, warnings), Map (one metric per unit; click a unit), Unit (window series of
GRACE / OL / DA, storages, increments, chi2), Groups (area-weighted series), Grid maps (trend / amplitude).
"""
import argparse, json, os, sys, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))
from src_postprocessing.da_evaluation import COMP, _load_obs, observation_windows, window_statistics  # noqa: E402

STORAGE_LABEL = {c[0]: c[1] for c in COMP}
STORAGE_COLOR = {c[0]: c[2] for c in COMP}
STORAGE_LABEL.update({'localwetlandstor': 'Local wetlands', 'globalwetlandstor': 'Global wetlands',
                      'locallakestor': 'Local lakes', 'globallakestor': 'Global lakes', 'reservoirstor': 'Reservoirs',
                      'canopystor': 'Canopy', 'tws': 'TWS', 'surface': 'Surface water'})
C_GR, C_OL, C_DA = '#1f1f1f', '#8c8c8c', '#2a78d6'
METRICS = {                                   # column of unit_metrics.csv: (label, colour scale, symmetric)
    'rms_reduction': ('RMS reduction 1 - RMS_DA / RMS_OL', 'RdBu', True),
    'rms_OL': ('RMS OL - GRACE [mm]', 'Viridis', False), 'rms_DA': ('RMS DA - GRACE [mm]', 'Viridis', False),
    'corr_OL': ('correlation OL - GRACE', 'Viridis', False), 'corr_DA': ('correlation DA - GRACE', 'Viridis', False),
    'chi2_median': ('median chi2 (1 = consistent)', 'RdBu', True), 'gain': ('effective gain', 'Viridis', False),
    'spread_fc_median': ('forecast spread [mm]', 'Viridis', False),
    'sigma_GRACE_median': ('GRACE sigma [mm]', 'Viridis', False),
    'innovation_rms': ('innovation RMS [mm]', 'Viridis', False), 'innovation_mean': ('mean innovation [mm]', 'RdBu', True),
    'trend_GRACE': ('trend GRACE [mm/yr]', 'RdBu', True), 'trend_OL': ('trend OL [mm/yr]', 'RdBu', True),
    'trend_DA': ('trend DA [mm/yr]', 'RdBu', True),
}


# ------------------------------------------------------------------------------------------------ arguments
def _args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--data', default=os.environ.get('PYGLDA_DATA', ''))
    p.add_argument('--case', default='GDA_Global')
    p.add_argument('--basin', default='GlobalBasins_v2.0')
    p.add_argument('--shp', default=None)
    p.add_argument('--tag', default=None)
    a, _ = p.parse_known_args()
    return a


# ------------------------------------------------------------------------------------------------ loaders (cached)
@st.cache_data(show_spinner=False)
def load_report(report_dir):
    rd = Path(report_dir)
    out = {}
    for name in ('unit_metrics', 'group_metrics', 'group_series'):
        f = rd / (name + '.csv')
        out[name] = pd.read_csv(f) if f.exists() else None
    f = rd / 'summary.json'
    out['summary'] = json.load(open(f)) if f.exists() else {}
    f = rd / 'summary.txt'
    out['summary_txt'] = open(f).read() if f.exists() else ''
    return out


@st.cache_data(show_spinner=False)
def load_geojson(shp_path, tolerance=0.05):
    """unit polygons as GeoJSON (simplified for the browser) + a table of their attributes"""
    import geopandas as gpd
    g = gpd.read_file(shp_path).sort_values('ID')
    if g.crs is not None and g.crs.to_epsg() != 4326:
        g = g.to_crs(4326)
    g['geometry'] = g.geometry.simplify(tolerance, preserve_topology=True)
    g['ID'] = g['ID'].astype(int)
    geo = json.loads(g[['ID', 'geometry']].to_json())
    attrs = g.drop(columns='geometry')
    c = g.geometry.representative_point()
    attrs['lat_c'], attrs['lon_c'] = c.y.values, c.x.values
    return geo, attrs


@st.cache_data(show_spinner=False)
def load_obs(obs_file):
    durs, sig_sub, sig_basin = _load_obs(obs_file)
    return [(a.isoformat(), b.isoformat()) for a, b in durs], sig_sub


@st.cache_data(show_spinner=False)
def h5_info(res_dir):
    """members, days, storages and first date of the daily files"""
    info = {}
    for stage in ('OL', 'DA'):
        f = Path(res_dir) / ('Res_%s.h5' % stage)
        if not f.exists():
            continue
        with h5py.File(f, 'r') as h:
            units = sorted([k for k in h.keys() if k.startswith('sub_basin_')], key=lambda s: int(s.split('_')[-1]))
            u0 = h[units[0]]
            storages = [v for v in u0.keys()]
            members = sorted(u0['tws'].keys(), key=int) if 'tws' in u0 else []
            nday = u0['tws'][members[0]].shape[0] if members else 0
            t = h['time'][:] if 'time' in h else None
        info[stage] = dict(n_units=len(units), storages=storages, members=members, nday=nday,
                           t0=float(t[0]) if t is not None else None)
    return info


@st.cache_data(show_spinner=False)
def load_unit_daily(res_dir, stage, unit_id, variables):
    """(nmem, nday) per variable for one unit, members 1..N (member 0 = unperturbed thread, left out)"""
    out = {}
    with h5py.File(Path(res_dir) / ('Res_%s.h5' % stage), 'r') as h:
        grp = h['sub_basin_%d' % unit_id]
        for v in variables:
            if v in grp:
                mem = sorted([m for m in grp[v].keys() if m != '0'], key=int)
                out[v] = np.array([grp[v][m][:] for m in mem])
    return out


@st.cache_data(show_spinner=False)
def load_unit_grace(res_dir, basin, unit_id):
    with h5py.File(Path(res_dir) / ('GRACE_%s.h5' % basin), 'r') as g:
        return g['original/basin_%d' % unit_id][:]


@st.cache_data(show_spinner=False)
def load_harmonic(res_dir, stage, q):
    import xarray as xr
    fn = Path(res_dir) / ('Harmonic_%s.nc' % stage)
    if not fn.exists():
        return None
    with xr.open_dataset(fn) as ds:
        for key in ('tws_%s_ensmean' % q, 'tws_%s' % q):
            if key in ds and ds[key].ndim == 2:
                a = ds[key].load()
                return dict(lat=a['lat'].values, lon=a['lon'].values, z=a.values)
    return None


def _decyear_to_date(y):
    year = int(np.floor(y))
    return dt.date(year, 1, 1) + dt.timedelta(days=(y - year) * (366 if _leap(year) else 365))


def _leap(y):
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


# ------------------------------------------------------------------------------------------------ figures
def fig_map(df, geo, column, label, scale, symmetric, selected=None):
    z = df[column].astype(float)
    if symmetric:
        v = float(np.nanpercentile(np.abs(z), 98))
        zmin, zmax = -v, v
    else:
        zmin, zmax = float(np.nanpercentile(z, 2)), float(np.nanpercentile(z, 98))
    hover = ['ID %d%s<br>%s: %.3g' % (i, (' - %s' % n) if isinstance(n, str) else '', label, val)
             for i, n, val in zip(df['ID'], df['NAME'] if 'NAME' in df else [None] * len(df), z)]
    fig = go.Figure(go.Choropleth(geojson=geo, locations=df['ID'], featureidkey='properties.ID', z=z,
                                  zmin=zmin, zmax=zmax, colorscale=scale, marker_line_width=0.2,
                                  marker_line_color='#555', colorbar=dict(title=label, thickness=12, len=0.7),
                                  text=hover, hoverinfo='text'))
    if selected is not None and 'lat_c' in df and 'lon_c' in df:
        r = df[df['ID'] == selected]
        if len(r):
            fig.add_trace(go.Scattergeo(lat=r['lat_c'], lon=r['lon_c'], mode='markers',
                                        marker=dict(size=12, color='red', symbol='circle-open', line=dict(width=3)),
                                        hoverinfo='skip', showlegend=False))
    fig.update_geos(projection_type='natural earth', showcountries=False, showcoastlines=True,
                    coastlinecolor='#999', showland=True, landcolor='#f4f4f4', showocean=True, oceancolor='#e8f0f7',
                    lataxis_range=[-60, 85])
    if len(df) <= 50:   # regional case: zoom to the sub-basins
        fig.update_geos(fitbounds='locations', lataxis_range=None, resolution=50, showrivers=True,
                        rivercolor='#9cc3e6', showcountries=True, countrycolor='#cccccc')
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=30, b=0), title=label)
    return fig


def fig_unit_windows(dates, s, unit_label):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=s['da'] + s['da_sd'], line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=dates, y=s['da'] - s['da_sd'], line=dict(width=0), fill='tonexty',
                             fillcolor='rgba(42,120,214,0.18)', name='DA spread (1 sd)', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=dates, y=s['gr'], mode='lines+markers', name='GRACE', line=dict(color=C_GR, width=2),
                             error_y=dict(type='data', array=s['sig'], visible=True, color='#aaa', thickness=1)))
    fig.add_trace(go.Scatter(x=dates, y=s['ol'], mode='lines', name='Open loop', line=dict(color=C_OL, width=2)))
    fig.add_trace(go.Scatter(x=dates, y=s['da'], mode='lines', name='DA', line=dict(color=C_DA, width=2.5)))
    fig.update_layout(title='%s - window means of TWS anomalies [mm]' % unit_label, height=380,
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation='h', y=-0.15), hovermode='x unified')
    return fig


def fig_unit_storages(days, OL, DA, storages):
    fig = go.Figure()
    for v in storages:
        if v in OL and v in DA:
            col = STORAGE_COLOR.get(v, None)
            o, d = OL[v].mean(0), DA[v].mean(0)
            o, d = o - o.mean(), d - o.mean()
            fig.add_trace(go.Scatter(x=days, y=o, name='%s OL' % STORAGE_LABEL.get(v, v),
                                     line=dict(color=col, width=1, dash='dot')))
            fig.add_trace(go.Scatter(x=days, y=d, name='%s DA' % STORAGE_LABEL.get(v, v), line=dict(color=col, width=2)))
    fig.update_layout(title='Storages, ensemble mean, anomalies on the OL mean [mm] (dotted: OL, solid: DA)',
                      height=420, margin=dict(l=10, r=10, t=40, b=10), hovermode='x unified',
                      legend=dict(orientation='h', y=-0.12))
    return fig


def fig_unit_increments(dates, inc_by_storage):
    fig = go.Figure()
    for v, inc in inc_by_storage.items():
        fig.add_trace(go.Bar(x=dates, y=inc, name=STORAGE_LABEL.get(v, v), marker_color=STORAGE_COLOR.get(v, None)))
    fig.update_layout(barmode='relative', title='Analysis increment per window and storage [mm]', height=340,
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation='h', y=-0.2))
    return fig


def fig_unit_filter(dates, s):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dates, y=s['chi2'], name='chi2', marker_color='#eb6834', opacity=0.7))
    fig.add_hline(y=1.0, line_dash='dash', line_color='#555')
    fig.add_trace(go.Scatter(x=dates, y=s['sd_fc'], name='forecast spread [mm]', yaxis='y2',
                             line=dict(color=C_DA)))
    fig.add_trace(go.Scatter(x=dates, y=s['sig'], name='GRACE sigma [mm]', yaxis='y2', line=dict(color=C_GR, dash='dot')))
    fig.update_layout(title='Filter consistency: chi2 = d^2 / (spread^2 + sigma^2)', height=320,
                      margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(title='chi2'),
                      yaxis2=dict(title='mm', overlaying='y', side='right'), legend=dict(orientation='h', y=-0.25))
    return fig


def fig_group(gs, name):
    g = gs[gs['group'] == name]
    x = pd.to_datetime(g['window_start'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=g['GRACE'], name='GRACE', line=dict(color=C_GR, width=2), mode='lines+markers'))
    fig.add_trace(go.Scatter(x=x, y=g['OL'], name='Open loop', line=dict(color=C_OL, width=2)))
    fig.add_trace(go.Scatter(x=x, y=g['DA'], name='DA', line=dict(color=C_DA, width=2.5)))
    fig.update_layout(title='%s - area-weighted TWS anomalies [mm]' % name, height=380, hovermode='x unified',
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation='h', y=-0.15))
    return fig


def fig_grid(grid, title, symmetric):
    z = grid['z']
    if symmetric:
        v = float(np.nanpercentile(np.abs(z), 98))
        zmin, zmax, scale = -v, v, 'RdBu_r'
    else:
        zmin, zmax, scale = float(np.nanpercentile(z, 2)), float(np.nanpercentile(z, 98)), 'Viridis'
    fig = go.Figure(go.Heatmap(z=z, x=grid['lon'], y=grid['lat'], zmin=zmin, zmax=zmax, colorscale=scale,
                               colorbar=dict(thickness=12, len=0.8)))
    fig.update_layout(title=title, height=420, margin=dict(l=10, r=10, t=40, b=10),
                      yaxis=dict(scaleanchor='x', scaleratio=1, range=[-60, 85]), xaxis=dict(range=[-180, 180]))
    return fig


# ------------------------------------------------------------------------------------------------ app
def main():
    st.set_page_config(page_title='PyGLDA v2 results', layout='wide', page_icon='🌍')
    a = _args()

    with st.sidebar:
        st.title('PyGLDA v2')
        data_root = Path(st.text_input('data root', a.data))
        cases = sorted([p.name for p in (data_root / 'Res').glob('*') if p.is_dir()]) if (data_root / 'Res').exists() else []
        case = st.selectbox('case', cases, index=cases.index(a.case) if a.case in cases else 0) if cases else a.case
        res_dir = data_root / 'Res' / case
        rep_root = res_dir / 'evaluation_global'
        tags = sorted([p.name for p in rep_root.glob('*') if (p / 'unit_metrics.csv').exists()]) if rep_root.exists() else []
        tag = st.selectbox('evaluation report', tags, index=tags.index(a.tag) if a.tag in tags else len(tags) - 1) if tags else None
        basin = st.text_input('basin / unit set', a.basin)
        shp_default = a.shp or str(data_root / 'Basin' / 'shp' / 'Global_v2.0' / ('%s.shp' % basin))
        shp_path = st.text_input('unit shapefile (map)', shp_default)
        obs_file = data_root / 'GRACE' / 'obs' / ('%s_obs_GRACE.hdf5' % basin)
        st.caption('results: %s' % res_dir)
        if st.button('reload files'):
            st.cache_data.clear()

    if tag is None:
        st.warning('No evaluation report found under %s. Run GDA.da_evaluation(tag=...) first; the Unit and Grid tabs '
                   'work without it.' % rep_root)
        rep = dict(unit_metrics=None, group_metrics=None, group_series=None, summary={}, summary_txt='')
    else:
        rep = load_report(str(rep_root / tag))
    um = rep['unit_metrics']

    geo, attrs = None, None
    if Path(shp_path).exists():
        try:
            geo, attrs = load_geojson(shp_path)
        except Exception as err:
            st.sidebar.error('shapefile not read: %s' % err)
    if um is not None and attrs is not None and 'NAME' not in um and 'NAME' in attrs:
        um = um.merge(attrs[['ID', 'NAME', 'lat_c', 'lon_c']], on='ID', how='left')

    if 'unit' not in st.session_state:
        st.session_state['unit'] = int(um['ID'].iloc[0]) if um is not None else 1

    tab_over, tab_map, tab_unit, tab_grp, tab_grid = st.tabs(['Overview', 'Map', 'Unit', 'Groups', 'Grid maps'])

    # ---------------------------------------------------------------- overview
    with tab_over:
        S = rep['summary']
        if S:
            st.subheader('%s - report %s' % (case, S.get('tag', tag)))
            st.caption('%s | %s members, %s units, %s windows | %s' % (S.get('period', ''), S.get('n_members', '?'),
                                                                        S.get('n_units', '?'), S.get('n_windows', '?'),
                                                                        S.get('method', '')))
            fg, fu, ff = S.get('fit_global', {}), S.get('units', {}), S.get('filter', {})
            c = st.columns(6)
            c[0].metric('RMS vs GRACE, global [mm]', '%.1f' % fg.get('rms_DA', np.nan),
                        'OL %.1f' % fg.get('rms_OL', np.nan), delta_color='off')
            c[1].metric('correlation, global', '%.3f' % fg.get('corr_DA', np.nan),
                        'OL %.3f' % fg.get('corr_OL', np.nan), delta_color='off')
            c[2].metric('median unit RMS reduction', '%.0f %%' % (100 * fu.get('rms_reduction_median', np.nan)),
                        '%.0f %% of units improved' % (100 * fu.get('frac_improved', np.nan)), delta_color='off')
            c[3].metric('median chi2', '%.2f' % ff.get('chi2_median_all', np.nan),
                        'mean %.2f' % ff.get('chi2_mean_all', np.nan), delta_color='off')
            c[4].metric('median gain', '%.2f' % ff.get('gain_median', np.nan))
            c[5].metric('forecast spread / GRACE sigma [mm]', '%.1f / %.1f' % (ff.get('spread_fc_median', np.nan),
                                                                                ff.get('sigma_GRACE_median', np.nan)))
            tr = 'trend [mm/yr]: GRACE %.2f | OL %.2f | DA %.2f' % (fg.get('trend_GRACE', np.nan),
                                                                     fg.get('trend_OL', np.nan), fg.get('trend_DA', np.nan))
            st.caption(tr)
            if S.get('adaptive_inflation'):
                ai = S['adaptive_inflation']
                st.caption('adaptive inflation: added sigma median %.1f mm, max %.1f mm, %.0f %% of windows at the '
                           'minimum, %.0f %% near the upper cap' % (ai.get('sigma_added_median', np.nan),
                                                                     ai.get('sigma_added_max', np.nan),
                                                                     100 * ai.get('frac_at_min', np.nan),
                                                                     100 * ai.get('frac_near_upper_cap', np.nan)))
            w = S.get('warnings', [])
            if w:
                with st.expander('warnings (%d)' % len(w), expanded=True):
                    for line in w:
                        st.markdown('- %s' % line)
            else:
                st.success('no warnings')
        if rep['group_metrics'] is not None:
            st.markdown('**Groups of units (area-weighted)**')
            gm = rep['group_metrics'].copy()
            st.dataframe(gm.style.format({c: '{:.3f}' for c in gm.columns if gm[c].dtype.kind == 'f'}),
                         use_container_width=True, height=380)
        if rep['summary_txt']:
            with st.expander('summary.txt'):
                st.code(rep['summary_txt'])

    # ---------------------------------------------------------------- map
    with tab_map:
        if um is None:
            st.info('no unit_metrics.csv')
        elif geo is None:
            st.info('no shapefile: the map needs the unit polygons (%s)' % shp_path)
        else:
            # increment diagnosis columns of unit_metrics.csv (da_evaluation.storage_increments)
            DIAG = {'inc_abs_': ('mean |increment| %s [mm]', 'Viridis', False),
                    'state_trend_': ('trend of DA - OL, %s [mm/yr]', 'RdBu', True),
                    'share_': ('share of the TWS correction trend in %s [-]', 'RdBu', True)}

            def diag_meta(c):
                for pre, (lab, sc, sy) in DIAG.items():
                    if c.startswith(pre):
                        return lab % STORAGE_LABEL.get(c[len(pre):], c[len(pre):]), sc, sy
                return None

            cols = [c for c in METRICS if c in um] + [c for c in um.columns if diag_meta(c) is not None]
            col = st.selectbox('metric', cols, format_func=lambda c: METRICS[c][0] if c in METRICS else diag_meta(c)[0])
            label, scale, sym = METRICS[col] if col in METRICS else diag_meta(col)
            fig = fig_map(um, geo, col, label, scale, sym, selected=st.session_state['unit'])
            try:                                   # click selection needs Streamlit >= 1.35
                ev = st.plotly_chart(fig, use_container_width=True, on_select='rerun', selection_mode='points',
                                     key='unit_map')
                pts = ev.selection.points if ev is not None else []
                if pts:
                    st.session_state['unit'] = int(pts[0].get('location', st.session_state['unit']))
            except TypeError:
                st.plotly_chart(fig, use_container_width=True)
                st.caption('(click selection needs streamlit >= 1.35; choose the unit in the Unit tab)')
            z = um[col].astype(float)
            c = st.columns(5)
            c[0].metric('median', '%.3g' % np.nanmedian(z)); c[1].metric('p10', '%.3g' % np.nanpercentile(z, 10))
            c[2].metric('p90', '%.3g' % np.nanpercentile(z, 90)); c[3].metric('min', '%.3g' % np.nanmin(z))
            c[4].metric('max', '%.3g' % np.nanmax(z))
            st.caption('click a unit to open it in the Unit tab; selected: ID %d' % st.session_state['unit'])
            with st.expander('table (sortable)'):
                st.dataframe(um, use_container_width=True, height=400)

    # ---------------------------------------------------------------- unit
    with tab_unit:
        info = h5_info(str(res_dir))
        if 'DA' not in info or 'OL' not in info:
            st.info('Res_OL.h5 / Res_DA.h5 not found under %s' % res_dir)
        else:
            n_unit = info['DA']['n_units']
            names = {}
            if attrs is not None and 'NAME' in attrs:
                names = dict(zip(attrs['ID'].astype(int), attrs['NAME']))
            uid = st.number_input('unit ID', 1, n_unit, int(st.session_state['unit']), step=1)
            st.session_state['unit'] = int(uid)
            unit_label = 'Unit %d%s' % (uid, (' - %s' % names[uid]) if uid in names else '')
            if um is not None:
                r = um[um['ID'] == uid]
                if len(r):
                    r = r.iloc[0]
                    c = st.columns(6)
                    c[0].metric('RMS OL / DA [mm]', '%.1f / %.1f' % (r['rms_OL'], r['rms_DA']))
                    c[1].metric('corr OL / DA', '%.2f / %.2f' % (r['corr_OL'], r['corr_DA']))
                    c[2].metric('chi2 median', '%.2f' % r['chi2_median'])
                    c[3].metric('gain', '%.2f' % r['gain'])
                    c[4].metric('spread / sigma [mm]', '%.1f / %.1f' % (r['spread_fc_median'], r['sigma_GRACE_median']))
                    c[5].metric('trend GRACE / OL / DA', '%.1f / %.1f / %.1f' % (r['trend_GRACE'], r['trend_OL'], r['trend_DA']))
                    extra = [a for a in ('REGION', 'BAS_CLASS', 'SNR', 'SUB_AREA', 'LAKE_FRAC', 'WET_FRAC', 'GLAC_EFF', 'ENDO')
                             if a in r]
                    if extra:
                        st.caption(' | '.join('%s %s' % (a, r[a]) for a in extra))
            storages = [v for v in info['DA']['storages'] if v != 'tws']
            with st.spinner('reading the daily series of unit %d' % uid):
                DA = load_unit_daily(str(res_dir), 'DA', int(uid), storages + ['tws'])
                OL = load_unit_daily(str(res_dir), 'OL', int(uid), storages + ['tws'])
            nday = DA['tws'].shape[1]
            t0 = info['DA']['t0']
            start = _decyear_to_date(t0) if t0 is not None else dt.date(2002, 1, 1)
            days = [start + dt.timedelta(days=i) for i in range(nday)]
            if obs_file.exists() and (res_dir / ('GRACE_%s.h5' % basin)).exists():
                durs_s, sig_sub = load_obs(str(obs_file))
                durs = [(dt.date.fromisoformat(x), dt.date.fromisoformat(y)) for x, y in durs_s]
                win = observation_windows(durs, start, nday)
                kk = np.array([w[0] for w in win])
                GR = load_unit_grace(str(res_dir), basin, int(uid))
                s = window_statistics(OL['tws'], DA['tws'], GR[kk], sig_sub[kk, uid - 1], win)
                wdates = [durs[k][0] + (durs[k][1] - durs[k][0]) / 2 for k in kk]
                st.plotly_chart(fig_unit_windows(wdates, s, unit_label), use_container_width=True)
                inc = {}
                for v in storages:
                    if v in DA and v in OL:
                        inc[v] = [float(np.mean((DA[v][:, i0] - DA[v][:, i0 - 1]) - (OL[v][:, i0] - OL[v][:, i0 - 1])))
                                  for _, i0, _ in win]
                st.plotly_chart(fig_unit_increments(wdates, inc), use_container_width=True)
                st.plotly_chart(fig_unit_filter(wdates, s), use_container_width=True)
            else:
                st.info('GRACE windows not available (%s / GRACE_%s.h5): daily storages only' % (obs_file, basin))
            sel = st.multiselect('storages', storages, default=[v for v in storages if v in STORAGE_COLOR] or storages)
            st.plotly_chart(fig_unit_storages(days, OL, DA, sel), use_container_width=True)

    # ---------------------------------------------------------------- groups
    with tab_grp:
        gs = rep['group_series']
        if gs is None:
            st.info('no group_series.csv')
        else:
            groups = list(dict.fromkeys(gs['group']))
            name = st.selectbox('group', groups)
            st.plotly_chart(fig_group(gs, name), use_container_width=True)
            if rep['group_metrics'] is not None:
                g = rep['group_metrics']
                row = g[g['group'] == name].iloc[0]
                st.dataframe(pd.DataFrame({'metric': row.index.astype(str), 'value': row.values.astype(str)}),
                             use_container_width=True, hide_index=True)

    # ---------------------------------------------------------------- grid maps
    with tab_grid:
        q = st.selectbox('quantity', ['trend', 'annual_amp', 'annual_peak_doy'],
                         format_func=lambda k: {'trend': 'trend [mm/yr]', 'annual_amp': 'annual amplitude [mm]',
                                                'annual_peak_doy': 'annual peak [day of year]'}[k])
        cols = st.columns(3)
        found = False
        for col, stage in zip(cols, ('OL', 'DA', 'GRACE')):
            grid = load_harmonic(str(res_dir), stage, q)
            if grid is None:
                col.info('Harmonic_%s.nc: %s not found' % (stage, q))
                continue
            found = True
            col.plotly_chart(fig_grid(grid, '%s %s' % (stage, q), symmetric=(q == 'trend')), use_container_width=True)
        if found:
            g1, g2 = load_harmonic(str(res_dir), 'DA', q), load_harmonic(str(res_dir), 'OL', q)
            if g1 is not None and g2 is not None and g1['z'].shape == g2['z'].shape:
                d = dict(lat=g1['lat'], lon=g1['lon'], z=g1['z'] - g2['z'])
                st.plotly_chart(fig_grid(d, 'DA - OL: %s' % q, symmetric=True), use_container_width=True)


if __name__ == '__main__':
    main()
