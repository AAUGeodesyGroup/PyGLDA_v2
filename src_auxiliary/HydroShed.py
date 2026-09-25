import datetime
import enum

import numpy as np
import geopandas as gpd
from shapely import box
from pathlib import Path
import shapely
import shapely.vectorized
import pandas as pd
import h5py
from GeoMathKit import GeoMathKit
import netCDF4 as nc


def demo_EU():
    box_area = (71.6, 36.1, -11.1, 42.1)

    # box_area = (71.6, 36.1, -11.1, 70.1)
    gdf = gpd.read_file(filename='/media/user/My Book/Fan/PyGLDA_v2_external_data/Extra/hybas_eu_lev01-12_v1c/hybas_eu_lev02_v1c.shp')

    import pygmt

    fig = pygmt.Figure()
    pygmt.config(MAP_HEADING_OFFSET=0, MAP_TITLE_OFFSET=-0.2)
    pygmt.config(FONT_ANNOT='10p', COLOR_NAN='white')
    pygmt.makecpt(cmap='wysiwyg', series=[0, 56], background='o')

    # region = [box_area[2] - 5, box_area[3] + 5, box_area[1] - 5, box_area[0] + 5]
    region = [box_area[2], box_area[3], box_area[1], box_area[0]]
    pj = "Q30/-20/12c"
    fig.basemap(region=region, projection=pj,
                frame=['WSne', 'xa10f5+lLongitude (\\260 E)', 'ya5f5+lLatitude (\\260 N)'])

    fig.coast(shorelines="1/0.2p", region=region, projection=pj)

    shortlist = []
    for i, el in gdf.HYBAS_ID.items():
        xy = gdf[gdf.HYBAS_ID == el].centroid

        if box_area[3] > xy.x.item() > box_area[2] and box_area[0] > xy.y.item() > box_area[1] and gdf.SUB_AREA[
            i] > 10000:
            fig.text(x=xy.x, y=xy.y, text="%s" % i, font='7p,black')
            fig.plot(data=gdf[gdf.HYBAS_ID == el].boundary, pen="0.5p,blue", projection=pj)

            shortlist.append(True)
        else:
            shortlist.append(False)

    nns = gdf[shortlist]
    nns.assign(ID = np.arange(1, len(nns)+1)).to_file('/media/user/My Book/Fan/PyGLDA_v2_external_data/Extra/shp/EU/EU.shp')
    fig.show()
    # fig.savefig('/home/user/Documents/ESA_SING_EXP/Europe/temp.png')
    pass


def demo_SA():
    """NOTE: box_area below is the European box copied from demo_EU -> selects nothing in South America.
    For the Amazon use demo_Amazon() (Pfafstetter-code based selection, no box needed)."""
    box_area = (71.6, 36.1, -11.1, 42.1)

    # box_area = (71.6, 36.1, -11.1, 70.1)
    gdf = gpd.read_file(filename='/media/user/My Book/Fan/PyGLDA_v2_external_data/Extra/hybas_sa_lev01-12_v1c/hybas_sa_lev02_v1c.shp')

    import pygmt

    fig = pygmt.Figure()
    pygmt.config(MAP_HEADING_OFFSET=0, MAP_TITLE_OFFSET=-0.2)
    pygmt.config(FONT_ANNOT='10p', COLOR_NAN='white')
    pygmt.makecpt(cmap='wysiwyg', series=[0, 56], background='o')

    # region = [box_area[2] - 5, box_area[3] + 5, box_area[1] - 5, box_area[0] + 5]
    region = [box_area[2], box_area[3], box_area[1], box_area[0]]
    pj = "Q30/-20/12c"
    fig.basemap(region=region, projection=pj,
                frame=['WSne', 'xa10f5+lLongitude (\\260 E)', 'ya5f5+lLatitude (\\260 N)'])

    fig.coast(shorelines="1/0.2p", region=region, projection=pj)

    shortlist = []
    for i, el in gdf.HYBAS_ID.items():
        xy = gdf[gdf.HYBAS_ID == el].centroid

        if box_area[3] > xy.x.item() > box_area[2] and box_area[0] > xy.y.item() > box_area[1] and gdf.SUB_AREA[
            i] > 10000:
            fig.text(x=xy.x, y=xy.y, text="%s" % i, font='7p,black')
            fig.plot(data=gdf[gdf.HYBAS_ID == el].boundary, pen="0.5p,blue", projection=pj)

            shortlist.append(True)
        else:
            shortlist.append(False)

    nns = gdf[shortlist]
    nns.assign(ID = np.arange(1, len(nns)+1)).to_file('/media/user/My Book/Fan/PyGLDA_v2_external_data/Extra/shp/SA/SA.shp')
    fig.show()
    # fig.savefig('/home/user/Documents/ESA_SING_EXP/Europe/temp.png')
    pass



def demo_EU_2():
    box_area = (71.6, 36.1, -11.1, 42.1)
    gdf = gpd.read_file(filename='/media/user/My Book/Fan/PyGLDA_v2_external_data/Extra/shp/EU/EU.shp')

    import pygmt

    fig = pygmt.Figure()
    pygmt.config(MAP_HEADING_OFFSET=0, MAP_TITLE_OFFSET=-0.2)
    pygmt.config(FONT_ANNOT='10p', COLOR_NAN='white')
    pygmt.makecpt(cmap='wysiwyg', series=[0, 56], background='o')

    region = [box_area[2] - 5, box_area[3] + 5, box_area[1] - 5, box_area[0] + 5]
    pj = "Q30/-20/12c"
    fig.basemap(region=region, projection=pj,
                frame=['WSne', 'xa10f5+lLongitude (\\260 E)', 'ya5f5+lLatitude (\\260 N)'])

    # fig.show()

    '''mask visualization'''
    h5 = h5py.File('/home/user/codes/PyGWater/res/mask/EU_res_0.1_crop.h5', 'r')
    # h5 = h5py.File('/home/user/codes/PyGWater/res/mask/EU_res_0.5_crop.h5', 'r')
    # h5 = h5py.File('/home/user/codes/PyGWater/res/mask/EU_res_1_crop.h5', 'r')
    lat = h5['lat'][:]
    lon = h5['lon'][:]
    mask = h5['sub_basin_50'][:]
    # mask = h5['basin'][:]
    res = h5['resolution'][()].item()

    '''load DA mask for EU'''
    box_id = h5['mask2global_id'][:]
    h52 = h5py.File('/media/user/My Book/Fan/ESA_SING/EuropeESM3ExpRes/Mask/mask_global.h5', 'r')

    err = res / 10
    lat2 = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
    lon2 = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
    lon2, lat2 = np.meshgrid(lon2, lat2)
    box_mask = np.zeros(np.shape(lon2))
    box_mask[box_id[0]:box_id[1], box_id[2]:box_id[3]] = mask

    # mask = box_mask * h52['mask']
    mask = box_mask
    mask = mask[box_id[0]:box_id[1], box_id[2]:box_id[3]]

    ''''''
    pygmt.makecpt(cmap='polar', series=[-1, 1], background='o')
    data_region = [min(lon[0]), max(lon[0]), min(lat[:, 0]), max(lat[:, 0])]

    ee = pygmt.xyz2grd(y=lat.flatten(), x=lon.flatten(), z=mask.flatten(),
                       spacing=(res, res), region=data_region)

    # pj = 'Q10c'
    fig.grdimage(
        grid=ee,
        cmap=True,
        # frame=['xaf', 'yaf'] + ['+t %s v.s. %s (%.1f mm)' % (dt1.name, miss2.name+'_'+dt2.name, statistic)],
        frame=['xaf', 'yaf'],
        # frame=['xaf', 'yaf'] + ['+t %s v.s. %s' % (miss1.name + '_' + dt1.name, miss2.name + '_' + dt2.name)],
        # dpi=100,
        projection=pj,
        interpolation='n',
        # nan_transparent=True,
        region=region
    )

    fig.coast(shorelines="1/0.2p", region=region, projection=pj, water='lightblue')
    # fig.coast(shorelines="1/0.2p", region=region, projection=pj)

    for i in range(1, gdf.shape[0] + 1):
        xy = gdf[gdf.ID == i].centroid
        fig.text(x=xy.x, y=xy.y, text="%s" % i, font='7p,black')

    fig.plot(data=gdf.boundary, pen="0.5p,blue", projection=pj)

    fig.show()
    # fig.savefig('/home/user/Documents/ESA_SING_EXP/Europe/temp.png')
    pass


def hydrobasins_subbasins(hybas_dir, continent='sa', pfaf_prefix='62', level=4, min_area_km2=100000.0,
                          split_above_km2=None, out_shp=None, verbose=True):
    """
    Extract one river basin and its sub-basins from HydroBASINS (v1c) and write a PyGLDA-ready shapefile.

    HydroBASINS uses Pfafstetter codes: every sub-basin of a basin shares the basin's PFAF_ID as prefix
    (e.g. '62' = Amazon, level-2 polygon HYBAS_ID 6020006540). Sub-basins smaller than `min_area_km2`
    are merged into their downstream neighbour (NEXT_DOWN) so that every remaining sub-basin is
    resolvable by GRACE. The result gets an 'ID' column (1..N), which shp2mask requires.

    Parameters
    ----------
    hybas_dir : str/Path   folder 'hybas_<continent>_lev01-12_v1c'
    continent : str        HydroBASINS region code: 'sa', 'eu', 'af', 'as', 'au', 'na', 'ar', 'gr', 'si'
    pfaf_prefix : str      Pfafstetter prefix of the basin ('62' Amazon, '2' Danube-region ..., see level-2/3 table)
    level : int            HydroBASINS level (3..12) from which the sub-basins are taken
    min_area_km2 : float   merge threshold; 0 or None = no merging
    split_above_km2 : float  polygons larger than this are replaced by their children from level+1
                           (same Pfafstetter prefix) before merging; None = no splitting
    out_shp : str/Path     output shapefile; None = do not write
    Returns geopandas.GeoDataFrame with columns ID, HYBAS_ID (list of merged ids), PFAF_ID, SUB_AREA, NEXT_DOWN, geometry
    """
    import numpy as np, geopandas as gpd, pandas as pd
    from pathlib import Path
    from shapely.ops import unary_union

    fn = Path(hybas_dir) / f'hybas_{continent}_lev{level:02d}_v1c.shp'
    gdf = gpd.read_file(fn)
    sub = gdf[gdf.PFAF_ID.astype(str).str.startswith(str(pfaf_prefix))].copy()
    assert len(sub) > 0, f'no polygon with PFAF_ID prefix {pfaf_prefix} in {fn}'
    if verbose:
        print(f'{fn.name}: {len(sub)} sub-basins with PFAF prefix {pfaf_prefix}, total area {sub.SUB_AREA.sum():.0f} km2')

    if split_above_km2:
        big = sub[sub.SUB_AREA > split_above_km2]
        if len(big):
            fn2 = Path(hybas_dir) / f'hybas_{continent}_lev{level + 1:02d}_v1c.shp'
            fine = gpd.read_file(fn2)
            parts = [sub[sub.SUB_AREA <= split_above_km2]]
            for _, r in big.iterrows():
                ch = fine[fine.PFAF_ID.astype(str).str.startswith(str(r.PFAF_ID))]
                if verbose:
                    print(f'  split {r.HYBAS_ID} (PFAF {r.PFAF_ID}, {r.SUB_AREA:.0f} km2) into {len(ch)} level-{level + 1} polygons')
                parts.append(ch)
            sub = pd.concat(parts, ignore_index=True)

    # --- working table: one dict per polygon, keyed by HYBAS_ID
    # 'parent' = Pfafstetter code at the base level; merging stays inside the parent whenever possible,
    # so that children of a split polygon do not leak into a neighbouring tributary.
    nd = len(str(sub.PFAF_ID.iloc[0]))            # digits at base level
    nd = min(len(str(p)) for p in sub.PFAF_ID)
    polys = {int(r.HYBAS_ID): dict(ids=[int(r.HYBAS_ID)], pfaf=int(r.PFAF_ID), area=float(r.SUB_AREA),
                                  down=int(r.NEXT_DOWN), geom=r.geometry, parent=str(r.PFAF_ID)[:nd])
             for _, r in sub.iterrows()}

    def survivor(hid):  # follow merges
        while hid in alias:
            hid = alias[hid]
        return hid
    alias = {}

    if min_area_km2:
        while True:
            small = [k for k, v in polys.items() if v['area'] < min_area_km2]
            if not small:
                break
            k = min(small, key=lambda x: polys[x]['area'])       # smallest first
            v = polys[k]
            down = survivor(v['down']) if v['down'] != 0 else None
            if down not in polys or down == k:
                down = None
            touching = [j for j, w in polys.items() if j != k and w['geom'].intersects(v['geom'])]
            same = [j for j in touching if polys[j]['parent'] == v['parent']]
            if down is not None and polys[down]['parent'] == v['parent']:
                target = down                                     # 1) downstream, same parent
            elif same:
                target = max(same, key=lambda j: polys[j]['area'])  # 2) largest neighbour, same parent
            elif down is not None:
                target = down                                     # 3) downstream, other parent
            elif touching:
                target = max(touching, key=lambda j: polys[j]['area'])  # 4) largest neighbour
            else:
                if verbose: print(f'  keep isolated small polygon {k} ({v["area"]:.0f} km2)')
                polys[k]['area'] = min_area_km2
                continue
            t = polys[target]
            if verbose:
                print(f'  merge {k} ({v["area"]:.0f} km2) -> {target} ({t["area"]:.0f} km2)')
            t['geom'] = unary_union([t['geom'], v['geom']])
            t['area'] += v['area']
            t['ids'] += v['ids']
            if t['area'] - v['area'] < v['area']: t['parent'] = t['parent']  # keep target parent
            alias[k] = target
            del polys[k]
            for w in polys.values():                              # re-point upstream neighbours
                if w['down'] in alias:
                    w['down'] = survivor(w['down'])

    rows = sorted(polys.values(), key=lambda v: v['pfaf'])
    out = gpd.GeoDataFrame({
        'ID': np.arange(1, len(rows) + 1),
        'HYBAS_ID': [v['ids'][0] for v in rows],
        'MERGED': [','.join(map(str, v['ids'])) for v in rows],
        'PFAF_ID': [v['pfaf'] for v in rows],
        'SUB_AREA': [v['area'] for v in rows],
        'NEXT_DOWN': [v['down'] for v in rows],
    }, geometry=[v['geom'] for v in rows], crs=sub.crs)
    if verbose:
        print(f'result: {len(out)} sub-basins, areas (10^3 km2): '
              + ', '.join(f'{a/1e3:.0f}' for a in sorted(out.SUB_AREA, reverse=True)))
    if out_shp is not None:
        Path(out_shp).parent.mkdir(parents=True, exist_ok=True)
        out.to_file(out_shp)
        if verbose: print('written', out_shp)
    return out

# Hydrologically defined Amazon sub-basins built from HydroBASINS Pfafstetter codes (mixed levels 4/5).
# Codes are read from hybas_sa_lev0{len(code)}: '62xx' -> level 4, '62xxx' -> level 5.
AMAZON_SUBBASINS = {
    # --- Andean / western system (Solimoes), peak Mar-Jun
    'Maranon':                                   ['62298'],
    'Ucayali':                                   ['62299'],
    'Solimoes mainstem (Napo, Putumayo, Javari, floodplain to Manaus)':
                                                 ['62297', '62295', '62293', '62291', '6227'],
    'Japura-Caqueta':                            ['62294'],
    'Jurua':                                     ['62296'],
    'Purus':                                     ['62292'],
    # --- Guiana-shield tributaries, northern-hemisphere regime (peak Jun-Aug)
    'Branco':                                    ['62284'],
    'Negro (excl. Branco)':                      ['62281', '62282', '62283', '62285', '62286', '62287', '62288', '62289'],
    'Middle Amazon north (Trombetas, Nhamunda, Uatuma)': ['62252', '62256', '62258'],
    'Lower Amazon north (Paru, Jari, Maicuru)':  ['62232', '62234', '62238', '6210'],
    # --- Madeira system, southern-hemisphere regime (peak Feb-Apr)
    'Madre de Dios-Beni':                        ['62266'],
    'Mamore':                                    ['62269'],
    'Guapore-Itenez':                            ['62268'],
    'Lower Madeira (Aripuana, Ji-Parana)':       ['62261', '62262', '62263', '62264', '62265', '62267'],
    # --- Brazilian-shield tributaries and the southern mainstem strip, peak Mar-May
    'Middle Amazon south (Canuma, Abacaxis, mainstem Manaus-Santarem)':
                                                 ['62254', '62251', '62253', '62255', '62257', '62259'],
    'Tapajos':                                   ['6224'],
    'Xingu':                                     ['6222'],
    'Lower Amazon south and estuary (Curua-Una, Marajo)':
                                                 ['62231', '62233', '62235', '62236', '62237', '62239', '6221', '6230'],
}

# Tocantins-Araguaia belongs to HydroBASINS basin 62 but drains to the Para river, not to the Amazon.
# Add these two entries to AMAZON_SUBBASINS if the whole HydroBASINS basin 62 is wanted.
TOCANTINS_SUBBASINS = {
    'Tocantins':                                 ['6244', '6243', '6242', '6241', '6250'],
    'Araguaia':                                  ['6245', '6246', '6247', '6248', '6249'],
}

def hydrobasins_named(hybas_dir, groups, continent='sa', out_shp=None, verbose=True):
    """Build a sub-basin shapefile from named groups of HydroBASINS Pfafstetter codes (mixed levels).
    groups: {name: [PFAF codes as strings]}; the level of each code equals its number of digits (PFAF '62' = level 2).
    Output columns: ID (1..N, in dict order), NAME, PFAF (codes), SUB_AREA (km2), geometry."""
    import geopandas as gpd, numpy as np, pandas as pd
    from pathlib import Path
    from shapely.ops import unary_union
    cache = {}
    def level(lev):
        if lev not in cache:
            cache[lev] = gpd.read_file(Path(hybas_dir) / f'hybas_{continent}_lev{lev:02d}_v1c.shp')
        return cache[lev]
    rows, used = [], []
    for name, codes in groups.items():
        geoms, area = [], 0.0
        for c in codes:
            g = level(len(c))
            m = g[g.PFAF_ID.astype(str) == c]
            assert len(m) == 1, f'{name}: PFAF {c} not found (or not unique) at level {len(c)}'
            geoms.append(m.geometry.iloc[0]); area += float(m.SUB_AREA.iloc[0]); used.append(c)
        rows.append(dict(NAME=name, PFAF=','.join(codes), SUB_AREA=area, geometry=unary_union(geoms)))
        if verbose: print(f'{name:55s} {area/1e3:7.0f} x10^3 km2  ({len(codes)} polygons)')
    out = gpd.GeoDataFrame(rows, crs=level(4).crs)
    out.insert(0, 'ID', np.arange(1, len(out) + 1))
    # completeness check: every code must be a leaf of the union tree (no double counting / gaps)
    assert len(used) == len(set(used)), 'a PFAF code is used twice'
    if verbose: print(f'total {out.SUB_AREA.sum():.0f} km2, {len(out)} sub-basins')
    if out_shp is not None:
        Path(out_shp).parent.mkdir(parents=True, exist_ok=True)
        out.to_file(out_shp)
        if verbose: print('written', out_shp)
    return out

def plot_subbasins(gdf, title='', region=None, fig_path=None, label_font='8p,Helvetica-Bold,black',
                   legend_font='8p,Helvetica,black'):
    """Check map of a sub-basin shapefile (pygmt): coloured polygons with the ID number at the
    representative point, and a legend column (ID, name, area) to the right of the map."""
    import pygmt
    if region is None:
        b = gdf.total_bounds
        region = [b[0] - 2, b[2] + 2, b[1] - 2, b[3] + 2]
    fig = pygmt.Figure()
    pygmt.config(FONT_ANNOT='10p', FONT_TITLE='14p', COLOR_NAN='white')
    pj = "Q%s/%s/13c" % (0.5 * (region[0] + region[1]), 0.5 * (region[2] + region[3]))
    fig.basemap(region=region, projection=pj,
                frame=['WSne+t%s' % title, 'xa10f5+lLongitude (\\260 E)', 'ya5f5+lLatitude (\\260 N)'])
    fig.coast(shorelines="1/0.2p", water='lightblue', region=region, projection=pj)
    pygmt.makecpt(cmap='categorical', series=[1, len(gdf) + 1, 1])
    for _, r in gdf.iterrows():
        fig.plot(data=gpd.GeoDataFrame(geometry=[r.geometry], crs=gdf.crs), fill='+z', zvalue=r.ID, cmap=True,
                 pen="0.5p,black", transparency=40, projection=pj)
    for _, r in gdf.iterrows():                      # numbers on top of all polygons
        c = r.geometry.representative_point()
        fig.text(x=c.x, y=c.y, text="%d" % r.ID, font=label_font, fill='white@30', pen='0.25p,black',
                 clearance='1p/1p+tO')
    # legend column to the right of the map
    names = gdf.NAME if 'NAME' in gdf.columns else ['' for _ in range(len(gdf))]
    lines = ["%d  %s  (%.0f x10@+3@+ km@+2@+)" % (r.ID, n.split('(')[0].strip(), r.SUB_AREA / 1e3)
             for (_, r), n in zip(gdf.iterrows(), names)]
    height_cm = 13.0 * (region[3] - region[2]) / (region[1] - region[0])   # approx. map height for Q
    fig.shift_origin(xshift='13.6c')
    n = len(lines)
    fig.text(x=[0.02] * n, y=[1 - (i + 0.5) / n for i in range(n)], text=lines, font=legend_font,
             justify='ML', no_clip=True, region=[0, 1, 0, 1], projection='X7c/%.2fc' % height_cm)
    if fig_path is not None:
        Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=300)
    fig.show()
    return fig

def demo_Amazon():
    """Amazon sub-basins for GRACE DA, hydrologically defined (major tributaries and mainstem reaches,
    grouped by flow regime): 18 sub-basins of 152,000 - 666,000 km2 (6.06 million km2 in total), built
    from HydroBASINS level-4/5 Pfafstetter codes (AMAZON_SUBBASINS). Tocantins-Araguaia (TOCANTINS_SUBBASINS)
    is excluded; merge the two dicts to obtain the complete HydroBASINS basin 62 (6.91 million km2)."""
    hybas_dir = '/media/user/My Book/Fan/PyGLDA_v2_external_data/Extra/hybas_sa_lev01-12_v1c'
    out_shp = '/media/user/My Book/Fan/PyGLDA_v2_external_data/Extra/shp/Amazon/Amazon.shp'
    fig_path = '/media/user/My Book/Fan/PyGLDA_v2_external_data/Extra/fig/Amazon_subbasins.png'
    gdf = hydrobasins_named(hybas_dir, AMAZON_SUBBASINS, continent='sa', out_shp=out_shp)
    # gdf = hydrobasins_named(hybas_dir, {**AMAZON_SUBBASINS, **TOCANTINS_SUBBASINS}, continent='sa', out_shp=out_shp)
    plot_subbasins(gdf, title='Amazon: %d sub-basins' % len(gdf), fig_path=fig_path)
    pass

def demo_Amazon_automatic():
    """Alternative: purely area-based division (level 4, split >800k km2 into level 5, merge <150k km2)."""
    hybas_dir = '/media/user/My Book/Fan/PyGLDA_v2_external_data/Extra/hybas_sa_lev01-12_v1c'
    gdf = hydrobasins_subbasins(hybas_dir, continent='sa', pfaf_prefix='62', level=4,
                                min_area_km2=150000, split_above_km2=800000, out_shp=None)
    plot_subbasins(gdf, title='Amazon (automatic): %d sub-basins' % len(gdf))
    pass

if __name__ == '__main__':
    demo_Amazon()

    pass
