"""
Sub-basin shapefiles for PyGLDA from HydroSHEDS / HydroBASINS (v1c, Lehner & Grill 2013).

HydroBASINS carries Pfafstetter codes: all polygons of one river basin share the basin's code as prefix
(Amazon '62' in region 'sa', Danube '227' in region 'eu'); the level of a code equals its number of digits
('2274' = level 4, '22799' = level 5). A PyGLDA sub-basin is a named group of such codes (mixed levels are
allowed); the polygons of a group are merged, and the result is written with the columns that shp2mask /
the DA need: ID (1..N, in dictionary order), NAME, PFAF (the codes), SUB_AREA [km2], geometry.

    from src_auxiliary.Hydroshed import HydroBASINS_subbasins, AMAZON_SUBBASINS, DANUBE_SUBBASINS
    hb = HydroBASINS_subbasins(hybas_dir='.../hybas_eu_lev01-12_v1c', continent='eu')
    gdf = hb.build(DANUBE_SUBBASINS, out_shp='.../Basin/shp/Danube/Danube.shp')
    hb.check_figure(gdf, basin_pfaf='227', fn='.../Extra/fig/Danube_subbasins.png')
    hb.list_units('227', level=5)        # table of the HydroBASINS polygons to design a new grouping

Groupings are hydrological units, sized for GRACE: Amazon 18 units of 150-670 x10^3 km2 (regimes: Andean /
Solimoes, Guiana shield with northern-hemisphere peak, Madeira and Brazilian-shield tributaries with
southern-hemisphere peak); Danube 6 units of 95-160 x10^3 km2 (Upper, Middle, Sava, Tisza, Lower west,
Lower east + delta). Smaller units are below the GRACE footprint and only make sense with a full error
covariance between them.
"""
from pathlib import Path
import numpy as np


# ------------------------------------------------------------------ named groupings
# Amazon (HydroBASINS South America, basin 62). Tocantins-Araguaia (6241..6250) drains to the Para river and
# is left out; add TOCANTINS_SUBBASINS for the whole HydroBASINS basin 62.
AMAZON_SUBBASINS = {
    'Maranon':                                   ['62298'],
    'Ucayali':                                   ['62299'],
    'Solimoes mainstem (Napo, Putumayo, Javari, floodplain to Manaus)':
                                                 ['62297', '62295', '62293', '62291', '6227'],
    'Japura-Caqueta':                            ['62294'],
    'Jurua':                                     ['62296'],
    'Purus':                                     ['62292'],
    'Branco':                                    ['62284'],
    'Negro (excl. Branco)':                      ['62281', '62282', '62283', '62285', '62286', '62287', '62288', '62289'],
    'Middle Amazon north (Trombetas, Nhamunda, Uatuma)': ['62252', '62256', '62258'],
    'Lower Amazon north (Paru, Jari, Maicuru)':  ['62232', '62234', '62238', '6210'],
    'Madre de Dios-Beni':                        ['62266'],
    'Mamore':                                    ['62269'],
    'Guapore-Itenez':                            ['62268'],
    'Lower Madeira (Aripuana, Ji-Parana)':       ['62261', '62262', '62263', '62264', '62265', '62267'],
    'Middle Amazon south (Canuma, Abacaxis, mainstem Manaus-Santarem)':
                                                 ['62254', '62251', '62253', '62255', '62257', '62259'],
    'Tapajos':                                   ['6224'],
    'Xingu':                                     ['6222'],
    'Lower Amazon south and estuary (Curua-Una, Marajo)':
                                                 ['62231', '62233', '62235', '62236', '62237', '62239', '6221', '6230'],
}
TOCANTINS_SUBBASINS = {
    'Tocantins':                                 ['6244', '6243', '6242', '6241', '6250'],
    'Araguaia':                                  ['6245', '6246', '6247', '6248', '6249'],
}

# Danube (HydroBASINS Europe, basin 227; 795 x10^3 km2, 8.2-29.7 E, 42.1-50.2 N). Coarse division:
DANUBE_SUBBASINS = {
    'Upper Danube to Bratislava (Inn, Traun, Enns, Morava)':              ['22799', '22798', '22797', '22796', '22795'],
    'Middle Danube Bratislava-Belgrade (Vah, Hron, Raba, Drava-Mura)':    ['22791', '22792', '22793', '22794', '2278', '2277', '2275'],
    'Sava':                                                               ['2274'],
    'Tisza (Somes, Bodrog, Sajo, Koros, Mures)':                          ['2276'],
    'Lower Danube west (Velika Morava, Banat, Jiu, Olt, NW Bulgaria)':    ['22736', '22737', '22738', '22739'],
    'Lower Danube east and delta (Arges, Ialomita, Yantra, Siret, Prut)': ['22731', '22732', '22733', '22734', '22735', '2272', '2271'],
}
# finer alternative (~10 units) kept for reference
DANUBE_SUBBASINS_10 = {
    'Upper Danube and Inn':                           ['22799', '22798'],
    'Austrian-Hungarian Danube (Traun, Enns, Morava, Vah, Raba)':
                                                      ['22797', '22796', '22795', '22794', '22793', '22792', '22791'],
    'Drava-Mura':                                     ['2278', '2277', '2275'],
    'Sava':                                           ['2274'],
    'Upper Tisza (Somes, Bodrog, Sajo)':              ['22769', '22768', '22767', '22766', '22765'],
    'Lower Tisza (Mures, Koros)':                     ['22764', '22763', '22762', '22761'],
    'Velika Morava and Banat':                        ['22738', '22739'],
    'Olt, Jiu and NW Bulgarian tributaries':          ['22736', '22737'],
    'Lower Danube east (Arges, Ialomita, Yantra, Dobruja)': ['22731', '22732', '22733', '22734', '22735'],
    'Siret, Prut and delta':                          ['2272', '2271'],
}


class HydroBASINS_subbasins:

    def __init__(self, hybas_dir, continent='eu'):
        """hybas_dir: folder hybas_<continent>_lev01-12_v1c ; continent: 'sa', 'eu', 'af', 'as', 'au', 'na', 'ar', 'gr', 'si'"""
        self.hybas_dir, self.continent = Path(hybas_dir), continent
        self._cache = {}

    def level(self, lev):
        import geopandas as gpd
        if lev not in self._cache:
            self._cache[lev] = gpd.read_file(self.hybas_dir / f'hybas_{self.continent}_lev{lev:02d}_v1c.shp')
        return self._cache[lev]

    # -------------------------------------------------------------- design helpers
    def find_basin(self, area_km2, level=3, tol=0.15):
        """polygons at `level` whose area is within tol of area_km2 (to identify a basin's Pfafstetter code)"""
        g = self.level(level)
        m = g[(g.SUB_AREA > area_km2 * (1 - tol)) & (g.SUB_AREA < area_km2 * (1 + tol))]
        return m[['HYBAS_ID', 'PFAF_ID', 'SUB_AREA', 'NEXT_DOWN']].assign(bounds=m.geometry.bounds.round(2).values.tolist())

    def list_units(self, basin_pfaf, level=5):
        """table (PFAF, area, downstream unit, centroid) of all polygons of a basin at one level"""
        import warnings
        g = self.level(level)
        s = g[g.PFAF_ID.astype(str).str.startswith(str(basin_pfaf))].copy()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            s['cx'], s['cy'] = s.geometry.centroid.x.round(2), s.geometry.centroid.y.round(2)
        t = s[['HYBAS_ID', 'PFAF_ID', 'SUB_AREA', 'NEXT_DOWN', 'cx', 'cy']].sort_values('PFAF_ID')
        print('level %d: %d polygons, %.0f km2' % (level, len(t), t.SUB_AREA.sum()))
        print(t.to_string(index=False))
        return t

    # -------------------------------------------------------------- build
    def build(self, groups: dict, out_shp=None, verbose=True):
        """groups: {name: [PFAF codes as strings]} -> GeoDataFrame (ID, NAME, PFAF, SUB_AREA, geometry); writes out_shp"""
        import geopandas as gpd
        from shapely.ops import unary_union
        rows, used = [], []
        for name, codes in groups.items():
            geoms, area = [], 0.0
            for c in codes:
                g = self.level(len(c))
                m = g[g.PFAF_ID.astype(str) == c]
                assert len(m) == 1, f'{name}: PFAF {c} not found (or not unique) at level {len(c)}'
                geoms.append(m.geometry.iloc[0]); area += float(m.SUB_AREA.iloc[0]); used.append(c)
            rows.append(dict(NAME=name, PFAF=','.join(codes), SUB_AREA=area, geometry=unary_union(geoms)))
            if verbose:
                print(f'{name:70s} {area / 1e3:7.1f} x10^3 km2  ({len(codes)} polygons)')
        out = gpd.GeoDataFrame(rows, crs=self.level(len(used[0])).crs)
        out.insert(0, 'ID', np.arange(1, len(out) + 1))
        assert len(used) == len(set(used)), 'a PFAF code is used twice'
        if verbose:
            print(f'total {out.SUB_AREA.sum():.0f} km2, {len(out)} sub-basins')
        if out_shp is not None:
            Path(out_shp).parent.mkdir(parents=True, exist_ok=True)
            out.to_file(out_shp)
            if verbose:
                print('written', out_shp)
        return out

    def check_completeness(self, gdf, basin_pfaf, level=3):
        """the union of the sub-basins must equal the basin polygon at `level` (no gaps, no overlaps)"""
        from shapely.ops import unary_union
        g = self.level(level)
        basin = g[g.PFAF_ID.astype(str) == str(basin_pfaf)].geometry.iloc[0]
        u = unary_union(gdf.geometry)
        ratio, sd = u.area / basin.area, u.symmetric_difference(basin).area
        print('union area / basin area = %.4f, symmetric difference %.2e deg2 -> %s' % (ratio, sd, 'OK' if sd < 1e-6 else 'CHECK'))
        return sd < 1e-6

    def check_figure(self, gdf, basin_pfaf, fn, ref_level=5, title=None, width=14.0, allow_pop_up=True,
                     rivers=('1/1.2p,mediumblue', '2/0.7p,mediumblue', '3/0.35p,royalblue')):
        """
        pygmt map of the sub-basins: coloured units with ID and area [10^3 km2], the basin outline, and the
        HydroBASINS polygons of ref_level in grey; ID badges on the map, a table ID | name | area to the right of the map.
        fn: output file without extension or with .png/.pdf (both are written); allow_pop_up shows the figure
        rivers: GMT river levels and pens drawn on top of the units (None = no river network)
        """
        import pygmt
        import geopandas as gpd

        l3 = self.level(len(str(basin_pfaf))); basin = l3[l3.PFAF_ID.astype(str) == str(basin_pfaf)]
        ref = self.level(ref_level); ref = ref[ref.PFAF_ID.astype(str).str.startswith(str(basin_pfaf))]
        x0, y0, x1, y1 = gdf.total_bounds
        region = [np.floor(x0 - 0.5), np.ceil(x1 + 0.5), np.floor(y0 - 0.5), np.ceil(y1 + 0.5)]
        colors = ['steelblue', 'seagreen', 'plum', 'pink', 'khaki', 'turquoise', 'salmon', 'gold', 'orchid',
                  'lightcoral', 'palegreen', 'skyblue', 'tan', 'lightpink', 'yellowgreen', 'thistle', 'wheat', 'aquamarine']
        title = title or 'Sub-basins from HydroBASINS'

        fig = pygmt.Figure()
        pygmt.config(FONT_TITLE="14p,5", MAP_TITLE_OFFSET="2p", MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY='10p,5',
                     FONT_LABEL='10p,5', MAP_TICK_LENGTH='5p', MAP_FRAME_PEN='1.5p')
        fig.basemap(region=region, projection=f'M{width}c', frame=['WSne+t' + title, 'xa4f1', 'ya2f1'])
        fig.coast(shorelines='0.4p,gray30', borders='1/0.25p,gray60', resolution='i', land='white', water='lightblue')
        # (rivers are drawn later, on top of the coloured units)
        fig.plot(data=ref, pen='0.3p,gray60')                                      # reference polygons
        for i, (_, r) in enumerate(gdf.iterrows()):
            one = gpd.GeoDataFrame(geometry=[r.geometry], crs=gdf.crs)
            fig.plot(data=one, fill=colors[i % len(colors)], transparency=40, pen='1p,black')
        fig.plot(data=basin, pen='2p,black')                                       # basin outline
        # river network from the GMT/GSHHG river database: 1 = permanent major rivers, 2 = additional major
        # rivers, 3 = additional rivers (drawn thinner); set rivers=None to switch it off
        if rivers:
            fig.coast(rivers=list(rivers) if not isinstance(rivers, str) else rivers, resolution='i')
        # ID badges on the map (white circle with the number); name and area go to the table on the right
        for _, r in gdf.iterrows():
            c = r.geometry.representative_point()
            fig.plot(x=c.x, y=c.y, style='c0.6c', fill='white', pen='1p,black')
            fig.text(x=c.x, y=c.y, text='%d' % r.ID, font='11p,5,black', justify='CM')

        # table to the right of the map, anchored at the map's top-right corner and drawn outside the frame:
        # columns ID | name (detail in a second, smaller line) | area, right-aligned
        # row height adapted to the map height so that the table never extends below the map
        merc = lambda lat: np.log(np.tan(np.pi / 4 + np.deg2rad(lat) / 2))
        map_h = width * (merc(region[3]) - merc(region[2])) / np.deg2rad(region[1] - region[0])
        n = len(gdf)
        row_h = min(0.95, max(0.45, (map_h - 2.6) / n))
        show_detail = row_h >= 0.8
        fs = '10p' if row_h >= 0.6 else '8p'
        x0, panel_w = 0.6, 9.5                                                     # cm
        y = 0.1
        fig.text(position='TR', text='ID   Sub-basin', font='10p,5,gray20', justify='TL',
                 offset='%.2fc/-%.2fc' % (x0, y), no_clip=True)
        fig.text(position='TR', text='Area [10@+3@+ km@+2@+]', font='10p,5,gray20', justify='TR',
                 offset='%.2fc/-%.2fc' % (x0 + panel_w, y), no_clip=True)
        y += 0.65
        for _, r in gdf.iterrows():
            main, _, detail = r.NAME.partition(' (')
            detail = detail.rstrip(')')
            fig.text(position='TR', text='%d' % r.ID, font=fs + ',5,black', justify='TL',
                     offset='%.2fc/-%.2fc' % (x0, y), no_clip=True)
            fig.text(position='TR', text=main, font=fs + ',5,black', justify='TL',
                     offset='%.2fc/-%.2fc' % (x0 + 0.8, y), no_clip=True)
            fig.text(position='TR', text='%.0f' % (r.SUB_AREA / 1e3), font=fs + ',5,black', justify='TR',
                     offset='%.2fc/-%.2fc' % (x0 + panel_w, y), no_clip=True)
            if detail and show_detail:
                fig.text(position='TR', text=detail, font='8p,4,gray35', justify='TL',
                         offset='%.2fc/-%.2fc' % (x0 + 0.8, y + 0.42), no_clip=True)
            y += row_h
        fig.text(position='TR', text='total', font=fs + ',5,black', justify='TL',
                 offset='%.2fc/-%.2fc' % (x0 + 0.8, y), no_clip=True)
        fig.text(position='TR', text='%.0f' % (gdf.SUB_AREA.sum() / 1e3), font=fs + ',5,black', justify='TR',
                 offset='%.2fc/-%.2fc' % (x0 + panel_w, y), no_clip=True)
        fig.text(position='TR', text='HydroBASINS v1c, basin PFAF %s; grey: level-%d polygons; blue: GMT river network' % (basin_pfaf, ref_level),
                 font='8p,4,gray35', justify='TL', offset='%.2fc/-%.2fc' % (x0, y + 0.8), no_clip=True)

        fn = Path(fn); base = fn.with_suffix('') if fn.suffix in ('.png', '.pdf') else fn
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(base) + '.png', dpi=200)
        fig.savefig(str(base) + '.pdf')
        print('figure', str(base) + '.png/.pdf')
        if allow_pop_up:
            fig.show()


# ------------------------------------------------------------------ demos
EXT = Path('/media/user/My Book/Fan/PyGLDA_v2_external_data')


def demo_Danube():
    hb = HydroBASINS_subbasins(hybas_dir=EXT / 'Extra/hybas_eu_lev01-12_v1c', continent='eu')
    gdf = hb.build(DANUBE_SUBBASINS, out_shp=EXT / 'Basin/shp/Danube/Danube.shp')
    hb.check_completeness(gdf, basin_pfaf='227')
    hb.check_figure(gdf, basin_pfaf='227', fn=EXT / 'Extra/fig/Danube_subbasins.png',
                    title='Danube sub-basins from HydroBASINS')


def demo_Amazon():
    hb = HydroBASINS_subbasins(hybas_dir=EXT / 'Extra/hybas_sa_lev01-12_v1c', continent='sa')
    gdf = hb.build(AMAZON_SUBBASINS, out_shp=EXT / 'Basin/shp/Amazon/Amazon.shp')
    hb.check_figure(gdf, basin_pfaf='62', fn=EXT / 'Extra/fig/Amazon_subbasins.png', ref_level=4,
                    title='Amazon sub-basins from HydroBASINS')


if __name__ == '__main__':
    demo_Danube()
