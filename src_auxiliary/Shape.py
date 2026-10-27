"""
Shapefile helpers for building assimilation units:
  box2shp          a one-polygon shapefile from a lat/lon box
  basin2grid_shp   cut a basin into regular grid cells (e.g. 2 x 2 degree), drop tiny or ocean cells, merge cells
  plot_units       quick look of a unit shapefile with the unit IDs (pygmt)

    bg = basin2grid_shp(grid=(2, 2)).load_original_shp('<basin>.shp').create_shp()
    bg.delete_tiny_grids().merge_grids([1, 2]).save2file(new_basin_name='<name>', out_dir='<folder>')
    plot_units(bg.new_shp)
"""
import numpy as np
import geopandas as gpd
from shapely import box
from pathlib import Path
import shapely
import shapely.vectorized
import pandas as pd
import h5py


def box2shp(box_area=(70.1, 33.9, -11.1, 45.1)):
    """
    Generate a shp file from a box boundary.

    For example: box = [76.1, 33.9, -11.1, 45.1] ==> [up (lat), down (lat), left (lon), right (lon)]
    """

    poly = box(xmin=box_area[2], ymin=box_area[0], xmax=box_area[3], ymax=box_area[1])
    d = {'ID': [0], 'geometry': [poly]}
    gdf = gpd.GeoDataFrame(d, crs='epsg:4326')

    return gdf


class basin2grid_shp:
    """
    Given a basin, to subdivide the basin into grid by grid.
    """

    def __init__(self, grid=(3, 3)):
        self.new_shp = None

        sub_basin = grid

        N = 180 // sub_basin[0]
        M = 360 // sub_basin[1]

        num_sub_basins = N * M

        lat_lu = np.array([90 - i * sub_basin[0] for i in range(int(N))])
        lon_lu = np.array([-180 + i * sub_basin[1] for i in range(int(M))])

        lat_ru = lat_lu.copy()
        lon_ru = lon_lu + sub_basin[1]

        lat_ld = lat_lu - sub_basin[0]
        lon_ld = lon_lu.copy()

        # lat_rd = lat_lu - sub_basin[0]
        # lon_rd = lon_lu + sub_basin[1]

        # lon_lu, lat_lu = np.meshgrid(lon_lu, lat_lu)
        lon_ru, lat_ru = np.meshgrid(lon_ru, lat_ru)
        lon_ld, lat_ld = np.meshgrid(lon_ld, lat_ld)
        # lon_rd, lat_rd = np.meshgrid(lon_rd, lat_rd)

        poly = box(xmin=lon_ld, ymin=lat_ld, xmax=lon_ru, ymax=lat_ru)
        ID = [i for i in range(1, int(num_sub_basins) + 1)]
        d = {'ID': ID, 'geometry': list(poly.flatten())}
        self.gdf = gpd.GeoDataFrame(d, crs='epsg:4326')
        pass

    def load_original_shp(self, original_shp):
        if isinstance(original_shp, str):
            self.__old_shp = gpd.read_file(original_shp)
            if self.__old_shp.crs is None:
                pass
            else:
                self.__old_shp = self.__old_shp.set_crs(crs='epsg:4326')
        else:
            self.__old_shp = original_shp

        return self

    def create_shp(self):
        basin_shp = self.__old_shp

        if basin_shp.union_all().geom_type == 'MultiPolygon':
            bb = max(basin_shp.union_all().geoms, key=lambda a: a.area)
        else:
            bb = basin_shp.union_all()

        basin_all = gpd.GeoDataFrame({'geometry': gpd.GeoSeries(bb)}, crs='epsg:4326')

        grid = self.gdf

        index = shapely.intersects(bb, grid.geometry).values

        new = []

        for i in np.arange(len(index)):
            if not index[i]:
                continue

            mm = grid[grid.ID == i + 1]
            mm = mm.drop(columns=['ID'])
            new.append(mm.overlay(basin_all, how='intersection', keep_geom_type=True))

        gdf = gpd.GeoDataFrame(pd.concat(new))
        ID = [i for i in range(1, len(gdf) + 1)]
        d = {'ID': ID, 'geometry': gdf.geometry}
        new_shp = gpd.GeoDataFrame(d, crs='epsg:4326')

        self.new_shp = new_shp

        return self

    def save2file(self, new_basin_name, out_dir):
        """writes <out_dir>/<new_basin_name>/<new_basin_name>.shp"""
        dp = Path(out_dir) / new_basin_name
        dp.mkdir(parents=True, exist_ok=True)
        self.new_shp.to_file(dp / ('%s.shp' % new_basin_name))
        pass

    def delete_tiny_grids(self):
        """
        The aim of this function is to delete tiny grids
        """
        gdf = self.new_shp

        '''requirement of minimal area'''
        propotion = 8  # This is totally empirical
        gdf = gdf[gdf.area >= (gdf.area.values.max()) / propotion].copy()

        '''must not be a multiPolygon'''
        # gdf = gdf[gdf.type == 'Polygon']

        '''requirement of the centroid, e.g., must be inside a box'''
        # lon_ld = -130
        # lat_ld = 20
        # lon_ru = -65
        # lat_ru = 50
        # poly = box(xmin=lon_ld, ymin=lat_ld, xmax=lon_ru, ymax=lat_ru)
        # gdf = gdf[poly.contains(gdf.centroid)]

        '''reorganization'''
        gdf['ID'] = np.arange(start=1, stop=gdf.shape[0] + 1)

        self.new_shp = gdf

        return self

    def delete_ocean_grid(self, land_mask_fn, min_land_fraction=0.16):
        """
        Keep only the grid cells with at least min_land_fraction land, from a 0.1-degree global land mask
        (land_mask_fn: h5 file with dataset 'mask', north-up, 1801 x 3600 as the W3RA land mask; the last row is
        dropped).
        """
        with h5py.File(land_mask_fn, 'r') as h:
            mask = h['mask'][:-1, :]
        gdf = self.new_shp

        invalids = []

        for id in range(1, gdf.ID.size + 1):
            tt = gdf[gdf.ID == id]
            minx = int(float(tt.bounds.minx.iloc[0]) / 0.1) + 1800
            maxx = int(float(tt.bounds.maxx.iloc[0]) / 0.1) + 1800
            maxy = 900 - int(float(tt.bounds.miny.iloc[0]) / 0.1)
            miny = 900 - int(float(tt.bounds.maxy.iloc[0]) / 0.1)
            vv = np.sum(mask[miny:maxy + 1, minx:maxx + 1])
            tg = mask[miny:maxy + 1, minx:maxx + 1].size

            flag = True

            '''in case the land area is too small'''
            if vv / tg < min_land_fraction:
                flag = False

            invalids.append(flag)

            pass

        n = gdf[invalids].copy()
        n.loc[:, 'ID'] = np.arange(np.sum(np.array(invalids))) + 1
        self.new_shp = n
        return self

    def merge_grids(self, grid_ids: list):
        """
        Merge grids based on the given grid IDs.
        :param grid_ids: List of grid IDs to merge.
        :return: self
        """
        gdf = self.new_shp
        grids_to_merge = gdf[gdf['ID'].isin(grid_ids)]

        if not grids_to_merge.empty:
            merged_geometry = grids_to_merge.union_all()
            new_id = min(grid_ids)

            # Create a new GeoDataFrame for the merged grid
            new_row = gpd.GeoDataFrame({'ID': [new_id], 'geometry': [merged_geometry]}, crs=gdf.crs)

            # Remove old grids and add the new merged grid
            gdf = gdf[~gdf['ID'].isin(grid_ids)]
            gdf = pd.concat([gdf, new_row], ignore_index=True)

            '''sort the ID and rename ID from 1 to n'''
            gdf = gdf.sort_values(by='ID').reset_index(drop=True)
            gdf['ID'] = np.arange(1, len(gdf) + 1)  # renumber the IDs consecutively

        self.new_shp = gdf

        return self


def plot_units(units, region=None, projection="Q30/-20/12c", show=True, fig_out=None):
    """
    Quick look of a unit shapefile (path or GeoDataFrame with column ID): unit boundaries with their IDs.
    region [lon_min, lon_max, lat_min, lat_max]; default: the bounds of the units plus 5 degrees.
    """
    import pygmt

    gdf = gpd.read_file(units) if isinstance(units, (str, Path)) else units
    if region is None:
        b = gdf.union_all().bounds
        region = [b[0] - 5, b[2] + 5, b[1] - 5, b[3] + 5]

    fig = pygmt.Figure()
    pygmt.config(MAP_HEADING_OFFSET=0, MAP_TITLE_OFFSET=-0.2, FONT_ANNOT='10p', COLOR_NAN='white')
    fig.basemap(region=region, projection=projection,
                frame=['WSne', 'xa10f5+lLongitude (\\260 E)', 'ya5f5+lLatitude (\\260 N)'])
    fig.coast(shorelines="1/0.2p", region=region, projection=projection, water='lightblue')
    for uid in gdf.ID:
        xy = gdf[gdf.ID == uid].geometry.representative_point()
        fig.text(x=float(xy.x.iloc[0]), y=float(xy.y.iloc[0]), text="%s" % uid, font='7p,black')
    fig.plot(data=gdf.boundary, pen="0.5p,blue", projection=projection)
    if fig_out is not None:
        fig.savefig(str(fig_out))
    if show:
        fig.show()
    return fig


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        sys.exit('usage: python -m src_auxiliary.Shape <units.shp>   (plots the units with their IDs)')
    plot_units(sys.argv[1])
