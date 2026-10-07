"""
Global basin units for a basin-based DA, built automatically from HydroSHEDS / HydroBASINS (v1c, Lehner & Grill 2013).

The manual groupings in HydroShed.py (Danube 6 units, Amazon 18 units) are generalised to the whole land surface
with one rule on the Pfafstetter codes (the level of a code = its number of digits; all polygons of a basin share
the basin's code as prefix):

  1) split, top-down: start from the level-1 polygons of each HydroBASINS region; a unit larger than area_max, or
     longer than extent_max (bounding-box diagonal, e.g. the Chilean or Norwegian coastal strips), is replaced by all
     its level+1 polygons (Pfafstetter: 4 largest tributaries = even digits, 5 inter-basins along the main stem = odd
     digits), until every unit is within both limits (or the atom level is reached)
     Optional (configure_coherence, with open-loop storage): a unit within the limits is also split if its level-6
     polygons vary incoherently (mean r^2 with the unit-mean series < c_split; not for archipelagos)
  2) merge, bottom-up: the smallest unit below area_min is merged into a neighbour, ranked by
        - within the limits: merged area <= merge_cap x area_max and merged extent <= extent_max,
        - merge_rule 'basin' (default): the longest common Pfafstetter prefix (siblings of the same river basin
          first; a small coastal basin goes to a neighbouring basin of the same level rather than into a part of a
          large river basin), then a flow connection (NEXT_DOWN), then the smallest merged area;
          merge_rule 'hybrid' (with configure_coherence): the highest correlation of the two unit-mean series,
          +0.1 for a sibling in the same river basin, +0.05 for a flow connection
     repeated until no unit < area_min is left. Neighbours are units that share a boundary (also across HydroBASINS
     regions, e.g. 'ar'/'na'). Units without any neighbour (islands) are kept if >= area_island, dropped otherwise
  3) the level-<atom_level> polygons of each unit are dissolved into one polygon
  4) optional glacier flag (configure_glacier): GLAC_IN / GLAC_EFF [%] ice area in / seen by GRACE in a unit,
     GLACIER = 1 (>= 0.5 %), 2 (>= 5 %): WaterGAP has no glacier mass change, GRACE does

Sizes, from the GRACE resolution and signal:
  - GRACE resolution in the literature: ~200 000 km2 (Rodell & Famiglietti 1999; Longuevergne et al. 2010),
    ~150 000 km2 (Rowlands et al. 2005), ~63 000 km2 at 2 cm error (Vishwakarma et al. 2018, Remote Sens. 10, 852;
    error inversely related to the area); mascons are solved on 3-deg caps (~300 km, JPL; Watkins et al. 2015)
  - area_min 6.3e4 km2 is the floor (Vishwakarma); configure_signal sets the size where it matters: a unit grows until
    SNR = std(unit-mean open-loop TWS) / sigma_noise(A) >= snr_min (3), sigma_noise = 20 mm (63 000 km2 / A)^0.5 or
    from GRACE error samples; strong-signal units are split (SNR >= 2 snr_min). Small units where TWS varies strongly
    (Amazon, Congo, monsoon Asia), large ones (up to merge_cap x area_max = 7.5e5 km2) in deserts
  - islands (land masses) < 1e5 km2 are dropped: below the GRACE resolution and surrounded by ocean signal; larger
    islands are dropped if their SNR stays < snr_min (they cannot merge with anything)
  - configure_glacier(remove=0.10): units whose GRACE signal contains >= 10 % ice area (GLAC_EFF) are dropped: the
    Canadian Arctic, Alaska coast, Iceland, Novaya Zemlya, Karakoram / western Tibet (WaterGAP has no glacier loss)
  - extent_max 2000 km keeps units compact
  - configure_model: the units on the WaterGAP 0.5 deg grid (a cell goes to the unit covering most of it); units
    with < 15 model cells or > 25 % global/regulated lake area (Great Lakes) are removed. The DA mask itself is
    written by global_shp2mask.py (same rules and format as shp2mask.py)

Result (v1c, 8 regions without 'gr'; signal CSR mascons 2002-2024, noise SaGEA DDK3 samples (3000), SNR >= 3;
hybrid coherence rule, c_split 0.8; river basins as a hard rule; neighbours only along >= 0.05 deg of shared boundary;
closed and open basins not grouped (same_drainage); land masses < 1e5 km2 (383 islands, incl. Cyprus, Shikoku,
Vancouver Island, Tierra del Fuego, Zealand) taken out before the split; islands below SNR 3 removed except Great
Britain and North / South Island NZ (snr_keep); units below SNR 1 removed (Rub al Khali, western Sahara coast, Namib);
glaciers >= 10 % (or >= 50 000 km2 and >= 5 %) and lake units > 25 % removed; WaterGAP grid): 772 units, area
p10/median/p90 71/112/232e3 km2, every unit one connected area. Every unit is part of one river basin (459 units,
48 % of the area), one whole basin (32, 5 %), a group of whole non-major basins of one drainage type (267, 44 %: coastal
strips and closed basins, e.g. the western Sahara 4.7e6 km2) or one basin + small coastal basins (14, 3 %); no unit
mixes parts of two rivers. 19 desert units (12e6 km2) stay below SNR 3. GRACE DDK3 noise of a unit mean falls as
~A^-0.21 (16.6 mm at 63 000 km2, 10.8 mm at 500 000 km2), ~15 % higher in the tropics and 20-40 % lower north of 50 N.
Coherence vs the same number of lat-lon boxes: R2 0.709 vs 0.704, C 0.636 vs 0.660. 3-degree boxes over the same
area: ~2000.

Output (written by save()):
  <out_dir>/<name>.shp          ID (1..N), NAME, PFAF, SUB_AREA [km2], REGION, LEVEL, N_PFAF, ENDO, COAST, MAIN_BAS
                                (+ SIGNAL [mm], SNR with configure_signal; GLAC_IN, GLAC_EFF, GLACIER with configure_glacier;
                                N_CELLS, MOD_AREA [km2], COVER, LAKE_FRAC, WET_FRAC with configure_model)
  <out_dir>/<name>_units.csv    the same table without geometry, with the full PFAF lists (the .shp field is cut at 254 chars)
  <out_dir>/<name>_dropped.csv  removed units with REASON (island < area_island, glacier, model); <name>_removed.shp
  (the mask: global_shp2mask.py -> Basin/mask/<name>/<name>_res_0.5.h5)
  <out_dir>/<name>.png          quick-look map (+ <name>_glacier.png)
  <out_dir>/<name>_coherence.csv  evaluate(): units vs lat-lon boxes
PFAF: the HydroBASINS codes of the unit (mixed levels, comma separated; complete sets of children are written as
their parent), i.e. the same convention as Basin/shp/<case>/<case>.shp, so that HydroShed.py can rebuild any unit.

Input: HydroBASINS v1c 'lev01-12' downloads per region (af, ar, as, au, eu, gr, na, sa, si) in hybas_dir, either
unzipped (hybas_<r>_lev01-12_v1c/) or as zip (hybas_<r>_lev01-12_v1c.zip, read without unzipping).
Greenland ('gr') is used only if present; Antarctica is not in HydroBASINS.
Ice: Extra/glacier/ne_10m_glaciated_areas.shp (Natural Earth 1:10m physical, public domain, from
github.com/nvkelso/natural-earth-vector); RGI 7.0 (NSIDC-0770, Earthdata login) is more accurate for mountain glaciers.
"""
import sys
import heapq
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REGIONS = ('af', 'ar', 'as', 'au', 'eu', 'gr', 'na', 'sa', 'si')


class hydrobasins_global:

    def __init__(self, hybas_dir, regions=REGIONS):
        """hybas_dir: folder with the HydroBASINS v1c downloads; regions: HydroBASINS regions to use (missing ones are skipped)"""
        self.hybas_dir = Path(hybas_dir)
        self.regions = [r for r in regions if self._source(r) is not None]
        missing = [r for r in regions if r not in self.regions]
        if missing:
            print('HydroBASINS regions not found in %s: %s' % (self.hybas_dir, ', '.join(missing)))

        '''defaults'''
        self.area_min, self.area_max = 1.0e5, 5.0e5            # km2
        self.area_island = 1.0e5                                # km2, smaller isolated units (islands) are dropped
        self.extent_max = 2000.0                                # km, bounding-box diagonal (splits long coastal strips)
        self.merge_cap = 1.5                                    # prefer merges that stay <= merge_cap * area_max
        self.atom_level = 6                                     # finest HydroBASINS level used (polygons, flow links)
        self.out_dir, self.name = None, 'GlobalBasins'
        self.glacier_shp = None                                 # glacier / ice-sheet polygons (configure_glacier)
        self.field = None                                       # TWS anomalies for the coherence rule (configure_coherence)
        self.merge_rule, self.c_split = 'basin', None
        self.basin_first = True                                 # river basins as a hard rule (configure_basin_rule)
        self.min_shared = 0.05                                  # minimum shared boundary of neighbours, deg (configure_basin_rule)
        self.same_drainage, self.protect_signal = True, False   # (configure_basin_rule)
        self.sig_field = None                                   # TWS for the signal-to-noise rule (configure_signal)
        self.snr_drop = None                                    # units below this SNR are removed (configure_signal)
        self.snr_keep = set()                                   # unit NAMEs kept despite a low SNR (configure_signal)
        self.glac_remove = None
        self.static_dir, self.label = None, None                # WaterGAP static input (configure_model)

        self.atoms = None                                       # GeoDataFrame of the level-<atom_level> polygons
        self.units = None                                       # GeoDataFrame of the final units
        self.dropped = None
        pass

    # ------------------------------------------------------------------ configuration
    def configure_size(self, area_min=1.0e5, area_max=5.0e5, area_island=1.0e5, extent_max=2000.0, merge_cap=1.5):
        """
        target unit size: units are split above area_max [km2] or extent_max [km, bounding-box diagonal] and
        merged below area_min [km2]; isolated units (islands) below area_island [km2] are dropped (GRACE resolves
        ~1-2e5 km2; an island is also surrounded by ocean signal leaking in)
        """
        assert area_max > area_min > 0 and area_island >= 0
        self.area_min, self.area_max, self.area_island, self.merge_cap = area_min, area_max, area_island, merge_cap
        self.extent_max = extent_max
        return self

    def configure_atom_level(self, level=6):
        """finest HydroBASINS level: the units are unions of these polygons (level 6: ~16 000 polygons, median ~5000 km2)"""
        assert 3 <= level <= 12
        self.atom_level = level
        return self

    def configure_signal(self, field, lat, lon, snr_min=3.0, snr_split=None, sigma0=20.0, a0=6.3e4, beta=0.5,
                         noise_atoms=None, debias=False, area_max_low=1.0e7, extent_max_low=8000.0, snr_drop=None,
                         snr_keep=()):
        """
        unit size adapted to the strength of the TWS signal: a unit is large enough when its signal-to-noise ratio
        SNR = std(unit-mean TWS of the open loop) / sigma_noise(unit) reaches snr_min, so units are small where TWS varies
        strongly (monsoon, Amazon) and large where it hardly varies (deserts, ice-free Arctic); area_min stays a floor
        and area_max (x merge_cap) a ceiling.
          field[time, lat, lon]: TWS whose variability is the signal, on a grid with lat from north to south:
                     GRACE itself (load_grace_mascon: CSR mascons, 2002-2024; includes the trends the model lacks) or
                     the open loop (load_ol_field(..., storages=('tws',)))
          snr_min:   SNR a unit must reach (merging continues until it does, within the area ceiling)
          snr_split: a unit within the limits is split into its Pfafstetter children if its SNR >= snr_split
                     (default 2 snr_min: the children can reach snr_min on their own)
          noise:     sigma_noise(A) = sigma0 (a0 / A)^beta [mm]; default anchored at 20 mm for 63 000 km2 (Vishwakarma
                     et al. 2018, Remote Sens. 10, 852) with the 1/sqrt(A) scaling of averaging; or
          area_max_low, extent_max_low: units that stay below snr_min at area_max x merge_cap (deserts) are aggregated
                     further, up to this area [km2] / bounding-box diagonal [km], preferably among themselves
          noise_atoms: .npz written by export_atom_noise (GRACE error samples, e.g. the SaGEA DDK3 Monte-Carlo
                     samples, averaged over every level-6 polygon): sigma_noise(unit) = std over the samples of the
                     unit mean, with the full spatial correlation and the latitude dependence of the GRACE errors
          debias:    the signal is GRACE (signal + noise): use sqrt(std^2 - sigma_noise^2) as signal
          snr_drop:  units that end below this SNR (e.g. 2.0) are removed: the river-basin rule or the ceiling kept
                     them from growing, and GRACE cannot separate their signal from the noise. Islands (no neighbour
                     to merge with) that stay below snr_min are always removed (Great Britain, New Zealand, Sulawesi)
          snr_keep:  unit NAMEs kept despite a low SNR (e.g. ('eu_233',): Great Britain, a common GRACE study region)
        """
        f = np.asarray(field, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.sig_field = f - np.nanmean(f, axis=0)
        self.s_lat, self.s_lon = np.asarray(lat), np.asarray(lon)
        self.snr_min = snr_min
        self.snr_split = 2.0 * snr_min if snr_split is None else snr_split
        self.sigma0, self.a0, self.beta = sigma0, a0, beta
        self.area_max_low, self.extent_max_low = area_max_low, extent_max_low
        self.noise_atoms, self.debias = noise_atoms, debias
        self.snr_drop = snr_drop
        self.snr_keep = set(snr_keep)
        return self

    def export_atom_noise(self, sample_files, out_npz, south_first=True, scale=1000.0, n_max=500, chunk=50):
        """
        reduce GRACE error samples to the level-6 polygons (run where the samples are; a few minutes per GB):
          sample_files: .npy files [n_sample, lat, lon] (any global grid, e.g. 1 deg: [2000, 180, 360]),
                        e.g. SaGEA/sample_DDK3/<YYYY-MM>_0.npy; use several
                        months (GRACE and GRACE-FO, different orbit configurations): the noise is the rms over months
          south_first:  the sample grid runs from -90 to 90 (as SaGEA, see prepare_GRACE.basin_COV); lon -180..180
          scale:        to mm (SaGEA samples are in m)
          n_max:        samples used per file (500 x 6 months x 16 000 polygons ~ 200 MB)
        writes out_npz (keys, w, noise[atom, sample] float32, mean removed per month) for configure_signal(noise_atoms=)
        """
        if self.atoms is None:
            self._load_tables()
        blocks = []
        M = None
        for fn in sample_files:
            arr = np.load(fn, mmap_mode='r')
            n = min(arr.shape[0], n_max)
            if arr.ndim == 3:
                nlat, nlon = arr.shape[1:]
            else:                                                  # [n, ncell]: grid from the cell number
                nlat = int(round(np.sqrt(arr[0].size / 2)))
                nlon = 2 * nlat
                arr = arr.reshape(arr.shape[0], nlat, nlon)
            res = 180.0 / nlat
            assert nlon * res == 360.0, '%s: %s is not a global grid' % (Path(fn).name, arr.shape)
            lat = np.arange(90 - res / 2, -90, -res)
            lon = np.arange(-180 + res / 2, 180, res)
            if M is None:
                print('  samples on a %g-degree grid (%d x %d)' % (res, nlat, nlon))
                first = np.asarray(arr[0])[::-1] if south_first else np.asarray(arr[0])
                M, has = self._atom_matrix(lat, lon, np.isfinite(first))
            out = np.zeros((len(self.atoms), n), dtype=np.float32)
            for i0 in range(0, n, chunk):
                x = np.asarray(arr[i0:min(i0 + chunk, n)], dtype=np.float64)
                if south_first:
                    x = x[:, ::-1, :]
                x = np.nan_to_num(x.reshape(len(x), -1)) * scale
                out[:, i0:i0 + len(x)] = np.asarray(M @ x.T)
            out -= out.mean(axis=1, keepdims=True)
            blocks.append(out)
            print('  %s: %d samples, median polygon noise %.1f mm' % (Path(fn).name, n, np.median(out.std(axis=1))))
        noise = np.hstack(blocks)
        np.savez(out_npz, keys=np.array(list(self.atoms.KEY), dtype='U20'),
                 w=np.where(has, self.atoms.SUB_AREA.values.astype(np.float64), 0.0),
                 noise=noise, files=np.array([Path(f).name for f in sample_files], dtype='U200'))
        print('written %s: %d polygons x %d samples' % (out_npz, noise.shape[0], noise.shape[1]))
        pass

    @staticmethod
    def load_grace_mascon(nc, begin='2002-04', end='2024-12'):
        """
        CSR GRACE/GRACE-FO RL06.x mascons (0.25 deg, cm, lat -90..90, lon 0..360) -> field[month, lat, lon] in mm on
        the 0.5 deg grid (lat 89.75 .. -89.75, lon -179.75 .. 179.75), with lat, lon
        """
        import xarray as xr
        with xr.open_dataset(nc, decode_times=False) as d:
            t0 = pd.Timestamp('2002-01-01')
            dates = t0 + pd.to_timedelta(d.time.values.astype(float), unit='D')
            sel = (dates >= pd.Timestamp(begin)) & (dates <= pd.Timestamp(end) + pd.offsets.MonthEnd(0))
            f = d.lwe_thickness.values[sel].astype(np.float64) * 10.0
        n = len(f)
        f = f.reshape(n, 360, 2, 720, 2).mean(axis=(2, 4))[:, ::-1, :]            # 0.5 deg, north first
        f = np.roll(f, 360, axis=2)                                               # lon 0..360 -> -180..180
        print('GRACE mascons: %d months %s .. %s from %s' % (n, dates[sel][0].strftime('%Y-%m'),
                                                              dates[sel][-1].strftime('%Y-%m'), Path(nc).name))
        return f, np.arange(89.75, -90, -0.5), np.arange(-179.75, 180, 0.5)

    def configure_glacier(self, shp, sigma=200.0, minor=0.005, major=0.05, remove=None, remove_area=5.0e4):
        """
        glacier flag of the units (WaterGAP has no glacier / ice-sheet mass change, GRACE does):
          shp:   glacier and ice-sheet polygons, one file or a list: Natural Earth 1:10m 'glaciated areas' (default,
                 includes the Greenland ice sheet; generalised, ~1.5x the RGI area in the Alps/Alaska), or better
                 the 19 regional RGI 7.0 files (NSIDC-0770) + a Greenland ice-sheet outline
          sigma: GRACE smoothing [km], Gaussian; ice outside a unit leaks in with 0.5*erfc(d / (sqrt(2) sigma)),
                 d = distance to the unit (half of the signal at the boundary, 0.1 % at 3 sigma)
          remove: units with GLAC_EFF >= remove are taken out (e.g. 0.10: the major glaciated regions whose GRACE trend
                 is dominated by ice loss); None = keep all, only flag
          remove_area: also taken out if the ice area GRACE sees (GLAC_EFF x area) >= remove_area km2 and GLAC_EFF >=
                 major, so that a large ice field is not kept only because it was aggregated into a large unit (a
                 plateau unit with a few % of ice stays)
          minor, major: thresholds on GLAC_EFF (ice area seen by GRACE / unit area) for GLACIER = 1, 2;
                 at ~0.5 m w.e./yr glacier loss, 0.5 % ~ 2.5 mm/yr and 5 % ~ 25 mm/yr apparent TWS trend
        """
        self.glac_remove, self.glac_remove_area = remove, remove_area
        shp = [shp] if isinstance(shp, (str, Path)) else list(shp)
        self.glacier_shp = [Path(f) for f in shp]
        self.sigma, self.glac_minor, self.glac_major = sigma, minor, major
        return self

    def configure_coherence(self, field, lat, lon, merge_rule='coherence', c_split=None):
        """
        TWS (or any storage) of the open loop, field[time, lat, lon] on the model grid, to build units whose cells vary
        together (the unit mean - what GRACE sees - then represents every cell of the unit):
          merge_rule: 'basin'     - neighbour ranked by Pfafstetter kinship (default without a field)
                      'coherence' - neighbour with the highest correlation of the unit-mean series
                      'hybrid'    - correlation + 0.1 for a sibling in the same river basin + 0.05 for a flow link
          c_split:    split a unit (> 2 area_min) whose coherence (mean r^2 of its atoms with the unit mean) is below
                      c_split into its Pfafstetter children (they are then merged again by merge_rule); None = off
        """
        f = np.asarray(field, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')                    # all-NaN cells (sea) -> NaN
            self.field, self.f_lat, self.f_lon = f - np.nanmean(f, axis=0), np.asarray(lat), np.asarray(lon)
        self.merge_rule, self.c_split = merge_rule, c_split
        return self

    def configure_model(self, static_dir, n_cells_min=15, lake_max=0.25, cover_range=(0.5, 2.0)):
        """
        put the units on the WaterGAP 2.2e grid (static_dir = Input_data/static_input: continental area, lake and
        wetland fractions) and select them by model criteria (see model_grid)
        """
        self.static_dir, self.n_cells_min, self.lake_max, self.cover_range = Path(static_dir), n_cells_min, lake_max, \
            tuple(cover_range)
        return self

    def configure_basin_rule(self, basin_first=True, min_shared=0.05, same_drainage=True, protect_signal=False):
        """
        same_drainage: closed (endorheic, HydroBASINS ENDO > 0) and open basins (draining to the sea) are not grouped;
                       only small pieces of the other type (< area_min / 2 in total) may attach. E.g. the Hai and Luan
                       (Bohai Sea) are not absorbed by the Gobi closed basins, the Atlantic rivers of Morocco not by
                       the Sahara, the Altiplano not by the Pacific coast
        protect_signal: a unit that already reaches snr_min is never absorbed by a low-signal unit that grows to reach
                       snr_min; low-signal units are aggregated only among themselves
        min_shared: two polygons are neighbours only if they share a boundary of at least this length (degrees,
        0.05 deg ~ 5 km); polygons that touch in a single point or along a few metres are not connected, so a unit
        is never two separate areas joined at a corner
        basin_first: river basins (HydroBASINS MAIN_BAS) are a hard rule in the merge step: a unit is either part of
        one river basin, or a group of whole basins none of which is a major river (> area_max), or one main basin
        plus whole small basins (< area_min, e.g. coastal strips at a river mouth). Parts of two different rivers are
        never merged, and a major river is never merged with another basin; signal and coherence only choose among
        these merges. A unit that cannot reach snr_min within its river keeps its size (the river stays whole).
        False: river basins are only a preference (Pfafstetter kinship in the ranking)
        """
        self.basin_first = basin_first
        self.min_shared = min_shared
        self.same_drainage, self.protect_signal = same_drainage, protect_signal
        return self

    def configure_output(self, out_dir, name='GlobalBasins'):
        self.out_dir, self.name = Path(out_dir), name
        return self

    # ------------------------------------------------------------------ input
    def _source(self, region):
        d = self.hybas_dir / f'hybas_{region}_lev01-12_v1c'
        if (d / f'hybas_{region}_lev01_v1c.dbf').exists():
            return d
        z = self.hybas_dir / f'hybas_{region}_lev01-12_v1c.zip'
        if z.exists():
            return z
        return None

    def _read(self, region, lev, geometry=False):
        import pyogrio
        src = self._source(region)
        fn = f'hybas_{region}_lev{lev:02d}_v1c'
        path = f'zip://{src}!{fn}.shp' if src.suffix == '.zip' else str(src / (fn + ('.shp' if geometry else '.dbf')))
        df = pyogrio.read_dataframe(path, read_geometry=geometry)
        df['PFAF'] = df.PFAF_ID.astype(np.int64).astype(str)
        df['REGION'] = region
        df['KEY'] = region + df.PFAF                            # PFAF codes are not unique across regions ('ar' has '353...' of 'si')
        return df

    def _load_tables(self):
        """attribute tables of levels 1..atom_level (KEY = region + PFAF -> area, children) and the atom polygons"""
        import geopandas as gpd
        self.area, self.children, self.top = {}, {}, []
        atoms = []
        for r in self.regions:
            for lev in range(1, self.atom_level + 1):
                geo = lev == self.atom_level
                df = self._read(r, lev, geometry=geo)
                for p, a in zip(df.KEY, df.SUB_AREA):
                    self.area[p] = float(a)
                    if lev > 1:
                        self.children.setdefault(p[:-1], []).append(p)
                if lev == 1:
                    self.top += list(df.KEY)
                if geo:
                    atoms.append(df)
            print('  %s: %d level-%d polygons, %.2f x10^6 km2' % (r, len(atoms[-1]), self.atom_level,
                                                                  atoms[-1].SUB_AREA.sum() / 1e6))
        a = gpd.GeoDataFrame(pd.concat(atoms, ignore_index=True), crs='EPSG:4326')
        a.geometry = a.geometry.make_valid()

        '''polygons at the date line: some reach beyond 180 E and have a wrong SUB_AREA (Fiji, au 552000: 175 335
        instead of ~18 000 km2) -> clip to -180..180 and use the geodesic area'''
        from pyproj import Geod
        import shapely
        b = a.geometry.bounds
        dl = np.flatnonzero(((b.maxx > 180) | (b.minx < -180)).values)
        geod = Geod(ellps='WGS84')
        for k in dl:
            g = shapely.make_valid(a.geometry.iloc[k].intersection(shapely.box(-180, -90, 180, 90)))
            area = abs(geod.geometry_area_perimeter(g)[0]) / 1e6
            if abs(area - a.SUB_AREA.iloc[k]) > 0.1 * area:
                print('  %s %s at the date line: SUB_AREA %.0f -> %.0f km2' % (a.REGION.iloc[k], a.PFAF.iloc[k],
                                                                         a.SUB_AREA.iloc[k], area))
                a.loc[a.index[k], 'SUB_AREA'] = area
            a.loc[a.index[k], 'geometry'] = g

        '''land masses: atoms connected by shared boundaries (>= min_shared). HydroBASINS puts islands into the
        Pfafstetter code of the nearby coast (Cyprus in the Anatolian coast, Shikoku, Vancouver Island, Tierra del
        Fuego), so a unit could be two separate areas. Land masses below area_island are taken out here (same rule as
        for isolated units: below the GRACE resolution, surrounded by ocean signal); the codes keep only their
        remaining atoms. Small islands inside a mainland atom polygon stay with it.'''
        i, j = self._shared_pairs(a.geometry, self.min_shared)
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        n_comp, comp = connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(len(a), len(a))),
                                            directed=False)
        land = pd.Series(a.SUB_AREA.values).groupby(comp).sum()
        small = np.isin(comp, land.index[land < self.area_island])
        rows = []
        for c, g in a[small].groupby(comp[small]):
            k = int(np.argmax(g.SUB_AREA.values))
            rows.append(dict(NAME='%s_%s' % (g.REGION.iloc[k], g.PFAF.iloc[k]), PFAF=','.join(g.PFAF),
                             SUB_AREA=float(g.SUB_AREA.sum()), REGION=','.join(sorted(set(g.REGION))),
                             N_PFAF=len(g), COAST=1, MAIN_BAS=int(g.MAIN_BAS.iloc[k]),
                             REASON='island < %.0f km2' % self.area_island,
                             geometry=shapely.union_all(g.geometry.values)))
        self._island_rows = rows
        print('  land masses: %d, %d islands below %.0f km2 taken out (%d polygons, %.2f x10^6 km2)'
              % (n_comp, len(rows), self.area_island, small.sum(), a.SUB_AREA[small].sum() / 1e6))
        keep = np.flatnonzero(~small)
        new = np.full(len(a), -1)
        new[keep] = np.arange(len(keep))
        ok = ~small[i] & ~small[j]
        a = a.iloc[keep].reset_index(drop=True)
        self._pairs_cache = ((len(a), self.min_shared), (new[i[ok]], new[j[ok]]))
        self.atoms = a

        '''areas of all codes as sums of their atoms (consistent with the corrected atoms); codes without atoms
        (islands taken out) are dropped from the tree'''
        self.area = {}
        for n in range(1, self.atom_level + 1):
            self.area.update(a.groupby(a.KEY.str[:n + 2].values).SUB_AREA.sum().to_dict())
        land = comp[keep]
        self._n_land = {}
        for n in range(1, self.atom_level + 1):
            self._n_land.update(pd.Series(land).groupby(a.KEY.str[:n + 2].values).nunique().to_dict())
        self.top = [p for p in self.top if p in self.area]
        self.children = {p: [c for c in ch if c in self.area] for p, ch in self.children.items() if p in self.area}
        self.children = {p: ch for p, ch in self.children.items() if ch}

        '''bounding box of every code (from its atoms); longitudes also in 0..360 for units across the date line'''
        b = a.geometry.bounds
        box = pd.DataFrame({'x0': b.minx, 'x1': b.maxx, 'y0': b.miny, 'y1': b.maxy,
                            'u0': b.minx % 360, 'u1': np.where(b.maxx < 0, b.maxx + 360, b.maxx)})
        box.loc[box.u1 < box.u0, ['u0', 'u1']] = [0.0, 360.0]   # atom itself across 0 E in 0..360
        self.box = {}
        for n in range(1, self.atom_level + 1):
            g = box.groupby(a.KEY.str[:n + 2].values)
            agg = pd.concat([g[['x0', 'y0', 'u0']].min(), g[['x1', 'y1', 'u1']].max()], axis=1)
            self.box.update({p: v for p, v in zip(agg.index, agg[['x0', 'x1', 'y0', 'y1', 'u0', 'u1']].values)})
        pass

    @staticmethod
    def _extent(bx):
        """diagonal [km] of a bounding box (x0, x1, y0, y1, u0, u1)"""
        dx = min(bx[1] - bx[0], bx[5] - bx[4])
        lat = np.deg2rad(0.5 * (bx[2] + bx[3]))
        return float(np.hypot(dx * 111.32 * np.cos(lat), (bx[3] - bx[2]) * 110.57))

    @staticmethod
    def _join(b1, b2):
        return np.array([min(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), max(b1[3], b2[3]),
                         min(b1[4], b2[4]), max(b1[5], b2[5])])

    # ------------------------------------------------------------------ 1) split
    def _split(self):
        units, stack = [], list(self.top)
        while stack:
            p = stack.pop()
            big = self.area[p] > self.area_max or self._extent(self.box[p]) > self.extent_max \
                or self._n_land.get(p, 1) > 1                           # two land masses (islands) in one code
            if not big and self.c_split is not None and self.area[p] > 2 * self.area_min and self._connected(p):
                big = self._coherence_of(p) < self.c_split            # (archipelagos are not split: isolated pieces)
            if not big and self.sig_field is not None and self.area[p] > 2 * self.area_min and self._connected(p):
                big = self._snr_of(p) >= self.snr_split               # strong signal: smaller units are resolved
            if big and len(p) - 2 < self.atom_level and p in self.children:
                stack.extend(self.children[p])
            else:
                units.append(p)
        return units

    # ------------------------------------------------------------------ 2) merge
    def _graph(self, units):
        """atom -> unit, unit adjacency (shared boundary) and flow links (NEXT_DOWN) between units"""
        a = self.atoms
        code_unit = {p: i for i, p in enumerate(units)}
        unit_of = np.empty(len(a), dtype=np.int64)
        for k, p in enumerate(a.KEY):
            for n in range(len(p), 2, -1):                     # longest prefix that is a unit
                if p[:n] in code_unit:
                    unit_of[k] = code_unit[p[:n]]
                    break
            else:
                raise ValueError(f'atom {p} is not in any unit')

        '''shared boundaries: polygons with a common edge of at least min_shared (not a single point)'''
        i, j = self._shared_pairs(a.geometry, self.min_shared)
        share = set(zip(i.tolist(), j.tolist()))
        ui, uj = unit_of[i], unit_of[j]
        keep = ui != uj
        adj = [set() for _ in units]
        for x, y in zip(ui[keep], uj[keep]):
            adj[x].add(y)

        '''flow links: atom -> NEXT_DOWN atom (only between atoms with a shared boundary)'''
        idx = pd.Series(np.arange(len(a)), index=a.HYBAS_ID.values)
        nd = a.NEXT_DOWN.values
        has = nd != 0
        src = np.flatnonzero(has)
        dst = idx.reindex(nd[has]).values
        ok = ~np.isnan(dst)
        flow = [set() for _ in units]
        for s0, d0 in zip(src[ok], dst[ok].astype(np.int64)):
            x, y = unit_of[s0], unit_of[d0]
            if x != y and (int(s0), int(d0)) in share:
                flow[x].add(y)
                flow[y].add(x)
                adj[x].add(y)
                adj[y].add(x)
        return unit_of, adj, flow

    @staticmethod
    def _common(p, q):
        n = 0
        for x, y in zip(p, q):
            if x != y:
                break
            n += 1
        return n

    def _merge(self, units):
        unit_of, adj, flow = self._graph(units)
        n = len(units)
        codes = [[p] for p in units]
        area = np.array([self.area[p] for p in units])
        main = list(units)                                     # code of the largest member (Pfafstetter identity)
        box = [self.box[p] for p in units]
        alive = np.ones(n, dtype=bool)
        parent = np.arange(n)
        cap = self.merge_cap * self.area_max

        '''river-basin bookkeeping (HydroBASINS MAIN_BAS of every atom): area of each basin inside each unit'''
        mb = self.atoms.MAIN_BAS.values
        sa = self.atoms.SUB_AREA.values
        bas_total = pd.Series(sa).groupby(mb).sum().to_dict()
        bdict = [dict() for _ in range(n)]
        for u_, b_, a_ in zip(unit_of, mb, sa):
            bdict[u_][b_] = bdict[u_].get(b_, 0.0) + a_
        major = self.area_max                                   # a basin larger than this is a 'major river'
        endo = pd.Series(sa * (self.atoms.ENDO.values > 0)).groupby(mb).sum()
        bas_endo = (endo / pd.Series(sa).groupby(mb).sum() > 0.5).to_dict()   # basin drains to a closed sink

        def basin_check(D):
            """2: hydrologically clean (part of one basin only, or whole basins of which none is a major river);
            1: one main basin (part of a basin, or a whole major river) + whole small basins < area_min (coastal
            strips at a river mouth); 0: not allowed (parts of two basins, or two major rivers)"""
            part = [b for b, a in D.items() if a < 0.999 * bas_total[b]]
            whole_major = [b for b, a in D.items() if a >= 0.999 * bas_total[b] and bas_total[b] > major]
            if len(D) == 1:
                return 2
            if self.same_drainage:                              # closed and open basins are not grouped, except
                a_endo = sum(a for b, a in D.items() if bas_endo[b])   # small pieces (< area_min / 2 in total)
                if min(a_endo, sum(D.values()) - a_endo) >= 0.5 * self.area_min:
                    return 0
            if not part and not whole_major:
                return 2
            mains = part + whole_major
            if len(mains) == 1 and sum(a for b, a in D.items() if b != mains[0]) < self.area_min:
                return 1
            return 0

        if self.field is not None:                             # unit sums of area x series (for the correlation)
            sx = np.zeros((n, self.series.shape[1]))
            sw = np.zeros(n)
            np.add.at(sx, unit_of, self.series_w[:, None] * self.series)
            np.add.at(sw, unit_of, self.series_w)

        sig = self.sig_field is not None
        if sig:                                                # unit sums of area x TWS (and x noise samples)
            ss = np.zeros((n, self.sig_series.shape[1]))
            sv = np.zeros(n)
            np.add.at(ss, unit_of, self.sig_w[:, None] * self.sig_series)
            np.add.at(sv, unit_of, self.sig_w)
            if self.noise_series is not None:
                sn = np.zeros((n, self.noise_series.shape[1]))
                np.add.at(sn, unit_of, self.sig_w[:, None] * self.noise_series)

        def noise_of(k):
            return (np.std(sn[k] / sv[k]) if self.noise_series is not None
                    else self.sigma0 * (self.a0 / area[k]) ** self.beta)

        def snr(k):
            if not sig or sv[k] <= 0:
                return np.inf
            return self._ratio(np.std(ss[k] / sv[k]), noise_of(k))

        def small(k):
            return area[k] < self.area_min or snr(k) < (self.snr_min if sig else -1)

        def prio(k):
            return snr(k) if sig else area[k]

        heap = [(prio(k), k) for k in range(n) if small(k)]
        heapq.heapify(heap)
        isolated, n_merge, n_low, n_basin = [], 0, 0, 0
        while heap:
            a_k, k = heapq.heappop(heap)
            if not alive[k] or a_k != prio(k) or not small(k):
                continue
            cand = [m for m in adj[k] if alive[m]]
            if not cand:
                isolated.append(k)
                continue
            if self.basin_first:
                '''river basins are a hard rule: only merges that keep the unit hydrologically clean'''
                chk = {m: basin_check({b: bdict[k].get(b, 0.0) + bdict[m].get(b, 0.0)
                                       for b in set(bdict[k]) | set(bdict[m])}) for m in cand}
                best_level = max(chk.values())
                if best_level == 0:
                    n_basin += 1
                    continue
                cand = [m for m in cand if chk[m] == best_level]
            '''rank the neighbours: within the size limits, same river basin (longest common Pfafstetter prefix;
            a sibling unit before a part of another basin), flow connection, small merged area.
            A unit that is large enough but below snr_min (deserts) is aggregated further, up to area_max_low and
            extent_max_low, preferably with neighbours that are also below snr_min (the units with a usable signal
            are not diluted)'''
            low = sig and area[k] >= self.area_min
            if low and self.protect_signal:
                cand = [m for m in cand if snr(m) < self.snr_min]   # never dilute a unit with a usable signal
                if not cand:
                    n_low += 1
                    continue
            c_area = self.area_max_low if low else cap
            c_ext = self.extent_max_low if low else self.extent_max

            def score(m):
                merged = area[k] + area[m]
                too_big = merged > c_area or self._extent(self._join(box[k], box[m])) > c_ext
                good = low and snr(m) >= self.snr_min
                common, sibling = max(((self._common(p, q), self._common(p, q) >= max(len(p), len(q)) - 1)
                                     for p in codes[k] for q in codes[m]))
                if self.merge_rule == 'basin' or self.field is None:
                    return (too_big, good, -common, not sibling, m not in flow[k], merged)
                r = self._corr(sx[k] / max(sw[k], 1e-9), sx[m] / max(sw[m], 1e-9))
                if self.merge_rule == 'coherence':
                    return (too_big, good, -r, -common, merged)
                return (too_big, good, -(r + 0.1 * sibling + 0.05 * (m in flow[k])), merged)       # hybrid
            m = min(cand, key=score)
            if score(m)[0] and area[k] >= self.area_min:       # only too-big merges left: keep k (low SNR)
                n_low += 1
                continue
            if self.field is not None:
                sx[m] += sx[k]
                sw[m] += sw[k]
            if sig:
                ss[m] += ss[k]
                sv[m] += sv[k]
                if self.noise_series is not None:
                    sn[m] += sn[k]

            '''merge k into m'''
            for b_, a_ in bdict[k].items():
                bdict[m][b_] = bdict[m].get(b_, 0.0) + a_
            codes[m] += codes[k]
            if area[k] > area[m]:
                main[m] = main[k]
            area[m] += area[k]
            box[m] = self._join(box[k], box[m])
            alive[k] = False
            parent[k] = m
            for x in adj[k]:
                adj[x].discard(k)
                if x != m:
                    adj[x].add(m)
                    adj[m].add(x)
            for x in flow[k]:
                flow[x].discard(k)
                if x != m:
                    flow[x].add(m)
                    flow[m].add(x)
            adj[m].discard(m)
            flow[m].discard(m)
            n_merge += 1
            if small(m):
                heapq.heappush(heap, (prio(m), m))

        '''resolve the atom -> final unit map'''
        def root(k):
            while parent[k] != k:
                k = parent[k]
            return k
        final = np.array([root(k) for k in range(n)])
        print('  merged %d units, %d isolated units (islands) left small, %d units below snr_min at the area ceiling%s'
              % (n_merge, len(isolated), n_low, ', %d merges refused by the river-basin rule' % n_basin
                 if self.basin_first else ''))
        self._bas_total = bas_total
        self._unit_snr = {k: snr(k) for k in range(n) if alive[k]}
        if sig:
            self._unit_sig = {k: float(np.std(ss[k] / max(sv[k], 1e-9))) for k in range(n) if alive[k]}
            self._unit_noise = {k: float(noise_of(k)) if sv[k] > 0 else np.nan for k in range(n) if alive[k]}
        return final[unit_of], codes, area, main, alive, set(isolated)

    # ------------------------------------------------------------------ coherence helpers
    @staticmethod
    def _corr(x, y):
        x, y = x - x.mean(), y - y.mean()
        d = np.sqrt((x * x).sum() * (y * y).sum())
        return float((x * y).sum() / d) if d > 0 else 0.0

    def _atom_matrix(self, lat, lon, good):
        """sparse matrix [atom, cell] of cos(lat) weights, rows normalised: the cells whose centre is inside the atom
        (small atoms: the cell at their representative point); atoms without a good cell get an empty row"""
        import shapely
        from scipy import sparse
        X, Y = np.meshgrid(lon, lat)
        wc = np.cos(np.deg2rad(Y))
        rows, cols, vals = [], [], []
        for k, g in enumerate(self.atoms.geometry.values):
            x0, y0, x1, y1 = g.bounds
            ci = np.flatnonzero((lon >= x0) & (lon <= x1))
            ri = np.flatnonzero((lat >= y0) & (lat <= y1))
            R = C = np.zeros(0, dtype=np.int64)
            if len(ci) and len(ri):
                R, C = np.meshgrid(ri, ci, indexing='ij')
                m = shapely.contains_xy(g, X[R, C], Y[R, C]) & good[R, C]
                R, C = R[m], C[m]
            if len(R) == 0:
                p = g.representative_point()
                r, c = int(np.argmin(np.abs(lat - p.y))), int(np.argmin(np.abs(lon - p.x)))
                if not good[r, c]:
                    continue
                R, C = np.array([r]), np.array([c])
            w = wc[R, C]
            rows.append(np.full(len(R), k)); cols.append(R * len(lon) + C); vals.append(w / w.sum())
        M = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(len(self.atoms), len(lat) * len(lon)))
        has = np.asarray(M.sum(axis=1)).ravel() > 0
        return M, has

    def _atom_series(self, field, lat, lon):
        """atom series [atom, time] of field[time, lat, lon] and the atom weights (area; 0 without a model cell)"""
        good = np.isfinite(field).all(axis=0)
        M, has = self._atom_matrix(lat, lon, good)
        F = np.nan_to_num(field.reshape(len(field), -1))
        return np.asarray(M @ F.T), np.where(has, self.atoms.SUB_AREA.values, 0.0), M, good

    def _prepare_series(self):
        """atom series for the coherence and the signal rules, atom adjacency and the atoms of every code"""
        if self.field is not None:
            self.series, self.series_w, _, _ = self._atom_series(self.field, self.f_lat, self.f_lon)
            print('  coherence: %d of %d atoms with model cells, %d time steps, merge rule %s, c_split %s'
                  % ((self.series_w > 0).sum(), len(self.atoms), self.field.shape[0], self.merge_rule, self.c_split))
        self.noise_series = None
        if self.sig_field is not None:
            self.sig_series, self.sig_w, M, good = self._atom_series(self.sig_field, self.s_lat, self.s_lon)
            if self.noise_atoms is not None:
                z = np.load(self.noise_atoms)
                pos = pd.Series(np.arange(len(z['keys'])), index=z['keys'])
                idx = pos.reindex(self.atoms.KEY.values).values
                assert np.isfinite(idx).all(), 'noise_atoms: polygons missing (other HydroBASINS regions / level?)'
                self.noise_series = z['noise'][idx.astype(int)].astype(np.float64)
            print('  signal: SNR >= %.1f (split at %.1f), noise %s%s' % (
                self.snr_min, self.snr_split, '%d GRACE error samples (%s)' % (self.noise_series.shape[1], Path(self.noise_atoms).name)
                if self.noise_atoms is not None else '%.0f mm x (%.0f km2 / A)^%.2f' % (self.sigma0, self.a0, self.beta),
                ', signal debiased' if self.debias else ''))
        keys = self.atoms.KEY.values
        i, j = self._shared_pairs(self.atoms.geometry, self.min_shared)
        self._atom_adj = [set() for _ in range(len(keys))]
        for x, y in zip(i, j):
            if x != y:
                self._atom_adj[x].add(int(y))
        self._atoms_of = {}
        for n in range(1, self.atom_level + 1):
            for p, idx in pd.Series(np.arange(len(keys))).groupby([k[:n + 2] for k in keys]):
                self._atoms_of[p] = idx.values
        pass

    def _snr_of(self, p):
        """signal-to-noise ratio of code p (area-weighted mean of its atoms)"""
        idx = self._atoms_of.get(p)
        w = self.sig_w[idx]
        if w.sum() <= 0:
            return np.inf
        s = (w[:, None] * self.sig_series[idx]).sum(axis=0) / w.sum()
        if self.noise_series is not None:
            noise = np.std((w[:, None] * self.noise_series[idx]).sum(axis=0) / w.sum())
        else:
            noise = self.sigma0 * (self.a0 / self.area[p]) ** self.beta
        return self._ratio(np.std(s), noise)

    def _ratio(self, signal, noise):
        """signal-to-noise ratio; with debias the signal (GRACE = true signal + noise) is reduced by the noise"""
        if self.debias:
            signal = np.sqrt(max(signal ** 2 - noise ** 2, 0.0))
        return float(signal / noise)

    def _shared_pairs(self, geom, min_shared, chunk=2000):
        """index pairs (i, j), i != j, of polygons whose common boundary is at least min_shared long (degrees);
        computed once per polygon set (cached), in chunks to keep the memory low"""
        key = (len(geom), min_shared)                           # only the atoms are passed here
        if getattr(self, '_pairs_cache', None) is not None and self._pairs_cache[0] == key:
            return self._pairs_cache[1]
        i, j = geom.sindex.query(geom, predicate='intersects')
        keep = i < j
        i, j = i[keep], j[keep]
        b = shapely.boundary(geom.values)
        length = np.zeros(len(i))
        for k in range(0, len(i), chunk):
            length[k:k + chunk] = shapely.length(shapely.intersection(b[i[k:k + chunk]], b[j[k:k + chunk]]))
        keep = length >= min_shared
        i, j = np.concatenate([i[keep], j[keep]]), np.concatenate([j[keep], i[keep]])
        self._pairs_cache = (key, (i, j))
        return i, j

    def _connected(self, p):
        """True if the atoms of code p form one connected piece (shared boundaries)"""
        idx = self._atoms_of.get(p)
        if idx is None or len(idx) < 2:
            return True
        inside, seen, stack = set(idx.tolist()), {int(idx[0])}, [int(idx[0])]
        while stack:
            k = stack.pop()
            for j in self._atom_adj[k]:
                if j in inside and j not in seen:
                    seen.add(j)
                    stack.append(j)
        return len(seen) == len(inside)

    def _coherence_of(self, p):
        """area-weighted mean r^2 of the atoms of code p with their mean series"""
        idx = self._atoms_of.get(p)
        if idx is None or len(idx) < 2:
            return 1.0
        w, s = self.series_w[idx], self.series[idx]
        if w.sum() <= 0:
            return 1.0
        mu = (w[:, None] * s).sum(axis=0) / w.sum()
        r = np.array([self._corr(x, mu) for x in s])
        return float((w * r ** 2).sum() / w.sum())

    # ------------------------------------------------------------------ PFAF list
    def _compress(self, codes):
        """replace complete sets of children by their parent, recursively; sorted"""
        s = set(codes)
        changed = True
        while changed:
            changed = False
            for p in sorted({c[:-1] for c in s if len(c) > 3}, key=len, reverse=True):
                ch = self.children.get(p, [])
                if ch and all(c in s for c in ch):
                    s.difference_update(ch)
                    s.add(p)
                    changed = True
        return sorted(s)

    # ------------------------------------------------------------------ run
    def run(self):
        import geopandas as gpd
        from shapely.ops import unary_union
        print('HydroBASINS global units: %s, area %.0f-%.0f x10^3 km2, extent <= %.0f km, atoms = level %d'
              % (','.join(self.regions), self.area_min / 1e3, self.area_max / 1e3, self.extent_max, self.atom_level))

        '''step-1: tables and atom polygons'''
        self._load_tables()

        if self.field is not None or self.sig_field is not None:
            self._prepare_series()

        '''step-2: split to <= area_max'''
        units = self._split()
        print('  split: %d units' % len(units))

        '''step-3: merge to >= area_min'''
        atom_unit, codes, area, main, alive, isolated = self._merge(units)

        '''step-4: dissolve and attributes'''
        a = self.atoms.assign(UNIT=atom_unit)
        rows, drop = [], []
        for u, g in a.groupby('UNIT'):
            row = dict(UNIT=u, PFAF=','.join(c[2:] for c in self._compress(codes[u])), SUB_AREA=float(g.SUB_AREA.sum()),
                       REGION=','.join(sorted(set(g.REGION))), LEVEL=len(main[u]) - 2, N_PFAF=len(codes[u]),
                       ENDO=int(g.SUB_AREA[g.ENDO > 0].sum() > 0.5 * g.SUB_AREA.sum()),
                       COAST=int(g.COAST.max()), MAIN_BAS=int(g.MAIN_BAS.iloc[np.argmax(g.SUB_AREA.values)]),
                       NAME='%s_%s' % (main[u][:2], main[u][2:]))
            if self.sig_field is not None:
                row['SIGNAL'] = round(self._unit_sig[u], 1)                     # std of unit-mean TWS [mm]
                row['NOISE'] = round(self._unit_noise[u], 1)                    # GRACE noise of the unit mean [mm]
                row['SNR'] = round(min(self._unit_snr[u], 999.0), 2)
            '''river-basin class of the unit (HydroBASINS MAIN_BAS)'''
            ba = g.groupby('MAIN_BAS').SUB_AREA.sum()
            tot = pd.Series(self._bas_total).reindex(ba.index)
            part = ba < 0.999 * tot
            row['N_BAS'] = len(ba)
            row['MAIN_SHR'] = round(float(ba.max() / ba.sum()), 3)
            if len(ba) == 1:
                row['BAS_CLASS'] = 'part of a basin' if part.iloc[0] else 'whole basin'
            elif not part.any() and not (tot > self.area_max).any():
                row['BAS_CLASS'] = 'whole basins'
            elif part.sum() + ((~part) & (tot > self.area_max)).sum() == 1:
                row['BAS_CLASS'] = 'basin + small coastal'
            else:
                row['BAS_CLASS'] = 'mixed'
            row['geometry'] = unary_union(g.geometry.values)
            if u in isolated and row['SUB_AREA'] < self.area_island:
                row['REASON'] = 'island < %.0f km2' % self.area_island
                drop.append(row)
                continue
            low_ok = row['NAME'] in self.snr_keep
            if self.sig_field is not None and u in isolated and row['SNR'] < self.snr_min and not low_ok:
                row['REASON'] = 'island, SNR < %.0f' % self.snr_min
                drop.append(row)
                continue
            if self.sig_field is not None and self.snr_drop is not None and row['SNR'] < self.snr_drop and not low_ok:
                row['REASON'] = 'low signal: SNR < %.1f' % self.snr_drop
                drop.append(row)
                continue
            rows.append(row)
        out = gpd.GeoDataFrame(rows, crs=self.atoms.crs)
        drop = self._island_rows + drop

        '''order: by region, then Pfafstetter code (neighbouring units get neighbouring IDs)'''
        out = out.sort_values(['REGION', 'NAME']).reset_index(drop=True)
        out.insert(0, 'ID', np.arange(1, len(out) + 1))
        cols = ['ID', 'NAME', 'PFAF', 'SUB_AREA', 'REGION', 'LEVEL', 'N_PFAF', 'ENDO', 'COAST', 'MAIN_BAS', 'BAS_CLASS',
                'N_BAS', 'MAIN_SHR']
        out = out[cols + [c for c in ('SIGNAL', 'NOISE', 'SNR') if c in out] + ['geometry']]
        self.units = out
        self.dropped = gpd.GeoDataFrame(drop, geometry='geometry', crs=self.atoms.crs) if drop else \
            gpd.GeoDataFrame(columns=['NAME', 'REASON', 'geometry'], geometry='geometry', crs=self.atoms.crs)

        '''step-5: glacier flag, removal of the major glaciated units'''
        if self.glacier_shp is not None:
            self.glacier()
            if self.glac_remove is not None:
                ice = self.units.GLAC_EFF / 100 * self.units.SUB_AREA
                rm = (self.units.GLAC_EFF >= 100 * self.glac_remove) | \
                     ((ice >= self.glac_remove_area) & (self.units.GLAC_EFF >= 100 * self.glac_major))
                d = self.units[rm].assign(REASON=np.where(self.units.GLAC_EFF[rm] >= 100 * self.glac_remove,
                                                          'glacier: GLAC_EFF >= %.0f %%' % (100 * self.glac_remove),
                                                          'glacier: ice seen >= %.0f km2 and >= %.0f %%'
                                                          % (self.glac_remove_area, 100 * self.glac_major)))
                print('  removed %d glaciated units (%.2f x10^6 km2): %s' % (rm.sum(), d.SUB_AREA.sum() / 1e6,
                                                                         ', '.join(d.NAME)))
                self.dropped = gpd.GeoDataFrame(pd.concat([self.dropped, d], ignore_index=True), geometry='geometry',
                                                crs=self.atoms.crs)
                self.units = self.units[~rm].reset_index(drop=True)
                self.units['ID'] = np.arange(1, len(self.units) + 1)

        '''step-6: the units on the WaterGAP grid, selection by model criteria'''
        if self.static_dir is not None:
            self.model_grid()
        self.summary()
        return self

    # ------------------------------------------------------------------ 6) WaterGAP model grid
    def model_grid(self):
        """
        every WaterGAP land cell (finite continental area) goes to the unit that covers the largest part of it; a
        cell whose largest cover is a removed unit (island, glacier) or that touches no unit (Greenland, small islands
        not in HydroBASINS) stays outside the DA. Per unit: N_CELLS, MOD_AREA (continental area of its cells, km2),
        COVER = MOD_AREA / SUB_AREA, LAKE_FRAC (global lakes + regulated lakes / reservoirs) and WET_FRAC (global
        wetlands) of MOD_AREA. Units are removed (REASON 'model: ...') if
          N_CELLS < n_cells_min  - too few model cells to carry a GRACE update
          LAKE_FRAC > lake_max   - storage dominated by a large lake (Great Lakes, Victoria, Ladoga): GRACE sees the
                                   lake level, WaterGAP has a simple lake balance; the update would go into the lake
          COVER outside cover_range - the model land does not represent the unit (coastline mismatch)
        (self.label[360, 720]: the cell assignment used for these statistics; the DA mask is written by
        global_shp2mask.py)
        """
        import shapely
        import xarray as xr
        import geopandas as gpd
        sd = self.static_dir

        def rd(fn, v):
            with xr.open_dataset(sd / fn, decode_times=False) as d:
                return np.asarray(d[v]).squeeze().astype(np.float64)
        ca = rd('watergap_22e_continentalarea.nc', 'continentalarea')
        lw = 'land_water_fractions/watergap_22e_%s.nc4'
        lake = (np.nan_to_num(rd(lw % 'glolak', 'glolak')) + np.nan_to_num(rd(lw % 'reglak', 'reglak'))) / 100
        wet = np.nan_to_num(rd(lw % 'glowet', 'glowet')) / 100
        valid = np.isfinite(ca)
        lat, lon = np.arange(89.75, -90, -0.5), np.arange(-179.75, 180, 0.5)
        R, C = np.nonzero(valid)
        x, y = lon[C], lat[R]
        cells = shapely.box(x - 0.25, y - 0.25, x + 0.25, y + 0.25)
        tree = shapely.STRtree(cells)

        '''coverage of every cell by every kept (+ID) and removed (-k) unit; the largest cover wins'''
        best, lab = np.zeros(len(R)), np.zeros(len(R), dtype=np.int64)
        polys = [(int(i), g) for i, g in zip(self.units.ID, self.units.geometry)]
        polys += [(-(k + 1), g) for k, g in enumerate(self.dropped.geometry) if g is not None]
        for i, g in polys:
            idx = tree.query(g, predicate='intersects')
            a = shapely.area(shapely.intersection(cells[idx], g))
            m = a > best[idx]
            best[idx[m]], lab[idx[m]] = a[m], i
        why_cell = np.full(len(R), 'no HydroBASINS unit (Greenland, small islands, Caspian Sea)', dtype=object)
        neg = lab < 0
        why_cell[neg] = [self.dropped.REASON.iloc[-i - 1].split(':')[0].split(' <')[0] for i in lab[neg]]
        why_cell[lab > 0] = ''
        lab[neg] = 0

        '''per-unit model statistics'''
        df = pd.DataFrame({'ID': lab, 'ca': ca[R, C], 'lake': lake[R, C] * ca[R, C], 'wet': wet[R, C] * ca[R, C]})
        st = df[df.ID > 0].groupby('ID').agg(N_CELLS=('ca', 'size'), MOD_AREA=('ca', 'sum'), lake=('lake', 'sum'),
                                             wet=('wet', 'sum'))
        u = self.units.set_index('ID')
        u['N_CELLS'] = st.N_CELLS.reindex(u.index).fillna(0).astype(int)
        u['MOD_AREA'] = st.MOD_AREA.reindex(u.index).fillna(0).round(0)
        u['COVER'] = (u.MOD_AREA / u.SUB_AREA).round(3)
        u['LAKE_FRAC'] = (st.lake / st.MOD_AREA).reindex(u.index).fillna(0).round(3)
        u['WET_FRAC'] = (st.wet / st.MOD_AREA).reindex(u.index).fillna(0).round(3)

        '''selection'''
        why = pd.Series('', index=u.index)
        why[u.N_CELLS < self.n_cells_min] = 'model: < %d cells' % self.n_cells_min
        why[(why == '') & (u.LAKE_FRAC > self.lake_max)] = 'model: lake > %.0f %%' % (100 * self.lake_max)
        why[(why == '') & ((u.COVER < self.cover_range[0]) | (u.COVER > self.cover_range[1]))] = \
            'model: cover outside %.1f-%.1f' % self.cover_range
        rm = why != ''
        d = u[rm].reset_index().assign(REASON=why[rm].values)
        if rm.any():
            print('  model grid: removed %d units: %s' % (rm.sum(), ', '.join('%s (%s)' % (n, r[7:]) for n, r
                                                                              in zip(d.NAME, d.REASON))))
        self.dropped = gpd.GeoDataFrame(pd.concat([self.dropped, d], ignore_index=True), geometry='geometry',
                                        crs=self.atoms.crs)
        keep = u[~rm].reset_index()
        new = dict(zip(keep.ID, np.arange(1, len(keep) + 1)))
        keep['ID'] = np.arange(1, len(keep) + 1)
        self.units = gpd.GeoDataFrame(keep, geometry='geometry', crs=self.atoms.crs)
        lab = np.array([new.get(i, 0) for i in lab])
        why_cell[(lab == 0) & (why_cell == '')] = 'model criteria (lakes, cells)'

        '''what is not assimilated'''
        t = pd.DataFrame({'why': why_cell, 'ca': ca[R, C]})[why_cell != ''].groupby('why').ca.agg(['size', 'sum'])
        for w, r in t.iterrows():
            print('    not assimilated: %-58s %6d cells %5.2f x10^6 km2' % (w, r['size'], r['sum'] / 1e6))

        '''label map on the global grid'''
        self.label = np.zeros(ca.shape, dtype=np.int32)
        self.label[R, C] = lab
        self.model_valid = valid
        n_out = int(valid.sum() - (lab > 0).sum())
        print('  model grid: %d of %d WaterGAP cells in %d units; %d cells (%.1f x10^6 km2) not assimilated'
              % ((lab > 0).sum(), valid.sum(), len(self.units), n_out, ca[valid & (self.label == 0)].sum() / 1e6))
        pass


    # ------------------------------------------------------------------ 5) glacier flag
    def glacier(self, res=0.05):
        """
        per unit: GLAC_IN  = ice area inside / unit area [%]
                  GLAC_EFF = (ice inside + ice outside x 0.5 erfc(d / sqrt(2) sigma)) / unit area [%]  (what GRACE sees)
                  GLACIER  = 0 (GLAC_EFF < minor), 1 (minor..major), 2 (>= major)
        the ice is sampled on a res-degree grid (cell centres inside the ice polygons, weighted by cell area); d is the
        distance of a sample to the unit in an azimuthal equidistant projection centred on the unit
        """
        import shapely
        import geopandas as gpd
        from pyproj import Transformer
        from scipy.special import erfc

        '''ice samples: cell centres of a res-degree grid inside each ice polygon, carrying the polygon's area in equal
        parts; polygons smaller than about a cell are one sample at their centroid (RGI has ~200 000 of them).
        Antarctica is skipped (far from any unit)'''
        ice = pd.concat([gpd.read_file(f).to_crs('EPSG:4326')[['geometry']] for f in self.glacier_shp], ignore_index=True)
        ice = gpd.GeoDataFrame(geometry=ice.geometry.make_valid().explode(index_parts=False).values, crs='EPSG:4326')
        ice = ice[ice.geom_type.isin(['Polygon', 'MultiPolygon']) & (ice.bounds.maxy > -60)].reset_index(drop=True)
        area = ice.to_crs('+proj=cea +units=km').area.values                      # km2
        b = ice.bounds.values
        small = (b[:, 2] - b[:, 0] < 2 * res) | (b[:, 3] - b[:, 1] < 2 * res)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')                    # centroid in lon/lat: fine for polygons < 2 cells
            c = ice.geometry[small].centroid
        lon, lat, wgt = [c.x.values], [c.y.values], [area[small]]
        for g, a in zip(ice.geometry[~small].values, area[~small]):
            x0, y0, x1, y1 = g.bounds
            X, Y = np.meshgrid(np.arange(np.floor(x0 / res) * res + res / 2, x1, res),
                               np.arange(np.floor(y0 / res) * res + res / 2, y1, res))
            m = shapely.contains_xy(g, X, Y)
            if m.sum() == 0:
                p = g.representative_point()
                X, Y, m = np.array([p.x]), np.array([p.y]), np.array([True])
            lon.append(X[m]); lat.append(Y[m]); wgt.append(np.full(m.sum(), a / m.sum()))
        plon, plat, parea = np.concatenate(lon), np.concatenate(lat), np.concatenate(wgt)
        tree = shapely.STRtree(shapely.points(plon, plat))
        print('  glacier flag: %d ice polygons, %.2f x10^6 km2 outside Antarctica (%d samples), sigma %.0f km'
              % (len(ice), parea.sum() / 1e6, len(plon), self.sigma))

        rmax = 3.0 * self.sigma
        g_in, g_eff = np.zeros(len(self.units)), np.zeros(len(self.units))
        for k, geom in enumerate(self.units.geometry.values):
            x0, y0, x1, y1 = geom.bounds
            dlat = rmax / 111.0 + 0.5
            la = min(89.0, max(abs(y0), abs(y1)) + dlat)
            dlon = min(360.0, rmax / (111.0 * np.cos(np.deg2rad(la))) + 0.5)
            boxes = [shapely.box(x0 - dlon + s, y0 - dlat, x1 + dlon + s, y1 + dlat) for s in (0, -360, 360)]
            idx = np.unique(np.concatenate([tree.query(b) for b in boxes]))
            if len(idx) == 0:
                continue
            c = geom.representative_point()
            tr = Transformer.from_crs('EPSG:4326', '+proj=aeqd +lat_0=%.4f +lon_0=%.4f +units=km' % (c.y, c.x),
                                      always_xy=True)
            u = shapely.transform(geom, lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))
            u = shapely.make_valid(u.simplify(2.0))
            px, py = tr.transform(plon[idx], plat[idx])
            d = shapely.distance(u, shapely.points(px, py))                  # km, 0 inside
            inside = d == 0
            g_in[k] = parea[idx][inside].sum() / u.area
            g_eff[k] = (parea[idx] * np.where(inside, 1.0, 0.5 * erfc(d / (np.sqrt(2) * self.sigma)))).sum() / u.area
        self.units['GLAC_IN'] = np.round(100 * g_in, 3)
        self.units['GLAC_EFF'] = np.round(100 * g_eff, 3)
        self.units['GLACIER'] = (g_eff >= self.glac_minor).astype(int) + (g_eff >= self.glac_major).astype(int)
        n = self.units.GLACIER.value_counts().to_dict()
        print('  GLACIER: %d units minor (GLAC_EFF >= %.1f %%), %d major (>= %.0f %%)'
              % (n.get(1, 0), 100 * self.glac_minor, n.get(2, 0), 100 * self.glac_major))
        pass

    # ------------------------------------------------------------------ report / save
    def summary(self):
        u = self.units
        q = np.percentile(u.SUB_AREA, [0, 10, 50, 90, 100]) / 1e3
        print('  %d units, %.2f x10^6 km2 (dropped %d units: islands, glaciers; %.0f km2)'
              % (len(u), u.SUB_AREA.sum() / 1e6, len(self.dropped),
                 self.dropped.SUB_AREA.sum() if len(self.dropped) else 0))
        print('  area [10^3 km2]: min %.0f, p10 %.0f, median %.0f, p90 %.0f, max %.0f' % tuple(q))
        print('  below area_min: %d (islands), above %.1f x area_max: %d'
              % ((u.SUB_AREA < self.area_min).sum(), self.merge_cap, (u.SUB_AREA > self.merge_cap * self.area_max).sum()))
        print('  per region:', u.REGION.value_counts().sort_index().to_dict())
        if 'BAS_CLASS' in u:
            c = u.groupby('BAS_CLASS').SUB_AREA.agg(['size', 'sum'])
            print('  river-basin classes: ' + ', '.join('%s %d (%.0f %%)' % (k, r['size'], 100 * r['sum'] / u.SUB_AREA.sum())
                                                       for k, r in c.iterrows()))
        pass

    def save(self, simplify=0.01, figure=True):
        """
        simplify: tolerance [deg] of a coverage simplification (shared edges stay shared: no gaps, no overlaps);
                  0.01 deg (~1 km) cuts the .shp from ~65 to ~19 MB, far below the 0.5 deg model grid; None = full HydroBASINS detail
        """
        import shapely
        assert self.out_dir is not None, 'configure_output first'
        self.out_dir.mkdir(parents=True, exist_ok=True)
        base = self.out_dir / self.name
        u = self.units.copy()
        u.geometry = u.geometry.make_valid()
        if simplify:
            u.geometry = shapely.coverage_simplify(u.geometry.values, simplify)
        u.drop(columns='geometry').to_csv(str(base) + '_units.csv', index=False)
        if len(self.dropped):
            dr = self.dropped.drop(columns=['UNIT'], errors='ignore')
            pd.DataFrame(dr.drop(columns='geometry')).to_csv(str(base) + '_dropped.csv', index=False)
            dr = dr[dr.geometry.notna()].copy()
            dr.geometry = dr.geometry.make_valid()
            dr['PFAF'] = dr.PFAF.astype(str).str[:250]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                dr[['NAME', 'PFAF', 'SUB_AREA', 'REASON', 'geometry']].to_file(str(base) + '_removed.shp')
        long = u.PFAF.str.len() > 254
        if long.any():
            print('  %d PFAF lists longer than 254 characters are cut in the .shp (full lists in %s_units.csv)'
                  % (long.sum(), self.name))
            u.loc[long, 'PFAF'] = u.loc[long, 'PFAF'].str[:250] + ',...'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            u.to_file(str(base) + '.shp')
        print('written', str(base) + '.shp')
        if figure:
            self.figure(str(base) + '.png')
            if 'GLACIER' in self.units:
                self.figure_glacier(str(base) + '_glacier.png')
        pass

    def figure_glacier(self, fn):
        """map of GLAC_EFF per unit (classes of the GLACIER flag outlined) and the ice polygons"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import geopandas as gpd
        u = self.units
        ice = pd.concat([gpd.read_file(f).to_crs('EPSG:4326')[['geometry']] for f in self.glacier_shp], ignore_index=True)
        ice = gpd.GeoDataFrame(ice, crs='EPSG:4326')
        bounds = [0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]
        cmap = plt.get_cmap('YlOrRd', len(bounds) - 1)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        fig, ax = plt.subplots(figsize=(20, 10))
        u.plot(ax=ax, column='GLAC_EFF', cmap=cmap, norm=norm, edgecolor='gray', linewidth=0.2)
        ice.plot(ax=ax, color='deepskyblue', linewidth=0)
        u[u.GLACIER == 1].boundary.plot(ax=ax, color='k', linewidth=0.6)
        u[u.GLACIER == 2].boundary.plot(ax=ax, color='k', linewidth=1.6)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.01, ticks=bounds)
        cb.set_label('GLAC_EFF: ice area seen by GRACE / unit area [%]')
        ax.set_xlim(-180, 180); ax.set_ylim(-60, 84); ax.set_aspect('equal')
        ax.set_title('Glacier flag, sigma %.0f km: %d units minor (>= %.1f %%, thin outline), %d major (>= %.0f %%, thick); blue: ice'
                     % (self.sigma, (u.GLACIER == 1).sum(), 100 * self.glac_minor, (u.GLACIER == 2).sum(),
                        100 * self.glac_major))
        fig.savefig(fn, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print('figure', fn)
        pass

    def figure(self, fn, ids=True):
        """quick-look map: units in random colours, area classes in the legend"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        u = self.units
        rng = np.random.default_rng(1)
        col = plt.cm.tab20(rng.permutation(len(u)) % 20)
        fig, ax = plt.subplots(figsize=(20, 10))
        u.plot(ax=ax, color=col, edgecolor='k', linewidth=0.25)
        if ids:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                pts = u.geometry.representative_point()
            for i, pt in zip(u.ID, pts):
                ax.text(pt.x, pt.y, str(i), fontsize=3.5, ha='center', va='center')
        ax.set_xlim(-180, 180); ax.set_ylim(-60, 84); ax.set_aspect('equal')
        rule = ', SNR >= %.0f' % self.snr_min if self.sig_field is not None else ''
        ax.set_title('%d basin units from HydroBASINS v1c (%.0f-%.0f x10$^3$ km$^2$%s, median %.0f)'
                     % (len(u), self.area_min / 1e3, self.merge_cap * self.area_max / 1e3, rule, u.SUB_AREA.median() / 1e3))
        fig.savefig(fn, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print('figure', fn)
        pass

    @staticmethod
    def load_ol_field(ol_dir, begin='2002-01', end='2007-06', day=15, trend_max=100.0,
                      storages=('canopystor', 'swe', 'soilmoist', 'groundwstor', 'locallakestor', 'localwetlandstor')):
        """
        monthly snapshots (day `day` of each month) of the open-loop storage from <ol_dir>/daily_output_YYYY-MM-DD.nc
        (e.g. OL_output/Ens_0) -> field[time, lat, lon] [mm], lat, lon. The default storages leave out rivers, global
        lakes / wetlands and reservoirs: point-like signals (Aral Sea, Lake Nasser, Three Gorges: up to 14 m std) that
        dominate the variance but not the regional regime. Cells with |trend| >= trend_max mm/yr (snow accumulating on
        glaciers in WaterGAP) are set to NaN
        """
        import xarray as xr
        months = pd.period_range(begin, end, freq='M')
        out, lat, lon = [], None, None
        for m in months:
            fn = Path(ol_dir) / ('daily_output_%s-%02d.nc' % (m, day))
            if not fn.exists():
                continue
            with xr.open_dataset(fn) as d:
                out.append(sum(d[v].values.astype(np.float64) for v in storages))
                lat, lon = d.lat.values, d.lon.values
        f = np.stack(out)
        t = np.arange(len(f))
        a = np.nan_to_num(f.reshape(len(f), -1))
        tr = np.polyfit(t, a, 1)[0].reshape(f.shape[1:]) * 12
        f[:, np.abs(tr) >= trend_max] = np.nan
        print('open-loop field: %d months from %s, %d cells (%d with |trend| >= %.0f mm/yr removed)'
              % (len(f), ol_dir, np.isfinite(f).all(axis=0).sum(), (np.abs(tr) >= trend_max).sum(), trend_max))
        return f, lat, lon

    def evaluate(self, field, lat, lon, grid_deg=(4, 5, 6, 7), fn=None):
        """
        coherence of the units against regular lat-lon boxes on the same model cells:
          C  = mean r^2 of each cell's series with the mean series of its unit (cos(lat) weights)
          R2 = variance of the cells explained by their unit mean
        both for the full anomalies and without the mean seasonal cycle. Compare units and boxes at the same number of
        regions (interpolate the box curve). Returns a DataFrame (written to fn as csv if given)
        """
        import shapely
        X, Y = np.meshgrid(lon, lat)
        W = np.cos(np.deg2rad(Y))
        F = np.asarray(field, dtype=np.float64)
        mon = np.arange(len(F)) % 12
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            F = F - np.nanmean(F, axis=0)
            S = F.copy()
            for m in range(12):
                S[mon == m] -= np.nanmean(F[mon == m], axis=0)
        L = np.zeros(X.shape, dtype=np.int64)
        for i, g in zip(self.units.ID, self.units.geometry):
            x0, y0, x1, y1 = g.bounds
            m = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
            L.ravel()[np.flatnonzero(m.ravel())[shapely.contains_xy(g, X[m], Y[m])]] = i
        mask = np.isfinite(F).all(axis=0) & (L > 0)

        def score(lab, G):
            l, w, x = lab[mask], W[mask], G[:, mask]
            ids, inv = np.unique(l, return_inverse=True)
            sw = np.bincount(inv, w)
            mu = np.stack([np.bincount(inv, w * x[t]) / sw for t in range(len(x))])[:, inv]
            xc, mc = x - x.mean(axis=0), mu - mu.mean(axis=0)
            r = (xc * mc).sum(axis=0) / np.sqrt((xc ** 2).sum(axis=0) * (mc ** 2).sum(axis=0) + 1e-12)
            return len(ids), np.average(r ** 2, weights=w), 1 - (w * (x - mu).var(axis=0)).sum() / (w * x.var(axis=0)).sum()

        rows = [('units',) + score(L, F) + score(L, S)[1:]]
        for d in grid_deg:
            G = ((90 - Y) // d).astype(np.int64) * 10000 + ((X + 180) // d).astype(np.int64) + 1
            rows.append(('grid %g deg' % d,) + score(G, F) + score(G, S)[1:])
        t = pd.DataFrame(rows, columns=['regions', 'n', 'C', 'R2', 'C_deseason', 'R2_deseason'])
        print(t.round(3).to_string(index=False))
        if fn is not None:
            t.to_csv(fn, index=False)
        return t

    def compare(self, shp, pfaf_prefix):
        """overlap of the units with a manual grouping (e.g. Basin/shp/Danube/Danube.shp, prefix '227')"""
        import geopandas as gpd
        ref = gpd.read_file(shp)
        u = self.units[self.units.PFAF.str.split(',').apply(lambda c: any(x.startswith(pfaf_prefix) or pfaf_prefix.startswith(x)
                                                                         for x in c))]
        print('manual %s: %d units, automatic: %d units' % (Path(shp).stem, len(ref), len(u)))
        for _, r in u.iterrows():
            print('  ID %4d %-12s %7.0f km2  PFAF %s' % (r.ID, r.NAME, r.SUB_AREA, r.PFAF[:90]))
        return u


# ------------------------------------------------------------------ demos
EXT = Path('/media/user/My Book/Fan/PyGLDA_v2_external_data')


def demo1():
    """
    global units: Pfafstetter + open-loop coherence + size from the GRACE signal-to-noise ratio; river basins as a
    hard rule; neighbours only along a shared boundary; islands < 1e5 km2 (or below SNR 3), units below SNR 1, the
    major glaciated units and the large lakes removed -> Basin/shp/Global/GlobalBasins.shp
    (run demo3 first for the GRACE noise; then global_shp2mask.demo2 for the mask)
    """
    ol = EXT / 'OL_output/Ens_0'
    f, lat, lon = hydrobasins_global.load_ol_field(ol, begin='2002-01', end='2007-06')             # coherence
    hg = hydrobasins_global(hybas_dir=EXT / 'Extra')
    hg.configure_size(area_min=6.3e4, area_max=5.0e5, area_island=1.0e5).configure_atom_level(6)
    hg.configure_coherence(f, lat, lon, merge_rule='hybrid', c_split=0.8)
    '''signal: CSR mascons (GRACE itself, incl. the trends WaterGAP lacks); noise: the SaGEA DDK3 error samples
    reduced by demo3 (falls back to the literature noise model if that file does not exist yet)'''
    g, glat, glon = hydrobasins_global.load_grace_mascon(
        EXT / 'GRACE/SaGEA/signal_Mascon/CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc')
    noise = EXT / 'Extra/grace_noise/atom_noise_DDK3.npz'
    hg.configure_signal(g, glat, glon, snr_min=3.0, debias=True, area_max_low=1.0e7, extent_max_low=8000.0,
                        snr_drop=1.0, snr_keep=('eu_233', 'au_571', 'au_572'), noise_atoms=noise if noise.exists() else None)   # Great Britain, North and South Island NZ
    hg.configure_glacier(shp=EXT / 'Extra/glacier/ne_10m_glaciated_areas.shp', remove=0.10)
    hg.configure_basin_rule(basin_first=True, min_shared=0.05, same_drainage=True)
    hg.configure_model(static_dir=EXT / 'Input_data/static_input', n_cells_min=15, lake_max=0.25)
    hg.configure_output(out_dir=EXT / 'Basin/shp/Global', name='GlobalBasins')
    hg.run().save()
    # the DA mask is made from GlobalBasins.shp by global_shp2mask.py (same rules and format as shp2mask.py)
    hg.evaluate(f, lat, lon, fn=EXT / 'Basin/shp/Global/GlobalBasins_coherence.csv')
    hg.compare(EXT / 'Basin/shp/Danube/Danube.shp', pfaf_prefix='227')
    hg.compare(EXT / 'Basin/shp/Amazon/Amazon.shp', pfaf_prefix='62')
    pass


def demo3():
    """
    GRACE noise per level-6 polygon from the SaGEA DDK3 Monte-Carlo samples (500 per month, ~1 GB per file; run on
    the machine that holds them, ~10 min) -> Extra/grace_noise/atom_noise_DDK3.npz (~200 MB), used by demo1.
    Six months: GRACE early / mid / late (degraded accelerometer) and GRACE-FO
    """
    smp = EXT / 'GRACE/SaGEA/sample_DDK3'
    files = [smp / ('%s_0.npy' % m) for m in ('2003-07', '2008-07', '2012-03', '2016-08', '2019-07', '2022-01')]
    out = EXT / 'Extra/grace_noise'
    out.mkdir(parents=True, exist_ok=True)
    hg = hydrobasins_global(hybas_dir=EXT / 'Extra').configure_atom_level(6)
    hg.export_atom_noise(files, out / 'atom_noise_DDK3.npz', south_first=True, scale=1000.0)
    pass


def demo2():
    """one region only (Europe), for a quick test"""
    hg = hydrobasins_global(hybas_dir=EXT / 'Extra', regions=('eu',))
    hg.configure_output(out_dir=EXT / 'Extra/test_global_basins', name='eu_basins')
    hg.run().save()
    pass


if __name__ == '__main__':
    demo3()
