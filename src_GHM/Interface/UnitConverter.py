import numpy as np


class UnitConverter:
    """
    Bidirectional unit converter that mirrors the logic in
    CreateandWritetoVariables.base_units().

    Internal model units  <-->  On-disk (NetCDF output) units
    -------------------------------------------------------
    vb_storages (canopystor, swe, soilmoist, smax)
        mm                 <-->  mm              (no change)

    vb_fluxes (potevap, netrad, canopy-evap, throughfall,
               snowfall, snm, snow-evap, qs, qrd, ...)
        mm/day             <-->  mm/s
        dimensionless      <-->  dimensionless   (lai-total, snowcover-frac)

    lb_storages (groundwstor, locallakestor, localwetlandstor,
                 globallakestor, globalwetlandstor, riverstor,
                 reservoirstor, tws)
        km³                <-->  mm

    lb_fluxes (consistent-precipitation, qg, qtot, qrf, qr,
               locallake-outflow, localwetland-outflow, globallake-outflow,
               globalwetland-outflow, ncrun, pot_cell_runoff, ...)
        km³/day            <-->  mm/s
        km³/day            <-->  m³/s            (dis, dis-from-upstream)
        km/day             <-->  m/s             (river-velocity)
        dimensionless      <-->  dimensionless   (get_neighbouring_cells_map,
                                                  land-area-fraction)

    Parameters
    ----------
    cell_area : np.ndarray
        Grid cell area in km². Must match the spatial shape of the state arrays
        being converted (box/cropped shape for regional DA, global shape otherwise).
    contfrac : np.ndarray or xarray.DataArray
        Continental fraction in percent (0–100). Same shape as cell_area.
        Accepts xarray DataArray; .values is called internally.
    """

    # ------------------------------------------------------------------ #
    #  Variable category sets — use the output/enum-style variable names  #
    # ------------------------------------------------------------------ #
    VB_STORAGES = {'canopystor', 'swe', 'soilmoist', 'smax'}

    LB_STORAGES = {
        'groundwstor', 'locallakestor', 'localwetlandstor',
        'globallakestor', 'globalwetlandstor',
        'riverstor', 'reservoirstor', 'tws'
    }

    # Vertical fluxes that carry no unit (pass-through)
    VB_FLUX_NO_CONVERSION = {'lai-total', 'snowcover-frac'}

    # Lateral fluxes that carry no unit (pass-through)
    LB_FLUX_NO_CONVERSION = {'get_neighbouring_cells_map', 'land-area-fraction'}

    # Discharge: km³/day <--> m³/s
    DISCHARGE = {'dis', 'dis-from-upstream'}

    # River velocity: km/day <--> m/s
    RIVER_VELOCITY = {'river-velocity'}

    # Conversion constants (match base_units exactly)
    _DAYS_TO_S  = 86400
    _KM_TO_M    = 1e3
    _KM3_TO_M3  = 1e9

    def __init__(self, cell_area, contfrac):
        cell_area = np.asarray(cell_area, dtype=np.float64)
        if hasattr(contfrac, 'values'):          # accept xarray DataArray
            contfrac = contfrac.values
        contfrac = np.asarray(contfrac, dtype=np.float64)

        # Store raw arrays so downstream code (e.g. EnKF) can crop them for a
        # basin-specific converter via UnitConverter.from_basin_mask().
        self.cell_area_global = cell_area
        self.contfrac_global  = contfrac

        self._km3_to_mm = 1e6 / (cell_area * (contfrac / 100))
        self._mm_to_km3 = 1.0 / self._km3_to_mm

    @classmethod
    def from_basin_mask(cls, local_mask: dict, cell_area_global, contfrac_global):
        """
        Construct a UnitConverter cropped to the basin bounding box,
        derived from the local_mask returned by load_mask().
        """
        gm = local_mask['global_2d'].astype(bool)
        b2 = np.asarray(local_mask['basin_2d'])

        if b2.shape == gm.shape:
            '''global unit mask (shp2mask.load_mask, attribute extent = 'global'): the box is the whole grid, no crop;
            the bounding box of the units (rows 24..279 for GlobalBasins v1.0) would not match basin_2d'''
            r0, r1, c0, c1 = 0, gm.shape[0], 0, gm.shape[1]
        else:
            '''regional mask: the box of load_mask is the bounding box of the basin cells'''
            row_idx = np.where(np.any(gm, axis=1))[0]
            col_idx = np.where(np.any(gm, axis=0))[0]
            r0, r1, c0, c1 = row_idx[0], row_idx[-1] + 1, col_idx[0], col_idx[-1] + 1

        cell_area_box = cell_area_global[r0:r1, c0:c1]
        contfrac_box = contfrac_global[r0:r1, c0:c1]

        assert cell_area_box.shape == b2.shape, \
            'UnitConverter.from_basin_mask: box %s of the model fields does not match basin_2d %s' % (
                cell_area_box.shape, b2.shape)

        instance = cls(cell_area_box, contfrac_box)
        instance._r0, instance._r1, instance._c0, instance._c1 = r0, r1, c0, c1

        return instance

    # ------------------------------------------------------------------ #
    #  Single-variable conversion                                         #
    # ------------------------------------------------------------------ #

    def to_disk(self, var_name: str, data: np.ndarray) -> np.ndarray:
        """Internal model units  -->  on-disk units (mirrors base_units)."""
        data = np.asarray(data, dtype=np.float64)

        # vb_storages: mm -> mm
        if var_name in self.VB_STORAGES:
            return data.copy()

        # lb_storages: km³ -> mm
        if var_name in self.LB_STORAGES:
            return data * self._km3_to_mm

        # vb_fluxes (dimensionless)
        if var_name in self.VB_FLUX_NO_CONVERSION:
            return data.copy()

        # lb_fluxes (dimensionless)
        if var_name in self.LB_FLUX_NO_CONVERSION:
            return data.copy()

        # discharge: km³/day -> m³/s
        if var_name in self.DISCHARGE:
            return (data * self._KM3_TO_M3) / self._DAYS_TO_S

        # river velocity: km/day -> m/s
        if var_name in self.RIVER_VELOCITY:
            return (data * self._KM_TO_M) / self._DAYS_TO_S

        # vb_fluxes (remaining): mm/day -> mm/s
        if var_name in self._vb_flux_names():
            return data / self._DAYS_TO_S

        # lb_fluxes (remaining): km³/day -> mm/s
        return (data * self._km3_to_mm) / self._DAYS_TO_S

    def to_internal(self, var_name: str, data: np.ndarray) -> np.ndarray:
        """On-disk units  -->  internal model units (inverse of base_units)."""
        data = np.asarray(data, dtype=np.float64)

        # vb_storages: mm -> mm
        if var_name in self.VB_STORAGES:
            return data.copy()

        # lb_storages: mm -> km³
        if var_name in self.LB_STORAGES:
            return data * self._mm_to_km3

        # vb_fluxes (dimensionless)
        if var_name in self.VB_FLUX_NO_CONVERSION:
            return data.copy()

        # lb_fluxes (dimensionless)
        if var_name in self.LB_FLUX_NO_CONVERSION:
            return data.copy()

        # discharge: m³/s -> km³/day
        if var_name in self.DISCHARGE:
            return (data * self._DAYS_TO_S) / self._KM3_TO_M3

        # river velocity: m/s -> km/day
        if var_name in self.RIVER_VELOCITY:
            return (data * self._DAYS_TO_S) / self._KM_TO_M

        # vb_fluxes (remaining): mm/s -> mm/day
        if var_name in self._vb_flux_names():
            return data * self._DAYS_TO_S

        # lb_fluxes (remaining): mm/s -> km³/day
        return (data * self._DAYS_TO_S) * self._mm_to_km3

    # ------------------------------------------------------------------ #
    #  Whole-dict convenience methods (for DA state dicts)                #
    # ------------------------------------------------------------------ #

    def state_dict_to_internal(self, state: dict, per_contfrac=None) -> dict:
        result = {}
        for k, v in state.items():
            if k in self.VB_STORAGES and per_contfrac is not None:
                v = np.asarray(v, dtype=np.float64).copy()
                pc = per_contfrac
                # Auto-crop if per_contfrac is global but v is box-shaped
                if pc.shape != v.shape and hasattr(self, '_r0'):
                    pc = pc[self._r0:self._r1, self._c0:self._c1]
                safe = np.isfinite(pc) & (pc > 0)
                v[safe] /= pc[safe]
                result[k] = v
            else:
                result[k] = self.to_internal(k, v)
        return result

    def state_dict_to_disk(self, state: dict) -> dict:
        """
        Convert an entire state dict (enum-named keys, internal units)
        to on-disk units.  Mirrors what base_units() does to the output
        variables before saving.
        """
        return {k: self.to_disk(k, v) for k, v in state.items()}

    # ------------------------------------------------------------------ #
    #  Internal helper                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _vb_flux_names() -> frozenset:
        """All vertical flux variable names (excluding dimensionless ones)."""
        return frozenset({
            'potevap', 'netrad', 'canopy-evap', 'throughfall',
            'snowfall', 'snm', 'snow-evap', 'qs', 'qrd',
            'consistent-precipitation'      # treated as flux in lb_fluxes
        })