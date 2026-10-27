import json
from pathlib import Path
from src_DA.EnumDA import HydroModel, FusionMethod, WaterGap_storage_variables


class config_DA:

    def __init__(self):
        self.basic = self.config_basic().__dict__
        self.obs = self.config_obs().__dict__
        self.model = self.config_model().__dict__
        self.method = self.config_method().__dict__

        pass

    @staticmethod
    def loadjson(js):
        Obj = config_DA()

        dict1 = json.load(open(js, 'r'))
        Obj.__dict__ = dict1
        return Obj

    def process(self):
        '''assign the settings into class attributes'''
        basic = self.basic
        self.basic = self.config_basic()
        self.basic.__dict__.update(basic)

        model = self.model
        self.model = self.config_model()
        self.model.__dict__.update(model)
        self.model.name = HydroModel[self.model.name]

        obs = self.obs
        self.obs = self.config_obs()
        self.obs.__dict__.update(obs)

        method = self.method
        self.method = self.config_method()
        self.method.__dict__.update(method)
        self.method.fusion_method = FusionMethod[self.method.fusion_method]

        return self

    def save_json(self, save_path):
        """
        Export the current configuration object to a new JSON file.
        This method safely serializes class instances and Enum objects back into standard JSON format.
        """

        # Helper function: Extract a dictionary copy from an object or dictionary
        def _get_dict(item):
            if hasattr(item, '__dict__'):
                return item.__dict__.copy()
            return item.copy()

        # 1. Revert internal configuration objects back to plain dictionaries
        out_dict = {
            'basic': _get_dict(self.basic),
            'obs': _get_dict(self.obs),
            'model': _get_dict(self.model),
            'method': _get_dict(self.method)
        }

        # 2. De-process Enum objects back to their string representations
        # Check if the attribute is an Enum (has a '.name' property)
        if hasattr(out_dict['model']['name'], 'name'):
            out_dict['model']['name'] = out_dict['model']['name'].name

        if hasattr(out_dict['method']['fusion_method'], 'name'):
            out_dict['method']['fusion_method'] = out_dict['method']['fusion_method'].name

        # 3. Safely dump the dictionary into a new JSON file
        save_path = Path(save_path)

        # Automatically create the parent directory if it does not exist
        # save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(out_dict, f, indent=4)

        return save_path

    class config_basic:
        """period, ensemble, case and the folders of a run; the paths are set by the drivers (src_FlowControl)"""
        def __init__(self):
            self.fromdate = '2002-01-01'
            self.todate = '2002-12-31'
            self.ensemble = 30
            self.case = None
            self.basin = None
            self.basin_shp = None
            self.basin_mask = None
            self.res_permanent = None
            self.OL_output_temp_dir = None
            self.DA_output_temp_dir = None
            self.Ensemble_ini_dir = None
            self.Ensemble_input_dir = None
            self.Auxiliary_dir = None

    class config_model:

        def __init__(self):
            self.name = HydroModel.WaterGap.name
            self.layer = {
                WaterGap_storage_variables.groundwstor.name: True,
                WaterGap_storage_variables.soilmoist.name: True,
                WaterGap_storage_variables.swe.name: True,
                WaterGap_storage_variables.locallakestor.name: True,
                WaterGap_storage_variables.localwetlandstor.name: True,
                WaterGap_storage_variables.globallakestor.name: True,
                WaterGap_storage_variables.globalwetlandstor.name: True,
                WaterGap_storage_variables.riverstor.name: True,
                WaterGap_storage_variables.reservoirstor.name: True,
                WaterGap_storage_variables.canopystor.name: True
            }

    class config_obs:
        """GRACE input; "GRACE" holds the folders and the product options (src_FlowControl.Regional_DA.get_GRACE_obs)"""
        def __init__(self):
            self.dir = None
            self.GRACE = {'kind': 'Mascon_monthly'}

    class config_method:

        def __init__(self):
            '''The filter is assembled from independent components (src_DA.filter_factory), each chosen with its own
            block; doc/configuration.md lists all options and their defaults.
              fusion_method       "EnKF_localized" (stochastic EnKF) | "LETKF" (deterministic ensemble transform)
              letkf               {"min_taper": 0.0}  LETKF only: observations with a taper weight below min_taper are
                                  dropped from a cell's local set (Gaussian localization)
              localization        {"kind": "none" | "block" | "gaussian", "length_km": 300, "cutoff": 2.0}
              inflation           {"kind": "none" | "multiplicative" | "rtps" | "additive" | "adaptive_additive", ...}
              increment_partition {"kind": "enkf" | "non_negative", ...}
              obs_error_inflation factor on the error variance of every sub-basin, e.g. 0.5; 1 = off
              obs_error_correlation "full" (as delivered) | "diagonal" (no error correlation between sub-basins; the
                                  member observations are re-perturbed consistently)
              obs_perturbation_centering  true: member GRACE perturbations shifted to zero mean
              soil_upper_bound    true: soil <= smax inside the bound-aware partition (the water soil cannot hold
                                  goes to the other storages); false: only the post-update threshold clips soil
              soil_daily_bounds   true: the soil increment of each day is shaped so that every day of the window
                                  stays within [0, smax] (src_DA.EnKF._soil_cap_config)
              increment_release   {"storages": ["riverstor"], "days": 10}: the increment of fast storages enters the
                                  next window over its first days (src_DA.EnKF._release_config)
              snow_bounds         limits of the analysed snow (read with snow_bounds() below)'''
            self.fusion_method = FusionMethod.EnKF_localized.name
            self.localization = {'kind': 'block'}
            self.inflation = {'kind': 'none'}
            self.increment_partition = {'kind': 'enkf'}
            self.obs_error_inflation = 1.0
            self.obs_error_correlation = 'full'
            self.obs_perturbation_centering = False
            self.soil_upper_bound = True
            self.soil_daily_bounds = False
            self.increment_release = None
            self.snow_bounds = {}
            self.letkf = {}


SNOW_BOUNDS_DEFAULT = {'lower_factor': 0.5, 'upper_factor': 2.0, 'upper_offset_mm': 20.0,
                       'envelope_factor': None, 'envelope_offset_mm': 10.0}


def snow_bounds(method):
    """
    The snow limits of the "method" block (object or dict) as one dict, the only reader of these settings:
        lower_factor                       snow >= f x forecast                                 (None = off)
        upper_factor, upper_offset_mm      snow <= f x forecast + o, per window                  (None = off)
        envelope_factor, envelope_offset_mm  snow <= f x max_OL(cell, calendar month) + o        (None = off)
    Missing entries take the defaults (envelope off); null / false switches a limit off.
    """
    get = method.get if isinstance(method, dict) else (lambda k, d=None: getattr(method, k, d))
    sb = dict(SNOW_BOUNDS_DEFAULT)
    new = get('snow_bounds', None)
    if isinstance(new, dict):
        unknown = [k for k in new if k not in SNOW_BOUNDS_DEFAULT]
        if unknown:
            raise ValueError('snow_bounds: unknown entries %s (allowed: %s)' % (unknown, list(SNOW_BOUNDS_DEFAULT)))
        sb.update(new)

    '''false / 0 -> off, numbers as float'''
    for k in ('lower_factor', 'upper_factor', 'envelope_factor'):
        sb[k] = float(sb[k]) if sb[k] not in (None, False) else None
    for k in ('upper_offset_mm', 'envelope_offset_mm'):
        sb[k] = float(sb[k] or 0.0)
    if sb['envelope_factor'] is not None and (sb['envelope_factor'] < 1.0 or sb['envelope_offset_mm'] < 0.0):
        raise ValueError('snow_bounds: envelope_factor must be >= 1 and envelope_offset_mm >= 0, got %s, %s'
                         % (sb['envelope_factor'], sb['envelope_offset_mm']))
    return sb
