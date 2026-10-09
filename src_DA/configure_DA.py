import json
from pathlib import Path
import numpy as np
from src_DA.EnumDA import HydroModel, FusionMethod, WaterGap_storage_variables


class config_DA:

    def __init__(self):
        self.basic = self.config_basic().__dict__
        self.obs = self.config_obs().__dict__
        self.model = self.config_model().__dict__
        self.method = self.config_method().__dict__

        pass

    @staticmethod
    def save_default(default_path = None):
        obj1 = config_DA()
        tc_dic = obj1.__dict__
        if default_path is None:
            default_path = Path.cwd().parent / 'settings' / 'DA_setting.json'
        else:
            default_path = Path(default_path)/'DA_setting.json'
        with open(default_path, 'w') as f:
            json.dump(tc_dic, f, indent=4)
        return default_path

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
        def __init__(self):
            self.fromdate = '2000-01-02'
            self.todate = '2000-01-31'
            self.ensemble = 30
            self.case = 'test'
            self.basin = 'Brahmaputra'
            self.basin_shp = '/media/user/My Book/Fan/WaterGap/Basin/shp/Brahmaputra/Brahmaputra.shp'
            self.basin_mask = '/media/user/My Book/Fan/WaterGap/Basin/mask/Brahmaputra/Brahmaputra_res_0.5.h5'
            self.res_permanent = '/media/user/My Book/Fan/WaterGap/Res'
            self.OL_output_temp_dir = '/media/user/My Book/Fan/WaterGap/OL_output'
            self.DA_output_temp_dir = '/media/user/My Book/Fan/WaterGap/DA_output'
            self.Ensemble_ini_dir = '/media/user/My Book/Fan/WaterGap/Ensemble_Initialization'
            self.Ensemble_input_dir = '/media/user/My Book/Fan/WaterGap/Ensemble_input'
            self.Auxiliary_dir = '/media/user/My Book/Fan/WaterGap/Auxiliary'
            pass

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

        def __init__(self):
            self.name = 'src_OBS'
            self.dir = '/media/user/My Book/Fan/WaterGap/GRACE/obs'
            self.GRACE = {
                'EWH_grid_dir': '/media/user/My Book/Fan/WaterGap/GRACE/SaGEA/signal_Mascon',
                'cov_dir': '/media/user/My Book/Fan/WaterGap/GRACE/SaGEA/sample_DDK3',
                'preprocess_res': '/media/user/My Book/Fan/WaterGap/GRACE/output',
                'OL_mean': '/media/user/My Book/Fan/WaterGap/GRACE/OLmean',
                'aux_for_time_epochs':'/media/user/My Book/Fan/WaterGap/GRACE/SaGEA/signal_Mascon',
                'kind': 'Mascon_monthly'
            }

    class config_method:

        def __init__(self):
            self.fusion_method = FusionMethod.EnKF_v0.name
            '''EnKF_localized: the filter is assembled from independent components (src_DA.filter_factory),
            each chosen with its own block; see doc/configuration.md for all options
              localization        {"kind": "none" | "block" | "gaussian", "length_km": 300, "cutoff": 2.0}
              inflation           {"kind": "none" | "multiplicative" | "rtps" | "additive" | "adaptive_additive", ...}
                                  e.g. {"kind": "rtps", "alpha": 0.7, "space": "state"}
                                       {"kind": "additive", "sigma": {"groundwstor": 15, "swe": 5}, "seed": 42}
                                       {"kind": "adaptive_additive", "split": {"groundwstor": 0.9, "swe": 0.1}}
              increment_partition {"kind": "enkf" | "non_negative"}
              obs_error_inflation {sub_basin_id (1-based, as string): factor on the error variance}, {} = off
              obs_error_correlation "full" (as delivered) | "diagonal" (no error correlation between sub-basins;
                                  the member observations are re-perturbed consistently)
              obs_perturbation_centering  true: member GRACE perturbations shifted to zero mean
              soil_upper_bound    true: soil <= smax (Auxiliary/smax.nc) inside the bound-aware partition, the water
                                  soil cannot hold goes to the other storages; false: only the post-update threshold
                                  clips soil at smax (the water is then lost)
              snow_bounds         all limits of the analysed snow in one block (read with snow_bounds() below):
                                  {"lower_factor": 0.5,                          snow >= f x forecast
                                   "upper_factor": 2.0, "upper_offset_mm": 20,   snow <= f x forecast + o (per window)
                                   "envelope_factor": 1.5, "envelope_offset_mm": 10}
                                                 snow <= f x the largest open-loop snow of the calendar month + o
                                                 (per cell; stops the growth of packs that never melt)
                                  a missing lower / upper entry takes the default above, null / false switches it off;
                                  without envelope_factor the envelope cap is off
            Old flat keys (inflation as a number, rtps_alpha, rtps_space, additive_inflation) are still read and
            translated by src_DA.filter_factory; the old snow keys (snow_upper_bound, snow_lower_bound,
            snow_envelope) by snow_bounds().'''
            self.localization = {'kind': 'block', 'length_km': 300, 'cutoff': 2.0}
            self.inflation = {'kind': 'none'}
            self.increment_partition = {'kind': 'enkf'}
            self.obs_error_inflation = {}
            self.obs_error_correlation = 'full'     # 'diagonal': R without correlations between sub-basins (test)
            self.obs_perturbation_centering = False  # True: member GRACE perturbations shifted to zero mean
            self.soil_upper_bound = True              # soil <= smax inside the non-negative partition
            self.snow_bounds = {'lower_factor': 0.5, 'upper_factor': 2.0, 'upper_offset_mm': 20.0}  # envelope off



SNOW_BOUNDS_DEFAULT = {'lower_factor': 0.5, 'upper_factor': 2.0, 'upper_offset_mm': 20.0,
                       'envelope_factor': None, 'envelope_offset_mm': 10.0}


def snow_bounds(method):
    """
    The snow limits of the "method" block (object or dict) as one dict, the only reader of these settings:
        lower_factor                       snow >= f x forecast                                 (None = off)
        upper_factor, upper_offset_mm      snow <= f x forecast + o, per window                  (None = off)
        envelope_factor, envelope_offset_mm  snow <= f x max_OL(cell, calendar month) + o        (None = off)
    New block "snow_bounds" (10 Oct 2026); missing entries take the defaults (envelope off), null / false switches a
    limit off. The old keys of earlier settings are translated and replace the corresponding entries:
        "snow_lower_bound": {"factor": f} | false
        "snow_upper_bound": {"factor": f, "offset_mm": o} | false
        "snow_envelope":    {"factor": f, "offset_mm": o} | false
    Results are identical to the old keys.
    """
    get = method.get if isinstance(method, dict) else (lambda k, d=None: getattr(method, k, d))
    sb = dict(SNOW_BOUNDS_DEFAULT)
    new = get('snow_bounds', None)
    if isinstance(new, dict):
        unknown = [k for k in new if k not in SNOW_BOUNDS_DEFAULT]
        if unknown:
            raise ValueError('snow_bounds: unknown entries %s (allowed: %s)' % (unknown, list(SNOW_BOUNDS_DEFAULT)))
        sb.update(new)

    '''old keys (settings before 10 Oct 2026): true / null = default, false / {} = off'''
    old = get('snow_lower_bound', None)
    if isinstance(old, dict) or old is False:
        sb['lower_factor'] = float(old.get('factor', 0.5)) if old else None
    old = get('snow_upper_bound', None)
    if isinstance(old, dict) or old is False:
        sb['upper_factor'] = float(old.get('factor', 2.0)) if old else None
        sb['upper_offset_mm'] = float(old.get('offset_mm', 20.0)) if old else sb['upper_offset_mm']
    old = get('snow_envelope', None)
    if isinstance(old, dict) or old is False:
        sb['envelope_factor'] = float(old.get('factor', 1.5)) if old else None
        sb['envelope_offset_mm'] = float(old.get('offset_mm', 10.0)) if old else sb['envelope_offset_mm']

    '''false / 0 -> off, numbers as float'''
    for k in ('lower_factor', 'upper_factor', 'envelope_factor'):
        sb[k] = float(sb[k]) if sb[k] not in (None, False) else None
    for k in ('upper_offset_mm', 'envelope_offset_mm'):
        sb[k] = float(sb[k] or 0.0)
    if sb['envelope_factor'] is not None and (sb['envelope_factor'] < 1.0 or sb['envelope_offset_mm'] < 0.0):
        raise ValueError('snow_bounds: envelope_factor must be >= 1 and envelope_offset_mm >= 0, got %s, %s'
                         % (sb['envelope_factor'], sb['envelope_offset_mm']))
    return sb


def demo1():
    dp = config_DA.save_default(default_path='/media/user/My Book/Fan/WaterGap/Extensions/DA_settings')
    a = config_DA.loadjson(dp).process()

    pass


if __name__ == '__main__':
    demo1()
