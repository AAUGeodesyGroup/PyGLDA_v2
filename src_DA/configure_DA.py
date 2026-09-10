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





def demo1():
    dp = config_DA.save_default(default_path='/media/user/My Book/Fan/WaterGap/Extensions/DA_settings')
    a = config_DA.loadjson(dp).process()

    pass


if __name__ == '__main__':
    demo1()
