import sys
import os

# Add ReWaterGAP directory to sys.path so WaterGAP's internal bare imports resolve.
# WaterGAP's source files use bare module names (e.g. 'import watergap_logger',
# 'from controller import ...') that assume ReWaterGAP/ is on the Python path.
# This must be done before any WaterGAP module is imported.
_rewatergap_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ReWaterGAP'))
if _rewatergap_dir not in sys.path:
    sys.path.insert(0, _rewatergap_dir)
