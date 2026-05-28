from pathlib import Path
import sys
import numpy as np
import pandas as pd
import importlib.util

# Load CellAnalyzer module from src/CellAnalyzer.py
root = Path(__file__).resolve().parents[1]
mod_path = root / 'src' / 'CellAnalyzer.py'
import types
import sys
cellpose_stub = types.ModuleType('cellpose')
cellpose_stub.core = types.SimpleNamespace(use_gpu=lambda: False)
cellpose_stub.denoise = types.SimpleNamespace(CellposeDenoiseModel=lambda **kw: None)
cellpose_stub.io = types.SimpleNamespace()
cellpose_stub.utils = types.SimpleNamespace(masks_to_outlines=lambda m: np.zeros_like(m))
sys.modules['cellpose'] = cellpose_stub

# Stub seaborn if missing
sns_stub = types.ModuleType('seaborn')
sys.modules['seaborn'] = sns_stub

# Stub aicsimageio if missing
aics_stub = types.ModuleType('aicsimageio')
aics_stub.AICSImage = lambda *args, **kwargs: None
sys.modules['aicsimageio'] = aics_stub

# Stub tifffile.tifffile.imwrite
tif_mod = types.ModuleType('tifffile')
tif_inner = types.SimpleNamespace(imwrite=lambda *a, **k: None)
tif_mod.tifffile = tif_inner
sys.modules['tifffile'] = tif_mod
# Also provide the submodule object expected by "from tifffile.tifffile import imwrite"
tif_inner_mod = types.ModuleType('tifffile.tifffile')
tif_inner_mod.imwrite = lambda *a, **k: None
sys.modules['tifffile.tifffile'] = tif_inner_mod

spec = importlib.util.spec_from_file_location('cellan', str(mod_path))
cellan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cellan)
CellAnalyzer = cellan.CellAnalyzer

# Monkeypatch utility outline function to keep things simple
try:
    cellan.utils.masks_to_outlines = lambda m: np.zeros_like(m)
except Exception:
    pass

# Dummy model to return artificially large masks
class DummyModel:
    def eval(self, img_list, diameter=None, channels=None):
        masks = []
        flows = []
        styles = []
        imgs_dn = []
        # Create masks with many unique labels (e.g., 40000 labels per image)
        for i in range(len(img_list)):
            h, w = 200, 200
            m = np.zeros((h, w), dtype=int)
            # Fill with sequential labels so max label == h*w
            flat = m.ravel()
            flat[:h*w] = np.arange(1, h*w+1)
            m = flat.reshape(h, w)
            masks.append(m)
            flows.append(None)
            styles.append(None)
            imgs_dn.append(None)
        return masks, flows, styles, imgs_dn

# Set up a CellAnalyzer instance
ca = CellAnalyzer('.')
# Replace heavy cellpose model with dummy
ca.cellpose_model = DummyModel()

# Prepare a small samples_df and projections list
n = 3
ca.samples_df = pd.DataFrame(index=range(n))
ca.samples_df['filename'] = [f'dummy_{i}.nd2' for i in range(n)]
ca.projections = [np.zeros((1, 10, 10), dtype=np.uint8) for _ in range(n)]
ca.samples_df['has_projection'] = [True] * n

# Run segmentation (this will use the DummyModel and should choose int32 for masks)
_ = ca.segment_cells(diameter=50, channels=[0,0], log=False, calculate_neighbours=False)

# Inspect results
for i, m in enumerate(ca.masks):
    print(f"Mask {i}: dtype={m.dtype}, max={int(m.max())}")

print('cells_df rows:', 0 if ca.cells_df is None else len(ca.cells_df))
print('samples_df num_cells:', ca.samples_df['num_cells'].tolist())

# Verify dtype is int32
dtypes_ok = all([m.dtype == np.int32 for m in ca.masks if m is not None])
print('All masks int32:', dtypes_ok)
