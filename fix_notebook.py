import json
import os

path = r"c:\Users\Vimal Sharma\Downloads\ram_99\pastis-crop-classification\notebooks\ultimate_geoai_eda.ipynb"
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Cell 1
        if "from src.data_loading import load_config" in source:
            source = source.replace(
                "from src.data_loading import load_config, load_metadata, get_patch_ids, load_patch_data",
                "from src.data_loading import DataManager\nmanager = DataManager()"
            )
            source = source.replace("config = load_config()", "config = manager.config")
            source = source.replace("metadata = load_metadata(config['paths']['metadata'])", "metadata = manager.load_metadata()")
            source = source.replace("patch_ids = get_patch_ids(metadata)", "patch_ids = manager.get_patch_ids(metadata)")
            
        # Other cells
        source = source.replace(
            "s, t = load_patch_data(p, s2_dir, target_dir)",
            "s, t = manager.load_patch_data(p)"
        )
        
        source = source.replace(
            "s2_sample, target_sample = load_patch_data(sample_pid, s2_dir, target_dir)",
            "s2_sample, target_sample = manager.load_patch_data(sample_pid)"
        )

        # Convert back to list of lines for JSON
        lines = [line + '\n' for line in source.split('\n')]
        if lines:
            lines[-1] = lines[-1].rstrip('\n')
        cell['source'] = lines

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Notebook successfully updated.")
