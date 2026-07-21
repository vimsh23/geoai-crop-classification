import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

def create_qml(output_path):
    # Class nomenclature
    CLASS_NAMES = {
        0: "Background", 1: "Meadow", 2: "Soft Winter Wheat", 3: "Corn",
        4: "Winter Barley", 5: "Winter Rapeseed", 6: "Spring Barley", 
        7: "Sunflower", 8: "Grapevine", 9: "Beet", 10: "Winter Triticale",
        11: "Winter Durum Wheat", 12: "Fruits/Vegetables/Flowers", 13: "Potatoes",
        14: "Leguminous Fodder", 15: "Soybeans", 16: "Orchard",
        17: "Mixed Cereal", 18: "Sorghum", 19: "Void",
    }
    
    # Generate colors using tab20 (matching the python visualizer)
    tab20 = plt.get_cmap('tab20')
    
    qgis = ET.Element('qgis', version="3.22.0-Białowieża", styleCategories="AllStyleCategories")
    pipe = ET.SubElement(qgis, 'pipe')
    rasterrenderer = ET.SubElement(pipe, 'rasterrenderer', 
                                   type="paletted", band="1", 
                                   opacity="1", alphaBand="-1")
    colorPalette = ET.SubElement(rasterrenderer, 'colorPalette')
    
    for class_id, name in CLASS_NAMES.items():
        if class_id == 0:
            color = "#000000" # Black for background
        elif class_id == 19:
            color = "#ffffff" # White for void
        else:
            color = to_hex(tab20(class_id))
            
        ET.SubElement(colorPalette, 'paletteEntry', 
                      value=str(class_id), color=color, alpha="255", label=name)
                      
    xml_str = minidom.parseString(ET.tostring(qgis)).toprettyxml(indent="  ")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)
        
    print(f"QGIS Style file generated at: {output_path}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    merge_lbl_dir = os.path.join(project_root, 'outputs', 'geotiff', 'merge', 'mosaic_label')
    
    os.makedirs(merge_lbl_dir, exist_ok=True)
    qml_path = os.path.join(merge_lbl_dir, "mosaic_label.qml")
    
    create_qml(qml_path)
