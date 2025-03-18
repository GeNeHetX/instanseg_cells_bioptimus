
import os
import glob
from pathlib import Path

image_dir = "/mnt/d/test_biopsy"
device = "cuda:0"


files = glob.glob(f"{image_dir}/*")

instanseg_cmd =f"python3.10 /mnt/c/Users/inserm/Documents/Tools/instanseg-main/instanseg-main/instanseg/scripts/inference.py --model_folder brightfield_nuclei --image_path {image_dir} --device {device} --save_geojson true "
#os.system(instanseg_cmd)

outdir = "."

for file in files : 
    file = Path(file)
    input_geojson = f"{file.parent}/{file.stem}_instanseg_prediction.geojson " 
    output_geojson = f"{file.parent}/{file.stem}_centroid.geojson"
    cleaned_geojson_cmd = f"python3.10 geojson_utils.py --input_geojson {input_geojson} --output_geojson {output_geojson}"
   #os.system(cleaned_geojson_cmd)


extract_features_cmd = f"python3.10 process_wsi.py  --out_dir {outdir} --wsi_dir {image_dir} --geojson_dir {image_dir} --model_path ./pytorch_model_bioptimus.bin --device {device} "

os.system(extract_features_cmd)