import os
import numpy as np
import h5py
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Demerdez-vous pour installer openslide sur votre machine
OPENSLIDE_PATH = r"c:\Users\inserm\Documents\openslide-bin-4.0.0.3-windows-x64\bin"


import openslide


from extract_tiles import parse_geojson
from extract_features import extract_features

from shapely.geometry import Point
from shapely.geometry import MultiPolygon, Polygon

from pathlib import Path
from torch import device

from argparse import ArgumentParser

from torch import device
import pandas as pd
import timm
import torch

def parse_arg():
    parser = ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(r"~/Documents/Beaujon/Thèse/"),
        help="Path to the temporary directory where the features will be saved",
        required=True,
    )
    parser.add_argument(
        "--wsi_dir",
        type=Path,
        default=Path(r"/mnt/d/01_HESCohortJulien/13AG/"),
        help="Path to the WSI. Can be a .svs, .ndpi, .qptiff",
    )
    parser.add_argument(
        "--geojson_dir",
        type=Path,
        default=Path(r"/Users/audreybfls/Documents/Beaujon/Thèse/svsFFX/"),
        help="Path to the geojson with cell detection",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path(r"pytorch_model_bioptimus.bin"),
        help="Path to the UNI model",
    )
    parser.add_argument(
        "--device", type=device, default="mps", help="Device to use for the predictions"
    ) 
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for the feature extraction")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for the feature extraction. Set to 0 if using windows.",
    )    
    parsed_args = parser.parse_args()
    return parser.parse_args()






def main(args):
    import glob
    from subscriptable_path import Path

    geo_dir = args.geojson_dir
    out_dir = args.out_dir

    files = glob.glob(f"{args.wsi_dir}/*.svs")

    nb_files = len(files)
    print(str(nb_files) + " SVS were found")
   
    
    model = timm.create_model("vit_giant_patch14_reg4_dinov2",img_size=224, patch_size=14, init_values=1e-5, num_classes=0, dynamic_img_size=True,global_pool="")
    model.load_state_dict(torch.load(args.model_path))
           
    model = model.to(str(args.device))
    i = 1
    for file in files :
        slidename = Path(os.path.basename(file)).stem
        print(str(i) + "/" + str(nb_files) + " : " +str(slidename))
   

        i = i +1

        if not os.path.exists(f"{out_dir}/{slidename}_cell_detection.h5") :
            #outdir = os.path.join(args.out_dir,slidename)
            
            geojson = f'{geo_dir}/{slidename}_centroid.geojson'
            
       
            

            print("\tExtract cell coord...")
            cells_coord = parse_geojson(geojson)
            
            print("\tExtracting tumoral features...")
            features_tum = extract_features(
                slide_path=file,
                device=args.device,
                batch_size=args.batch_size,
                outdir = args.out_dir,
           
                num_workers=args.num_workers,
                checkpoint_path = model,
                geojson = cells_coord , patientName=slidename,
                extractor = "optimus"
            )

            print("\tDone")
        else :
            print('\tFeatures already exists')

    print("Features are stored in " + str(args.out_dir))



if __name__ == "__main__":
    args = parse_arg()
    main(args)
