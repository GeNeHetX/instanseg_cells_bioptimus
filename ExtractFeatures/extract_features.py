from openslide.deepzoom import DeepZoomGenerator
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from torch.utils.data import Dataset
from tqdm import tqdm

from pathlib import Path
import os

OPENSLIDE_PATH = r"c:\Users\inserm\Documents\openslide-bin-4.0.0.3-windows-x64\bin"


import openslide
    
import xml.etree.ElementTree as ET

import numpy as np
import openslide
import torch
from openslide.deepzoom import DeepZoomGenerator
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm
import torch
from torchvision.models.resnet import Bottleneck, ResNet
import timm
from torchvision import transforms

import pandas as pd

import h5py
from shapely.geometry import Point
from shapely.geometry import MultiPolygon, Polygon

class TilesDataset(Dataset):
    def __init__(self, slide: openslide.OpenSlide, tiles_coords: np.ndarray, extractor : str) -> None:
        self.slide = slide
        self.tiles_coords = tiles_coords
        self.extractor = extractor 

        if self.extractor == "optimus" : 
            self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)
                ),
            ])
        self.dz = DeepZoomGenerator(slide, tile_size=224, overlap=0)
        file_extension = Path(self.slide._filename).suffix
        if file_extension == ".svs":
            self.magnification = int(self.slide.properties["openslide.objective-power"])
        elif file_extension == ".qptiff":
            r = (
                ET.fromstring(slide.properties["openslide.comment"])
                .find("ScanProfile")
                .find("root")
                .find("ScanResolution")
            )
            self.magnification = float(r.find("Magnification").text)
        elif file_extension == ".ndpi":
            self.magnification = int(self.slide.properties["openslide.objective-power"])
        elif file_extension == ".tiff": # experimental support of ome.tiff
            root = ET.fromstring(slide.properties["openslide.comment"])
            self.magnification = int(float(root[0][0].attrib["NominalMagnification"]))
        else:
            try :
                self.magnification = int(self.slide.properties["openslide.objective-power"])
            except:
                raise ValueError(f"File extension {file_extension} not supported")
        # We want the second highest level so as to have 112 microns tiles / 0.5 microns per pixel
        if self.magnification == 20 :
            self.level = self.dz.level_count - 1
        elif self.magnification == 40 or self.dz.level_count==18:
            self.level = self.dz.level_count - 2
            self.magnification = 20
        else:
            raise ValueError(f"Objective power {self.magnification}x not supported")
        self.z = self.level

        
    def __getitem__(self, item: int):
        #tile_coord = (int(self.tiles_coords[item][0]-(112)),int(self.tiles_coords[item][1]-(112)))
        if self.extractor =="UNI":
            tile_coord = (int(self.tiles_coords[item][0]-(112)),int(self.tiles_coords[item][1]-(112)))
        if self.extractor =="optimus":
            tile_coord = (int(self.tiles_coords[item][0]-(112)),int(self.tiles_coords[item][1]-(112)))
        try:
            im = self.slide.read_region(location=tile_coord,level=0, size=(224,224))
        except ValueError:
            print(f"ValueError: impossible to open tile {tile_coord} from {self.slide}")
            raise ValueError


        # im = ToTensor()(im)
        # if im.shape != torch.Size([3, 224, 224]):
        #     print(f"Image shape is {im.shape} for tile {tile_coords}. Padding...")
        #     # PAD the image in white to reach 224x224
        #     im = torch.nn.functional.pad(im, (0, 224 - im.shape[2], 0, 224 - im.shape[1]), value=1)
        im = im.convert("RGB")
        im = self.transform(im)
        return im

    def __len__(self) -> int:
        return len(self.tiles_coords)



def get_features(
    slide: openslide.OpenSlide,
    model: torch.nn.Module,
    tiles_coords: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 0,
    prefetch_factor: int = 8,
    extractor : str = "optimus"
) -> np.ndarray:

    dataset = TilesDataset(slide=slide, tiles_coords=tiles_coords,extractor=extractor)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,  # num_workers=0 is necessary when using windows
        pin_memory=True ,
        drop_last=False
    )


    features_l = []
    dtype = next(model.parameters()).dtype

    if dataset.extractor == "optimus": 
        with torch.autocast(device_type="cuda:0", dtype=torch.float16):
            with torch.inference_mode():
                for batch in tqdm(dataloader, leave = False):
                    features = model(batch.type(dtype).to(device))
                    cls_tokens = features[:,0].cpu().detach()
                    patches = features[:,model.num_prefix_tokens:]
                    #features_pool = patches[0,119,:].cpu().detach()
                    features_pool = torch.mean(patches[:,(119,  120, 135, 136),:],1).cpu().detach() #optimus
                    _ = [features_l.append(i) for i in features_pool]
    
    if dataset.extractor == "UNI" : 
        for batch in tqdm(dataloader, leave = False):
            features = model(batch.type(dtype).to(device))
            cls_tokens = features[0,0].cpu().detach()
            patches = features[:,model.num_prefix_tokens:]
            features_pool = patches[0,90,:].cpu().detach()
            #features_pool = torch.mean(features_b[0,(90,  91, 104, 105),:],0).cpu().detach() #UNI   
            features_l.append(features_pool)
    

    features_l = torch.concat(features_l).cpu().numpy()
    return features_l





def extract_features(
    slide_path: Path,
    device: torch.device,
    checkpoint_path , patientName, geojson ,
    batch_size: int = 32,
    outdir: Path = None,
    num_workers: int = 0,
    extractor : str = "UNI",


):  
    

    try : 
        slide = openslide.OpenSlide(str(slide_path))

    except : 
        print(f"\t Image {patientName} couldn't be opened ")
        return []
    else :  

        features = get_features(
            slide=slide,
            model=checkpoint_path,
            tiles_coords=geojson,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            extractor = extractor
        )



        with h5py.File(f"{outdir}/{patientName}_cell_detection.h5", 'w') as f2 :
            f2["position"]=geojson
            f2["features"]=features
            f2["name"]=f"{patientName}"
            f2["tool"]= f"{extractor}_patches_cell_embeding"
            f2["width"] = "20x"

        
    return []
