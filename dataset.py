from osgeo import gdal
from torch.utils.data import Dataset
import os
import json
import numpy as np

def read_metadata(json_path):
    f = open(json_path)
    metadata = json.load(f)
    f.close()
    return {'labels': metadata['labels'], 's2_patch': metadata['corresponding_s2_patch']}

def normalize(arr):
    m = np.min(arr)
    M = np.max(arr)
    return (arr - m)/(M-m)

class S1_S2_Dataset(Dataset):

    def __init__(self, s1_path, s2_path, size = -1, transform = None):
        self.s1_path = s1_path
        self.s2_path = s2_path
        self.s1_images = sorted(os.listdir(self.s1_path))[0:size]
        self.s2_images = sorted(os.listdir(self.s2_path))[0:size]
        self.transform = transform

    def __len__(self):
        return len(self.s2_images)

    def __getitem__(self, idx):

        # Load Sentinel-1 image

        s1_image = self.s1_images[idx]
        s1_image_path = os.path.join(self.s1_path, s1_image, s1_image + '_VH.tif')
        s1_data = gdal.Open(s1_image_path).ReadAsArray()
        s1_data = normalize(s1_data)

        # Get label and corresponding Sentinel-2 patch from metdata
        metadata = read_metadata(os.path.join(self.s1_path, s1_image, s1_image + '_labels_metadata.json'))

        # Load corresponsing Sentinel-2 image
        s2_image = metadata['s2_patch']
        r = gdal.Open(os.path.join(self.s2_path, s2_image, s2_image + '_B04.tif')).ReadAsArray()
        g = gdal.Open(os.path.join(self.s2_path, s2_image, s2_image + '_B03.tif')).ReadAsArray()
        b = gdal.Open(os.path.join(self.s2_path, s2_image, s2_image + '_B02.tif')).ReadAsArray()
        s2_data = np.asarray([normalize(r), normalize(g), normalize(b)]).transpose(1,2,0)

        return {'s1_data': s1_data, 's2_data': s2_data, 'labels': metadata['labels']}




