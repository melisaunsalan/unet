import torch
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
from dataset import S1_S2_Dataset
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def plot_side_by_side(data):
    plt.subplot(121)
    plt.imshow(data['s1_data'], cmap = 'gray')
    plt.title('Sentinel 1 image')
    plt.subplot(122)
    plt.imshow(data['s2_data'])
    plt.title('Sentinel 2 image')
    plt.suptitle(data['labels'])
    plt.show()

if __name__ == '__main__':

    data = S1_S2_Dataset(s1_path='data/BigEarthNet-S1-v1.0', s2_path = 'data/BigEarthNet-S2-v1.0', size = 10)
    print(len(data))
    plot_side_by_side(data[9])