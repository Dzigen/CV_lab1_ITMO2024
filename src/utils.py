from dataclasses import dataclass
import pandas as pd
import os 
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import cv2
from transformers import AutoImageProcessor
import torch

from datasets import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from PIL import Image

TT_INFO = './data/tt_union_fronts_info.csv'
CONFIG_FILE_JSON = './src/learning_conf.json'

@dataclass
class LearningConfig:
    lr: int
    epochs: int
    batch_size: int
    device: str
    model_name: str
    classes: int

    base_dir: str
    run_name: str
    to_save: bool


class CustomCarDataset(Dataset):
    def __init__(self, part_name, fronts_info_path, processor):
        data_info = pd.read_csv(fronts_info_path, sep=';')
        self._data = data_info.loc[data_info['part'] == part_name, :].reset_index(drop=True)
        self.uniq_labels = data_info['label'].unique()
        self.labels_map = {label: i for i, label in enumerate(self.uniq_labels)}
        self.processor = processor

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        image_path = f"{self._data['relative_path'][idx]}/{self._data['image_name'][idx]}"
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.processor(image)
        image.close()
        
        label = self.labels_map[self._data['label'][idx]]

        return image_tensor, label
    
    def __getitems__(self, idxs):
        return [self.__getitem__(idx) for idx in idxs]
        

def custom_collate(data):

    images = torch.cat([torch.unsqueeze(item[0], 0) for item in data], 0)
    labels = torch.tensor([item[1] for item in data])

    return {
        "images": images, 
        "labels": labels
    }