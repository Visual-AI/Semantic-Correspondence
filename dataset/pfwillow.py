import os
import warnings
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from .dataset import CorrespondenceDataset


class PFWillowDataset(CorrespondenceDataset):
    benchmark = 'pfwillow'

    def __init__(self, cfg, split, category='all', transform=None):
        super().__init__(cfg, split=split, transform=transform)
        self.cls = ['car(G)', 'car(M)', 'car(S)', 'duck(S)', 'motorbike(G)', 'motorbike(M)', 'motorbike(S)', 'winebottle(M)', 'winebottle(wC)', 'winebottle(woC)']

        if split != "test":
            warnings.warn(f"Invalid split: {split}. Split will be forced to 'test'.")
            split = 'test'

        self.data = pd.read_csv(self.spt_path)

        self.filter_category(category)
        self.load_annotations()

    def filter_category(self, category):
        if category != "all":
            if category not in self.cls:
                raise ValueError(f"Invalid category: {category}, select from {self.cls + ['all']}.")
            self.data = self.data[self.data['imageA'].str.contains(category, regex=False)]

    def load_annotations(self):
        self.src_imnames = self.data['imageA'].values.tolist()
        self.trg_imnames = self.data['imageB'].values.tolist()

        self.src_kps = torch.from_numpy(self.data.iloc[:, 2:22].values).float().floor()
        self.src_kps = self.src_kps.view(-1, 2, 10).permute(0, 2, 1) # (data_len, 10, 2)
        self.trg_kps = torch.from_numpy(self.data.iloc[:, 22:].values).float().floor()
        self.trg_kps = self.trg_kps.view(-1, 2, 10).permute(0, 2, 1) # (data_len, 10, 2)

        self.src_bbox = self.get_bounding_boxes(self.src_kps)
        self.trg_bbox = self.get_bounding_boxes(self.trg_kps)

        self.cls_ids = [self.cls.index(name.split('/')[1]) for name in self.src_imnames]
        self.src_imnames = [os.path.join(*name.split('/')[1:]) for name in self.src_imnames]
        self.trg_imnames = [os.path.join(*name.split('/')[1:]) for name in self.trg_imnames]
        
        self.src_ids = [f"{self.cls[ids]}-{name.split('/')[1][:-4]}" for ids, name in zip(self.cls_ids, self.src_imnames)]
        self.trg_ids = [f"{self.cls[ids]}-{name.split('/')[1][:-4]}" for ids, name in zip(self.cls_ids, self.trg_imnames)]

    @staticmethod
    def get_bounding_boxes(keypoints):
        """(data_len, 10, 2) -> (data_len, 4)"""
        min_coords, _ = torch.min(keypoints, dim=1)
        max_coords, _ = torch.max(keypoints, dim=1)
        return torch.cat([min_coords, max_coords], dim=1)

    def get_image(self, imnames, idx):
        """Loads and returns an RGB image given its name and index."""
        return Image.open(os.path.join(self.img_path, imnames[idx])).convert('RGB')
    

class PFWillowImageDataset(Dataset):
    benchmark = 'pfwillow'

    def __init__(self, cfg, split, category, transform):
        self.root = os.path.join(cfg.DATASET.ROOT, 'pf-willow')
        self.transform = transform
        self.cls = ['car(G)', 'car(M)', 'car(S)', 'duck(S)', 'motorbike(G)', 'motorbike(M)', 'motorbike(S)', 'winebottle(M)', 'winebottle(wC)', 'winebottle(woC)']

        if category not in self.cls + ["all"]:
            raise ValueError(f"Invalid category: {category}, select from {self.cls + ['all']}.")

        if split != "test":
            warnings.warn(f"Invalid split: {split}. Split will be forced to 'test'.")
            split = 'test'

        image_pairs = pd.read_csv(os.path.join(self.root, 'test_pairs.csv'))

        unique = set()
        for _, pair in image_pairs.iterrows():
            cls = pair['imageA'].split("/")[1]
            if cls == category or category == "all":
                for img in [pair['imageA'], pair['imageB']]:
                    unique.add(f"{cls}-{img.split('/')[-1][:-4]}")

        unique = sorted(unique)
        
        self.imnames = unique
        self.impaths = [os.path.join(self.root, 'PF-dataset', cat, f"{img}.png") 
                        for cat, img in (name.split('-') for name in self.imnames)]
    def __len__(self):
        return len(self.imnames)
    
    def __getitem__(self, index):
        """
        Returns:
        dict: A dictionary containing:
            - 'pixel_values' (torch.Tensor): The image data.
            - 'id' (str): A unique identifier for the image. e.g. aeroplane-2010_000541
            - 'impath' (str): The file path of the image. e.g. root/SPair-71k/JPEGImages/aeroplane/2010_000541.jpg
        """
        img = Image.open(self.impaths[index]).convert('RGB')
        img = self.transform(img)

        return {
            "pixel_values": img,
            "id": self.imnames[index],
            "impath": self.impaths[index]
        }
