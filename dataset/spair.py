import os
import re
import json
from glob import glob
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

from .dataset import CorrespondenceDataset


class SPairDataset(CorrespondenceDataset):
    benchmark = 'spair'

    def __init__(self, cfg, split, category='all', transform=None):
        super().__init__(cfg, split=split, transform=transform)
        self.cls = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "train", "tvmonitor"]

        self.cls_dict = {cat: i for i, cat in enumerate(self.cls)}

        if split not in ["trn", "val", "test"]:
            raise ValueError(f"Invalid split: {split}, select from ('trn', 'val', 'test')")
        
        self.data = open(self.spt_path).read().strip().split('\n') # e.g. 000001-2008_001546-2008_004704:aeroplane
        self.filter_category(category)

        self.src_imnames = [f"{x.split('-')[1]}.jpg" for x in self.data] # e.g. 2008_001546.jpg
        self.trg_imnames = [f"{x.split('-')[2].split(':')[0]}.jpg" for x in self.data] # e.g. 2008_004704.jpg

        self.seg_path = os.path.join(os.path.dirname(self.img_path), 'Segmentation')
        self.load_annotations()

    def filter_category(self, category):
        """Filters the data to include only pairs from the specified category."""
        options = self.cls + ["all"]
        if category not in options:
            raise ValueError(f"Invalid category: {category}, select from {options}.")
        self.data = [pair for pair in self.data if category == "all" or category in pair]

    def load_annotations(self):
        anntn_files = [glob(f'{self.ann_path}/{data_name}.json')[0] for data_name in self.data]
        self.cls_ids = []
        # self.vpvar, self.scvar, self.trncn, self.occln = [], [], [], []

        for anntn_file in tqdm(anntn_files, desc="Reading SPair-71k information"):
            with open(anntn_file) as f:
                anntn = json.load(f)
            self.src_kps.append(torch.tensor(anntn['src_kps']).float()) # n_pts, 2
            self.trg_kps.append(torch.tensor(anntn['trg_kps']).float()) # n_pts, 2
            self.src_bbox.append(torch.tensor(anntn['src_bndbox']).float()) # 4,
            self.trg_bbox.append(torch.tensor(anntn['trg_bndbox']).float()) # 4,
            self.cls_ids.append(self.cls_dict[anntn['category']])

            # Not commonly used
            # self.vpvar.append(torch.tensor(anntn['viewpoint_variation']))
            # self.scvar.append(torch.tensor(anntn['scale_variation']))
            # self.trncn.append(torch.tensor(anntn['truncation']))
            # self.occln.append(torch.tensor(anntn['occlusion']))

        self.src_ids = [f"{self.cls[ids]}-{name[:-4]}" for ids, name in zip(self.cls_ids, self.src_imnames)]
        self.trg_ids = [f"{self.cls[ids]}-{name[:-4]}" for ids, name in zip(self.cls_ids, self.trg_imnames)]

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)

        # Not commonly used
        # batch['src_mask'] = self.get_mask(batch, batch['src_imname'], (Hs, Ws))
        # batch['trg_mask'] = self.get_mask(batch, batch['trg_imname'], (Ht, Wt))
        # batch.update({'vpvar': self.vpvar[idx], 'scvar': self.scvar[idx], 'trncn': self.trncn[idx], 'occln': self.occln[idx]})

        return batch

    def get_image(self, img_names, idx):
        """Loads and returns an RGB image given its name and index."""
        path = os.path.join(self.img_path, self.cls[self.cls_ids[idx]], img_names[idx])
        return Image.open(path).convert('RGB')
    
    def get_mask(self, batch, imname, scaled_imsize):
        """Loads a Binary mask (0 for background, 255 for object) at scaled_imsize."""
        mask_path = os.path.join(self.seg_path, batch['category'], f"{imname.split('.')[0]}.png")
        tensor_mask = torch.from_numpy(np.array(Image.open(mask_path)))
        class_id = self.cls_dict[batch['category']] + 1
        tensor_mask = (tensor_mask == class_id).float().unsqueeze(0).unsqueeze(0) 
        return F.interpolate(tensor_mask, size=scaled_imsize, mode='nearest').int().squeeze() * 255

    
class SPairImageDataset(Dataset):
    benchmark = 'spair'

    def __init__(self, cfg, split, category, transform):
        """
        A PyTorch Dataset for loading images from the SPair-71k dataset.

        Args:
            root (str): Root directory of the dataset.
            split (str): Dataset split ('trn', 'val', 'test', or 'all').
            category (str): Image category or 'all' for all categories.
            transform (callable): Transform to be applied on batch.
        """
        self.root = os.path.join(cfg.DATASET.ROOT, 'SPair-71k')
        self.transform = transform
        self.cls = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "train", "tvmonitor"]

        options = self.cls + ["all"]
        if category not in options:
            raise ValueError(f"Invalid category: {category}, select from {options}.")

        self._load_data(split, category)

    def _load_data(self, split, category):
        if split not in ["trn", "val", "test", "all"]:
            raise ValueError(f"Invalid split: {split}, select from ('trn', 'val', 'test', 'all')")    
        splits = ["trn", "val", "test"] if split == "all" else [split]
    
        layout_dir = os.path.join(self.root, 'Layout', 'large')
        imnames = set()
        for split in splits:
            with open(os.path.join(layout_dir, f'{split}.txt')) as f:
                for line in f:
                    _, img1, img2, cat = re.split('[-:]', line.strip())
                    if category == "all" or cat == category:
                        imnames.update([f"{cat}-{img1}", f"{cat}-{img2}"])

        self.imnames = sorted(imnames)
        self.impaths = [os.path.join(self.root, 'JPEGImages', *x.split('-')) + '.jpg' for x in self.imnames]

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
            "identifier": self.imnames[index],
            "impath": self.impaths[index]
        }