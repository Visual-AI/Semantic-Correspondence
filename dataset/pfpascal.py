import os

from tqdm import tqdm
import scipy.io as sio
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .dataset import CorrespondenceDataset


class PFPascalDataset(CorrespondenceDataset):
    benchmark = 'pfpascal'

    def __init__(self, cfg, split, category="all", transform=None):
        super().__init__(cfg, split=split, transform=transform)
        self.cls = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        if split not in ["trn", "val", "test"]:
            raise ValueError(f"Invalid split: {split}, select from ('trn', 'val', 'test')")

        self.data = pd.read_csv(self.spt_path)
        self.filter_category(category)

        self.src_imnames = [os.path.basename(path) for path in self.data['source_image'].tolist()]
        self.trg_imnames = [os.path.basename(path) for path in self.data['target_image'].tolist()]

        self.load_annotations()

    def filter_category(self, category):
        """Filters the data to include only pairs from the specified category."""
        if category != "all":
            if category not in self.cls:
                raise ValueError(f"Invalid category: {category}, select from {self.cls + ['all']}.")
            self.data = self.data[self.data['class'].apply(lambda x: self.cls[x-1] == category)]

    def load_annotations(self):
        self.cls_ids = (self.data['class'] - 1).tolist()
        self.flip = self.data['flip'].tolist() if 'flip' in self.data.columns else [0] * len(self.data)

        for src_imname, trg_imname, cls_id in tqdm(zip(self.src_imnames, self.trg_imnames, self.cls_ids), 
                                                   total=len(self.data), desc="Reading PF-Pascal information"):
            src_anns = os.path.join(self.ann_path, self.cls[cls_id], src_imname[:-4] + '.mat')
            trg_anns = os.path.join(self.ann_path, self.cls[cls_id], trg_imname[:-4] + '.mat')

            src_data = sio.loadmat(src_anns)
            trg_data = sio.loadmat(trg_anns)

            src_kps = torch.from_numpy(src_data['kps']).float().floor()
            trg_kps = torch.from_numpy(trg_data['kps']).float().floor()
            src_box = torch.from_numpy(src_data['bbox'].squeeze().astype(np.float32))
            trg_box = torch.from_numpy(trg_data['bbox'].squeeze().astype(np.float32))

            valid_mask = torch.isfinite(src_kps).all(dim=1) & torch.isfinite(trg_kps).all(dim=1)
            src_kps = src_kps[valid_mask]
            trg_kps = trg_kps[valid_mask]

            self.src_kps.append(src_kps)
            self.trg_kps.append(trg_kps)
            self.src_bbox.append(src_box)
            self.trg_bbox.append(trg_box)

            self.src_ids = [f"{self.cls[ids]}-{name[:-4]}-{flip}" for ids, name, flip in zip(self.cls_ids, self.src_imnames, self.flip)]
            self.trg_ids = [f"{self.cls[ids]}-{name[:-4]}-{flip}" for ids, name, flip in zip(self.cls_ids, self.trg_imnames, self.flip)]
    
    def __getitem__(self, idx):
        """Constructs and returns a batch for PF-PASCAL dataset"""
        batch = super().__getitem__(idx)
        if self.flip[idx]:
            self.horizontal_flip(batch)
        batch['flip'] = torch.tensor([self.flip[idx]])
        return batch
    
    def get_image(self, imnames, idx):
        """Loads and returns an RGB image given its name and index."""
        return Image.open(os.path.join(self.img_path, imnames[idx])).convert('RGB')

    def horizontal_flip(self, batch):
        """Horizontally flips images, bounding boxes, and keypoints in the batch."""
        x_max, n_pts = self.img_size[1], batch['n_pts']

        batch['src_bbox'][0], batch['src_bbox'][2] = x_max - batch['src_bbox'][2], x_max - batch['src_bbox'][0]
        batch['trg_bbox'][0], batch['trg_bbox'][2] = x_max - batch['trg_bbox'][2], x_max - batch['trg_bbox'][0]

        batch['src_kps'][:n_pts, 0] = x_max - batch['src_kps'][:n_pts, 0]
        batch['trg_kps'][:n_pts, 0] = x_max - batch['trg_kps'][:n_pts, 0]

        batch['src_img'] = torch.flip(batch['src_img'], dims=(2,))
        batch['trg_img'] = torch.flip(batch['trg_img'], dims=(2,))


class PFPascalImageDataset(Dataset):
    benchmark = 'pfpascal'

    def __init__(self, cfg, split, category, transform):
        """
        A PyTorch Dataset for loading images from the PF-Pascal dataset.

        Args:
            root (str): Root directory of the dataset.
            split (str): Dataset split ('trn', 'val', 'test', or 'all').
            category (str): Image category or 'all' for all categories.
            transform (callable): Transform to be applied on sample.
        """
        self.root = os.path.join(cfg.DATASET.ROOT, 'pf-pascal')
        self.transform = transform
        self.cls = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        if category not in self.cls + ["all"]:
            raise ValueError(f"Invalid category: {category}. Select from {self.cls + ['all']}.")

        self._load_data(split, category)

    def _load_data(self, split, category):
        valid_splits = ["trn", "val", "test", "all"]
        if split not in valid_splits:
            raise ValueError(f"Invalid split: {split}. Select from {valid_splits}.")
        
        splits = ["trn", "val", "test"] if split == "all" else [split]
        
        unique = set()
        for split in splits:
            image_pairs = pd.read_csv(os.path.join(self.root, f'{split}_pairs.csv'))
            for _, pair in image_pairs.iterrows():
                src, trg = [p.split("/")[-1][:-4] for p in [pair['source_image'], pair['target_image']]]
                flip = pair['flip'] if split == "trn" else 0
                cls = self.cls[pair["class"]-1]
                if cls == category or category == "all":
                    unique.update([f"{cls}-{src}-{flip}", f"{cls}-{trg}-{flip}"])

        self.imnames = sorted(unique)
        self.impaths = [os.path.join(self.root, 'PF-dataset-PASCAL/JPEGImages', f"{x.split('-')[1]}.jpg") for x in self.imnames]
        self.category = [x.split('-')[0] for x in self.imnames]

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
        if int(self.imnames[index][-1]):
            img = torch.flip(img, (2,)) # h-flip

        return {
            "pixel_values": img,
            "id": self.imnames[index],
            "impath": self.impaths[index]
        }