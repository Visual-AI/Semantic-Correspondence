import json
from glob import glob
from tqdm import tqdm
import os

import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from .dataset import CorrespondenceDataset


class AP10KDataset(CorrespondenceDataset):
    benchmark = 'ap10k'

    def __init__(self, cfg, split, category='all', transform=None, subsample=None):
        super().__init__(cfg, split=split, transform=transform)
        self.cls = ['alouatta', 'antelope', 'beaver', 'bison', 'bobcat', 'brown bear', 'buffalo', 'cat', 'cheetah', 'chimpanzee', 'cow', 'deer', 'dog', 'elephant', 'fox', 'giraffe', 'gorilla', 'hamster', 'hippo', 'horse', 'jaguar', 'leopard', 'lion', 'marmot', 'monkey', 'moose', 'mouse', 'noisy night monkey', 'otter', 'panda', 'panther', 'pig', 'polar bear', 'rabbit', 'raccoon', 'rat', 'rhino', 'sheep', 'skunk', 'snow leopard', 'spider monkey', 'squirrel', 'tiger', 'uakari', 'weasel', 'wolf', 'zebra']
        self.cls_dict = {cat: i for i, cat in enumerate(self.cls)}
        
        if split not in ["trn", "val", "test"]:
            raise ValueError(f"Invalid split: {split}, select from ('trn', 'val', 'test')")
        
        # data = sorted(glob(f'{self.ann_path}/{split}/*.json'))
        self.data = []
        path = 'asset/ap-10k'
        for cat_idx, cat in tqdm(enumerate(self.cls), total=len(self.cls), desc="Processing Categories"):
            np.random.seed(42)
            pairs = sorted(glob(f'{path}/PairAnnotation/{split}/*:{cat}.json'))
            if subsample is not None and subsample > 0:
                pairs = [pairs[ix] for ix in np.random.choice(len(pairs), subsample)]
            self.data.extend(pairs)


        # self.filter_category(category)

        self.seg_path = os.path.join(os.path.dirname(self.img_path), 'features')
        self.load_annotations()

    # def filter_category(self, category):
    #     """Filters the data to include only pairs from the specified category."""
    #     options = self.cls + ["all"]
    #     if category not in options:
    #         raise ValueError(f"Invalid category: {category}, select from {options}.")
    #     self.data = [pair for pair in self.data if pair.split(':')[1].split('.')[0] not in self.exclude_cls]
    #     if category != 'all':
    #         self.data = [pair for pair in self.data if category == pair.split(':')[1].split('.')[0]]
            
    def load_annotations(self):
        self.src_impaths, self.trg_impaths, self.cls_ids = [], [], []

        for pair in tqdm(self.data, desc="Reading AP-10k information"):
            with open(pair) as f:
                pair_data = json.load(f)

            src_jspath, trg_jspath = pair_data["src_json_path"], pair_data["trg_json_path"]
            self.src_impaths.append(src_jspath.replace("json", "jpg").replace('ImageAnnotation', 'JPEGImages'))
            self.trg_impaths.append(trg_jspath.replace("json", "jpg").replace('ImageAnnotation', 'JPEGImages'))

            with open(src_jspath) as f:
                src_file = json.load(f)
            with open(trg_jspath) as f:
                trg_file = json.load(f)

            self.src_bbox.append(self._process_bbox(src_file)) # 4,
            self.trg_bbox.append(self._process_bbox(trg_file)) # 4,
            self.cls_ids.append(self.cls_dict[src_file['name']])

            src_kps, src_is_visible = self._process_keypoints(src_file)
            trg_kps, trg_is_visible = self._process_keypoints(trg_file)
            both_visible = src_is_visible & trg_is_visible
            self.src_kps.append(src_kps[both_visible])
            self.trg_kps.append(trg_kps[both_visible])

        self.src_imnames = [os.path.basename(path) for path in self.src_impaths]
        self.trg_imnames = [os.path.basename(path) for path in self.trg_impaths]
        self.src_ids = [f"{self.cls[ids]}-{name[:-4]}" for ids, name in zip(self.cls_ids, self.src_imnames)]
        self.trg_ids = [f"{self.cls[ids]}-{name[:-4]}" for ids, name in zip(self.cls_ids, self.trg_imnames)]

    @staticmethod
    def _process_keypoints(file):
        kps, visibility = torch.tensor(file["keypoints"]).view(-1, 3).float().split(2, dim=-1)
        return kps, visibility.squeeze() == 2.
    
    @staticmethod
    def _process_bbox(file):
        x, y, width, height = file['bbox']
        return torch.tensor([x, y, x + width, y + height], dtype=torch.float32)

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)

        # Not commonly used
        # batch['src_mask'] = self.get_mask(batch, self.src_impaths[idx], (Hs, Ws))
        # batch['trg_mask'] = self.get_mask(batch, self.trg_impaths[idx], (Ht, Wt))

        return batch

    def get_image(self, imnames, idx):
        """Loads and returns an RGB image given image names and index."""
        imname = imnames[idx]
        impath = self.src_impaths[idx] if imname in self.src_impaths[idx] else self.trg_impaths[idx]
        return Image.open(impath).convert('RGB')
    
    def get_mask(self, batch, img_path, scaled_imsize):
        """Loads a Binary mask (0 for background, 255 for object) at scaled_imsize."""
        mask_path = img_path.replace("JPEGImages", "features").replace(".jpg", "_mask.png")
        tensor_mask = torch.from_numpy(np.array(Image.open(mask_path))).float().unsqueeze(0).unsqueeze(0) 
        return F.interpolate(tensor_mask, size=scaled_imsize, mode='nearest').int().squeeze()


class AP10KImageDataset(Dataset):
    benchmark = 'ap10k'
    
    def __init__(self, cfg, split, category, transform, subsample=None):
        """
        A PyTorch Dataset for loading images from the AP-10K dataset.

        Args:
            cfg (object): Configuration object containing dataset settings.
            split (str): Dataset split ('trn', 'val', 'test', or 'all').
            category (str): Image category or 'all' for all categories.
            transform (callable): Transform to be applied on batch.
        """
        super().__init__()
        self.root = os.path.join(cfg.DATASET.ROOT, 'ap-10k')
        self.transform = transform
        
        self.cls = ['alouatta', 'antelope', 'beaver', 'bison', 'bobcat', 'brown bear', 'buffalo', 'cat', 'cheetah', 'chimpanzee', 'cow', 'deer', 'dog', 'elephant', 'fox', 'giraffe', 'gorilla', 'hamster', 'hippo', 'horse', 'jaguar', 'leopard', 'lion', 'marmot', 'monkey', 'moose', 'mouse', 'noisy night monkey', 'otter', 'panda', 'panther', 'pig', 'polar bear', 'rabbit', 'raccoon', 'rat', 'rhino', 'sheep', 'skunk', 'snow leopard', 'spider monkey', 'squirrel', 'tiger', 'uakari', 'weasel', 'wolf', 'zebra']

        options = self.cls + ["all"]
        if category not in options:
            raise ValueError(f"Invalid category: {category}, select from {options}.")
        
        self._load_data(split, category, subsample)

    def _load_data(self, split, category, subsample):
        if split not in ["trn", "val", "test", "all"]:
            raise ValueError(f"Invalid split: {split}, select from ('trn', 'val', 'test', 'all')")    
        splits = ["trn", "val", "test"] if split == "all" else [split]
        cats = self.cls if category == "all" else [category]
        
        # impaths = set()
        impaths = []
        for split in splits:
            for cat in cats:
                np.random.seed(42)
                pairs = sorted(glob(os.path.join(self.root, 'PairAnnotation', split, f'*:{cat}.json')))
                if subsample is not None and subsample > 0:
                    pairs = [pairs[ix] for ix in np.random.choice(len(pairs), subsample)]
                for pair in pairs:
                    with open(pair) as f:
                        data = json.load(f)
                    for json_path in [data["src_json_path"], data["trg_json_path"]]:
                        impath = json_path.replace("json", "jpg").replace('ImageAnnotation', 'JPEGImages')
                        impaths.append(impath)

        # self.impaths = sorted(impaths)
        self.impaths = impaths
        self.imnames = [f"{os.path.basename(os.path.dirname(path))}-{os.path.splitext(os.path.basename(path))[0]}" for path in self.impaths]
    
    def __len__(self):
        return len(self.imnames)
    
    def __getitem__(self, index):
        """
        Returns:
        dict: A dictionary containing:
            - 'pixel_values' (torch.Tensor): The image data.
            - 'id' (str): A unique identifier for the image. e.g. cat-000000033072
            - 'impath' (str): The file path of the image.
        """
        img = Image.open(self.impaths[index]).convert('RGB')
        img = self.transform(img)

        return {
            "pixel_values": img,
            "identifier": self.imnames[index],
            "impath": self.impaths[index]
        }