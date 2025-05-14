import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .augmentation.augmentation import Augmentation
from utils.geometry_data import scaling_coordinates, regularize_coordinates


class CorrespondenceDataset(Dataset):
    def __init__(self, cfg, split, thres="auto", transform=None):
        self.metadata = {
            'pfwillow': ('pf-willow', 'test_pairs.csv', 'PF-dataset', '', 'bbox'),
            'pfpascal': ('pf-pascal', f'{split}_pairs.csv', 'PF-dataset-PASCAL/JPEGImages', 'PF-dataset-PASCAL/Annotations', 'img'),
            'spair':    ('SPair-71k', 'Layout/large', 'JPEGImages', 'PairAnnotation', 'bbox'),
            'ap10k':    ('ap-10k', '', 'JPEGImages', 'PairAnnotation', 'bbox')
        }

        benchmark = self.benchmark
        self.split = split
        self.img_size = (cfg.DATASET.IMG_SIZE, cfg.DATASET.IMG_SIZE)
        self.thres = self.metadata[benchmark][4] if thres == 'auto' else thres

        base_path = os.path.join(os.path.abspath(cfg.DATASET.ROOT), self.metadata[benchmark][0])
        self.img_path = os.path.join(base_path, self.metadata[benchmark][2])
        self.ann_path = os.path.join(base_path, self.metadata[benchmark][3], split if benchmark == 'spair' else '')

        file_ext = {'pfwillow': '', 'pfpascal': '', 'spair': f'{split}.txt', 'ap10k': ''}
        self.spt_path = os.path.join(base_path, self.metadata[benchmark][1], file_ext[benchmark]).rstrip('/')

        self.max_pts = 20
        self.transform = Augmentation(cfg, totensor=transform, _device='cpu')
        if split != 'trn':
            self.transform.enable_color_aug = False
            self.transform.enable_geo_aug = False

        self.data = []
        self.src_imnames, self.trg_imnames = [], []
        self.cls, self.cls_ids = [], []
        self.src_kps, self.trg_kps = [], []
        self.src_bbox, self.trg_bbox = [], []
        self.src_pckthres, self.trg_pckthres = [], []
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves and processes a single batch from the SPair-71k dataset.

        Args:
            idx (int): Index of the batch to retrieve.

        Returns:
            dict: A dictionary containing the processed batch with the following keys:
                - 'src_img' (torch.Tensor): Source image tensor of shape (3, H, W).
                - 'trg_img' (torch.Tensor): Target image tensor of shape (3, H, W).
                - 'src_kps' (torch.Tensor): Source keypoints tensor of shape (max_pts, 2).
                - 'trg_kps' (torch.Tensor): Target keypoints tensor of shape (max_pts, 2).
                - 'n_pts' (torch.Tensor): Number of valid keypoints in 'src_kps' and 'trg_kps'.
                - 'src_bbox' (torch.Tensor): Source bounding box tensor of shape (4,), (x1, y1, x2, y2).
                - 'trg_bbox' (torch.Tensor): Target bounding box tensor of shape (4,), (x1, y1, x2, y2).
                - 'src_imname' (str): Filename of the source image, e.g. '2008_001546.jpg'.
                - 'trg_imname' (str): Filename of the target image.
                - 'category' (str): Category name of the object. e.g. 'aeroplane'.
                - 'category_id' (int): Category ID of the object.
                - 'src_imsize' (tuple): Original size of the source image (H, W).
                - 'trg_imsize' (tuple): Original size of the target image (H, W).
                - 'src_pckthres' (torch.Tensor): PCK threshold for the source image.
                - 'trg_pckthres' (torch.Tensor): PCK threshold for the target image.
                - 'pckthres' (torch.Tensor): PCK threshold (same as trg_pckthres).
                - 'src_ids' (str): Unique identifier for the source image. e.g. 'aeroplane-2008_001546'.
                - 'trg_ids' (str): Unique identifier for the target image.

        Note:
            - Keypoints and bounding boxes are scaled to match the resized image dimensions.
            - PCK thresholds are calculated based on the target bounding box or image size.
            - Images are augmented using the transform pipeline defined in self.transform.
        """
        
        batch = {
            'src_imname': self.src_imnames[idx],
            'trg_imname': self.trg_imnames[idx],
            'category_id': self.cls_ids[idx],
            'category': self.cls[self.cls_ids[idx]],
        }

        src_pil = self.get_image(self.src_imnames, idx)
        trg_pil = self.get_image(self.trg_imnames, idx)

        # original image size (H, W)
        batch['src_imsize'] = src_pil.size[::-1] 
        batch['trg_imsize'] = trg_pil.size[::-1]

        src_kps, n_pts = self.get_points(self.src_kps, idx, batch['src_imsize'], self.img_size)
        trg_kps, _ = self.get_points(self.trg_kps, idx, batch['trg_imsize'], self.img_size)
        batch['n_pts'] = n_pts = torch.tensor(n_pts)

        src_out = self.transform(np.array(src_pil), src_kps.numpy())
        trg_out = self.transform(np.array(trg_pil), trg_kps.numpy())
        batch["src_img"], batch["src_kps"] = src_out["image"], src_out["keypoints"]
        batch["trg_img"], batch["trg_kps"] = trg_out["image"], trg_out["keypoints"]
        
        batch['src_bbox'] = self.get_bbox(self.src_bbox, idx, batch['src_imsize'], self.img_size)
        batch['trg_bbox'] = self.get_bbox(self.trg_bbox, idx, batch['trg_imsize'], self.img_size)

        batch['trg_pckthres'] = self.get_pckthres({'trg_img': batch['trg_img'], 'trg_bbox': batch['trg_bbox']}) # pckthres if matching to trg_img
        batch['src_pckthres'] = self.get_pckthres({'trg_img': batch['src_img'], 'trg_bbox': batch['src_bbox']}) # pckthres if matching to src_img
        batch['pckthres'] = batch['trg_pckthres'].clone() # default pckthres is the trg_pckthres, as by default we are matching in src -> trg order

        batch['src_kps'][:n_pts] = regularize_coordinates(batch['src_kps'][:n_pts], self.img_size, eps=1e-4)
        batch['trg_kps'][:n_pts] = regularize_coordinates(batch['trg_kps'][:n_pts], self.img_size, eps=1e-4)

        batch['src_identifier'] = self.src_ids[idx]
        batch['trg_identifier'] = self.trg_ids[idx]

        return batch

    def get_bbox(self, bbox_list, idx, ori_imsize, scaled_imsize):
        """Scales a bounding box from original image size (Ho, Wo) to target size (Ht, Wt)."""
        bbox = bbox_list[idx].view(2, 2) # (4,) -> (2, 2)
        return scaling_coordinates(bbox, ori_imsize, scaled_imsize).view(4,)

    def get_pckthres(self, batch):
        """Calculate PCK threshold based on bounding box or image size."""
        if self.thres == 'bbox':
            bbox = batch['trg_bbox'].squeeze(0) if batch['trg_bbox'].dim() == 2 else batch['trg_bbox']
            pckthres = max(bbox[2] - bbox[0], bbox[3] - bbox[1]).item()
        elif self.thres == 'img':
            pckthres = max(batch['trg_img'].size()[-2:])
        else:
            raise ValueError(f'Invalid pck threshold type: {self.thres}')
        
        return torch.tensor(pckthres, dtype=torch.float)

    def get_points(self, pts_list, idx, ori_imsize, scaled_imsize):
        """
        Scale keypoints from `ori_imsize` to `scaled_imsize` and pad to max_pts.

        Returns:
            tuple: (scaled and padded keypoints array (self.max_pts, 2), number of actual points)
        """
        pts = pts_list[idx] # (n_pts, 2)
        n_pts = pts.shape[0]
        scaled_pts = scaling_coordinates(pts, ori_imsize, scaled_imsize)
        
        # Pad the scaled points tensor
        pad_size = self.max_pts - n_pts
        padded_pts = F.pad(scaled_pts, (0, 0, 0, pad_size), mode='constant', value=-2) # (self.max_pts, 2)
        
        return padded_pts, n_pts