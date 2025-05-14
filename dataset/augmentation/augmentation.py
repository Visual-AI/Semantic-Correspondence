import albumentations as A
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import torch
import torch.nn.functional as F
from argparse import Namespace
from utils.geometry_data import normalize_coordinates, create_grid
from .utils_flow.flow_and_mapping_operations import convert_flow_to_mapping, convert_mapping_to_flow
from .utils_data.geometric_transformation_sampling.synthetic_warps_sampling import SynthecticAffHomoTPSTransfo
import time 

class Augmentation:
    '''
    A class contains augmentation used in data preprocessing
    '''
    
    def __init__(self, config, totensor=None, _device='cpu'):
        self.device = _device

        self.enable_color_aug = config.DATASET.COLOR_AUG
        self.enable_geo_aug = config.DATASET.GEO_AUG

        self.color_aug= A.Compose([
                A.Posterize(p=0.5),
                A.Equalize(p=0.5),
                A.augmentations.transforms.Sharpen(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.ColorJitter(p=0.5)
                ])

        self.geometric_aug = GeometricTransformation(config, _device)

        self.totensor = totensor or transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((config.DATASET.IMG_SIZE, config.DATASET.IMG_SIZE)),
                transforms.Normalize(mean=config.DATASET.MEAN,
                                    std=config.DATASET.STD)
            ])
        
    def __call__(self, image: np.ndarray, keypoints: np.ndarray):
        """
        IN:
            image[np.ndarray](H1xW1x3)
            keypoints[np.ndarray](nptsx2)
        OUT:
            image[torch.Tensor](3xH2xW2)
            kps[torch.Tensor](nptsx2)
        """

        kps = torch.from_numpy(keypoints).float().to(self.device)

        if self.enable_color_aug:
            image = self.color_aug(image=image)["image"]
        
        image = self.totensor(image).to(self.device)

        if self.enable_geo_aug:
            image = image[None, :]
            kps = kps[None, :]
            out = self.geometric_aug(image=image, keypoints=kps)
            image = out["image"].squeeze()
            kps = out["keypoints"].squeeze()

        return {"image": image, "keypoints":kps}
        


class GeometricTransformation:
    '''
    Geoemtric transformation of the augmentation of image
    '''
    def __init__(self, config, _device):
        
        settings = {'flow_size': int(512), # to avoid a large overhead when sampling the random transform, we 
                                           # sample the transform at 512x512 and scale this tranform to the required size
            'parametrize_with_gaussian': False,
            'transformation_types': ['hom', 'tps', 'afftps'],
            'random_t': 0.15,
            'random_s': 0.15,
            'random_alpha': np.pi / 20,
            'random_t_tps_for_afftps': 0.15,
            'random_t_hom': 0.15,
            'random_t_tps': 0.15,
            'device': 'cpu'}

        settings = Namespace(**settings)

        self.sample_transfo = SynthecticAffHomoTPSTransfo(size_output_flow=settings.flow_size, random_t=settings.random_t,
                                                random_s=settings.random_s,
                                                random_alpha=settings.random_alpha,
                                                random_t_tps_for_afftps=settings.random_t_tps_for_afftps,
                                                random_t_hom=settings.random_t_hom, random_t_tps=settings.random_t_tps,
                                                transformation_types=settings.transformation_types,
                                                parametrize_with_gaussian=settings.parametrize_with_gaussian,
                                                proba_horizontal_flip=0,
                                                _device=_device)
                                                

        self.enable_geoemtric_aug = config.DATASET.GEO_AUG


    def __call__(self, image, keypoints=None, mask=None):
        '''
        Apply geometric transformation to the image. Optional arguments are keypoints and mask. 
        REMEMBER: all images are going through the same geometric transformation
        IN:
            image[torch.Tensor](B, 3, H, W): Image to be augmented
            keypoints[torch.Tensor](B, Nq, 2): (Optional) Keypoints to be augmented
            mask[torch.Tensor](B, 1, H, W): (Optional) Mask to be augmented
        OUT:
            image[torch.Tensor](B, 3, H, W): Augmented image
            keypoints[torch.Tensor](B, Nq, 2): (Optional) Augmented keypoints
            mask[torch.Tensor](B, 1, H, W): (Optional) Augmented mask
        '''
        B, _, H1, W1 = image.shape
        _device = image.device
        
        if self.enable_geoemtric_aug:
            # Sample a random dense flow. This dense flow is the gt flow from aug_img to img, not from img to aug_img. The flow 
            # is larger than img to make sure that it covers the whole image. 

            for i in range(300):
                synthetic_flow = self.sample_transfo()  # the primary flow is 512x512
                synthetic_flow = synthetic_flow.to(_device).requires_grad_(False)
                # resize the flow to 2*imgsize to cover the whole image
                synthetic_flow = F.interpolate(synthetic_flow, (2*H1, 2*W1), mode="bilinear", align_corners=True)
                # then scaled by the ratio of interpolation
                # synthetic_flow[:, 0, :, :] = synthetic_flow[:, 0, :, :] / 300 * W1
                # synthetic_flow[:, 1, :, :] = synthetic_flow[:, 1, :, :] / 300 * H1
                synthetic_flow[:, 0, :, :].div_(300).mul_(W1)
                synthetic_flow[:, 1, :, :].div_(300).mul_(H1)

                ok = self.verify_flow(image, synthetic_flow)

                if ok:
                    break

            # warp the image and produce dense flow from img to aug_img
            aug_image, mapping = self.warp_image(image, synthetic_flow)
                
            output = {'image': aug_image}

            if mask is not None:
                aug_mask = self.transform_mask(mask, mapping)
                output.update({'mask': aug_mask})

            if keypoints is not None:
                aug_keypoints = self.transform_kps(keypoints, mapping)
                output.update({'keypoints': aug_keypoints})
        
        else:
            output = {'image': image, 'keypoints': keypoints, 'mask': mask}

        return output


    def verify_flow(self, image, flow):
        '''
        Verify whether warped boundaries have four corners in the image area
        '''
        B, _, H1, W1 = image.shape
        _, _, H2, W2 = flow.shape
        _device = image.device

        # create meshgrid
        grid = create_grid(H2, W2, device=_device)
        grid = grid.permute(2, 0, 1)

        # convert flow to mapping
        mapping = convert_flow_to_mapping(flow).contiguous()    # (1, 2, H2, W2)

        # shift the mapping to make the image centered in the flow
        delta = (H2 - H1) // 2
        mapping = mapping - delta

        leftborder = torch.abs(mapping[:, 0, :, :] - 0) < 1   # warped left vertical side of image
        rightborder = torch.abs(mapping[:, 0, :, :] - (W1-1)) < 1   # warped right vertical side of image
        upborder = torch.abs(mapping[:, 1, :, :] - 0) < 1   # warped up horizontal side of image
        downborder = torch.abs(mapping[:, 1, :, :] - (H1-1)) < 1   # warped down horizontal side of image
        leftup_corner = (leftborder & upborder).sum()
        rightup_corner = (rightborder & upborder).sum()
        leftdown_corner = (leftborder & downborder).sum()
        rightdown_corner = (rightborder & downborder).sum()

        if leftup_corner > 0 and rightup_corner > 0 and leftdown_corner > 0 and rightdown_corner > 0:
            return True
        else:
            return False


    def warp_image(self, image, flow):
        '''
        All images would undergo the same geometric transformation
        IN:
            image[torch.Tensor](B, 3, H1, W1): Image to be augmented
            flow[torch.Tensor](1, 2, H2, W2): Flow to warp the image
        OUT:
            aug_image[torch.Tensor](B, 3, H1, W1): Augmented image
            mapping[torch.Tensor](1, H1, W1, 2): mapping from aug_img to image
        '''
        B, _, H1, W1 = image.shape
        _, _, H2, W2 = flow.shape
        _device = image.device

        # create meshgrid
        grid = create_grid(H2, W2, device=_device).permute(2, 0, 1)
        # grid = grid.permute(2, 0, 1)

        # convert flow to mapping
        mapping = convert_flow_to_mapping(flow).contiguous()    # (1, 2, H2, W2)

        # shift the mapping to make the image centered in the flow
        delta = (H2 - H1) // 2
        mapping = mapping - delta

        leftborder = torch.abs(mapping[:, 0, :, :] - 0) < 1   # warped left vertical side of image
        rightborder = torch.abs(mapping[:, 0, :, :] - (W1-1)) < 1   # warped right vertical side of image
        upborder = torch.abs(mapping[:, 1, :, :] - 0) < 1   # warped up horizontal side of image
        downborder = torch.abs(mapping[:, 1, :, :] - (H1-1)) < 1   # warped down horizontal side of image
        leftup_corner = (leftborder & upborder).unsqueeze(1).expand(-1, 2, -1, -1)
        rightup_corner = (rightborder & upborder).unsqueeze(1).expand(-1, 2, -1, -1)
        leftdown_corner = (leftborder & downborder).unsqueeze(1).expand(-1, 2, -1, -1)
        rightdown_corner = (rightborder & downborder).unsqueeze(1).expand(-1, 2, -1, -1)

        # find crop boundary
        left, right, up, down = [], [], [], []
        lu = grid[leftup_corner[0]].view(2, -1).mean(dim=1).long()
        ru = grid[rightup_corner[0]].view(2, -1).mean(dim=1).long()
        ld = grid[leftdown_corner[0]].view(2, -1).mean(dim=1).long()
        rd = grid[rightdown_corner[0]].view(2, -1).mean(dim=1).long()

        # clip the border line
        leftborder[:, :lu[1]] = False
        leftborder[:, ld[1]:] = False
        rightborder[:, :ru[1]] = False
        rightborder[:, rd[1]:] = False
        upborder[:, :, :lu[0]] = False
        upborder[:, :, ru[0]:] = False
        downborder[:, :, :ld[0]] = False
        downborder[:, :, rd[0]:] = False

        # find crop boundary
        x = torch.arange(W2).to(_device)
        left.append(x[leftborder[0].sum(dim=0) > 0].min())
        right.append(x[rightborder[0].sum(dim=0) > 0].max())
        y = torch.arange(H2).to(_device)
        up.append(y[upborder[0].sum(dim=1) > 0].min())
        down.append(y[downborder[0].sum(dim=1) > 0].max())
        
        left = torch.stack(left, dim=0)
        right = torch.stack(right, dim=0)
        up = torch.stack(up, dim=0)
        down = torch.stack(down, dim=0)
        
        # crop the flow and sample image
        mapping = mapping[:, :, up[0]:down[0]+1, left[0]:right[0]+1]
        mapping = F.interpolate(mapping, size=(H1, W1), mode='bilinear', align_corners=True).permute(0, 2, 3, 1).expand(B, -1, -1, -1)
        mapping_normalised = normalize_coordinates(mapping, (H1, W1))
        aug_img = F.grid_sample(image, mapping_normalised, mode='bilinear', padding_mode='border', align_corners=True)

        return aug_img, mapping[0:1]

    def transform_mask(self, mask, mapping):
        '''
        Transform the mask according to the mapping
        IN:
            mask[torch.Tensor](B, 1, H1, W1): Mask to be transformed
            mapping[torch.Tensor](1, H1, W1, 2): mapping from aug_img to image
        OUT:
            aug_mask[torch.Tensor]
        '''
        B, C, H1, W1 = mask.shape
        mapping = mapping.expand(B, -1, -1, -1)
        mapping_normalised = normalize_coordinates(mapping, H1, W1)
        aug_mask = F.grid_sample(mask, mapping_normalised, mode='nearest', padding_mode='zeros', align_corners=True)

        return aug_mask

    def transform_kps(self, kps, mapping):
        '''
        Transform the keypoints according to the mapping, using four corner interpolation
        IN:
            kps[torch.Tensor](B, Nq, 2): Mask to be transformed
            mapping[torch.Tensor](1, H1, W1, 2): mapping from aug_img to image
        OUT:
            aug_kps[torch.Tensor]
        '''
        B, Nq, _ = kps.shape
        h, w = mapping.shape[1:3]
        kps = kps.unsqueeze(2)
        mapping = mapping.expand(B, -1, -1, -1).view(B, -1, 2).unsqueeze(1)

        diff = torch.norm(kps - mapping, dim=-1)    # (B, Nq, h*w)

        values, indices = torch.min(diff, dim=-1)

        x = indices % w
        y = indices // w
        aug_kps = torch.stack([x, y], dim=-1).float()    # (B, Nq, 2)

        return aug_kps