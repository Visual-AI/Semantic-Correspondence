import torch
import torch.nn as nn
import numpy as np
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def expand_tensor(tensor, num_dims):

    current_dims = tensor.dim()

    if current_dims >= num_dims:
        return tensor
    else:
        for _ in range(num_dims - current_dims):
            tensor = tensor.unsqueeze(-1)
        return tensor


def softmax_with_temperature(x, beta, dim):
    M, _ = x.max(dim=dim, keepdim=True)
    x = x - M
    if isinstance(beta, float):
        exp_x = torch.exp(x/beta)
    elif isinstance(beta, torch.Tensor):
        if beta.shape[0] != x.shape[0]:
            raise ValueError("The batch size of beta must match with the batch size of x")
        else:
            beta = expand_tensor(beta, x.dim())
            exp_x = torch.exp(x/beta)

    exp_x_sum = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / exp_x_sum


def torch_to_numpy(img, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225]):
    '''
    Convert the image from tensor to numpy and denormalise the image.
    Ipnut:
        img: [Tensor] (Bx3xHxW) Image to be de-normalised in torch tensor.
        norm_mean: [List] The mean of the normalisation.
        norm_std; [List] The std of the normalisation.
    Output:
        img: [Array] (BxHxWx3)
    '''

    img = img.clone()
    img = img.cpu()

    mean = torch.Tensor(norm_mean)
    std = torch.Tensor(norm_std)
    mean = mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    std = std.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    mean = mean.expand_as(img)
    std = std.expand_as(img)

    img = img * std + mean
    img = img.permute(0, 2, 3, 1).numpy()

    img = img*255
    img = img.astype(np.uint8)

    return img


def move_batch_to(batch, _device):

    for k, v in batch.items():

        if isinstance(v, torch.Tensor):
            batch[k] = v.to(_device)


img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.tensor([10.], device=x.device))
img2psnr = lambda x, y: mse2psnr(img2mse(x, y))

class SSIM(nn.Module):
    '''
    Layer to compute the SSIM between a pair of images
    '''
    def __init__(self):
        super(SSIM, self).__init__()
        k = 7
        self.mu_x_pool = nn.AvgPool2d(k, 1)
        self.mu_y_pool = nn.AvgPool2d(k, 1)
        self.sig_x_pool = nn.AvgPool2d(k, 1)
        self.sig_y_pool = nn.AvgPool2d(k, 1)
        self.sig_xy_pool = nn.AvgPool2d(k, 1)
        self.refl = nn.ReflectionPad2d(k//2)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * \
            (sigma_x + sigma_y + self.C2)
        # return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1) # this is DSSIM, disimilarity of SSIM. Or SSIM loss
        return torch.clamp((SSIM_n/SSIM_d), 0, 1)
    

def set_default(cfg, attr, default):
    """
    Retrieve a nested attribute from a configuration object, returning a default value if not found.

    Examples:
        >>> cfg = CN(); cfg.EVALUATOR = CN(); cfg.EVALUATOR.ALPHA = 0.1
        >>> set_default(cfg, 'EVALUATOR.ALPHA', 0.05)
        0.1
        >>> set_default(None, 'EVALUATOR.ALPHA', 0.05)
        0.05
    """
    if cfg is None:
        return default
    attrs = attr.split('.')
    value = cfg
    for a in attrs:
        value = getattr(value, a, None)
        if value is None:
            return default
    return value

def move_dict_to(batch, *args, ignore_keys=None, **kwargs):
    """
    Move tensors in a batch to specified device/dtype, preserving non-tensor elements.

    Args:
        batch (dict): Input dictionary with tensor and non-tensor elements.
        *args: Positional arguments for torch.Tensor.to().
        ignore_keys (list): Keys to ignore when moving tensors. Default: [].
        **kwargs: Keyword arguments for torch.Tensor.to().
    """
    if ignore_keys is None:
        ignore_keys = []

    return {
        k: v.to(*args, **kwargs) if isinstance(v, torch.Tensor) and k not in ignore_keys else v
        for k, v in batch.items()
    }