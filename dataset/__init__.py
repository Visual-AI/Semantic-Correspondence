from .pfpascal import PFPascalDataset, PFPascalImageDataset
from .pfwillow import PFWillowDataset, PFWillowImageDataset
from .spair import SPairDataset, SPairImageDataset
from .ap10k import AP10KDataset, AP10KImageDataset
# from .augmentation.augmentation import Augmentation

CorrDatasets = {
    'pfwillow': (PFWillowDataset, PFWillowImageDataset),
    'pfpascal': (PFPascalDataset, PFPascalImageDataset),
    'spair': (SPairDataset, SPairImageDataset),
    'ap10k': (AP10KDataset, AP10KImageDataset)
}