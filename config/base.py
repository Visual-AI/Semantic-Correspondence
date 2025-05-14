from yacs.config import CfgNode as CN

_CN = CN()

_CN.SIMSC_ROOT = "asset/SimSC"

# 0 dataset configuration
_CN.DATASET = CN()
_CN.DATASET.NAME = 'spair'
_CN.DATASET.ROOT = 'asset/'
_CN.DATASET.IMG_SIZE = 768
_CN.DATASET.MEAN = [0.485, 0.456, 0.406]
_CN.DATASET.STD = [0.229, 0.224, 0.225]

_CN.DATASET.GEO_AUG = False
_CN.DATASET.COLOR_AUG = False
_CN.DATASET.CROP_AUG = False

# 1 Feature extractor configuration
_CN.FEATURE_EXTRACTOR = CN()
_CN.FEATURE_EXTRACTOR.NAME = "dino"  # Options: "sd", "dino", "combined"
_CN.FEATURE_EXTRACTOR.ENABLE_L2_NORM = True
_CN.FEATURE_EXTRACTOR.IF_FINETUNE = True

# DINO configuration
_CN.DINO = CN()
_CN.DINO.IMG_SIZE = 840 #  Options: 840 448 224
_CN.DINO.MEAN = [0.485, 0.456, 0.406]
_CN.DINO.STD = [0.229, 0.224, 0.225]
_CN.DINO.MODEL = "base"  # Options: dinov2-small, dinov2-base, dinov2-large, dinov2-giant
_CN.DINO.LAYER = "stage12"  # Options: stage1 to stage12
_CN.DINO.FINETUNE_FROM = "stage11"  # Fine-tune blocks from this layer onwards. "" for no tuning, "all" for all blocks

# SD configuration
_CN.SD = CN()
_CN.SD.IMG_SIZE = 224
_CN.SD.MEAN = [0.5, 0.5, 0.5]
_CN.SD.STD = [0.5, 0.5, 0.5]
_CN.SD.SELECT_TIMESTEP = 261  # Select from 1-1000
_CN.SD.ENSEMBLE_SIZE = 1  # If > 1, average features over multiple denoising processes
_CN.SD.PROMPT = 'single' # Options: "none", "single", "cpm"

# SD4Match configuration
_CN.SD4MATCH = CN()
_CN.SD4MATCH.TOKEN_LENGTH = 75
_CN.SD4MATCH.PROMPT_DIM = 1024
_CN.SD4MATCH.CPM_COND_LENGTH = 50 
_CN.SD4MATCH.CPM_GLOBAL_LENGTH = _CN.SD4MATCH.TOKEN_LENGTH - _CN.SD4MATCH.CPM_COND_LENGTH
_CN.SD4MATCH.CPM_IN_DIM = 768
_CN.SD4MATCH.CPM_N_PATCH = 256

# 0 dataset configuration

# 2 Temperature predictor configuration
_CN.TEMPERATURE = 0.03  # Options: SimSC-Single, SimSC-MLP, SimSC-PerPixel, [float]


# Evaluator configuration
_CN.EVALUATOR = CN()
_CN.EVALUATOR.ALPHA = (0.01, 0.05, 0.1, 0.15)  
_CN.EVALUATOR.BY = 'image'                          # select between ('image', 'point'), PCK per image or PCK per point
_CN.EVALUATOR.SOFTMAX_TEMP = 0.04
_CN.EVALUATOR.KERNELSOFTMAX_SIGMA = 7


# Loss configuration
_CN.LOSS = CN()
_CN.LOSS.KERNEL_SIZE = 7
_CN.LOSS.SOFTMAX_TEMP = 0.03

BASE_CONFIG = _CN.clone()