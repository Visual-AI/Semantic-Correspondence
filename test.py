import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import time
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
from utils.misc import *
from utils.matching import *
from utils.geometry import *
from utils.evaluator import PCKEvaluator
from dataset import CorrDatasets

from config.base import BASE_CONFIG
from src.feature_extractor import FeatureExtractor

from tools.log import setup_logger

os.chdir(os.path.dirname(os.path.realpath(__file__)))

def float_or_string(input):
    try:
        return float(input)
    except:
        if isinstance(input, str):
            return input
        else:
            raise TypeError(f"Type for temperature must be either string or float, but got {type(input)} instead.")

def parse_arguments():
    parser = argparse.ArgumentParser()
    # dataset setting
    parser.add_argument('--dataset', default='spair', type=str) # spair ap10k
    parser.add_argument('--val_sample', type=int, default=2)     # AP-10K sample 20 pairs for each category for testing, set to 0 to use all pairs
    parser.add_argument('--split', default='test', type=str) # test val
    parser.add_argument('--resolution', default=840, type=int)
    parser.add_argument('--category', default='all', type=str)

    # model setting
    parser.add_argument('--method', default='dino', type=str, help="choose between dino | sd | combined | dinov1")
    parser.add_argument('--if_finetune_backbone', default=True, action='store_true')

    parser.add_argument('--prompt_type', default='none', type=str, help="single | cpm | none")
    parser.add_argument('--temperature', default=0.03, type=float_or_string) # 'SimSC-Single'

    parser.add_argument('--pre_extract', default=True, action='store_true', help='Pre-extract image features to enable faster validation')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--device', default=0, type=int)

    # model save 
    parser.add_argument('--ckpt_dir', default='./dino_spair_840_85.11', help='Path to save checkpoints and logs')

    return parser.parse_args()


def initialize_config(args):
    cfg = BASE_CONFIG.clone()

    # override cfg in cfg.DATASET
    cfg.DATASET.NAME = args.dataset
    cfg.TEMPERATURE = args.temperature
    cfg.DATASET.IMG_SIZE = args.resolution
    cfg.DINO.IMG_SIZE = args.resolution
    cfg.DATASET.GEO_AUG = False
    cfg.DATASET.COLOR_AUG = False

    cfg.FEATURE_EXTRACTOR.NAME = args.method

    cfg.SD.PROMPT = args.prompt_type
    cfg.SD.SELECT_TIMESTEP = 261
    cfg.SD.ENSEMBLE_SIZE = 8

    cfg.FEATURE_EXTRACTOR.IF_FINETUNE = args.if_finetune_backbone

    if args.method == 'sd' or args.method == 'combined':
        cfg.DATASET.MEAN = [0.5, 0.5, 0.5]
        cfg.DATASET.STD = [0.5, 0.5, 0.5]

    return cfg


def log_training_info(logger, args, cfg):
    logger.info("Args settings:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    logger.info("Configuration settings:")
    logger.info(cfg.dump())


def load_dataset(cfg, args):
    # create dataset and dataloader
    transform = T.Compose([
                T.ToTensor(),
                T.Resize((cfg.DATASET.IMG_SIZE, cfg.DATASET.IMG_SIZE)),
                T.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD)
    ])

    Dataset, ImageDataset = CorrDatasets[args.dataset]

    if args.dataset == 'ap10k':
        dataset = Dataset(cfg, args.split, args.category, transform, args.val_sample)
    else: 
        dataset = Dataset(cfg, args.split, args.category, transform)

    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    img_loader = None
    if args.pre_extract:
        if args.dataset == 'ap10k':
            img_dataset = ImageDataset(cfg, args.split, args.category, transform, args.val_sample)
        else:
            img_dataset = ImageDataset(cfg, args.split, args.category, transform)

        img_loader = DataLoader(img_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)

    return loader, img_loader

def pre_extract_features(feature_extractor, img_loader, device='cuda'):
    """Extract or load feature maps for images, returning a dict of image IDs to feature tensors."""

    featmap_dict = {}
    for batch in tqdm(img_loader, desc='Caching feature maps'):
        move_batch_to(batch, device)
        identifier = batch["identifier"][0]
        
        with torch.autocast(device_type=device, dtype=torch.float16):
            fmap = feature_extractor(image=batch["pixel_values"])

        featmap_dict[identifier] = fmap.float()
    
    return featmap_dict

def extract_validation_features(batch, feature_extractor, cfg, loader, featmap_dict, args, pre_extract=False, device='cuda'):
    if not pre_extract:
        with torch.autocast(device_type=device, dtype=torch.float16):
            if cfg.SD.PROMPT == 'cpm':
                fmap0 = feature_extractor(image=batch["src_img"], image2 = batch["trg_img"])
                fmap1 = feature_extractor(image=batch["trg_img"], image2 = batch["src_img"])
            else:
                fmap0 = feature_extractor(image=batch['src_img'])
                fmap1 = feature_extractor(image=batch['trg_img'])
    else: 
        fmap0 = torch.cat([featmap_dict[imname] for imname in batch['src_identifier']], dim=0)
        fmap1 = torch.cat([featmap_dict[imname] for imname in batch['trg_identifier']], dim=0)

    batch['src_featmaps'] = fmap0
    batch['trg_featmaps'] = fmap1

    return batch


def main():
    args = parse_arguments()

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(args.device)

    timestamp = time.strftime('%m%d_%H%M', time.localtime())
    log_file = os.path.join(args.ckpt_dir, f'{args.method}_eval_{args.dataset}_{args.resolution}_{timestamp}.log')
    logger = setup_logger(log_file)

    cfg = initialize_config(args)

    log_training_info(logger, args, cfg)

    loader, img_loader = load_dataset(cfg, args)

    feature_extractor = FeatureExtractor(cfg)
    
    if args.ckpt_dir:
        if os.path.exists(os.path.join(args.ckpt_dir, 'best_weights.pt')):
            feature_extractor.load_trainable_state_dict(torch.load(os.path.join(args.ckpt_dir, 'best_weights.pt')))
        if os.path.exists(os.path.join(args.ckpt_dir, 'weights.pt')):
            feature_extractor.load_trainable_state_dict(torch.load(os.path.join(args.ckpt_dir, 'weights.pt')))
        # feature_extractor.load_trainable_state_dict(torch.load(os.path.join(args.ckpt_dir, 'weights.pt')))  # SD4Match
    feature_extractor = feature_extractor.to('cuda', dtype=torch.float16).eval()

    evaluator = PCKEvaluator(cfg, logger)

    with torch.no_grad():
        if args.pre_extract:
            featmap_dict = pre_extract_features(feature_extractor, img_loader, device='cuda')
        else:
            featmap_dict = None

        logger.info("Do the real matching...")
        for idx, batch in enumerate(tqdm(loader, desc='Matching')):
            move_batch_to(batch, "cuda")
            batch = extract_validation_features(batch, feature_extractor, cfg, loader, featmap_dict, args, pre_extract=False, device='cuda')

            if isinstance(args.temperature, float):
                temp = args.temperature

            evaluator.evaluate_feature_map(batch, enable_l2_norm=True, softmax_temp=temp)

        evaluator.print_summarize_result()


if __name__ == "__main__":
    main()