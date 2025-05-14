import os
import torch
import argparse
import numpy as np
from datetime import timedelta
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from tools.log import setup_logger
from config import BASE_CONFIG
from dataset import CorrDatasets
from src.feature_extractor import FeatureExtractor
from utils.misc import move_batch_to
from src.loss import GaussianCrossEntropyLoss
from utils.evaluator import PCKEvaluator
from tool import float_or_string
from tabulate import tabulate
import time

def parse_args():
    parser = argparse.ArgumentParser()

    # dataset setting
    parser.add_argument('--dataset', default='ap10k', type=str) # spair ap10k
    parser.add_argument('--train_sample', type=int, default=1) # Only for ap10k, set to 0 to use all pairs
    parser.add_argument('--val_sample', type=int, default=2) # Only for ap10k
    # dataset augmentation
    parser.add_argument('--color_aug', default=0, type=int) # 0-False, 1-True
    parser.add_argument('--geo_aug', default=0, type=int) # 0-False, 1-True
    parser.add_argument('--crop_aug', default=0, type=int) # 0-False, 1-True
    # resolution
    parser.add_argument('--resolution', default=224, type=int)
    parser.add_argument('--sd_resolution', default=224, type=int)

    # model setting
    parser.add_argument('--method', default='dino', type=str, help="dino | sd | combined")
    parser.add_argument('--if_finetune_backbone', default=True, action='store_true')

    parser.add_argument('--prompt_type', default='none', type=str, help="single | cpm | none")
    parser.add_argument('--temperature', default=0.03, type=float) 

    # training setting
    parser.add_argument('--save_thre', default=0.8, type=float)
    parser.add_argument('--eval_interval', default=0.2, type=float)

    parser.add_argument('--pre_extract', default=True, action='store_true', help='Pre-extract image features to enable faster validation')

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Gradient accumulation steps before update')
    parser.add_argument('--epochs', default=1, type=int)

    # learning rate
    parser.add_argument('--init_lr', default=0.0001, type=float) 
    parser.add_argument('--dino_lr', default=0.0001, type=float)

    parser.add_argument("--scheduler", type=str, default="constant", help='Choose between ["linear", "constant", "piecewise_constant"]')
    parser.add_argument("--scheduler_power", type=float, default=1.0)
    parser.add_argument("--scheduler_step_rules", type=str, default=None)
    parser.add_argument('--num_workers', default=0, type=int)

    # model save 
    parser.add_argument('--ckpt_dir', default='./0_ap10k', help='Path to save checkpoints and logs')
    parser.add_argument('--resume_dir', type=str, default=None, help='Path to a checkpoint to resume training from')

    return parser.parse_args()

def initialize_config(args):
    cfg = BASE_CONFIG.clone()

    if args.color_aug == 1:
        cfg.DATASET.COLOR_AUG = True
    if args.geo_aug == 1:
        cfg.DATASET.GEO_AUG = True
    if args.crop_aug == 1:
        cfg.DATASET.CROP_AUG = True

    cfg.DATASET.IMG_SIZE = args.resolution
    cfg.DINO.IMG_SIZE = args.resolution
    cfg.SD.IMG_SIZE = args.sd_resolution
    cfg.DATASET.NAME = args.dataset

    cfg.FEATURE_EXTRACTOR.NAME = args.method

    cfg.TEMPERATURE = args.temperature

    cfg.SD.PROMPT = args.prompt_type

    cfg.FEATURE_EXTRACTOR.IF_FINETUNE = args.if_finetune_backbone

    return cfg

def log_training_info(logger, args, cfg, start_epoch):
    logger.info("----------------Args settings: -----------------")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    logger.info("-----------Configuration settings: -------------")
    logger.info(cfg.dump())
    logger.info(f"Starting training from epoch {start_epoch}")

def create_transforms(cfg):
    return T.Compose([
           T.ToTensor(),
           T.Resize((cfg.DATASET.IMG_SIZE, cfg.DATASET.IMG_SIZE)),
           T.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD)
    ])

def load_dataset(cfg, args):
    transforms = create_transforms(cfg)

    Dataset, ImageDataset = CorrDatasets[args.dataset]

    if args.dataset == 'ap10k':
        trn_dataset = Dataset(cfg, 'trn', 'all', transforms, args.train_sample)
        val_dataset = Dataset(cfg, 'val', 'all', transforms, args.val_sample)
    else:
        trn_dataset = Dataset(cfg, 'trn', 'all', transforms)
        val_dataset = Dataset(cfg, 'val', 'all', transforms)

    if args.pre_extract:
        if args.dataset == 'ap10k':
            val_img_dataset = ImageDataset(cfg, 'val', 'all', transforms, args.val_sample)
        else:
            val_img_dataset = ImageDataset(cfg, 'val', 'all', transforms)

    return trn_dataset, val_dataset, val_img_dataset

def create_optimizer_and_scheduler(model, cfg, args):

    learned_param = []
    model_params = model.trainable_state_dict
    for name, param_group in model_params.items():
        if name == 'dino':
            learned_param.append({"params":param_group, "lr":args.dino_lr})
        elif name == 'token_embed':
            learned_param.append({"params":param_group, "lr":args.captioner_lr})
        else:
            raise ValueError(f"Invalid name: {name}")

    optimizer = AdamW(learned_param, lr=args.init_lr,
                      weight_decay=0.05)

    import torch.optim.lr_scheduler as lr_scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[1,2], gamma=0.5, verbose=True)

    return optimizer, scheduler

def load_checkpoint(args, feature_extractor, optimizer, lr_scheduler, logger):
    if not args.resume_dir or args.resume_dir == 'None':
        logger.info("No saved state found, start training from scratch.")
        return 0, 0
    else:
        logger.info(f"Found resume path, loading state from {args.resume_dir}")

        model_state_dict = torch.load(os.path.join(args.resume_dir, 'best_weights.pt'), map_location='cuda')
        training_state = torch.load(os.path.join(args.resume_dir, 'best_trn_states.pt'), map_location='cuda')

        feature_extractor.module.load_trainable_state_dict(model_state_dict)

        optimizer.load_state_dict(training_state['optimizer'])
        lr_scheduler.load_state_dict(training_state['scheduler'])
        epoch = training_state['epoch'] + 1
        best_pck = training_state.get('best_pck', 0)

        return epoch, best_pck

def show_trainable_parameters(model, logger):
    logger.info("Trainable parameters: ")
    trainable_params = []
    total_params = total_trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params.append([name, 'x'.join(map(str, param.shape)), param.numel(), param.device])
            total_trainable_params += param.numel()
    
    trainable_params.sort(key=lambda x: x[0])
    trainable_params.append(["Total", "", total_trainable_params, "", ""])

    table = tabulate(trainable_params, 
                     headers=["Parameter", "Shape", "# Parameters", "Device"], tablefmt="grid")
    summary = f"\nTotal Parameters: {total_params:,}\nTrainable Parameters: {total_trainable_params:,}\nPercent Trainable: {(total_trainable_params / total_params) * 100:.2f}%"
    
    output = "\n" + table + summary
    logger.info(output) if logger else print(output)


def cache_featmaps(img_loader, model, cfg, logger, device='cuda'):
    featmap_dict = {}
    logger.info("Prompt only depend on individual images, so we are caching all featmaps first...")

    for idx, batch in enumerate(tqdm(img_loader)):
        move_batch_to(batch, device)
        identifier = batch["identifier"][0]
        
        with torch.autocast(device_type=device, dtype=torch.float16):
            fmap = model(image=batch["pixel_values"])

        featmap_dict[identifier] = fmap.float()
    
    return featmap_dict

def extract_validation_features(batch, feature_extractor, cfg, featmap_dict=None, faster_evaluation=False, device='cuda'):
    if not faster_evaluation:
        with torch.autocast(device_type=device, dtype=torch.float16):
            if cfg.SD.PROMPT == 'cpm':
                fmap0 = feature_extractor(image=batch["src_img"], image2 = batch["trg_img"])
                fmap1 = feature_extractor(image=batch["trg_img"], image2 = batch["src_img"])
            else:
                fmap0 = feature_extractor(batch['src_img'])
                fmap1 = feature_extractor(batch['trg_img'])
    else: 
        fmap0 = torch.cat([featmap_dict[imname] for imname in batch['src_identifier']], dim=0)
        fmap1 = torch.cat([featmap_dict[imname] for imname in batch['trg_identifier']], dim=0)

    batch['src_featmaps'] = fmap0
    batch['trg_featmaps'] = fmap1

    return batch

@torch.no_grad()
def evaluate(args, cfg, img_loader, val_loader, feature_extractor, evaluator, logger):
    feature_extractor.eval()
    cfg.SD.SELECT_TIMESTEP = 50
    cfg.SD.ENSEMBLE_SIZE = 8

    if args.pre_extract: # img_loader
        featmap_dict = cache_featmaps(img_loader, feature_extractor, cfg, logger, device='cuda')
    else:
        featmap_dict = None
                 
    logger.info("Do the real matching...")

    for idx, batch in enumerate(tqdm(val_loader)):
        move_batch_to(batch, "cuda")

        batch = extract_validation_features(batch, feature_extractor, cfg, featmap_dict, args.pre_extract)

        if isinstance(cfg.TEMPERATURE, float):
            temp = cfg.TEMPERATURE

        # corr_volume = compute_corr_volume(batch['src_featmaps'], batch['trg_featmaps'])

        evaluator.evaluate_feature_map(batch, enable_l2_norm=True, softmax_temp=temp)

    pck = np.array(evaluator.result["kernelsoftmax_pck0.1"]["all"]).mean()
    evaluator.print_summarize_result()
    feature_extractor.train()
    
    cfg.SD.SELECT_TIMESTEP = 261
    cfg.SD.ENSEMBLE_SIZE = 1

    return pck

def main():
    args = parse_args()
    np.random.seed(0)
    torch.manual_seed(0)

    # Set up the logger
    timestamp = time.strftime('%m%d_%H%M', time.localtime())

    ckpt_dir = os.path.join(args.ckpt_dir, f'{timestamp}_{args.method}_{args.if_finetune_backbone}_{args.dataset}_{args.resolution}_{args.init_lr}')

    os.makedirs(ckpt_dir, exist_ok=True)
    logger = setup_logger(os.path.join(ckpt_dir, 'train.log'))
    
    cfg = initialize_config(args)

    model_name = f"{args.dataset}-{cfg.DATASET.IMG_SIZE}-{cfg.FEATURE_EXTRACTOR.NAME}-{cfg.TEMPERATURE}"

    # Set up the directory for logging
    current_training_dir = os.path.join(cfg.SIMSC_ROOT, args.dataset, timestamp)
    
    # Set up the Accelerator
    kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=72))
    accelerator = Accelerator(kwargs_handlers=[kwargs], 
                              mixed_precision='fp16', 
                              gradient_accumulation_steps=args.grad_accum_steps)

    # Load the dataset
    train_dataset, val_dataset, img_dataset = load_dataset(cfg, args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size= 1 , shuffle=False)
    img_loader = DataLoader(img_dataset, batch_size=1, shuffle=False)

    # Create the model, optimizer and scheduler
    feature_extractor = FeatureExtractor(cfg)
    feature_extractor.to(accelerator.device)

    optimizer, lr_scheduler = create_optimizer_and_scheduler(feature_extractor, cfg, args)

    feature_extractor, train_loader, optimizer, lr_scheduler = \
        accelerator.prepare(feature_extractor, train_loader, optimizer, lr_scheduler)

    start_epoch, best_pck = load_checkpoint(args, feature_extractor, optimizer, lr_scheduler, logger)
    end_epoch = start_epoch + args.epochs
    accelerator.wait_for_everyone() 

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=current_training_dir)
        log_training_info(logger, args, cfg, start_epoch)
        show_trainable_parameters(feature_extractor, logger)

    loss_fn = GaussianCrossEntropyLoss()
    evaluator = PCKEvaluator(cfg, logger)
    best_pck = 0
    progress_bar_epoch = tqdm(range(start_epoch, end_epoch), 
                              disable=not accelerator.is_main_process)

    # Training loop
    for epoch in range(start_epoch, end_epoch):
        logger.info(f"Epoch {epoch}: ----------------------------------------------------")
        feature_extractor.train()
        evaluator.clear_result()
        progress_bar = tqdm(range(len(train_loader)), disable=not accelerator.is_main_process)
        timestep = len(train_loader)
        save_threshold = timestep * args.save_thre
        eval_interval = max(1, int(timestep * args.eval_interval))

        # Training
        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            if args.prompt_type == 'cpm':
                fmap0 = feature_extractor(image=batch["src_img"], image2 = batch["trg_img"])
                fmap1 = feature_extractor(image=batch["trg_img"], image2 = batch["src_img"])
            else:
                fmap0 = feature_extractor(image=batch['src_img'])
                fmap1 = feature_extractor(image=batch['trg_img'])

            if isinstance(cfg.TEMPERATURE, float):
                temp = cfg.TEMPERATURE
            else:
                temp = 0.03
                
            lossfn_input = {
                    'src_featmaps': fmap0,
                    'trg_featmaps': fmap1,
                    'src_kps': batch['src_kps'],
                    'trg_kps': batch['trg_kps'],
                    'src_imgsize': batch['src_img'].shape[2:],
                    'trg_imgsize': batch['trg_img'].shape[2:],
                    'npts': batch['n_pts'],
                    'softmax_temp': temp,
                    'enable_l2_norm': True
                }
            loss = loss_fn(**lossfn_input)

            # Backward
            accelerator.backward(loss)
            log_loss = accelerator.gather(loss).mean().item()

            # eval and save
            if accelerator.is_main_process:
                writer.add_scalar("train_loss", log_loss, epoch * len(train_loader) + idx)
                writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch * len(train_loader) + idx)

                if idx % eval_interval == 0:
                    logger.info(f"Idx {idx}, Calculated loss: {log_loss:.4f}")
                if (epoch==0 and idx > save_threshold and idx % eval_interval == 0) or (epoch>0 and idx>1 and idx % eval_interval == 0) : 
                    pck = evaluate(args, cfg, img_loader, val_loader, feature_extractor, evaluator, logger)
                    logger.info(f"Step {idx}: Validation PCK is {pck}")
                    if pck > best_pck:
                        best_pck = pck
                        unwrapped = accelerator.unwrap_model(feature_extractor)
                        accelerator.save(unwrapped.trainable_state_dict, 
                                        os.path.join(ckpt_dir, 'best_weights.pt'))
                        accelerator.save({
                            'optimizer': optimizer.state_dict(),
                            'scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'best_pck': best_pck
                        }, os.path.join(ckpt_dir, 'best_trn_states.pt'))
                        logger.info(f"Saved best model at {ckpt_dir}")
            
            optimizer.step()
            progress_bar.update(1)
            progress_bar.set_postfix(loss=log_loss, lr=optimizer.param_groups[0]['lr'])

        lr_scheduler.step()
        accelerator.wait_for_everyone()
        
        progress_bar_epoch.update(1)
        progress_bar_epoch.set_postfix(epoch=epoch)
        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()

    if accelerator.is_main_process:
        logger.info("Training finished.")
    accelerator.end_training()

if __name__ == "__main__":
    main()
