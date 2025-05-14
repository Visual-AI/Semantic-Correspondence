import os
import torch

def float_or_string(input):
    try:
        return float(input)
    except:
        if isinstance(input, str):
            return input
        else:
            raise TypeError(f"Type for temperature \
                            must be either string or float, but got {type(input)} instead.")

def save_checkpoint(model, cfg, epoch, optimizer, lr_scheduler, logging_dir, logger, best=False):
    state_dict = model.module.get_state_dict_to_save()
    
    saving_dict = {
        "state_dict": state_dict,
        "cfg": cfg,
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict()
    }
    
    filename = "best_ckpt.pt" if best else f"{epoch}_epoch.pt"
    save_path = os.path.join(logging_dir, filename)

    torch.save(saving_dict, save_path)
    
    logger.info(f"{filename} saved at {save_path}")

def save_idx_checkpoint(model, cfg, epoch, optimizer, lr_scheduler, logging_dir, logger, idx):

    state_dict = model.module.get_state_dict_to_save()
    
    saving_dict = {
        "state_dict": state_dict,
        "cfg": cfg,
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict()
    }
    
    filename = f"{epoch}_{idx}.pt"
    save_path = os.path.join(logging_dir, filename)
    
    torch.save(saving_dict, save_path)
    logger.info(f"{filename} saved at {save_path}")