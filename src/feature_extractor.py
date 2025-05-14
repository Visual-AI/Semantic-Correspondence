import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import make_dino_backbone, make_sd_backbone
from .sd4match import make_captioner
from utils.misc import set_default
from utils.matching import l2_norm
from .vit import ViTExtractor

class FeatureExtractor(nn.Module):

    def __init__(self, cfg=None):
        super().__init__()
        # 0. Load configuration
        self.cfg = cfg
        self.backbone_name = set_default(cfg, 'FEATURE_EXTRACTOR.NAME', 'combined')
        self.enable_l2_norm = set_default(cfg, 'FEATURE_EXTRACTOR.ENABLE_L2_NORM', True)
        self.sd_img_size = (set_default(cfg, 'SD.IMG_SIZE', 768),) * 2
        self.dino_img_size = (set_default(cfg, 'DINO.IMG_SIZE', 840),) * 2
        to_tensor = lambda x: torch.tensor(x).view(1, 3, 1, 1)
        self.register_buffer('sd_mean', to_tensor(set_default(cfg, 'SD.MEAN', [0.5, 0.5, 0.5])))
        self.register_buffer('sd_std', to_tensor(set_default(cfg, 'SD.STD', [0.5, 0.5, 0.5])))
        self.register_buffer('dino_mean', to_tensor(set_default(cfg, 'DINO.MEAN', [0.485, 0.456, 0.406])))
        self.register_buffer('dino_std', to_tensor(set_default(cfg, 'DINO.STD', [0.229, 0.224, 0.225])))
        if self.backbone_name not in ['sd', 'dino', 'dinov1', 'combined']:
            raise ValueError(f"Invalid backbone name: {self.backbone_name}")
        
        # Initialize backbones
        self.if_finetune = cfg.FEATURE_EXTRACTOR.IF_FINETUNE  

        self.sd_backbone = make_sd_backbone(cfg) if self.backbone_name in ['sd', 'combined'] else None
        self.dino_backbone = make_dino_backbone(cfg) if self.backbone_name in ['dino', 'combined'] else None
        self.captioner = make_captioner(cfg, self.sd_backbone.text_encoder, self.sd_backbone.tokenizer, self.dino_backbone)  if self.sd_backbone else None

        if self.backbone_name == 'dinov1':
            self.dinov1_backbone = ViTExtractor()
        else:
            self.dinov1_backbone = None
    
    @property
    def trainable_state_dict(self):
        """Return a dictionary of trainable components."""
        result = {}

        # DINOv2
        if self.dino_backbone :
            finetune_from = self.cfg.DINO.FINETUNE_FROM
            all_stages = self.dino_backbone.stage_names
            finetune_idx = all_stages.index(finetune_from)
            dino_trainable_params = (
                [p for layer in self.dino_backbone.encoder.layer[finetune_idx-1:] for p in layer.parameters()]
            )

        # DINOv1
        if self.dinov1_backbone and self.if_finetune:
            dino_trainable_params = []
            fine_tune_from = 1
            for param in self.dinov1_backbone.parameters():
                param.requires_grad = False
            for param in self.dinov1_backbone.blocks[-fine_tune_from:].parameters():
                param.requires_grad = True
                dino_trainable_params.append(param)
            # result["dinov1"] = dino_trainable_params

        # SD
        if self.sd_backbone:
            sd_trainable_params = []
            for param in self.sd_backbone.parameters(): # SD freeze all
                param.requires_grad = False
            
            if self.captioner is None:
                sd_trainable_params = None
            else: # single or CPM
                if self.captioner.mode == 'single':
                    for component in self.captioner.single_embed: # captioner trainable
                        if isinstance(component, nn.Parameter):
                            component.requires_grad = True
                    sd_trainable_params = self.captioner.single_embed
                elif self.captioner.mode == 'cpm': # CPM
                    for name, param in self.captioner.cpm.named_parameters():
                        if param.requires_grad:
                            sd_trainable_params.append(param)

        if self.backbone_name == "dino":
            return {"dino": dino_trainable_params}
        elif self.backbone_name == "dinov1":
            return {"dino": dino_trainable_params}
        elif self.backbone_name == "sd":
            return {"token_embed": sd_trainable_params}
            # return {"single_embed": sd_trainable_params} # for SD4Match
        elif self.backbone_name == "combined":
            if sd_trainable_params is not None:
                return {"token_embed": sd_trainable_params,
                    "dino": dino_trainable_params}
            else:
                return {"dino": dino_trainable_params}
        else:
            raise ValueError(f"Invalid backbone name: {self.backbone_name}")

    def load_trainable_state_dict(self, state_dict):
        """Load the state of trainable components from a state dictionary."""
        for name, component in self.trainable_state_dict.items():
            if isinstance(component, nn.Parameter):
                component.data.copy_(state_dict[name])
            elif isinstance(component, list): 
                for idx, param in enumerate(component):
                    param.data.copy_(state_dict[name][idx])
            else:
                component.load_state_dict(state_dict[name])
    
    def transform_normalization(self, image):
        """Transform a PyTorch image tensor from SD normalization to DINO normalization."""
        return (image * self.sd_std + self.sd_mean - self.dino_mean) / self.dino_std

    def forward(self, image, image2=None):
        """
        Extract features from the input image using the specified backbone(s).

        Args:
            image (torch.Tensor): Input image tensor (B x C x H x W).
        """
        if self.dino_backbone:
            dino_image = F.interpolate(image, size=self.dino_img_size, mode='bilinear', align_corners=True)
            if self.backbone_name == 'combined':
                dino_image = self.transform_normalization(dino_image)

            dino_featmaps = feat1 = self.dino_backbone(dino_image).feature_maps[0]
            if self.enable_l2_norm:
                dino_featmaps = l2_norm(dino_featmaps, dim=1)

        if self.sd_backbone:
            sd_image = F.interpolate(image, size=self.sd_img_size, mode='bilinear', align_corners=True)

            prompt_embeds = None
            if self.captioner:
                feat1 = feat1 if self.dino_backbone else None
                prompt_embeds = self.captioner(image1=image, image2=image2, feat1=feat1)

            sd_featmaps = self.sd_backbone(sd_image, prompt_embeds=prompt_embeds)
            if self.enable_l2_norm:
                sd_featmaps = l2_norm(sd_featmaps, dim=1)
        
        if self.backbone_name == 'sd':
            return sd_featmaps # 1,1280,48,48
        elif self.backbone_name == 'dino':
            return dino_featmaps
        elif self.backbone_name == 'dinov1':
            dinov1_fmap = self.dinov1_backbone(image)
            return dinov1_fmap
        
        elif self.backbone_name == 'combined':
            dino_featmaps = F.interpolate(dino_featmaps, size=sd_featmaps.shape[-2:], mode='bilinear', align_corners=True)
            return torch.cat([sd_featmaps, dino_featmaps], dim=1) * 0.5
        else:
            raise ValueError(f"Invalid backbone name: {self.backbone_name}")