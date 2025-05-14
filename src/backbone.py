import torch
import torch.nn as nn
from transformers import Dinov2Backbone
from diffusers import StableDiffusionPipeline

from utils.misc import set_default

def make_dino_backbone(cfg=None):
    model = set_default(cfg, 'DINO.MODEL', 'base')
    layer = set_default(cfg, 'DINO.LAYER', 'stage12')
    finetune_from = set_default(cfg, 'DINO.FINETUNE_FROM', 'stage12')

    dino_path = f"facebook/dinov2-{model}"
    backbone = Dinov2Backbone.from_pretrained(dino_path)

    all_stages = backbone.stage_names
    layer_idx = all_stages.index(layer)

    # Truncate the backbone
    backbone.out_features = [layer]
    backbone.encoder.layer = backbone.encoder.layer[:layer_idx]
    backbone.stage_names = all_stages[:layer_idx+1]
    
    if finetune_from != "all":
        finetune_idx = all_stages.index(finetune_from)
        for param in backbone.embeddings.parameters():
            param.requires_grad = False
        for i in range(finetune_idx-1):
            for param in backbone.encoder.layer[i].parameters():
                param.requires_grad = False

    # Disable requires_grad for mask_token as it's not used in loss calculation
    backbone.embeddings.mask_token.requires_grad = False

    return backbone

def make_frozen_dino_backbone(cfg=None):
    model = set_default(cfg, 'DINO.MODEL', 'base')
    layer = set_default(cfg, 'DINO.LAYER', 'stage12')

    dino_path = f"facebook/dinov2-{model}"
    backbone = Dinov2Backbone.from_pretrained(dino_path)

    all_stages = backbone.stage_names
    layer_idx = all_stages.index(layer)

    # Truncate the backbone
    backbone.out_features = [layer]
    backbone.encoder.layer = backbone.encoder.layer[:layer_idx]
    backbone.stage_names = all_stages[:layer_idx + 1]
    
    # Freeze all parameters in the backbone
    for param in backbone.parameters():
        param.requires_grad = False

    # Disable requires_grad for mask_token as it's not used in loss calculation
    backbone.embeddings.mask_token.requires_grad = False

    return backbone

def make_sd_backbone(cfg=None):
    sd_path = "stabilityai/stable-diffusion-2-1"
    
    t = set_default(cfg, 'SD.SELECT_TIMESTEP', 261)
    ensemble = set_default(cfg, 'SD.ENSEMBLE_SIZE', 8)  

    backbone = SDFeatureExtractor.from_pretrained(sd_path)    
    backbone.t, backbone.ensemble = t, ensemble
    return backbone


class SDFeatureExtractor(nn.Module):
    """Extracts SD2-1 features for semantic point matching (DIFT)."""

    t = 261
    ensemble = 8

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        instance = cls()
        pipeline = StableDiffusionPipeline.from_pretrained(*args, **kwargs)
        
        # Truncate up blocks of UNet, and avoid post-process when calling UNet.
        pipeline.unet.up_blocks = pipeline.unet.up_blocks[:2]
        pipeline.unet.conv_norm_out = pipeline.unet.conv_act = None
        pipeline.unet.conv_out = nn.Identity()
        
        # Save memory (optional)
        pipeline.enable_attention_slicing()
        pipeline.enable_xformers_memory_efficient_attention()
        pipeline.vae.decoder = None
        
        instance.encode_prompt = pipeline.encode_prompt
        for key, value in pipeline.__dict__.items():
            setattr(instance, key, value)
        return instance

    def __call__(self, img_tensor, prompt=None, prompt_embeds=None):
        """
        Extract features from the input image using Stable Diffusion.

        Args:
            img_tensor (torch.Tensor): Input image tensor of shape (B, C, H, W).
            prompt (str or List[str], optional): Text prompt(s) for guidance. If a list, must have length B.
            prompt_embeds (torch.Tensor, optional): Pre-computed text embeddings of shape (B, 77, 1024).

        Returns:
            torch.Tensor: Extracted features of shape (B, 1280, H // 16, W // 16).
        """
        device = self.unet.device
        B = img_tensor.shape[0]
        t = torch.tensor([self.t], dtype=torch.long, device=device)

        # Validate inputs
        if prompt_embeds is not None:
            assert B == prompt_embeds.shape[0], f"Batch size mismatch: prompt_embeds ({prompt_embeds.shape[0]}) vs img_tensor ({B})."
        elif prompt is not None:
            if isinstance(prompt, list):
                assert len(prompt) == B, f"Number of prompts ({len(prompt)}) must match batch size ({B})."
            else:
                prompt = [prompt] * B 
        else:
            prompt = [""] * B

        # Encode input image to latent space and add noise
        latents = self.vae.encode(img_tensor).latent_dist.mode() * self.vae.config.scaling_factor
        latents = latents.repeat_interleave(self.ensemble, dim=0)
        latents_noisy = self.scheduler.add_noise(latents, torch.randn_like(latents), t)

        # Prepare text embeddings
        if prompt_embeds is None:
            prompt_embeds = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
            )[0]
        prompt_embeds = prompt_embeds.repeat_interleave(self.ensemble, dim=0)

        # Average features across ensemble
        feature = self.unet(latents_noisy, t, encoder_hidden_states=prompt_embeds).sample
        return feature.view(B, self.ensemble, *feature.shape[1:]).mean(dim=1)