import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from transformers import Dinov2Backbone
from .backbone import make_frozen_dino_backbone

from utils.misc import set_default

def encode_token_embed(text_encoder, token_embed):
    last_hidden_state = text_encoder.text_model.encoder(inputs_embeds=token_embed, return_dict=False)[0]
    return text_encoder.text_model.final_layer_norm(last_hidden_state)

def make_captioner(cfg=None, text_encoder=None, tokenizer=None, dino=None):
    captioner_type = set_default(cfg, 'SD.PROMPT', 'none')

    if captioner_type not in ['single', 'cpm']:
        return None

    if text_encoder is None or (captioner_type == 'cpm' and tokenizer is None):
        sd = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
        text_encoder, tokenizer = sd.text_encoder, sd.tokenizer

    if captioner_type == 'single':
        return Captioner(text_encoder, cfg, tokenizer=tokenizer, mode='single')
    elif captioner_type == 'cpm':
        dino_CPM = make_frozen_dino_backbone(cfg)
        return Captioner(text_encoder, cfg, tokenizer=tokenizer, dino=dino_CPM, mode='cpm')

class CPM(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        global_length = set_default(cfg, 'SD4MATCH.CPM_GLOBAL_LENGTH', 25)
        cond_length = set_default(cfg, 'SD4MATCH.CPM_COND_LENGTH', 50)
        in_dim = set_default(cfg, 'SD4MATCH.CPM_IN_DIM', 768)
        out_dim = set_default(cfg, 'SD4MATCH.PROMPT_DIM', 1024)
        n_patch = set_default(cfg, 'SD4MATCH.CPM_N_PATCH', 256)

        self.global_embeds = nn.Parameter(torch.randn(1, global_length, out_dim) * 0.02)

        self.positional_embedding = nn.Parameter(torch.randn(1, cond_length, out_dim) * 0.02)
        self.linear_C = nn.Linear(in_dim * 2, out_dim)
        self.linear_N = nn.Linear(n_patch, n_patch)
        self.adaptive_pool = nn.AdaptiveMaxPool1d(cond_length)
        self.alpha_cond = nn.Parameter(torch.randn(1, cond_length, out_dim) * 0.02)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for linear layers."""
        nn.init.xavier_uniform_(self.linear_C.weight)
        nn.init.xavier_uniform_(self.linear_N.weight)
        nn.init.zeros_(self.linear_C.bias)
        nn.init.zeros_(self.linear_N.bias)

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat1, feat2 (torch.Tensor): DINO feature tensor of shape (B, n_patch, in_dim).

        Returns:
            torch.Tensor: Combined embedding of shape (B, global_length + cond_length, out_dim).
        """
        feat = torch.cat([feat1, feat2], dim=2)
        x = self.linear_C(feat)
        x = self.linear_N(x.transpose(1, 2)).transpose(1, 2)
        x = self.adaptive_pool(x.transpose(1, 2)).transpose(1, 2)
        x = self.positional_embedding + x
        cond_embeds = torch.tanh(self.alpha_cond) * x

        global_embeds = self.global_embeds.expand(cond_embeds.size(0), -1, -1)

        return torch.cat([global_embeds, cond_embeds], dim=1)


class Captioner(nn.Module):
    def __init__(self, text_encoder, cfg=None, tokenizer=None, dino=None, mode='single'):
        """
        A captioner that can operate in either 'single' or 'hybrid' mode.

        Args:
            text_encoder: The text encoder model.
            cfg (dict, optional): Configuration dictionary.
            tokenizer (optional): Tokenizer for text inputs.
            dino (optional): DINO model for feature extraction (required for 'hybrid' mode).
            mode (str): Mode of operation, either 'single' or 'hybrid'.
        """
        super().__init__()
        self.mode = mode
        self.text_encoder = text_encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_model_embeddings = self.text_encoder.text_model.embeddings

        ids = tokenizer("", truncation=True, return_tensors="pt")["input_ids"]
        sos_eos_embeds = self.text_encoder.text_model.embeddings(input_ids=ids).split(1, dim=1)
        self.register_buffer('sos_embed', sos_eos_embeds[0])
        self.register_buffer('eos_embed', sos_eos_embeds[1])

        if self.mode == 'single':
            token_length = set_default(cfg, 'SD4MATCH.TOKEN_LENGTH', 75)
            prompt_dim = set_default(cfg, 'SD4MATCH.PROMPT_DIM', 1024)
            #! learning params
            self.single_embed = nn.Parameter(torch.randn(1, token_length, prompt_dim) * 0.02)
        elif self.mode == 'cpm':
            self.dino = dino
            self.cpm = CPM(cfg) #! learning params
        else:
            raise ValueError("Invalid mode for Captioner. Choose 'single' or 'cpm'.")

    def forward(self, batch_size=1, image1=None, image2=None, feat1=None, feat2=None):
        """
        Forward pass for the Captioner.

        Args:
            batch_size (int): Batch size (used in 'single' mode if no image or feature is provided).
            image1, image2 (torch.Tensor, optional): Input images (B, 3, H, W).
            feat1, feat2 (torch.Tensor, optional): Pre-computed dinov2 features (B, in_dim, H, W).

        Returns:
            torch.Tensor: Encoded token embeddings.
        """
        if self.mode == 'single':
            input_batch_size = next((x.shape[0] for x in (image1, image2, feat1, feat2) if x is not None), batch_size)
            token_embed = torch.cat([self.sos_embed, self.single_embed, self.eos_embed], dim=1)
            prompt_embed = encode_token_embed(self.text_encoder, token_embed)
            return prompt_embed.repeat(input_batch_size, 1, 1)

        elif self.mode == 'cpm':
            def process_input(image, feat):
                if feat is None:
                    if image is None:
                        raise ValueError("Either image or feature must be provided in a hybrid captioner.")
                    image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=True) if image.shape[-2:] != (224, 224) else image
                    feat = self.dino(image).feature_maps[0]
                feat = F.interpolate(feat, size=(16, 16), mode='bilinear', align_corners=True) if feat.shape[-2:] != (16, 16) else feat
                return feat.view(-1, 768, 256).transpose(1, 2)

            feat1, feat2 = map(process_input, (image1, image2), (feat1, feat2))
            cpm_embed = self.cpm(feat1, feat2)
            # expand
            self.sos_embed = self.sos_embed.expand(cpm_embed.size(0), -1, -1)
            self.eos_embed = self.eos_embed.expand(cpm_embed.size(0), -1, -1)

            token_embed = torch.cat([self.sos_embed, cpm_embed, self.eos_embed], dim=1)
            return encode_token_embed(self.text_encoder, token_embed)