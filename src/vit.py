import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import timm

class PatchEmbed(nn.Module):
    """ 
    A dummy PatchEmbed that remove image size restriction. Create based on base_model
    """
    def __init__(self, base_model):
        super().__init__()
        self.img_size = base_model.patch_embed.img_size
        self.patch_size = base_model.patch_embed.patch_size
        self.grid_size = base_model.patch_embed.grid_size
        self.num_patches = base_model.patch_embed.num_patches
        self.flatten = base_model.patch_embed.flatten

        self.proj = base_model.patch_embed.proj
        self.norm = base_model.patch_embed.norm

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = self.norm(x)
        return x


class ViTExtractor(nn.Module):
    def __init__(self, model_name = 'vit_base_patch16_224_dino-token_9', bin_level=0, bin_mode='avgpool_8'):
        '''
        model_name: [str] base_model-[facet]_[layer]
            base_model: (vit_small_patch16_224_dino | vit_small_patch8_224_dino | vit_base_patch16_224_dino | vit_base_patch8_224_dino
                        vit_base_patch8_224 | vit_base_patch16_224)
            facet: (key | query | value | attn | token)
            layer: [int] 0-11
            bin: [int] if bin == 0 it means no binning
        '''
        super(ViTExtractor, self).__init__()
        self.base_model, config = model_name.split('-')
        config = config.split('_')
        
        self.facet = config[0]
        self.layer = int(config[1])
        self.bin_level = bin_level
        self.bin_mode = bin_mode
        
        base_model_ = timm.create_model(self.base_model, pretrained=True)

        self.create_from_base_model(base_model_, facet=self.facet, layer=self.layer)
        del base_model_

        self.p = self.patch_embed.patch_size
        self.stride = self.patch_embed.proj.stride

        self.load_size = None
        self.num_patches = None

    def create_from_base_model(self, base_model, facet, layer) -> None:

        # copy various modules from base model
        self.num_features = base_model.embed_dim
        self.cls_token = base_model.cls_token
        self.pos_drop = base_model.pos_drop
        self.pos_embed = base_model.pos_embed
        # self.norm = base_model.norm

        # create customed patch_embed, which removed the restriction on image size
        self.patch_embed = PatchEmbed(base_model)

        # cut block module
        self.blocks = ViTExtractor.fix_blocks(base_model.blocks, facet, layer)


    @staticmethod
    def fix_blocks(blocks, facet, layer):
        out = []
        for i in range(layer):
            out.append(blocks[i])
        if facet == 'token':
            out.append(blocks[layer])
        elif facet == 'query' or facet == 'key' or facet == 'value' or facet == 'attn':
            out.append(ViTExtractor.partial_block(blocks[layer], facet))
        return nn.Sequential(*out)

    @staticmethod
    def partial_block(block, facet):
        if facet == 'query':
            facet_idx = 0
        elif facet == 'key':
            facet_idx = 1
        elif facet == 'value':
            facet_idx = 2
        elif facet == 'attn':
            facet_idx = None
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        class PartialBlock(nn.Module):
            def __init__(self):
                super(PartialBlock, self).__init__()
                self.norm1 = block.norm1
                self.attn = block.attn
            def forward(self, x):
                x = self.norm1(x)
                B, N, C = x.shape
                qkv = self.attn.qkv(x).reshape(B, N, 3, self.attn.num_heads, C // self.attn.num_heads).permute(2, 0, 3, 1, 4)
                if facet == 'query' or facet == 'key' or facet == 'value':
                    return qkv[facet_idx]
                elif facet == 'attn':
                    q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
                    attn = (q @ k.transpose(-2, -1)) * self.attn.scale
                    attn = attn.softmax(dim=-1)
                    attn = self.attn.attn_drop(attn)
                    return attn
                else:
                    raise TypeError(f"{facet} is not a supported facet.")

        return PartialBlock()
                
    @staticmethod
    def fix_pos_embed(pos_embed, patch_size, stride, h, w):

        N = pos_embed.shape[1] - 1
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = pos_embed.shape[-1]

        # compute number of tokens taking stride into account
        w0 = 1 + (w - patch_size) // stride
        h0 = 1 + (h - patch_size) // stride

        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
            align_corners=False, recompute_scale_factor=False
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = self.num_patches[1]
        h0 = self.num_patches[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


    def prepare_tokens(self, x, mask=None):
        B, nc, h, w = x.shape
        # patch linear embedding
        x = self.patch_embed(x)
        # recalculate number of patches
        self.num_patches = (h // self.patch_embed.patch_size[0], w // self.patch_embed.patch_size[1]) # note that this number is different from self.patch_embed.num_patchs
                                                                                                # which is the updated (old) num_patch.

        # mask image modeling
        if mask is not None:
            x = self.mask_model(x, mask)
        x = x.flatten(2).transpose(1, 2)

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)


    def _extract_features(self, x):
        B, C, H, W = x.shape
        
        x = self.prepare_tokens(x)

        x = self.blocks(x)

        self.load_size = (H, W)
        self.num_patches = (1 + (H - self.p[0]) // self.stride[0], 1 + (W - self.p[1]) // self.stride[1])
        return x


    def extract_descriptors(self, batch, facet='key', include_cls=False):
        """
        extract descriptors from the model
        :param batch: batch to extract descriptors for. Has shape BxCxHxW.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']
        :param bin: apply log binning to the descriptor. default is 0.
        :return: tensor of descriptors. Bx1xtxd' where d' is the dimension of the descriptors.
        """
        assert facet in ['key', 'query', 'value', 'token'], f"""{facet} is not a supported facet for descriptors. 
                                                            choose from ['key' | 'query' | 'value' | 'token'] """
    
        x = self._extract_features(batch)
        
        if facet == 'token':
            x.unsqueeze_(dim=1) #Bx1xtxd
        
        if not include_cls:
            x = x[:, :, 1:, :]  # remove cls token
        else:
            assert self.bin_level==0, "bin > 0 and include_cls = True are not supported together, set one of them False."

        x = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1)  # Bx(t-1)x(dxh)
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], x.shape[1], self.num_patches[0], self.num_patches[1])
        
        if self.bin_level > 0:
            x = log_bin(x, self.bin_level, self.bin_mode)

        return x


    def extract_saliency_maps(self,img, reduction='mean', heads=[0, 2, 4, 5]):
        """
        extract saliency maps. The saliency maps are extracted by averaging several attention heads from the last layer
        in of the CLS token. All values are then normalized to range between 0 and 1.
        :param img: batch to extract saliency maps for. Has shape BxCxHxW.
        :return: a tensor of saliency maps. has shape Bxt-1
        """
        assert self.base_model == 'vit_small_patch8_224_dino', 'According to DINO paper, the only model support attn map is ViT-S/8'
        feat = self._extract_features(img) #Bxhxtxt
        cls_attn_map = feat[:, heads, 0, 1:]    #Bxhx(t-1)
        if reduction == 'mean':
            cls_attn_map = cls_attn_map.mean(dim=1)   #Bx(t-1)
            temp_mins, temp_maxs = cls_attn_map.min(dim=1, keepdim=True)[0], cls_attn_map.max(dim=1, keepdim=True)[0]
            cls_attn_maps = (cls_attn_map - temp_mins) / (temp_maxs - temp_mins)  # normalize to range [0,1]
            cls_attn_maps = cls_attn_maps.reshape(cls_attn_maps.shape[0], self.num_patches[0], self.num_patches[1])
        elif reduction is None:
            temp_mins, temp_maxs = cls_attn_map.min(dim=2, keepdim=True)[0], cls_attn_map.max(dim=2, keepdim=True)[0]
            cls_attn_maps = (cls_attn_map - temp_mins) / (temp_maxs - temp_mins)  # normalize to range [0,1]
            cls_attn_maps = cls_attn_maps.reshape(cls_attn_maps.shape[0], cls_attn_maps.shape[1], self.num_patches[0], self.num_patches[1])
        return cls_attn_maps


    def forward(self, x):
        if self.facet != 'attn':
            feat = self.extract_descriptors(x, facet=self.facet)
        else:
            feat = self.extract_saliency_maps(x)

        return feat