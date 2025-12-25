import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import SwinTransformer


class SwinBackbone(nn.Module):
    """Wrap timm's SwinTransformer to provide multi-scale features for HBFormer.
    
    This module outputs four feature maps (C1..C4) corresponding to Swin stages,
    which will be projected to the channels expected by HBFormer encoder/decoder.
    """
    def __init__(self, pretrained_path=None, img_size=256, window_size=8):
        super().__init__()
        # adjust img_size to be divisible by window_size * patch_size (patch_size=4)
        base = window_size * 4
        adjusted_img_size = int(math.ceil(img_size / base) * base)
        self.init_img_size = adjusted_img_size
        self.orig_img_size = img_size
        self.window_size = window_size
        self.pad_h = self.init_img_size - self.orig_img_size
        self.pad_w = self.init_img_size - self.orig_img_size

        # Use Swin-Tiny configuration compatible with swin_tiny_patch4_window7_224.pth
        self.backbone = SwinTransformer(
            img_size=self.init_img_size,
            patch_size=4,
            in_chans=3,
            num_classes=0,  # no classification head
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            window_size=window_size,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True
        )

        if pretrained_path is not None:
            state = torch.load(pretrained_path, map_location="cpu")
            # support both pure state_dict and {'model': state_dict}
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            model_state = self.backbone.state_dict()
            filtered = {}

            def resize_rel_pos_bias(key, tensor, target_shape):
                # tensor: [L_orig, nH], target_shape: [L_tgt, nH]
                nH = tensor.shape[1]
                src_len = tensor.shape[0]
                tgt_len = target_shape[0]
                src_size = int(math.sqrt(src_len))
                tgt_size = int(math.sqrt(tgt_len))
                if src_size * src_size != src_len or tgt_size * tgt_size != tgt_len:
                    return None
                if src_size == tgt_size:
                    return tensor
                # reshape to [1, nH, src_size, src_size] -> interpolate -> reshape back
                bias = tensor.transpose(0,1).view(1, nH, src_size, src_size)
                bias = F.interpolate(bias, size=(tgt_size, tgt_size), mode="bicubic", align_corners=False)
                bias = bias.view(nH, tgt_size * tgt_size).transpose(0,1)
                return bias

            for k, v in state.items():
                if k not in model_state:
                    continue
                if v.shape == model_state[k].shape:
                    filtered[k] = v
                elif "relative_position_bias_table" in k and v.shape[1] == model_state[k].shape[1]:
                    resized = resize_rel_pos_bias(k, v, model_state[k].shape)
                    if resized is not None and resized.shape == model_state[k].shape:
                        filtered[k] = resized
                # other shape-mismatch keys are skipped

            msg = self.backbone.load_state_dict(filtered, strict=False)
            missing = set(model_state.keys()) - set(filtered.keys())
            print(f"Loaded Swin pretrained weights from {pretrained_path}: {msg}")
            if missing:
                print(f"Skipped {len(missing)} keys due to shape mismatch or absence (kept {len(filtered)}/{len(model_state)}).")

    def forward(self, x):
        """Return multi-scale features from Swin backbone.

        x: (B, 3, H, W)
        returns: list of 4 feature maps [C1, C2, C3, C4]
        """
        # Manually unroll Swin forward to capture features before downsample at each stage
        B, C, H, W = x.shape
        # pad input to initialized size so that window partition in timm constructor is valid
        if self.pad_h > 0 or self.pad_w > 0:
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h))
        x = self.backbone.patch_embed(x)
        if x.dim() == 4:
            pe = self.backbone.patch_embed
            embed_dim = getattr(pe, "embed_dim", None)
            if embed_dim is None and hasattr(pe, "num_features"):
                embed_dim = getattr(pe, "num_features")
            if embed_dim is None and hasattr(pe, "proj") and hasattr(pe.proj, "out_channels"):
                embed_dim = pe.proj.out_channels
            # Newer timm versions return (B, H, W, C); older return (B, C, H, W)
            if embed_dim is not None and x.shape[-1] == embed_dim:
                B, h, w, C = x.shape
            else:
                B, C, h, w = x.shape
                x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        else:
            B, L, C = x.shape
            h = w = int(L ** 0.5)
            x = x.view(B, h, w, C)
        # Some timm SwinTransformer versions do not expose `ape` / `absolute_pos_embed`.
        # Guard attribute access to stay compatible across versions.
        ape = getattr(self.backbone, "ape", False)
        if ape:
            abs_pos = getattr(self.backbone, "absolute_pos_embed", None)
            if abs_pos is not None:
                x = x + abs_pos
        pos_drop = getattr(self.backbone, "pos_drop", None)
        if pos_drop is not None:
            x = pos_drop(x)

        features = []
        B, H, W, C = x.shape
        feat0 = x.permute(0, 3, 1, 2).contiguous()
        # crop back to original patch resolution
        if self.pad_h > 0 or self.pad_w > 0:
            crop_h = h - self.pad_h // 4
            crop_w = w - self.pad_w // 4
            feat0 = feat0[:, :, :crop_h, :crop_w]
            H, W = crop_h, crop_w
            x = feat0.permute(0, 2, 3, 1).contiguous()
        features.append(feat0)

        # downstream Swin stages; Swin blocks internally handle window padding
        for idx, layer in enumerate(self.backbone.layers):
            x = layer(x)
            B, H, W, C = x.shape
            feat = x.permute(0, 3, 1, 2).contiguous()

            # crop to original resolution at this stage
            if self.pad_h > 0 or self.pad_w > 0:
                # each stage halves spatial dim after layer0's downsample; factor = 2**idx after stage0
                div = 2 ** (idx + 1)  # layer0 outputs 1/4->1/4? stage numbering: after patch embed:1/4; layer0 keeps 1/4; layer1 ->1/8 etc.
                crop_h = H - self.pad_h // div
                crop_w = W - self.pad_w // div
                feat = feat[:, :, :crop_h, :crop_w]
                H, W = crop_h, crop_w
                x = feat.permute(0, 2, 3, 1).contiguous()

            features.append(feat)

        # drop the patch-embed feature, keep four Swin stages [C1..C4]
        return features[1:]
