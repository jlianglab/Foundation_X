# Developed by Nathaniel Alberti

import torch
import torch.nn as nn
import timm.models.swin_transformer as swin
from timm.models.helpers import load_state_dict

class SwinTransformer(swin.SwinTransformer):
    def __init__(self, num_classes, projector_features=None, use_mlp=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert num_classes is not None
        
        self.projector = None
        if projector_features:
            print(f"[DEBUG]: Projector features = {projector_features}")
            encoder_features = self.num_features
            self.num_features = projector_features
            if use_mlp:
                self.projector = nn.Sequential(
                    nn.Linear(encoder_features, self.num_features),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.num_features, self.num_features)
                )
            else:
                self.projector = nn.Linear(encoder_features, self.num_features)
        else:
            # if projector_features is None, use self.num_features as is
            print("[DEBUG]: Projector features = NONE")
            encoder_features = self.num_features

        self.head = nn.ModuleList(
            nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)  # [B, L, C]
        x = x.transpose(1, 2)  # [B, C, L]
        x = self.avgpool(x)  # [B, C, 1]
        x = x.squeeze(-1)    # [B, C]
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.projector:
            x = self.projector(x)
        return head(x)

    def generate_embeddings(self, x, after_proj=True):
        x = self.forward_features(x)
        if after_proj and self.projector:
            x = self.projector(x)
        return x

def build_model(pretrained_weights, num_classes=0, img_size=224, projector_features=None, use_mlp=False):
  print("=== Initializing with Foundation X pretrained weights ===")
    # Foundation X uses swin_base
    model = SwinTransformer(
        num_classes_list,
        projector_features,
        use_mlp,
        patch_size=4,
        window_size=7,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        img_size=img_size,
    )

  state_dict = torch.load(pretrained_weights, map_location='cpu')
  state_dict = state_dict["teacher_model"]
  new_state_dict = {}
  for key, value in state_dict.items():
      if "head" in key or "attn_mask" in key:
          continue
      if "backbone" in key:
          new_key = key.replace('backbone.0.', '')
          new_state_dict[new_key] = value
  status = model.load_state_dict(new_state_dict, strict=False)
  print(status)
        
  return model

### Example usage ###
"""

from load_weights import build_model

pretrained_weights = "path/to/weights/ckpt.pth"
foundationx_model = build_model(pretrained_weights, num_classes = 1)

foundationx_model.train()
...

"""
