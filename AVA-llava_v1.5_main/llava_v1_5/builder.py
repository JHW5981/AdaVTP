import os
import torch.nn as nn
import re
from .clip_encoder import CLIPVisionTower
from .adaptor import IdentityMap

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'vision_tower', None)
    if vision_tower is None or not os.path.exists(vision_tower):
        raise ValueError(f'Vision tower is not correctly specified')
    return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

def build_vision_projector(config):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
