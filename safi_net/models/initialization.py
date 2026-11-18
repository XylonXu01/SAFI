"""Reproducible task-only initialization, independent of optional backbone construction."""
import hashlib
import math

import torch
from torch import nn

FROZEN_PREFIXES = ('dino.', 'ir_motion.', 'vis_motion.')


def initialize_task_parameters(model, seed):
    """Leave loaded assets intact; equal named task tensors share equal initial values."""
    def generator(name):
        value=int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:8], 'little')
        return torch.Generator(device='cpu').manual_seed(value % (2**63))

    with torch.no_grad():
        for name,module in model.named_modules():
            if (name+'.').startswith(FROZEN_PREFIXES):
                continue
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), generator=generator(name+'.weight'))
                if module.bias is not None:
                    fan_in=module.weight.shape[1]*math.prod(module.kernel_size)
                    nn.init.uniform_(module.bias,-1/math.sqrt(fan_in),1/math.sqrt(fan_in),generator=generator(name+'.bias'))
            elif isinstance(module, (nn.GroupNorm,nn.LayerNorm)):
                if module.weight is not None:nn.init.ones_(module.weight)
                if module.bias is not None:nn.init.zeros_(module.bias)
        if model.semantic_adapter is not None:
            model.semantic_adapter.layer_logits.zero_()
        for head in (model.decoder.ir,model.decoder.vis):
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        if model.mode=='a3_plain_decoder':
            nn.init.xavier_uniform_(model.decoder.rgb_head.weight,generator=generator('decoder.rgb_head.weight'))
            nn.init.zeros_(model.decoder.rgb_head.bias)
        else:
            nn.init.zeros_(model.decoder.fused.weight);nn.init.zeros_(model.decoder.fused.bias)
        if model.mode=='a2_separate_weights':
            model.initialize_interpolation_heads()
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        if not name.startswith(FROZEN_PREFIXES):
            digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return {'kind':'pretrained_backbones_only','task_initialization':'named_parameter_v1',
            'seed':seed,'task_state_sha256':digest.hexdigest(),
            'reset_optimizer':True,'reset_steps':True,'reset_best_and_early_stopping':True}
