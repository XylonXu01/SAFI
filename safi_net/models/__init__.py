from pathlib import Path

from ..paths import PACKAGE_ROOT
from .backbones import DINOv3Backbone, RIFEBackbone
from .model import SAFINet
from .ablation import mode_of


def build_model(config, pretrained_root=None, *, test_backbones=None, load_backbone_weights=True):
    """Explicit test injection order: (IR motion, VIS motion, semantic backbone)."""
    cfg = dict(config.get('model', config))
    if cfg.get('allow_mock'):
        raise ValueError('Mock fallback is unsupported; tests must inject test_backbones explicitly')
    if test_backbones is None:
        root = Path(pretrained_root) if pretrained_root is not None else PACKAGE_ROOT / 'pretrained'
        rife_path = root / 'rife/flownet.pkl' if load_backbone_weights else None
        dino_path = root / 'dinov3/model.safetensors' if load_backbone_weights else None
        ir, vis = RIFEBackbone(rife_path), RIFEBackbone(rife_path)
        dino = (None if mode_of(cfg) == 'a1_no_dino' else
                DINOv3Backbone(dino_path, person_checkpoint=None,
                              person_config=cfg.get('perception')))
    else:
        ir, vis, dino = test_backbones
        cfg['_test_backbones'] = True
        if ir is vis or {p.data_ptr() for p in ir.parameters()} & {p.data_ptr() for p in vis.parameters()}:
            raise ValueError('IR and VIS motion backbones must have independent storage')
    if mode_of(cfg) == 'a1_no_dino':
        dino = None
    model = SAFINet(cfg, ir, vis, dino)
    configure_stage(model, 'joint')
    return model


def configure_stage(model, stage):
    if stage not in ('a', 'b', 'c', 'joint'):
        raise ValueError('Stage must be a, b, c or joint')
    model.requires_grad_(True)
    if model.dino is not None:
        model.dino.requires_grad_(False).eval()
    if hasattr(model.vis_motion, 'set_trainable'):
        model.vis_motion.set_trainable(stage == 'a')
    else:
        model.vis_motion.requires_grad_(stage == 'a')
    model.vis_motion.train(model.training and stage == 'a')
    if hasattr(model.ir_motion, 'set_trainable'):
        model.ir_motion.set_trainable(stage == 'a')
    else:
        model.ir_motion.requires_grad_(stage == 'a')
    model.ir_motion.train(model.training and stage == 'a')
    if stage == 'a':
        for module in (model.semantic_adapter, model.semantic_projection, model.reliability_heads,
                       model.transport, model.decoder.fused):
            if module is not None:
                module.requires_grad_(False)
    if not model.ablation['semantic_features']:
        for module in (model.semantic_adapter, model.semantic_projection, model.transport):
            if module is not None:
                module.requires_grad_(False)
    model.stage = stage
    return model


def parameter_groups(model, stage):
    configure_stage(model, stage)
    flow_ids = {id(p) for module in (model.ir_motion, model.vis_motion) for p in module.parameters()}
    task = [p for p in model.parameters() if p.requires_grad and id(p) not in flow_ids]
    flow = [p for p in model.ir_motion.parameters() if p.requires_grad]
    groups = [{'name': 'task', 'params': task, 'lr': 5e-6 if stage == 'c' else (1e-5 if stage in ('b', 'joint') else 1e-4)}]
    if flow:
        groups.append({'name': 'ir_flow', 'params': flow, 'lr': 1e-5})
    vis_flow = [p for p in model.vis_motion.parameters() if p.requires_grad]
    if vis_flow:
        groups.append({'name': 'vis_flow', 'params': vis_flow, 'lr': 1e-5})
    return groups


__all__ = ['SAFINet', 'build_model', 'configure_stage', 'parameter_groups']
