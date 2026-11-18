"""FP32 video inference with cached observation features."""
from ..checkpoints import load_inference_model
from .runtime import FastDINO, FastModel, FastRIFE
from .transport import InferenceTransport


def load_fast_model(path, pretrained_root=None, device='cpu', *, return_aux=False):
    model = load_inference_model(path, pretrained_root=pretrained_root)
    settings = model.ablation
    if (model.mode != 'full' or settings['cascade']
            or not settings['semantic_features'] or settings['aggregation'] != 'proposed'):
        raise ValueError('Fast inference requires the full SAFI-Net model with the standard architecture')

    # IR uses the flow-only path. VIS keeps the refinement modules for missing frames.
    for motion in (model.ir_motion, model.vis_motion):
        motion.flownet.block_tea = None
        motion.__class__ = FastRIFE
    model.ir_motion.flownet.contextnet = None
    model.ir_motion.flownet.unet = None
    model.dino.model.embeddings.register_parameter('mask_token', None)
    model.dino.__class__ = FastDINO
    model.transport.__class__ = InferenceTransport
    model.__class__ = FastModel
    model.return_aux = return_aux
    return model.requires_grad_(False).to(device).eval()
