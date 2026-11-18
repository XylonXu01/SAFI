from types import SimpleNamespace

from PIL import Image
import pytest
import torch


def test_fast_cli_uses_the_selected_checkpoint_and_enables_aux(monkeypatch, capsys):
    from safi_net import infer_video

    calls = []

    def run(args):
        calls.append((args.checkpoint, args.fast, args.precision, args.save_aux))
        return {}

    monkeypatch.setattr(infer_video, 'run_inference', run)
    arguments = ['--ir-input', 'ir', '--vis-input', 'vis',
                 '--checkpoint', 'one.pt', '--output-dir', 'out']
    infer_video.main(arguments + ['--fast', '--save-aux'])
    infer_video.main(arguments)
    assert calls == [('one.pt', True, 'float32', True), ('one.pt', False, 'bf16', False)]
    with pytest.raises(SystemExit):
        infer_video.main(arguments + ['--fast', '--precision', 'bf16'])


def test_static_fusion_pads_one_observation_and_preserves_image_size(tmp_path, monkeypatch):
    from safi_net import fuse_image

    ir, vis, output = (tmp_path / name for name in ('ir.png', 'vis.png', 'fused.png'))
    Image.new('L', (31, 33), 100).save(ir)
    Image.new('RGB', (31, 33), (83, 124, 149)).save(vis)

    def load(path, pretrained_root, device):
        assert path == str(tmp_path / 'one.pt')
        assert str(device) == 'cpu'

        def predict(inputs):
            assert inputs.ir.shape == (1, 2, 1, 64, 32)
            assert inputs.vis.shape == (1, 2, 3, 64, 32)
            assert inputs.ir_valid.tolist() == inputs.vis_valid.tolist() == [[True, False]]
            assert inputs.query_time.tolist() == [0]
            assert inputs.ir_times.dtype == inputs.vis_times.dtype == torch.float64
            assert inputs.ir[:, 1].count_nonzero() == inputs.vis[:, 1].count_nonzero() == 0
            return SimpleNamespace(fused=inputs.vis[:, 0])

        return predict

    monkeypatch.setattr(fuse_image, 'load_inference_model', load)
    args = ['--ir-input', str(ir), '--vis-input', str(vis), '--checkpoint',
            str(tmp_path / 'one.pt'), '--output', str(output), '--device', 'cpu']
    fuse_image.main(args)
    with Image.open(output) as result:
        assert result.mode == 'RGB'
        assert result.size == (31, 33)
        assert result.getpixel((30, 32)) == (83, 124, 149)

    with pytest.raises(FileExistsError):
        fuse_image.main(args)

    Image.new('L', (31, 32)).save(ir)
    args[args.index(str(output))] = str(tmp_path / 'mismatch.png')
    with pytest.raises(ValueError, match='dimensions'):
        fuse_image.main(args)
    assert not (tmp_path / 'mismatch.png').exists()


def test_portable_checkpoint_uses_embedded_state(tmp_path, monkeypatch):
    from safi_net import checkpoints, models

    checkpoint = tmp_path / 'one.pt'
    state = {'weight': torch.tensor([[0.25]])}
    torch.save({'format': checkpoints.INFERENCE_FORMAT, 'architecture': 'safi_net',
                'model_config': {}, 'model': state}, checkpoint)

    def build(config, pretrained_root=None, *, load_backbone_weights=True):
        assert not load_backbone_weights
        assert pretrained_root is None
        return torch.nn.Linear(1, 1, bias=False)

    monkeypatch.setattr(models, 'build_model', build)
    model = checkpoints.load_inference_model(checkpoint)
    assert not model.training
    assert torch.equal(model.weight, state['weight'])

    damaged = tmp_path / 'damaged.pt'
    torch.save({'format': checkpoints.INFERENCE_FORMAT, 'architecture': 'safi_net',
                'model_config': {}, 'model': {}}, damaged)
    with pytest.raises(RuntimeError, match='Missing key'):
        checkpoints.load_inference_model(damaged)


def test_historical_previews_are_removed_from_training_config():
    from safi_net.runtime import load_config

    assert 'video_validation' not in load_config(overrides={'video_validation': {'enabled': True}})


@pytest.mark.parametrize('switches', [
    {'cascade': True, 'semantic_features': True, 'aggregation': 'proposed'},
    {'cascade': False, 'semantic_features': False, 'aggregation': 'proposed'},
    {'cascade': False, 'semantic_features': True, 'aggregation': 'standard'},
])
def test_fast_rejects_incompatible_checkpoint_graph(monkeypatch, switches):
    from safi_net import fast

    monkeypatch.setattr(fast, 'load_inference_model',
                        lambda *args, **kwargs: SimpleNamespace(mode='full', ablation=switches))
    with pytest.raises(ValueError, match='full SAFI-Net'):
        fast.load_fast_model('other.pt')
