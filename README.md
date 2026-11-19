<div align="center">

<h1>🎬 SAFI-Net</h1>

<p><strong>Semantic Guidance for Joint Infrared–Visible Video Fusion and Frame Interpolation</strong></p>

<p><em>Fuse and interpolate in one pass, guided by visible semantics.</em></p>

<p>
  <a href="https://arxiv.org/abs/2611.24277"><img alt="arXiv paper 2611.24277" src="https://img.shields.io/badge/arXiv-2611.24277-b31b1b.svg"></a>
  <img alt="Python 3.10 or newer" src="https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&amp;logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C.svg?logo=pytorch&amp;logoColor=white">
  <img alt="DINOv3 and RIFE_m backbones" src="https://img.shields.io/badge/Backbones-DINOv3%20%2B%20RIFE__m-6f42c1.svg">
</p>

<p>
  <a href="#why-safi-net">✨ Why SAFI-Net?</a> ·
  <a href="#method">🧩 Method</a> ·
  <a href="#quick-start">⚡ Quick Start</a> ·
  <a href="#inference">▶️ Inference</a> ·
  <a href="#training">🧪 Training</a> ·
  <a href="#weight-distribution-and-credits">📦 Weights &amp; Credits</a>
</p>

<img src="assets/overview.png" alt="Graphical abstract comparing cascaded methods and SAFI-Net for same-rate and different-rate infrared–visible inputs">

<p><em>SAFI-Net reconstructs missing observations and fuses infrared and visible frames at a shared target time.</em></p>

</div>

## Why SAFI-Net?

When one camera runs slower than the other, a fused video still needs aligned infrared and visible information at each target time. Cascading interpolation and fusion can propagate reconstruction errors and introduce brightness fluctuations. SAFI-Net uses visible semantics to guide both tasks in a single model.

| Challenge | How SAFI-Net handles it |
| --- | --- |
| Missing observations | Reconstructs missing frames and produces a fused frame at the same target time. |
| Different input rates | Uses the observed visible frame to guide missing infrared reconstruction when the visible stream runs at twice the infrared rate. |
| Brightness fluctuations | Applies bounded thermal enhancement and trains with a temporal brightness consistency loss. |

On HDO, M3SVD, and VTMOT, SAFI-Net improves visual quality at lower computational cost than the evaluated cascades. It also improves temporal consistency for different-rate inputs and supports downstream multi-object tracking on VTMOT under missing observations.

## Method

![SAFI-Net architecture showing motion-guided alignment, shared source aggregation, and bounded thermal enhancement](assets/architecture.png)

1. **🎯 Target-time semantic guidance.** SAFI-Net extracts [DINOv3](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) features directly from an observed visible frame. When that frame is missing, motion-guided local matching propagates semantics from neighboring frames.
2. **🔀 Shared source selection.** Separate [RIFE_m](https://github.com/hzwer/ECCV2022-RIFE) branches align infrared and visible features. Matching confidence helps weight the aligned sources, and both reconstruction and fusion use those weights at each scale.
3. **💡 Temporal brightness consistency.** The decoder adds bounded thermal enhancement to a visible reference to preserve its color. Training penalizes brightness inconsistencies across frames while accounting for changes in thermal contrast.

## Quick Start

### Install

Use Python 3.10 or newer and a PyTorch build for your device. Video inference also requires FFmpeg with `libx264` and `ffprobe`.

```sh
python -m pip install -e .
```

### Download the checkpoint

The self-contained [`weights/safi_net.pt`](weights/safi_net.pt) checkpoint bundles the task network, DINOv3, and both trained RIFE_m branches. Download it from the file page and place it in `weights/`, or run the following from your checkout with [Git LFS](https://git-lfs.com/) installed.

```sh
git lfs install --local
git lfs pull --include="weights/safi_net.pt"
```

### Run the demo

> [!TIP]
> 🎞️ The bundled demo contains nine synthetic, spatially aligned IR–VIS pairs at 256 × 192 pixels.

```sh
python -m safi_net.infer_video \
  --ir-input demo/ir --vis-input demo/vis --fps-ir 10 --fps-vis 10 \
  --checkpoint weights/safi_net.pt --output-dir outputs/demo --device cuda:0
```

Look in `outputs/demo/` for 17 PNGs under `fused/`, the video `fused.mp4`, and `summary.json` with output timestamps and checkpoint identity.

## Inference

Use spatially registered inputs with matching dimensions. The video CLI reads frame rates and timestamps from video files by default. Pass `--fps-ir` and `--fps-vis` to specify nominal rates when the metadata is ambiguous. For frame directories, pass the input sampling rates with those flags, or supply JSON timestamps with `--timestamps-ir` and `--timestamps-vis`.

| Mode | Setup |
| --- | --- |
| **Same-rate video** | Use equal input rates, such as `--fps-ir 10 --fps-vis 10`. The output includes observed infrared times and the midpoints between them. |
| **Different-rate video** | Use a 2:1 VIS-to-IR rate, such as `--fps-ir 5 --fps-vis 10`. The output follows the visible timeline within the range covered by both inputs. |
| **Fast video** | Add `--fast` for the FP32 runtime that reuses features from repeated observations. Add `--save-aux` to save IR and VIS interpolation frames. |
| **Single image fusion** | Run `python -m safi_net.fuse_image` with an aligned image pair. The command saves an RGB PNG at the input size. |

Standard video inference defaults to BF16 on CUDA. To use FP32, pass `--precision float32`. For CPU video inference, use `--device cpu --precision float32`. Single-image fusion accepts `--device cpu` on its own.

<details>
<summary><strong>📋 Commands for fast video, the different-rate demo, and image fusion</strong></summary>

### Fast video

```sh
python -m safi_net.infer_video --fast --save-aux \
  --ir-input demo/ir --vis-input demo/vis --fps-ir 10 --fps-vis 10 \
  --checkpoint weights/safi_net.pt --output-dir outputs/demo_fast --device cuda:0
```

### Different-rate demo

Keep every other infrared frame and all nine visible frames so the two streams stay aligned.

```sh
mkdir -p outputs/demo_inputs/ir_5fps
for frame in 0000 0002 0004 0006 0008; do
  cp "demo/ir/${frame}.png" outputs/demo_inputs/ir_5fps/
done

python -m safi_net.infer_video \
  --ir-input outputs/demo_inputs/ir_5fps --vis-input demo/vis \
  --fps-ir 5 --fps-vis 10 \
  --checkpoint weights/safi_net.pt --output-dir outputs/demo_async --device cuda:0
```

### Single image fusion

```sh
python -m safi_net.fuse_image \
  --ir-input demo/ir/0000.png --vis-input demo/vis/0000.png \
  --checkpoint weights/safi_net.pt --output outputs/fused.png --device cuda:0
```

</details>

Choose a new output directory for each video run. An existing SAFI-Net training checkpoint such as `best.pt` also works with the original backbone weights under `pretrained/`.

## Training

Use a VideoFusion-VFI dataset root containing `annotations/sequences.jsonl` and paired infrared and visible frames under `train/<sequence>/` and `test/<sequence>/`. Training defaults to [`artifacts/split_async.json`](artifacts/split_async.json). To create a split for a different dataset copy, see `python -m safi_net.split_manifest --help` and pass the resulting file with `--manifest`.

1. **Verify the pretrained backbones.**

   ```sh
   (cd pretrained && sha256sum -c SHA256SUMS)
   ```

2. **Start joint training.**

   ```sh
   python -m safi_net.pipeline \
     --data-root /path/to/VideoFusion-VFI --run-dir runs/safi \
     --init-from-pretrained --device cuda:0 --batch-size 1 --accumulation-steps 1
   ```

   Adjust batch size and accumulation for your GPU memory. Training saves `runs/safi/joint/latest.pt` and selects `best.pt` using sampled validation.

3. **Resume training with optimizer and RNG state.**

   ```sh
   python -m safi_net.pipeline \
     --data-root /path/to/VideoFusion-VFI --resume-run runs/safi --device cuda:0
   ```

4. **Export a standalone checkpoint from your run.**

   ```sh
   python -m safi_net.export_weights \
     --checkpoint runs/safi/joint/best.pt --output weights/safi_net_custom.pt
   ```

<details>
<summary><strong>🧰 Run the test suite</strong></summary>

```sh
python -m pip install -e '.[dev]'
python -m pytest
```

</details>

## Weight Distribution and Credits

- **Checksum.** Verify the downloaded checkpoint with `(cd weights && sha256sum -c safi_net.pt.sha256)`.
- **Backbones.** SAFI-Net uses [DINOv3 ViT-S/16](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) ([license](weights/DINOv3-LICENSE.md)) and [RIFE_m](https://github.com/hzwer/ECCV2022-RIFE) ([license](weights/RIFE-LICENSE)).
- **Third-party notices.** Vendored RIFE and [PMT](https://github.com/tue-mps/pmt) notices are available in [`safi_net/third_party/`](safi_net/third_party/).
