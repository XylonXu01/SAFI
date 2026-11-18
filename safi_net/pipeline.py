"""SAFI-Net single-stage joint full-image fusion and temporal training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import torch.distributed as dist
import yaml

from .checkpoints import load_checkpoint, validate_training_checkpoint
from .early_stopping import early_stopped_checkpoint
from .runtime import (add_runtime_arguments, apply_runtime_arguments, load_config,
                      merge_config, positive_int, resolve_manifest, validate_config)
from .train import train_stage


def parse_configuration(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    initialize = parser.add_mutually_exclusive_group()
    initialize.add_argument('--resume-run')
    initialize.add_argument('--init-from', help='SAFI-Net joint weights for a fresh run')
    initialize.add_argument('--init-from-v6', help='Explicit v6 weights and frozen reference migration')
    initialize.add_argument('--init-from-pretrained',action='store_true',help='Only pretrained DINO/RIFE; fresh task parameters')
    parser.add_argument('--start-stage', choices=('joint',), default='joint')
    parser.add_argument('--validate-only', action='store_true',
                        help='run sampled validation without updating weights')
    parser.add_argument('--gpus', help='CUDA_VISIBLE_DEVICES, e.g. 0,1,2')
    parser.add_argument('--allow-world-size-change', action='store_true',
                        help='Explicitly resize DDP on resume while retaining effective clips per update')
    for name in ('batch-size', 'accumulation-steps', 'max-steps'):
        parser.add_argument('--' + name, '--' + name + '-joint', dest=name.replace('-', '_'), type=positive_int)
    args = parser.parse_args(argv)
    if args.run_dir and args.resume_run:
        parser.error('--run-dir and --resume-run are mutually exclusive')
    if not any((args.resume_run, args.init_from, args.init_from_v6,args.init_from_pretrained)):
        parser.error('Choose --init-from-pretrained, --init-from, --init-from-v6 or --resume-run')
    if args.gpus is not None:
        if not args.gpus.strip() or any(not value.strip() for value in args.gpus.split(',')):
            parser.error('--gpus must list visible devices')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    config = load_config()
    checkpoint = None
    if args.resume_run:
        args.run_dir = str(Path(args.resume_run).expanduser().resolve())
        checkpoint = load_checkpoint(Path(args.run_dir) / 'joint/latest.pt')
        config = merge_config(config, checkpoint['config'])
    if args.config:
        with Path(args.config).open() as handle:
            config = merge_config(config, yaml.safe_load(handle) or {})
    config = apply_runtime_arguments(config, args)
    config['pipeline_start_stage'] = 'joint'
    for argument, field in (('batch_size', 'batch_size'), ('accumulation_steps', 'accumulation'), ('max_steps', 'max_steps')):
        value = getattr(args, argument)
        if value is not None:
            config['stages']['joint'][field] = value
    validate_config(config)
    if checkpoint is not None:
        validate_training_checkpoint(checkpoint, config)
    return config, args


def run_pipeline(config, resume_run=False, device='cuda', *, start_stage=None, init_from=None, init_from_v6=None,
                 validate_only=False, allow_world_size_change=False,init_from_pretrained=False):
    if (start_stage or config.get('pipeline_start_stage', 'joint')) != 'joint':
        raise ValueError('SAFI-Net has one joint stage')
    if sum(bool(x) for x in (resume_run, init_from, init_from_v6,init_from_pretrained)) != 1:
        raise ValueError('Specify exactly one initialization or resume mode')
    validate_config(config)
    manifest = resolve_manifest(config)
    root = Path(config['run_dir']) / 'joint'
    latest, best = root / 'latest.pt', root / 'best.pt'
    resume = None
    policy = config.get('checkpoint_policy', 'best')
    if latest.exists():
        if not resume_run:
            raise FileExistsError(f'Stage exists; use --resume-run: {latest}')
        checkpoint = load_checkpoint(latest, manifest_identity=manifest['fingerprint'])
        validate_training_checkpoint(checkpoint, config)
        if checkpoint['stage'] != 'joint':
            raise ValueError('Only joint checkpoints can resume v7')
        if early_stopped_checkpoint(checkpoint) or (checkpoint['completed'] and checkpoint['step'] == config['stages']['joint']['max_steps']):
            selected = best if policy == 'best' else latest
            selected_checkpoint = load_checkpoint(selected, manifest_identity=manifest['fingerprint'])
            validate_training_checkpoint(selected_checkpoint, config)
            return [{'stage': 'joint', 'latest': str(latest), 'best': str(best), 'selected': str(selected),
                     'skipped': True, 'step': checkpoint['step'], 'termination_reason': checkpoint.get('termination_reason')}]
        resume = latest
    elif resume_run:
        raise FileNotFoundError(f'No joint checkpoint to resume: {latest}')
    result = train_stage(config, 'joint', resume=resume, init_from=init_from,
                         init_from_v6=init_from_v6, device=device, validate_only=validate_only,
                         **({'allow_world_size_change': True} if allow_world_size_change else {}),
                         **({'init_from_pretrained':True} if init_from_pretrained else {}))
    result.update(skipped=False, selected=result[policy], checkpoint_policy=policy)
    return [result]


def main(argv=None):
    from .preflight import preflight
    actual_argv = sys.argv[1:] if argv is None else argv
    config, args = parse_configuration(actual_argv)
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if args.device == 'cuda' and world_size == 1 and torch.cuda.device_count() > 1:
        subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                        '--nproc-per-node', str(torch.cuda.device_count()), '-m',
                        'safi_net.pipeline', *actual_argv], check=True)
        return
    preflight(config, device=args.device)
    try:
        results = run_pipeline(config, resume_run=bool(args.resume_run), device=args.device,
                               start_stage='joint', init_from=args.init_from, init_from_v6=args.init_from_v6,
                               validate_only=args.validate_only,
                               init_from_pretrained=args.init_from_pretrained,
                               allow_world_size_change=args.allow_world_size_change)
        if int(os.environ.get('RANK', '0')) == 0:
            print(json.dumps({'phase': 'validation_completed' if args.validate_only else 'pipeline_completed', 'stages': results}, allow_nan=False), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
