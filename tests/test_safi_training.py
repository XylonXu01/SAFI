"""Joint clip accounting, gradient accumulation, replay and single-stage policy."""
from copy import deepcopy
from pathlib import Path
import shutil

import pytest
import torch

from test_foundation import make_dataset
from test_model import TestMotion, TestSemantic
from safi_net import checkpoints, train
from safi_net.data import AsyncClipDataset, balanced_sample_key, collate_clips
from safi_net.losses import TemporalReference, sum_statistics
from safi_net.models import build_model, parameter_groups
from safi_net.runtime import (load_config, GlobalSampleSampler, StageObjective, scheduled_counts,
    observation_counts, effective_update_objective, statistic_denominators, portable_config)


def model(config):
    torch.manual_seed(17)
    return build_model(config['model'], test_backbones=(TestMotion(.1), TestMotion(-.2), TestSemantic()))


@pytest.fixture
def case(tmp_path):
    root=tmp_path/'data';manifest=make_dataset(root)
    config=load_config(overrides={
        'model': {'base_channels':4,'semantic_dim':8,'transport':{'radius':1}},
        'data':{'root':str(root),'manifest':manifest,'crop_size':32},
        'run_dir':str(tmp_path/'run'),'precision':'float32','workers':0,
        'video_validation':{'enabled':False},'validation_clips':6,
        'log_every':1,'validate_every':1,'save_every':1,
        'stages':{'joint':{'batch_size':2,'max_steps':2,'accumulation':1,'workers':0}}})
    datasets={split:AsyncClipDataset(root,manifest,split,crop_size=32,training=split=='train')
              for split in ('train','val')}
    return config,datasets


def reference(net, config):
    return TemporalReference(net.ir_motion,net.vis_motion,fusion_config=config['fusion'])


def test_joint_loader_has_three_queries_no_motion_context_or_teacher(case):
    config,ds=case
    keys=list(GlobalSampleSampler(ds['train'],0,4,2,stage='joint'))
    assert all(k.query_index is None for k in keys)
    samples=[ds['train'][k] for k in keys]
    assert all('teacher_inputs' not in s and len(s['inputs'])==3 for s in samples)
    assert all(q.motion_context is None for s in samples for q in s['inputs'])
    batch=collate_clips(samples)
    assert observation_counts(batch,'joint')==scheduled_counts(ds['train'],'joint',0,4)
    assert observation_counts(batch,'joint')['queries']==12


def test_joint_gradient_accumulation_matches_effective_batch(case):
    config,ds=case;net=model(config);parameter_groups(net,'joint');ref=reference(net,config)
    samples=[ds['train'][balanced_sample_key(ds['train'],i)] for i in range(2)]
    objective=StageObjective(net,'joint',config['fusion'])
    weights={**objective.loss.weights,'temporal':.1,'temporal_lowfreq':.1}
    counts=scheduled_counts(ds['train'],'joint',0,2)
    stats=objective(collate_clips(samples),reference=ref,return_stats=True)
    assert stats['temporal'][1]==stats['temporal_lowfreq'][1]==2
    assert stats['target_luma'][1]==6
    loss,_=effective_update_objective(stats,weights,counts);loss.backward()
    expected={name:p.grad.clone() for name,p in net.named_parameters() if p.grad is not None}
    net.zero_grad(set_to_none=True)
    for sample in samples:
        stats=objective(collate_clips([sample]),reference=ref,return_stats=True)
        loss,_=effective_update_objective(stats,weights,counts);loss.backward()
    for name,p in net.named_parameters():
        if name in expected:
            torch.testing.assert_close(p.grad,expected[name],atol=3e-5,rtol=3e-4)
    assert all(p.grad is None for module in (net.ir_motion,net.vis_motion,net.dino) for p in module.parameters())


def test_joint_schedules_are_reproducible_and_validation_weights_fixed(case):
    config,_=case;settings=config['stages']['joint'];settings['max_steps']=15000
    assert train.joint_learning_rate(0,settings)==pytest.approx(1e-5)
    assert train.joint_learning_rate(14999,settings)==pytest.approx(5e-6)
    assert train.temporal_weight_factor(0)==.1
    assert train.temporal_weight_factor(250)==pytest.approx(.55)
    assert train.temporal_weight_factor(500)==1
    assert train.temporal_weight_factor(900)==1


def initial_checkpoint(config, net, path, manifest):
    state={'format_version':1,'architecture':checkpoints.ARCHITECTURE,
        'training_recipe':checkpoints.TRAINING_RECIPE,'config':portable_config(config),
        'model_config':deepcopy(config['model']),'model':net.state_dict(),
        'loss_fingerprint':checkpoints.loss_fingerprint(config),'asset_fingerprints':{},
        'manifest_identity':manifest['fingerprint'],'stage':'joint','step':0,
        'temporal_reference':reference(net,config).state_dict()}
    checkpoints.atomic_save(state,path)


def test_joint_train_resume_replays_next_update(case,tmp_path,monkeypatch):
    config,ds=case;source=tmp_path/'init.pt';initial_checkpoint(config,model(config),source,ds['train'].manifest)
    archive=tmp_path/'interrupted';archive.mkdir()
    real_save=train.atomic_save
    def save(state,path):
        real_save(state,path)
        if state is not None and state['step']<=1:
            if Path(path).name=='best.pt' or state['step']==1:
                shutil.copy2(path,archive/Path(path).name)
    monkeypatch.setattr(train,'atomic_save',save)
    result=train.train_stage(config,'joint',model=model(config),datasets=ds,init_from=source,device='cpu')
    full=checkpoints.load_checkpoint(result['latest'])
    monkeypatch.setattr(train,'atomic_save',real_save)
    resumed_config=deepcopy(config);resumed_config['run_dir']=str(tmp_path/'resumed')
    result=train.train_stage(resumed_config,'joint',model=model(config),datasets=ds,
                             resume=archive/'latest.pt',device='cpu')
    resumed=checkpoints.load_checkpoint(result['latest'])
    assert full['stage']==resumed['stage']=='joint'
    assert full['step']==resumed['step']==2
    assert full['logical_sample_count']==resumed['logical_sample_count']==4
    assert full['schedule']==resumed['schedule']
    for name in full['model']:
        torch.testing.assert_close(full['model'][name],resumed['model'][name],atol=0,rtol=0)
    assert full['optimizer']['param_groups']==resumed['optimizer']['param_groups']


def test_pipeline_calls_joint_once_and_never_advances(case,monkeypatch):
    from safi_net import pipeline
    config,ds=case;calls=[]
    def fake(config,stage,**kwargs):
        calls.append((stage,kwargs));return {'best':'best','latest':'latest'}
    monkeypatch.setattr(pipeline,'train_stage',fake)
    result=pipeline.run_pipeline(config,init_from_v6='fixed_v6.pt',device='cpu')
    assert len(calls)==len(result)==1 and calls[0][0]=='joint'
    assert calls[0][1]['init_from_v6']=='fixed_v6.pt'


def test_validation_requires_support_for_each_enabled_temporal_loss():
    from safi_net.metrics import selection_eligible
    support={name:{'sum':1} for name in ('target_support','temporal_support','temporal_lowfreq_support')}
    assert selection_eligible('joint',support,{'temporal':.1,'temporal_lowfreq':.1})
    support['temporal_lowfreq_support']['sum']=0
    assert not selection_eligible('joint',support,{'temporal':.1,'temporal_lowfreq':.1})
    assert selection_eligible('joint',support,{'temporal':.1,'temporal_lowfreq':0})


def _ddp_worker(rank,config,rendezvous,destination):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from contextlib import nullcontext
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    try:
        ds=AsyncClipDataset(config['data']['root'],config['data']['manifest'],'train',32,True)
        net=model(config);parameter_groups(net,'joint');ref=reference(net,config)
        objective=StageObjective(net,'joint',config['fusion'])
        wrapped=DistributedDataParallel(objective,find_unused_parameters=True)
        counts=scheduled_counts(ds,'joint',0,4)
        weights={**objective.loss.weights,'temporal':.1,'temporal_lowfreq':.1}
        for micro in range(2):
            sample=ds[balanced_sample_key(ds,2*micro+rank)]
            with wrapped.no_sync() if micro==0 else nullcontext():
                stats=wrapped(collate_clips([sample]),reference=ref,return_stats=True)
                loss,_=effective_update_objective(stats,weights,counts,world_size=2)
                loss.backward()
        if rank==0:
            torch.save({n:p.grad for n,p in net.named_parameters() if p.grad is not None},destination)
    finally:
        dist.destroy_process_group()


def test_ddp_two_rank_accumulation_matches_global_batch(case,tmp_path):
    import torch.multiprocessing as mp
    config,ds=case;net=model(config);parameter_groups(net,'joint');ref=reference(net,config)
    objective=StageObjective(net,'joint',config['fusion'])
    samples=[ds['train'][balanced_sample_key(ds['train'],i)] for i in range(4)]
    stats=objective(collate_clips(samples),reference=ref,return_stats=True)
    weights={**objective.loss.weights,'temporal':.1,'temporal_lowfreq':.1}
    loss,_=effective_update_objective(stats,weights,scheduled_counts(ds['train'],'joint',0,4));loss.backward()
    destination=tmp_path/'ddp_grads.pt'
    mp.spawn(_ddp_worker,args=(config,str(tmp_path/'rendezvous'),str(destination)),nprocs=2,join=True)
    grads=torch.load(destination,weights_only=True)
    expected={n:p.grad for n,p in net.named_parameters() if p.grad is not None}
    assert grads.keys()==expected.keys()
    for name in grads:
        torch.testing.assert_close(grads[name],expected[name],atol=3e-5,rtol=5e-4)


def test_validate_only_never_builds_loader_or_steps_optimizer_and_can_resume(case,tmp_path,monkeypatch):
    import json
    config,ds=case;source=tmp_path/'init_validate.pt';initial= model(config)
    initial_checkpoint(config,initial,source,ds['train'].manifest)
    expected={name:value.detach().clone() for name,value in initial.state_dict().items()}
    def forbidden(*args,**kwargs):
        raise AssertionError('validate-only must not create a training loader or update the optimizer')
    with monkeypatch.context() as scoped:
        scoped.setattr(torch.optim.AdamW,'step',forbidden)
        scoped.setattr(train,'DataLoader',forbidden)
        result=train.train_stage(config,'joint',model=model(config),datasets=ds,
                                 init_from=source,device='cpu',validate_only=True)
    checkpoint=checkpoints.load_checkpoint(result['latest'])
    assert result['validation_only'] and result['phase']=='validated'
    assert checkpoint['step']==checkpoint['logical_sample_count']==0
    assert checkpoint['completed'] is False and checkpoint['termination_reason'] is None
    assert checkpoint['early_stopping']['stopped'] is False
    assert checkpoint['early_stopping']['last_validation_step'] is None
    assert checkpoint['optimizer']['state']=={}
    for name,value in expected.items():
        torch.testing.assert_close(checkpoint['model'][name],value,atol=0,rtol=0)
    status=json.loads((Path(result['latest']).parent/'status.json').read_text())
    assert status['phase']=='validated' and status['validation_only'] is True
    resumed=train.train_stage(config,'joint',model=model(config),datasets=ds,
                              resume=result['latest'],device='cpu')
    assert resumed['step']==2 and resumed['logical_sample_count']==4
    assert resumed['validation_only'] is False


def test_pipeline_propagates_validate_only(case,monkeypatch):
    from safi_net import pipeline
    config,_=case;calls=[]
    def fake(config,stage,**kwargs):
        calls.append(kwargs);return {'best':'best','latest':'latest'}
    monkeypatch.setattr(pipeline,'train_stage',fake)
    pipeline.run_pipeline(config,init_from_v6='fixed_v6.pt',device='cpu',validate_only=True)
    assert len(calls)==1 and calls[0]['validate_only'] is True
