import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from examples.tiny_batch import tiny_batch
from omnigpo import StandardConfig, compute_credit, critic_loss, compute_policy_loss, compute_turn_ppo_gae
from gigpo.bepo_neural_credit import raw_step_rewards, attach_bepo_neural_credit
from gigpo.core_bepo import compute_bepo_advantage, _normalize_per_uid
from gigpo.graph_value_teacher import graph_value_aux_loss, attach_graph_teacher


def test_defaults_match_the_locked_standard_algorithm_recipe():
    c=StandardConfig()
    assert (c.q,c.beta,c.gamma,c.lam,c.step_weight,c.cliprange_value)==(.25,.1,.95,.8,1.5,.5)


@pytest.mark.parametrize('q',[0.,.25,1.])
def test_fixed_credit_matches_the_three_channel_equation(q):
    b=tiny_batch();cfg=StandardConfig(q=q)
    _,_,_,parts=compute_bepo_advantage(b.batch['token_level_rewards'],b.batch['response_mask'],
        b.non_tensor_batch['anchor_obs'],b.non_tensor_batch['rewards'],b.non_tensor_batch['uid'],
        b.non_tensor_batch['traj_uid'],cfg=cfg.bepo(),step_advantage_w=cfg.step_weight,
        gigpo_mode='mean_std_norm',step_idx=b.non_tensor_batch['step_idx'],
        terminal_kind=b.non_tensor_batch['terminal_kind'],return_components=True)
    raw=raw_step_rewards(b)
    neural,returns,mask=compute_turn_ppo_gae(raw,b.batch['values'],b.batch['response_mask'],
        b.non_tensor_batch['traj_uid'],b.non_tensor_batch['step_idx'],cfg.gamma,cfg.lam)
    n=torch.tensor(_normalize_per_uid(neural[:,0].numpy(),b.non_tensor_batch['uid'],
        'mean_std_norm')).unsqueeze(-1)*b.batch['response_mask']
    expected=parts['outcome_weight']*parts['outcome']+cfg.step_weight*((1-q)*parts['graph_gate']*parts['fp_step']+q*n)
    original=b.batch['token_level_rewards'].clone()
    compute_credit(b,cfg)
    torch.testing.assert_close(b.batch['advantages'],expected,atol=1e-7,rtol=1e-6)
    torch.testing.assert_close(b.batch['returns'],returns,atol=0,rtol=0)
    torch.testing.assert_close(b.batch['value_mask'],mask,atol=0,rtol=0)
    torch.testing.assert_close(b.batch['token_level_rewards'],original,atol=0,rtol=0)
    assert raw.sum()==10. and original.sum()>29.


def test_q_zero_exactly_recovers_native_graph_actor_credit():
    b=tiny_batch();cfg=StandardConfig(q=0.)
    compute_credit(b,cfg)
    torch.testing.assert_close(b.batch['advantages'],b.batch['bepo_reference_advantages'],atol=0,rtol=0)


@pytest.mark.parametrize('q',[float('nan'),float('inf'),-.1,1.1])
def test_invalid_q_is_rejected(q):
    with pytest.raises(ValueError):StandardConfig(q=q)


@pytest.mark.parametrize('field,value',[('beta',-.1),('gamma',1.2),('lam',-.1),('step_weight',-.1)])
def test_invalid_coefficients_are_rejected(field,value):
    with pytest.raises(ValueError):replace(StandardConfig(),**{field:value})


def test_raw_rewards_do_not_repeat_episode_outcome_per_turn():
    b=tiny_batch();original=b.batch['token_level_rewards'].clone()
    compute_credit(b)
    assert b.meta_info['bepo_metrics']['credit_route/raw_step_reward_total_unique']==10.
    torch.testing.assert_close(original,b.batch['token_level_rewards'])


def test_turn_gae_crosses_observations_and_reads_before_action_value():
    r=torch.tensor([[0.,0.,0.],[0.,1.,0.]])
    v=torch.tensor([[.2,99.,0.],[.4,99.,0.]])
    m=torch.tensor([[1.,1.,0.],[1.,1.,0.]])
    a,t,vm=compute_turn_ppo_gae(r,v,m,np.array(['a','a']),np.array([0,1]),.9,1.)
    torch.testing.assert_close(a,torch.tensor([[.7,.7,0.],[.6,.6,0.]]))
    torch.testing.assert_close(t,torch.tensor([[.9,0.,0.],[1.,0.,0.]]))
    torch.testing.assert_close(vm,torch.tensor([[1.,0.,0.],[1.,0.,0.]]))


def test_missing_turn_is_not_silently_bootstrapped():
    z=torch.zeros(2,2)
    with pytest.raises(ValueError,match='missing'):
        compute_turn_ppo_gae(z,z,torch.ones_like(z),np.array(['a','a']),np.array([0,2]),.95,.8)


def test_actor_uses_token_ratios_not_joint_action_ratio():
    old=torch.zeros(1,2);new=torch.full((1,2),float(np.log(1.1)),requires_grad=True)
    loss,clip,*_=compute_policy_loss(old,new,torch.ones_like(old),torch.ones_like(old),cliprange=.2)
    torch.testing.assert_close(loss,torch.tensor(-1.1));assert clip.item()==0.
    loss.backward();assert new.grad.count_nonzero()==2


def test_graph_supervision_targets_are_detached():
    pred=torch.tensor([[2.,20.],[4.,40.]],requires_grad=True)
    target=torch.tensor([[1.,99.],[0.,99.]],requires_grad=True)
    conf=torch.tensor([[1.,0.],[0.,0.]],requires_grad=True)
    mask=torch.tensor([[1.,0.],[1.,0.]])
    loss=graph_value_aux_loss(pred,target,conf,mask);loss.backward()
    torch.testing.assert_close(loss,torch.tensor(.5))
    torch.testing.assert_close(pred.grad,torch.tensor([[1.,0.],[0.,0.]]))
    assert target.grad is None and conf.grad is None


def test_critic_uses_base_loss_plus_fixed_beta_auxiliary():
    b=compute_credit(tiny_batch());v=b.batch['values'].clone().requires_grad_()
    total,metrics=critic_loss(v,b)
    torch.testing.assert_close(total,metrics['value_loss']+.1*metrics['graph_aux_loss'])
    total.backward();assert torch.isfinite(v.grad).all()
    assert not b.batch['advantages'].requires_grad and not b.batch['graph_value_targets'].requires_grad


def test_beta_zero_disables_only_the_critic_auxiliary():
    b=compute_credit(tiny_batch());v=b.batch['values'].clone().requires_grad_()
    total,metrics=critic_loss(v,b,StandardConfig(beta=0.))
    torch.testing.assert_close(total,metrics['value_loss'])
    assert b.batch['bepo_fp_credit'].abs().sum()>0


def test_padding_copies_do_not_inflate_graph_support():
    b=tiny_batch()
    for name,value in b.batch.items():b.batch[name]=torch.cat([value,value[:1]],dim=0)
    for name,value in b.non_tensor_batch.items():b.non_tensor_batch[name]=np.concatenate([value,value[:1]])
    compute_credit(b)
    assert b.meta_info['graph_teacher_metrics']['graph_teacher/padding_copies']==1
    assert b.batch['graph_confidence'][-1].count_nonzero()==0
    assert b.meta_info['bepo_metrics']['credit_route/raw_step_reward_total_unique']==10.


def test_graph_reconvergence_transfers_downstream_information():
    b=compute_credit(tiny_batch())
    assert abs(b.batch['graph_value_targets'][1,0].item()-4.75)<1e-5
    assert abs(b.batch['graph_value_targets'][4,0].item()-4.75)<1e-5


@pytest.mark.parametrize('key',['neural_credit_router','mc_direct_supported'])
def test_exploratory_routes_are_not_accepted_by_fixed_core(key):
    cfg=StandardConfig();bepo=cfg.bepo();bepo[key]={'kind':'unused'} if key.endswith('router') else True
    with pytest.raises(ValueError,match='fixed-q'):
        attach_bepo_neural_credit(tiny_batch(),bepo,cfg.teacher(),gamma=.95,lam=.8,
            step_advantage_w=1.5,gigpo_mode='mean_std_norm')
