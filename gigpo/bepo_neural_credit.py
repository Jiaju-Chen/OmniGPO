"""Fixed-q BEPO/learned-critic credit routing, with a separate graph auxiliary.

Actor outcome and graph channels are the original native BEPO computation.
The neural GAE and critic targets use one RAW environment reward per turn,
matching BEPO's graph. Invalid-action penalties remain in native outcome credit;
they are not duplicated into a newly invented step reward.
"""
import math
from pathlib import Path

import numpy as np
import torch

from gigpo.core_bepo import compute_bepo_advantage, _normalize_per_uid


def neural_credit_enabled(algorithm):
    return (algorithm.get('adv_estimator') == 'bepo'
            and (algorithm.get('bepo') or {}).get('neural_credit_q') is not None)


def validate_configuration(config):
    if not neural_credit_enabled(config.algorithm):
        return
    cfg = config.algorithm.bepo
    q = float(cfg.neural_credit_q)
    if not math.isfinite(q) or not 0. <= q <= 1.:
        raise ValueError('Neural-credit q must be finite in [0,1]')
    if config.actor_rollout_ref.actor.policy_loss.loss_mode != 'vanilla':
        raise ValueError('Keep original BEPO token-level PPO, not turn-level ratio')
    if config.reward_model.use_step_rewards or config.algorithm.use_kl_in_reward:
        raise ValueError('Native outcome rewards and separate KL loss must be preserved')
    if config.algorithm.gamma != cfg.gamma or config.algorithm.lam != cfg['lambda']:
        raise ValueError('Neural and graph GAE must use the same gamma/lambda')
    if cfg.combine_mode != 'add' or cfg.gate_mode != 'hard' or not cfg.normalize_step:
        raise ValueError('Preserve original additive hard-gated normalized BEPO')
    if not config.algorithm.turn_ppo.graph_aux_enabled:
        raise ValueError('The graph-to-critic learning loss is required')
    beta=float(config.critic.graph_aux_beta)
    alpha=float(config.critic.get('graph_edge_alpha',0.))
    # Original .1 recipes remain valid. Only the two author-approved additional
    # FIXED node/edge combinations are admitted; no adaptive beta or schedule.
    # Nov 2 author-approved missing control: retain the graph Actor pathway and
    # teacher diagnostics, but give its Critic auxiliary exactly zero weight.
    zero_control=(beta==0. and q==.25 and alpha==0.
                  and cfg.get('neural_credit_router') is None
                  and not cfg.get('mc_direct_supported',False))
    approved=(zero_control or beta==.1 or any(math.isclose(beta,b,rel_tol=0.,abs_tol=1e-12)
        and math.isclose(alpha,a,rel_tol=0.,abs_tol=1e-12)
        for b,a in [(.15,2./3.),(.2,.5)]))
    if config.critic.graph_aux_schedule is not None or not approved:
        raise ValueError('Use an approved fixed auxiliary weight, including fixed-q.25 beta0 control; no adaptive beta')
    route = cfg.get('neural_credit_router')
    if cfg.get('mc_direct_supported', False) and route is None:
        raise ValueError('Supported MC-direct control requires the same delayed router for fallback')
    if route is not None:
        if route.get('kind') != 'delayed_mc_credit_ls' or float(route['momentum']) not in (0., .9):
            raise ValueError('Only the authorized momentum0/.9 pair is supported')
        if float(route['initial_q']) != .25 or q != .25:
            raise ValueError('Both experiments start with q=.25')


def raw_step_rewards(data):
    mask = data.batch['response_mask'].bool()
    if not bool(mask.any(-1).all()):
        raise ValueError('Every turn needs at least one valid response token')
    rewards = np.asarray(data.non_tensor_batch['rewards'], dtype=np.float64).reshape(-1)
    if len(rewards) != mask.shape[0] or not np.isfinite(rewards).all():
        raise ValueError('Invalid per-turn environment reward vector')
    positions = torch.arange(mask.shape[-1], device=mask.device).expand_as(mask)
    last = positions.masked_fill(~mask, -1).max(-1).values
    output = torch.zeros_like(data.batch['token_level_rewards'])
    output[torch.arange(len(rewards), device=mask.device), last] = torch.as_tensor(
        rewards, device=mask.device, dtype=output.dtype)
    return output


@torch.no_grad()
def attach_bepo_neural_credit(data, bepo_cfg, graph_cfg, *, gamma, lam,
                              step_advantage_w, gigpo_mode):
    from gigpo.ppo import compute_turn_ppo_gae
    from gigpo.graph_value_teacher import attach_graph_teacher
    if bepo_cfg.get('neural_credit_router') is not None or bepo_cfg.get('mc_direct_supported', False):
        raise ValueError('This core release supports fixed-q routing only')
    q = float(bepo_cfg['neural_credit_q'])
    if not math.isfinite(q) or not 0. <= q <= 1.:
        raise ValueError('Invalid fixed q')
    if gamma != bepo_cfg['gamma'] or lam != bepo_cfg['lambda']:
        raise ValueError('Mismatched credit horizons')
    if graph_cfg.get('graph_aux_gate_mode') is not None:
        raise ValueError('No adaptive graph-beta gate in the fixed-q experiment')
    if not graph_cfg.get('graph_aux_enabled', False):
        raise ValueError('Graph auxiliary must remain enabled')
    original_rewards = data.batch['token_level_rewards']
    reference, _, metrics, components = compute_bepo_advantage(
        token_level_rewards=original_rewards, eos_mask=data.batch['response_mask'],
        anchor_obs=data.non_tensor_batch['anchor_obs'],
        rewards=data.non_tensor_batch['rewards'], index=data.non_tensor_batch['uid'],
        traj_index=data.non_tensor_batch['traj_uid'], cfg=bepo_cfg,
        step_advantage_w=step_advantage_w, gigpo_mode=gigpo_mode,
        step_idx=data.non_tensor_batch.get('step_idx'),
        action_text=data.non_tensor_batch.get('action_text'),
        is_action_valid=data.non_tensor_batch.get('is_action_valid'),
        terminal_kind=data.non_tensor_batch.get('terminal_kind'), return_components=True)
    step_rewards = raw_step_rewards(data)
    neural, returns, value_mask = compute_turn_ppo_gae(
        step_rewards, data.batch['values'], data.batch['response_mask'],
        data.non_tensor_batch['traj_uid'], data.non_tensor_batch['step_idx'],
        gamma=gamma, lam=lam, whiten_advantages=False)
    rows = torch.arange(neural.shape[0], device=neural.device)
    first = data.batch['response_mask'].bool().long().argmax(-1)
    raw_neural = neural[rows, first].float().cpu().numpy()
    normalized = _normalize_per_uid(raw_neural, data.non_tensor_batch['uid'], gigpo_mode)
    ncredit = torch.as_tensor(normalized, device=neural.device, dtype=neural.dtype).unsqueeze(-1)
    ncredit = ncredit * data.batch['response_mask']
    graph_credit = components['graph_gate'] * components['fp_step']
    # Attach the unchanged full graph before routing so its existing grouped-MC
    # values can calibrate the Actor credit without another graph or forward.
    data.batch['returns'] = returns.detach()
    data.batch['value_mask'] = value_mask
    data.batch['token_level_rewards'] = step_rewards
    try:
        attach_graph_teacher(data, gamma, graph_cfg)
    finally:
        data.batch['token_level_rewards'] = original_rewards
    route = bepo_cfg.get('neural_credit_router')
    if route is not None:
        from gigpo.credit_router import compute_routing
        def boundary(key):
            return data.batch[key][rows, first].detach().float().cpu().numpy()
        supported = ((boundary('graph_confidence') > 0)
                     & (components['graph_gate'][rows, first].cpu().numpy() > 0))
        q, next_state, route_metrics, proxy = compute_routing(
            graph_credit[rows, first].cpu().numpy(), ncredit[rows, first].cpu().numpy(),
            boundary('graph_mc_targets'), data.non_tensor_batch['rewards'], supported,
            data.non_tensor_batch['uid'], data.non_tensor_batch['traj_uid'],
            data.non_tensor_batch['step_idx'], gamma=gamma,mode=gigpo_mode,cfg=route,
            state=data.meta_info.get('credit_router_state'),
            outer_step=data.meta_info['credit_router_outer_step'])
        data.meta_info['credit_router_state_next'] = next_state
        data.meta_info['credit_router_record'] = route_metrics
        metrics.update(route_metrics)
        data.batch['bepo_mc_credit_proxy'] = torch.as_tensor(
            proxy, device=neural.device, dtype=neural.dtype).unsqueeze(-1)*data.batch['response_mask']
    # Exact legacy identity at q=0 avoids a cancellation-rounding artifact.
    actor = reference if q == 0. else (
        components['outcome_weight'] * components['outcome']
        + float(step_advantage_w) * ((1.-q)*graph_credit + q*ncredit))
    # Intervention only where the existing full-graph teacher AND native BEPO
    # gate support a real turn. Padding copies keep the original native fallback;
    # do not fix/reweight the upstream padded Actor layout in this experiment.
    direct = bool(bepo_cfg.get('mc_direct_supported', False))
    mc_mask = torch.zeros_like(data.batch['response_mask'], dtype=torch.bool)
    if direct:
        if route is None:
            raise ValueError('MC-direct requires the matched proxy/router computation')
        mc_mask = torch.as_tensor(supported, device=actor.device).unsqueeze(-1).bool()
        mc_mask = mc_mask & data.batch['response_mask'].bool()
        direct_actor = (components['outcome_weight'] * components['outcome']
                        + float(step_advantage_w) * data.batch['bepo_mc_credit_proxy'])
        actor = torch.where(mc_mask, direct_actor, actor)
    if not torch.isfinite(actor).all() or not torch.isfinite(returns).all():
        raise ValueError('Nonfinite actor credit or critic target')
    data.batch['advantages'] = actor.detach()
    data.batch['returns'] = returns.detach()
    data.batch['value_mask'] = value_mask
    data.batch['bepo_reference_advantages'] = reference.detach()
    data.batch['bepo_fp_credit'] = graph_credit.detach()
    data.batch['bepo_neural_credit'] = ncredit.detach()
    data.batch['bepo_graph_gate'] = components['graph_gate'].detach()
    data.batch['bepo_mc_direct_mask'] = mc_mask.detach()
    mask = data.batch['response_mask'].bool()
    gates = components['graph_gate'][rows, first]
    lookup = {}
    for i in range(len(rows)):
        key = (str(data.non_tensor_batch['uid'][i]), str(data.non_tensor_batch['traj_uid'][i]),
               int(np.asarray(data.non_tensor_batch['step_idx'][i]).reshape(-1)[0]))
        lookup.setdefault(key, i)
    ids = list(lookup.values())
    umask = torch.as_tensor(ids, device=neural.device)
    g_unique = graph_credit[umask, first[umask]]
    n_unique = ncredit[umask, first[umask]]
    eligible = (reference.abs() > .05) & (actor.abs() > .05) & mask
    actual_graph = ((1.-q)*float(step_advantage_w)*graph_credit).masked_fill(mc_mask, 0.)
    actual_neural = (q*float(step_advantage_w)*ncredit).masked_fill(mc_mask, 0.)
    actual_mc = (float(step_advantage_w)*data.batch['bepo_mc_credit_proxy']).masked_fill(~mc_mask, 0.) if route is not None else torch.zeros_like(actor)
    metrics.update({
        'credit_route/q': q,
        'credit_route/graph_supported_fraction': float(gates.mean()),
        'credit_route/neural_raw_std': float(np.std(raw_neural)),
        'credit_route/neural_credit_abs_mean': float(ncredit[mask].abs().mean()),
        'credit_route/actor_credit_abs_mean': float(actor[mask].abs().mean()),
        'credit_route/actor_sign_change_fraction': float(
            (((actor*reference) < 0) & eligible).sum()/eligible.sum().clamp_min(1)),
        'credit_route/unique_turns': len(ids),
        'credit_route/unique_trajectories': len({key[:2] for key in lookup}),
        'credit_route/raw_step_reward_total_unique': float(step_rewards[ids].sum()),
        'credit_route/raw_reward_matches_bepo_graph': 1.,
        'credit_route/graph_abs_mean_unique':float(g_unique.abs().mean()),
        'credit_route/neural_abs_mean_unique':float(n_unique.abs().mean()),
        'credit_route/weighted_graph_abs_mean_unique':float(actual_graph[umask, first[umask]].abs().mean()),
        'credit_route/weighted_neural_abs_mean_unique':float(actual_neural[umask, first[umask]].abs().mean()),
        'credit_route/weighted_mc_abs_mean_unique':float(actual_mc[umask, first[umask]].abs().mean()),
        'credit_route/mc_direct_enabled':float(direct),
        'credit_route/mc_direct_unique_fraction':float(mc_mask[umask,first[umask]].float().mean()),
        'credit_route/weighted_outcome_abs_mean_unique':float((components['outcome_weight']*components['outcome'])[umask,first[umask]].abs().mean()),
    })
    data.meta_info['bepo_metrics'] = metrics
    return data


def save_credit_snapshot(data, path):
    """Boundary-only diagnostics with the actual critic reward convention."""
    from gigpo.graph_value_teacher import save_teacher_snapshot
    if 'bepo_neural_credit' not in data.batch:
        return save_teacher_snapshot(data, path)
    original = data.batch['token_level_rewards']
    data.batch['token_level_rewards'] = raw_step_rewards(data)
    try:
        save_teacher_snapshot(data, path)
    finally:
        data.batch['token_level_rewards'] = original
    path = Path(path)
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    mask = data.batch['value_mask'].bool()
    rows = torch.arange(mask.shape[0], device=mask.device)
    pos = mask.long().argmax(-1)
    for key in ('bepo_reference_advantages','bepo_fp_credit','bepo_neural_credit','bepo_graph_gate','bepo_mc_direct_mask'):
        arrays[key] = data.batch[key][rows, pos].detach().float().cpu().numpy()
    if 'bepo_mc_credit_proxy' in data.batch:
        arrays['bepo_mc_credit_proxy'] = data.batch['bepo_mc_credit_proxy'][rows,pos].detach().float().cpu().numpy()
    np.savez_compressed(path, **arrays)
