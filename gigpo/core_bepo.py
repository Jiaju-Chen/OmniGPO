"""Gated-BEPO advantage estimation on an empirical rollout graph."""

from collections import Counter, defaultdict

import numpy as np
import torch

TERMINAL_STATE = "__terminal__"
TERMINAL_SUCCESS = "__terminal_success__"
TERMINAL_FAILURE = "__terminal_failure__"
TERMINAL_TRUNCATED = "__terminal_truncated__"
_TERMINAL_STATES = frozenset(
    {TERMINAL_STATE, TERMINAL_SUCCESS, TERMINAL_FAILURE, TERMINAL_TRUNCATED}
)


def _to_hashable(value):
    if isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return tuple(_to_hashable(item) for item in value.reshape(-1))
    if isinstance(value, (list, tuple)):
        return tuple(_to_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _to_hashable(item)) for key, item in value.items()))
    raise TypeError(f"Unsupported state type: {type(value)}")


def _episode_advantage(
    token_level_rewards, response_mask, index, epsilon, remove_std
):
    """Normalize one episode score per rollout group and broadcast to tokens."""
    episode_scores = token_level_rewards.sum(dim=-1)
    normalized = torch.zeros_like(episode_scores)
    for group in np.unique(index):
        rows = np.where(index == group)[0]
        row_tensor = torch.as_tensor(rows, device=episode_scores.device)
        scores = episode_scores[row_tensor]
        if scores.numel() == 1:
            values = scores
        else:
            centered = scores - scores.mean()
            values = centered if remove_std else centered / (scores.std() + epsilon)
        normalized[row_tensor] = values
    return normalized.unsqueeze(-1) * response_mask


def is_terminal_state(state):
    return state in _TERMINAL_STATES


def _terminal_sentinel(kind):
    return {
        "success": TERMINAL_SUCCESS,
        "failure": TERMINAL_FAILURE,
        "truncated": TERMINAL_TRUNCATED,
    }.get(kind, TERMINAL_STATE)


def get_state_key(observation):
    return _to_hashable(observation)


def _group_rows_by_traj(uid_rows, traj_index, step_idx=None):
    """Recover temporal trajectories after batch balancing reorders rows."""
    trajectories = defaultdict(list)
    for row in uid_rows:
        trajectories[_to_hashable(traj_index[row])].append(int(row))
    for rows in trajectories.values():
        if step_idx is None:
            rows.sort()
        else:
            rows.sort(key=lambda row: int(np.asarray(step_idx[row]).reshape(-1)[0]))
    return trajectories


def _as_action_valid(values, batch_size):
    if values is None:
        return np.ones(batch_size, dtype=bool)
    if len(values) != batch_size:
        raise ValueError(f"is_action_valid has {len(values)} rows, expected {batch_size}")
    return np.asarray(
        [bool(np.asarray(value).reshape(-1).min()) for value in values],
        dtype=bool,
    )


def _build_empirical_mdp(
    trajectories,
    anchor_obs,
    rewards,
    step_cost,
    action_valid=None,
    terminal_kind=None,
):
    """Build per-state transitions while retaining every row for scoring."""
    edges_by_state = defaultdict(list)
    step_state = {}
    step_next_state = {}

    for rows in trajectories.values():
        for position, row in enumerate(rows):
            state = get_state_key(anchor_obs[row])
            if position + 1 == len(rows):
                kind = None if terminal_kind is None else terminal_kind[row]
                next_state = _terminal_sentinel(kind)
            else:
                next_state = get_state_key(anchor_obs[rows[position + 1]])
            step_state[row] = state
            step_next_state[row] = next_state
            if action_valid is None or action_valid[row]:
                edges_by_state[state].append((float(rewards[row]) - step_cost, next_state))
    return edges_by_state, step_state, step_next_state


def _init_values(
    edges_by_state,
    step_state,
    trajectories,
    rewards,
    method,
    gamma,
    step_cost,
    action_valid=None,
):
    values = {state: 0.0 for state in edges_by_state}
    values.update({state: 0.0 for state in _TERMINAL_STATES})
    if method != "mc":
        return values

    state_returns = defaultdict(list)
    for rows in trajectories.values():
        running_return = 0.0
        for row in reversed(rows):
            if action_valid is not None and not action_valid[row]:
                continue
            running_return = float(rewards[row]) - step_cost + gamma * running_return
            state_returns[step_state[row]].append(running_return)
    for state, returns in state_returns.items():
        values[state] = float(np.mean(returns))
    return values


def _run_value_iteration(edges_by_state, values, gamma, max_iter, tol):
    """Solve the empirical Bellman expectation equation."""
    last_difference = 0.0
    actual_iterations = 0
    for iteration in range(max_iter):
        actual_iterations = iteration + 1
        updated = dict(values)
        last_difference = 0.0
        for state, edges in edges_by_state.items():
            targets = [
                reward + gamma * (0.0 if is_terminal_state(next_state) else values.get(next_state, 0.0))
                for reward, next_state in edges
            ]
            new_value = float(np.mean(targets)) if targets else 0.0
            last_difference = max(last_difference, abs(new_value - values.get(state, 0.0)))
            updated[state] = new_value
        values = updated
        if last_difference < tol:
            break
    return values, actual_iterations, last_difference


def _normalize_per_uid(advantages, index, mode, epsilon=1e-6):
    if mode not in {"mean_norm", "mean_std_norm"}:
        raise ValueError(f"Unknown Gated-BEPO normalization mode: {mode!r}")
    normalized = advantages.copy()
    for group in np.unique(index):
        rows = np.where(index == group)[0]
        values = advantages[rows]
        centered = values - values.mean()
        normalized[rows] = (
            centered if mode == "mean_norm" else centered / (values.std() + epsilon)
        )
    return normalized


def _gigpo_step_baseline(
    uid_rows, anchor_obs, rewards, trajectories, gamma, mode, epsilon=1e-6
):
    """State-grouped Monte Carlo fallback for undersized rollout groups."""
    returns = {}
    for rows in trajectories.values():
        running_return = 0.0
        for row in reversed(rows):
            running_return = float(rewards[row]) + gamma * running_return
            returns[row] = running_return

    grouped_rows = defaultdict(list)
    for row in uid_rows:
        grouped_rows[get_state_key(anchor_obs[row])].append(int(row))
    row_to_local = {int(row): local for local, row in enumerate(uid_rows)}
    output = np.zeros(len(uid_rows), dtype=np.float32)
    for rows in grouped_rows.values():
        scores = np.asarray([returns[row] for row in rows], dtype=np.float32)
        centered = scores - scores.mean()
        if mode == "mean_std_norm" and len(rows) > 1:
            centered = centered / (scores.std() + epsilon)
        for row, value in zip(rows, centered):
            output[row_to_local[row]] = value
    return output


def compute_state_gate(
    step_state,
    step_next_state,
    uid_rows,
    min_state_count=2,
    min_successors=2,
    action_valid=None,
):
    """Hard confidence gate based on repeated states with distinct successors."""
    counts = Counter()
    successors = defaultdict(set)
    for row in uid_rows:
        row = int(row)
        if action_valid is not None and not action_valid[row]:
            continue
        state = step_state[row]
        counts[state] += 1
        successors[state].add(step_next_state[row])

    trusted_states = {
        state
        for state, count in counts.items()
        if count >= min_state_count and len(successors[state]) >= min_successors
    }
    gate = {
        int(row): float(
            (action_valid is None or action_valid[int(row)])
            and step_state[int(row)] in trusted_states
        )
        for row in uid_rows
    }
    return gate, len(trusted_states)


def _group_skew_eta(trajectory_rewards, trajectories):
    values = [float(trajectory_rewards[rows[0]]) for rows in trajectories.values()]
    if not values:
        return 1.0
    success_rate = float(np.clip(np.mean(values), 0.0, 1.0))
    return float(4.0 * success_rate * (1.0 - success_rate))


def _validate_inputs(batch_size, **fields):
    for name, values in fields.items():
        if values is not None and len(values) != batch_size:
            raise ValueError(f"{name} has {len(values)} rows, expected {batch_size}")


@torch.no_grad()
def compute_bepo_advantage(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    anchor_obs: np.ndarray,
    rewards: np.ndarray,
    index: np.ndarray,
    traj_index: np.ndarray,
    cfg,
    step_advantage_w: float = 1.0,
    gigpo_mode: str = "mean_norm",
    epsilon: float = 1e-6,
    step_idx: np.ndarray = None,
    action_text: np.ndarray = None,
    is_action_valid: np.ndarray = None,
    terminal_kind: np.ndarray = None,
    return_components: bool = False,
):
    """Compute Gated-BEPO token-level advantages.

    Gated-BEPO estimates a Bellman-consistent potential on the empirical graph
    formed by a rollout group. Its hard gate enables step credit only at states
    that are both repeated and observed with multiple successors.
    """
    del action_text  # Kept in the public signature for trainer compatibility.
    batch_size, response_length = eos_mask.shape
    _validate_inputs(
        batch_size,
        anchor_obs=anchor_obs,
        rewards=rewards,
        index=index,
        traj_index=traj_index,
        step_idx=step_idx,
        terminal_kind=terminal_kind,
    )

    def setting(name, default):
        if cfg is None:
            return default
        return cfg.get(name, default) if hasattr(cfg, "get") else getattr(cfg, name, default)

    gamma = float(setting("gamma", 0.99))
    lam = float(setting("lambda", 0.8))
    use_gae = bool(setting("use_gae", True))
    max_iter = int(setting("max_iter", 20))
    tol = float(setting("tol", 1e-6))
    init = str(setting("init", "mc"))
    step_cost = float(setting("step_cost", 0.0))
    combine_mode = str(setting("combine_mode", "add"))
    normalize_step = bool(setting("normalize_step", True))
    clip_advantage = setting("clip_advantage", None)
    min_traj_per_uid = int(setting("min_traj_per_uid", 1))
    fallback = str(setting("fallback", "gigpo"))
    exclude_invalid_edges = bool(setting("exclude_invalid_edges", False))
    gate_mode = str(setting("gate_mode", "none"))
    eta_min = float(setting("eta_min", 0.5))
    gate_min_state_count = int(setting("gate_min_state_count", 2))
    gate_min_successors = int(setting("gate_min_successors", 2))
    gate_use_group_skew = bool(setting("gate_use_group_skew", False))

    if not 0.0 <= gamma < 1.0:
        raise ValueError(f"bepo.gamma must be in [0, 1), got {gamma}")
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"bepo.lambda must be in [0, 1], got {lam}")
    if init not in {"zero", "mc"}:
        raise ValueError(f"Unknown bepo.init: {init!r}")
    if combine_mode not in {"add", "step_only"}:
        raise ValueError(f"Unknown bepo.combine_mode: {combine_mode!r}")
    if fallback not in {"none", "zero", "gigpo"}:
        raise ValueError(f"Unknown bepo.fallback: {fallback!r}")
    if gate_mode not in {"none", "hard"}:
        raise ValueError(f"Unknown bepo.gate_mode: {gate_mode!r}")
    if not 0.0 <= eta_min <= 1.0:
        raise ValueError(f"bepo.eta_min must be in [0, 1], got {eta_min}")

    rewards = np.asarray(rewards, dtype=np.float32)
    action_valid = _as_action_valid(is_action_valid, batch_size)
    episode_advantage = _episode_advantage(
        token_level_rewards,
        eos_mask,
        index,
        epsilon,
        gigpo_mode == "mean_norm",
    )

    step_advantage = np.zeros(batch_size, dtype=np.float32)
    gate_by_row = np.zeros(batch_size, dtype=np.float32)
    episode_weight = np.ones(batch_size, dtype=np.float32)
    trajectory_rewards = token_level_rewards.sum(dim=-1).cpu().numpy()
    diagnostics = defaultdict(list)
    fallback_groups = 0

    for uid in np.unique(index):
        uid_rows = np.where(index == uid)[0]
        trajectories = _group_rows_by_traj(uid_rows, traj_index, step_idx)
        if len(trajectories) < min_traj_per_uid:
            fallback_groups += 1
            if fallback == "gigpo":
                step_advantage[uid_rows] = _gigpo_step_baseline(
                    uid_rows,
                    anchor_obs,
                    rewards,
                    trajectories,
                    gamma,
                    gigpo_mode,
                    epsilon,
                )
            continue

        valid_for_graph = action_valid if exclude_invalid_edges else None
        edges, step_state, step_next_state = _build_empirical_mdp(
            trajectories,
            anchor_obs,
            rewards,
            step_cost,
            action_valid=valid_for_graph,
            terminal_kind=terminal_kind,
        )
        values = _init_values(
            edges,
            step_state,
            trajectories,
            rewards,
            init,
            gamma,
            step_cost,
            action_valid=valid_for_graph,
        )
        values, iterations, difference = _run_value_iteration(
            edges, values, gamma, max_iter, tol
        )

        deltas = {}
        for row in uid_rows:
            row = int(row)
            state = step_state[row]
            next_state = step_next_state[row]
            next_value = 0.0 if is_terminal_state(next_state) else values.get(next_state, 0.0)
            delta = float(rewards[row]) - step_cost + gamma * next_value - values.get(state, 0.0)
            deltas[row] = delta
            diagnostics["deltas"].append(delta)
            if state == next_state:
                diagnostics["self_loop_deltas"].append(delta)

        if use_gae:
            for rows in trajectories.values():
                gae = 0.0
                for row in reversed(rows):
                    gae = deltas[row] + gamma * lam * gae
                    step_advantage[row] = gae
        else:
            for row, delta in deltas.items():
                step_advantage[row] = delta

        diagnostics["values"].extend(
            value for state, value in values.items() if not is_terminal_state(state)
        )
        diagnostics["iterations"].append(iterations)
        diagnostics["difference"].append(difference)
        diagnostics["states"].append(len(edges))
        diagnostics["edges"].append(sum(len(state_edges) for state_edges in edges.values()))

        if gate_mode == "hard" and combine_mode == "add":
            gate, branch_count = compute_state_gate(
                step_state,
                step_next_state,
                uid_rows,
                min_state_count=gate_min_state_count,
                min_successors=gate_min_successors,
                action_valid=valid_for_graph,
            )
            group_weight = (
                _group_skew_eta(trajectory_rewards, trajectories)
                if gate_use_group_skew
                else 1.0
            )
            for row in uid_rows:
                row = int(row)
                gate_by_row[row] = gate[row]
                episode_weight[row] = group_weight * (
                    eta_min + (1.0 - eta_min) * (1.0 - gate[row])
                )
            diagnostics["branch_states"].append(branch_count)
            diagnostics["gate"].extend(gate.values())
            diagnostics["eta"].extend(episode_weight[uid_rows])

    if clip_advantage is not None:
        step_advantage = np.clip(
            step_advantage, -float(clip_advantage), float(clip_advantage)
        )
    if normalize_step:
        step_advantage = _normalize_per_uid(
            step_advantage, index, gigpo_mode, epsilon
        )

    step_tensor = torch.as_tensor(
        step_advantage, dtype=token_level_rewards.dtype, device=token_level_rewards.device
    ).unsqueeze(-1) * eos_mask
    if combine_mode == "step_only":
        scores = step_tensor
    elif gate_mode == "hard":
        gate_tensor = torch.as_tensor(
            gate_by_row, dtype=token_level_rewards.dtype, device=token_level_rewards.device
        ).unsqueeze(-1) * eos_mask
        weight_tensor = torch.as_tensor(
            episode_weight, dtype=token_level_rewards.dtype, device=token_level_rewards.device
        ).unsqueeze(-1) * eos_mask
        scores = weight_tensor * episode_advantage + step_advantage_w * gate_tensor * step_tensor
    else:
        scores = episode_advantage + step_advantage_w * step_tensor

    def mean(name):
        return float(np.mean(diagnostics[name])) if diagnostics[name] else 0.0

    metrics = {
        "bepo/gate_mode": 1.0 if gate_mode == "hard" else 0.0,
        "bepo/rho_mean": mean("gate"),
        "bepo/rho_one_ratio": mean("gate"),
        "bepo/eta_mean": mean("eta") if diagnostics["eta"] else 1.0,
        "bepo/num_branch_states_mean": mean("branch_states"),
        "bepo/mean_delta": mean("deltas"),
        "bepo/mean_step_adv": float(step_advantage.mean()) if batch_size else 0.0,
        "bepo/std_step_adv": float(step_advantage.std()) if batch_size else 0.0,
        "bepo/mean_value": mean("values"),
        "bepo/num_states_mean": mean("states"),
        "bepo/num_edges_mean": mean("edges"),
        "bepo/value_iter_steps_mean": mean("iterations"),
        "bepo/value_iter_max_diff_mean": mean("difference"),
        "bepo/self_loop_delta_mean": mean("self_loop_deltas"),
        "bepo/fallback_uid_ratio": fallback_groups / max(len(np.unique(index)), 1),
        "bepo/final_adv_abs_mean": float(scores[eos_mask.bool()].abs().mean())
        if eos_mask.any()
        else 0.0,
    }
    if return_components:
        if gate_mode != "hard" or combine_mode != "add":
            raise ValueError("Credit routing requires original hard-gated additive BEPO")
        return scores, scores, metrics, {
            "outcome": episode_advantage,
            "outcome_weight": weight_tensor,
            "graph_gate": gate_tensor,
            "fp_step": step_tensor,
        }
    return scores, scores, metrics
