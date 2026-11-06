"""Raw, current-policy empirical graph values for a boundary-Critic pilot."""
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from gigpo.core_bepo import (
    _build_empirical_mdp, _group_rows_by_traj, _init_values,
    _run_value_iteration, get_state_key, is_terminal_state,
)


@torch.no_grad()
def attach_graph_teacher(data, gamma, cfg):
    rewards = data.batch["token_level_rewards"].float().sum(-1).cpu().numpy()
    mask = data.batch["value_mask"].bool()
    size = len(rewards)
    uid = data.non_tensor_batch["uid"]
    traj = data.non_tensor_batch["traj_uid"]
    steps = data.non_tensor_batch["step_idx"]
    anchor = data.non_tensor_batch["anchor_obs"]
    terminal = data.non_tensor_batch.get("terminal_kind")
    targets = np.zeros(size, dtype=np.float32)
    mc_targets = np.zeros(size, dtype=np.float32)
    confidence = np.zeros(size, dtype=np.float32)
    unique = np.zeros(size, dtype=bool)
    groups = defaultdict(list)
    representatives = {}
    for row in range(size):
        key = (str(uid[row]), str(traj[row]), int(np.asarray(steps[row]).reshape(-1)[0]))
        if key in representatives:
            reference = representatives[key]
            if not np.isclose(rewards[row], rewards[reference], atol=1e-5):
                raise ValueError("Padding copy changed transition reward")
            if get_state_key(anchor[row]) != get_state_key(anchor[reference]):
                raise ValueError("Padding copy changed anchor")
            continue
        representatives[key] = row
        unique[row] = True
        groups[str(uid[row])].append(row)
    residuals, iterations = [], []
    for rows in groups.values():
        trajectories = _group_rows_by_traj(rows, traj, steps)
        for ordered in trajectories.values():
            ids = [int(np.asarray(steps[r]).reshape(-1)[0]) for r in ordered]
            if ids != list(range(len(ids))):
                raise ValueError(f"Teacher trajectory has missing steps: {ids}")
        edges, states, successors = _build_empirical_mdp(
            trajectories, anchor, rewards, 0.0, terminal_kind=terminal)
        mc = _init_values(edges, states, trajectories, rewards, "mc", gamma, 0.0)
        fp, count, _ = _run_value_iteration(
            edges, mc, gamma, int(cfg.get("graph_max_iter", 200)),
            float(cfg.get("graph_tol", 1e-6)))
        residual = max((abs(fp[s] - np.mean([
            r + gamma * (0.0 if is_terminal_state(n) else fp.get(n, 0.0))
            for r, n in outgoing])) for s, outgoing in edges.items()), default=0.0)
        if not all(np.isfinite(v) for v in fp.values()):
            raise ValueError("Nonfinite graph value")
        residuals.append(residual)
        iterations.append(count)
        distinct_trajs, branches = defaultdict(set), defaultdict(set)
        for row in rows:
            distinct_trajs[states[row]].add(str(traj[row]))
            branches[states[row]].add(successors[row])
        for row in rows:
            s = states[row]
            targets[row], mc_targets[row] = fp[s], mc[s]
            confidence[row] = float(
                len(distinct_trajs[s]) >= int(cfg.get("graph_min_trajectories", 2))
                and len(branches[s]) >= int(cfg.get("graph_min_successors", 2))
                and residual <= float(cfg.get("graph_max_residual", 1e-4)))
    boundary = mask.to(torch.float32)
    for name, array in [("graph_value_targets", targets), ("graph_mc_targets", mc_targets),
                        ("graph_confidence", confidence)]:
        vector = torch.as_tensor(array, device=boundary.device)
        data.batch[name] = vector.unsqueeze(-1) * boundary
    rows = torch.arange(size, device=boundary.device)
    positions = mask.long().argmax(-1)
    pred = data.batch["values"][rows, positions].float().detach().cpu().numpy()
    supported = confidence > 0
    n_support = int(supported.sum())
    delta = np.abs(targets - mc_targets)
    metrics = {
        "graph_teacher/unique_turns": int(unique.sum()),
        "graph_teacher/padding_copies": int((~unique).sum()),
        "graph_teacher/support_fraction": n_support / max(int(unique.sum()), 1),
        "graph_teacher/fp_mc_mae": float(delta[unique].mean()) if unique.any() else 0.0,
        "graph_teacher/fp_mc_mae_support": float(delta[supported].mean()) if n_support else 0.0,
        "graph_teacher/fp_mc_changed_fraction_support": float((delta[supported] > 1e-3).mean()) if n_support else 0.0,
        "graph_teacher/old_critic_fp_mse_support": float(np.square(pred[supported] - targets[supported]).mean()) if n_support else 0.0,
        "graph_teacher/old_critic_mc_mse_support": float(np.square(pred[supported] - mc_targets[supported]).mean()) if n_support else 0.0,
        "graph_teacher/bellman_residual_max": max(residuals, default=0.0),
        "graph_teacher/iterations_mean": float(np.mean(iterations)) if iterations else 0.0,
    }
    data.meta_info["graph_teacher_metrics"] = metrics
    # A label-only matched control. Actor BEPO credit, Critic PPO targets,
    # confidence/support mask, normalization and sample count do not change.
    teacher = cfg.get("graph_aux_teacher", "fp")
    if teacher not in ("fp", "mc"):
        raise ValueError("Unknown fixed auxiliary teacher")
    if teacher == "mc":
        data.batch["graph_value_targets"] = data.batch["graph_mc_targets"].clone()
    metrics["graph_teacher/auxiliary_teacher_is_mc"] = float(teacher == "mc")


def graph_value_aux_loss(vpreds, targets, confidence, value_mask):
    """E_boundary[c(s)(V-target)^2], including unsupported states in denominator."""
    if not (vpreds.shape == targets.shape == confidence.shape == value_mask.shape):
        raise ValueError("Graph auxiliary tensors must share shape")
    weights = value_mask.float() * confidence.detach().float()
    errors = vpreds.float() - targets.detach().float()
    return (errors.square() * weights).sum() / value_mask.float().sum().clamp_min(1.0)


def save_teacher_snapshot(data, path):
    """Small boundary-only fixture. Training-target agreement is not held-out accuracy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = data.batch["value_mask"].bool()
    rows = torch.arange(mask.shape[0], device=mask.device)
    positions = mask.long().argmax(-1)
    arrays = {key: data.batch[key][rows, positions].detach().float().cpu().numpy()
              for key in ["values", "returns", "advantages", "graph_value_targets",
                          "graph_mc_targets", "graph_confidence"]}
    arrays["reward"] = data.batch["token_level_rewards"].float().sum(-1).cpu().numpy()
    for key in ["uid", "traj_uid", "step_idx"]:
        arrays[key] = np.asarray(data.non_tensor_batch[key]).astype(str)
    np.savez_compressed(path, **arrays)
