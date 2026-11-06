"""Small portable interface over the experiment-matched algorithm kernels."""
from dataclasses import dataclass, field
import math
from typing import Any

import torch

from gigpo.bepo_neural_credit import attach_bepo_neural_credit
from gigpo.graph_value_teacher import graph_value_aux_loss
from gigpo.ppo import compute_value_loss


@dataclass
class Batch:
    """Minimal DataProto-compatible container. Rows represent environment turns."""
    batch: dict[str, torch.Tensor]
    non_tensor_batch: dict[str, Any]
    meta_info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StandardConfig:
    """Fixed coefficients, not learned gates. Defaults follow the standard recipe."""
    q: float = .25
    beta: float = .1
    gamma: float = .95
    lam: float = .8
    step_weight: float = 1.5
    cliprange_value: float = .5

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.q,self.beta,self.gamma,self.lam,
                                              self.step_weight,self.cliprange_value)):
            raise ValueError("Configuration coefficients must be finite")
        if not 0 <= self.q <= 1 or not 0 <= self.gamma <= 1 or not 0 <= self.lam <= 1:
            raise ValueError("q, gamma and lambda must lie in [0, 1]")
        if self.beta < 0 or self.step_weight < 0 or self.cliprange_value < 0:
            raise ValueError("Loss weights and value clip must be nonnegative")

    def bepo(self):
        return dict(gamma=self.gamma,**{"lambda":self.lam},use_gae=True,
            max_iter=20,tol=1e-6,init="mc",step_cost=0.,combine_mode="add",
            normalize_step=True,clip_advantage=None,min_traj_per_uid=1,
            fallback="gigpo",exclude_invalid_edges=False,gate_mode="hard",
            eta_min=.5,gate_min_state_count=2,gate_min_successors=2,
            gate_use_group_skew=False,neural_credit_q=self.q)

    def teacher(self):
        return dict(graph_aux_enabled=True,graph_aux_teacher="fp",graph_max_iter=200,
            graph_tol=1e-6,graph_max_residual=1e-4,
            graph_min_trajectories=2,graph_min_successors=2)


def compute_credit(data, config: StandardConfig | None = None):
    """Attach detached actor credit, PPO critic targets and FP teacher targets.

    Accepts Batch or an existing DataProto-compatible object. Does not construct
    an actor/critic network, sample actions, change a policy, or run optimizers.
    """
    config = config or StandardConfig()
    for name in ("response_mask", "token_level_rewards", "values"):
        if name not in data.batch:
            raise ValueError(f"Missing tensor: {name}")
    mask = data.batch["response_mask"]
    if mask.ndim != 2 or mask.shape != data.batch["values"].shape or mask.shape != data.batch["token_level_rewards"].shape:
        raise ValueError("Mask, rollout values and token rewards must share [N, L] shape")
    if not bool(((mask == 0) | (mask == 1)).all()) or not bool(mask.bool().any(-1).all()):
        raise ValueError("Each turn needs at least one valid token in a binary mask")
    if not torch.isfinite(data.batch["values"]).all() or not torch.isfinite(data.batch["token_level_rewards"]).all():
        raise ValueError("Values and rewards must be finite")
    for name in ("uid", "traj_uid", "step_idx", "anchor_obs", "rewards"):
        if name not in data.non_tensor_batch or len(data.non_tensor_batch[name]) != len(mask):
            raise ValueError(f"Missing or wrong-length turn metadata: {name}")
    return attach_bepo_neural_credit(data,config.bepo(),config.teacher(),
        gamma=config.gamma,lam=config.lam,step_advantage_w=config.step_weight,
        gigpo_mode="mean_std_norm")


def critic_loss(new_values: torch.Tensor, data, config: StandardConfig | None = None):
    """Native clipped PPO value loss + beta * graph point-supervision loss.

    ``new_values`` must come from the trainable critic after rollout. Actor credit
    and all targets are detached; this function returns a differentiable scalar.
    The native base value loss has no added 1/2 factor.
    """
    config = config or StandardConfig()
    b = data.batch
    if new_values.shape != b["values"].shape:
        raise ValueError("New and rollout critic values must share shape")
    base, _ = compute_value_loss(new_values,b["returns"].detach(),b["values"].detach(),
        b["value_mask"],config.cliprange_value,loss_agg_mode="token-mean")
    auxiliary = graph_value_aux_loss(new_values,b["graph_value_targets"],
        b["graph_confidence"],b["value_mask"])
    total = base + config.beta * auxiliary
    return total, {"value_loss":base.detach(),"graph_aux_loss":auxiliary.detach(),
                   "weighted_graph_aux_loss":(config.beta*auxiliary).detach()}
