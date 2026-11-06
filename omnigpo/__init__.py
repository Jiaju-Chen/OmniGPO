"""OmniGPO: graph-assisted, fixed-q neural credit and critic supervision."""
from .api import Batch, StandardConfig, compute_credit, critic_loss
from gigpo.core_bepo import compute_bepo_advantage
from gigpo.ppo import compute_policy_loss, compute_turn_ppo_gae

__all__ = ["Batch", "StandardConfig", "compute_credit", "critic_loss",
           "compute_bepo_advantage", "compute_policy_loss", "compute_turn_ppo_gae"]
__version__ = "0.1.0"
