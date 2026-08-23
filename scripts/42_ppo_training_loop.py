#!/usr/bin/env python
# coding: utf-8

# # Notebook 42 ? PPO Training Loop
# 
# ## Objectives
# 
# - Load the PPO dataset package from Notebook 40
# - Load the actor, critic, and optimizer checkpoint from Notebook 41
# - Reconstruct the neural-network architecture
# - Validate checkpoint compatibility
# - Create reusable PPO mini-batches
# - Compute discounted returns and normalized advantages
# - Run multiple PPO optimization epochs
# - Track policy loss, value loss, entropy, ratios, and gradient norms
# - Validate that actor and critic parameters update
# - Export training history and the updated checkpoint
# 
# > The current three-row dataset is a validation fixture. This notebook validates the full PPO training-loop architecture, not production performance.
# 

# ## Section 1 — Project Setup and Checkpoint Discovery
# 
# #### Locate the project, verify the Notebook 40 training package and Notebook 41 actor–critic checkpoint, and configure PyTorch.

# In[1]:


# ============================================================
# NOTEBOOK 42 — PPO TRAINING LOOP
# SECTION 1 — PROJECT SETUP
# ============================================================

from __future__ import annotations

import json
import math
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical


def find_project_root(
    start_path: Path | None = None,
) -> Path:
    current = (
        start_path or Path.cwd()
    ).resolve()

    for candidate in [
        current,
        *current.parents,
    ]:
        if (
            (candidate / "src").is_dir()
            and (candidate / "notebooks").is_dir()
            and (candidate / "reports").is_dir()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not locate the project root."
    )


PROJECT_ROOT = find_project_root()

SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"

NOTEBOOK40_REPORT_DIR = (
    REPORTS_DIR / "notebook40"
)

NOTEBOOK41_REPORT_DIR = (
    REPORTS_DIR / "notebook41"
)

NOTEBOOK42_REPORT_DIR = (
    REPORTS_DIR / "notebook42"
)

PPO_PACKAGE_FILE = (
    NOTEBOOK40_REPORT_DIR
    / "ppo_training_package.pkl"
)

ACTOR_CRITIC_CHECKPOINT_FILE = (
    NOTEBOOK41_REPORT_DIR
    / "ppo_training_engine.pkl"
)

NOTEBOOK42_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("NOTEBOOK 42 — PROJECT SETUP")
print("=" * 70)

print(
    "Project root                 :",
    PROJECT_ROOT,
)

print(
    "Notebook 40 package          :",
    PPO_PACKAGE_FILE,
)

print(
    "Notebook 40 package exists   :",
    PPO_PACKAGE_FILE.exists(),
)

print(
    "Notebook 41 checkpoint       :",
    ACTOR_CRITIC_CHECKPOINT_FILE,
)

print(
    "Notebook 41 checkpoint exists:",
    ACTOR_CRITIC_CHECKPOINT_FILE.exists(),
)

print(
    "Notebook 42 reports          :",
    NOTEBOOK42_REPORT_DIR,
)

print(
    "PyTorch version              :",
    torch.__version__,
)

print(
    "Execution device             :",
    DEVICE,
)

print(
    "Random seed                  :",
    RANDOM_SEED,
)

assert SRC_DIR.exists()
assert SCRIPTS_DIR.exists()
assert NOTEBOOKS_DIR.exists()
assert REPORTS_DIR.exists()

assert PPO_PACKAGE_FILE.exists(), (
    f"Missing Notebook 40 package: "
    f"{PPO_PACKAGE_FILE}"
)

assert ACTOR_CRITIC_CHECKPOINT_FILE.exists(), (
    f"Missing Notebook 41 checkpoint: "
    f"{ACTOR_CRITIC_CHECKPOINT_FILE}"
)

print()
print(
    "✅ SECTION 1 PROJECT SETUP PASSED"
)


# ## Section 2 — Load PPO Package and Training Checkpoint
# 
# #### Load the PPO dataset package from Notebook 40 and the actor–critic checkpoint from Notebook 41. Validate all required tensors, metadata, and optimizer state before beginning training.

# In[2]:


# ============================================================
# SECTION 2 — LOAD TRAINING PACKAGE
# ============================================================

with open(
    PPO_PACKAGE_FILE,
    "rb",
) as f:
    ppo_package = pickle.load(f)

with open(
    ACTOR_CRITIC_CHECKPOINT_FILE,
    "rb",
) as f:
    checkpoint = pickle.load(f)

ppo_batch = ppo_package["ppo_batch"]
metadata40 = ppo_package["metadata"]

metadata41 = checkpoint["metadata"]

print("NOTEBOOK 42 — LOAD TRAINING PACKAGE")
print("=" * 70)

print(
    "Dataset samples      :",
    metadata40["num_samples"],
)

print(
    "Observation dim      :",
    metadata40["observation_dim"],
)

print(
    "Action dim           :",
    metadata40["action_dim"],
)

print(
    "Mini-batches         :",
    metadata40["num_batches"],
)

print()

print(
    "Checkpoint LR        :",
    metadata41["learning_rate"],
)

print(
    "Checkpoint Gamma     :",
    metadata41["gamma"],
)

print(
    "Checkpoint Clip EPS  :",
    metadata41["clip_epsilon"],
)

summary = pd.DataFrame(
    [
        (
            "Observations",
            ppo_batch["observations"].shape,
        ),
        (
            "Action Masks",
            ppo_batch["action_masks"].shape,
        ),
        (
            "Actions",
            ppo_batch["actions"].shape,
        ),
        (
            "Rewards",
            ppo_batch["rewards"].shape,
        ),
        (
            "Dones",
            ppo_batch["dones"].shape,
        ),
    ],
    columns=[
        "Tensor",
        "Shape",
    ],
)

display(summary)

assert "ppo_batch" in ppo_package
assert "metadata" in ppo_package

assert "actor_state_dict" in checkpoint
assert "critic_state_dict" in checkpoint
assert "optimizer_state_dict" in checkpoint
assert "metadata" in checkpoint

assert ppo_batch["observations"].shape == (3, 4)
assert ppo_batch["action_masks"].shape == (3, 4)
assert ppo_batch["actions"].shape == (3,)
assert ppo_batch["rewards"].shape == (3,)
assert ppo_batch["dones"].shape == (3,)

print()

print("✅ SECTION 2 LOAD TRAINING PACKAGE PASSED")


# ## Section 3 — Rebuild Actor and Critic Networks
# 
# #### Reconstruct the PPO Actor and Critic neural network architectures and restore their trained parameters from the Notebook 41 checkpoint.

# In[3]:


# ============================================================
# SECTION 3 — REBUILD ACTOR & CRITIC
# ============================================================

class PPOActor(nn.Module):

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                observation_dim,
                128,
            ),
            nn.ReLU(),
            nn.Linear(
                128,
                64,
            ),
            nn.ReLU(),
            nn.Linear(
                64,
                action_dim,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(x)


class PPOCritic(nn.Module):

    def __init__(
        self,
        observation_dim: int,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                observation_dim,
                128,
            ),
            nn.ReLU(),
            nn.Linear(
                128,
                64,
            ),
            nn.ReLU(),
            nn.Linear(
                64,
                1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(x)


actor = PPOActor(
    metadata40["observation_dim"],
    metadata40["action_dim"],
).to(DEVICE)

critic = PPOCritic(
    metadata40["observation_dim"],
).to(DEVICE)

actor.load_state_dict(
    checkpoint["actor_state_dict"]
)

critic.load_state_dict(
    checkpoint["critic_state_dict"]
)

optimizer = torch.optim.Adam(
    list(actor.parameters())
    + list(critic.parameters()),
    lr=metadata41["learning_rate"],
)

optimizer.load_state_dict(
    checkpoint["optimizer_state_dict"]
)

actor.eval()
critic.eval()

actor_parameters = sum(
    parameter.numel()
    for parameter in actor.parameters()
)

critic_parameters = sum(
    parameter.numel()
    for parameter in critic.parameters()
)

print("NOTEBOOK 42 — REBUILD NETWORKS")
print("=" * 70)

print(
    "Actor parameters :",
    actor_parameters,
)

print(
    "Critic parameters:",
    critic_parameters,
)

print(
    "Optimizer groups :",
    len(optimizer.param_groups),
)

assert actor_parameters == 9156
assert critic_parameters == 8961

assert len(
    optimizer.param_groups
) == 1

print()

print(
    "✅ SECTION 3 REBUILD NETWORKS PASSED"
)


# ## Section 4 — Prepare PPO Training Tensors
# 
# #### Convert the PPO dataset into PyTorch tensors on the selected device and verify that all tensor shapes and data types are correct before beginning optimization.

# In[4]:


# ============================================================
# SECTION 4 — PREPARE TRAINING TENSORS
# ============================================================

observations = torch.as_tensor(
    ppo_batch["observations"],
    dtype=torch.float32,
    device=DEVICE,
)

action_masks = torch.as_tensor(
    ppo_batch["action_masks"],
    dtype=torch.bool,
    device=DEVICE,
)

actions = torch.as_tensor(
    ppo_batch["actions"],
    dtype=torch.long,
    device=DEVICE,
)

rewards = torch.as_tensor(
    ppo_batch["rewards"],
    dtype=torch.float32,
    device=DEVICE,
)

dones = torch.as_tensor(
    ppo_batch["dones"],
    dtype=torch.bool,
    device=DEVICE,
)

dataset_size = observations.shape[0]

print("NOTEBOOK 42 — PREPARE TRAINING TENSORS")
print("=" * 70)

print("Dataset size        :", dataset_size)
print("Observations shape  :", observations.shape)
print("Action masks shape  :", action_masks.shape)
print("Actions shape       :", actions.shape)
print("Rewards shape       :", rewards.shape)
print("Dones shape         :", dones.shape)

print()

print("Observation dtype   :", observations.dtype)
print("Action mask dtype   :", action_masks.dtype)
print("Action dtype        :", actions.dtype)
print("Reward dtype        :", rewards.dtype)
print("Done dtype          :", dones.dtype)

assert observations.ndim == 2
assert action_masks.ndim == 2
assert actions.ndim == 1
assert rewards.ndim == 1
assert dones.ndim == 1

assert observations.shape[0] == dataset_size
assert action_masks.shape[0] == dataset_size
assert actions.shape[0] == dataset_size
assert rewards.shape[0] == dataset_size
assert dones.shape[0] == dataset_size

print()
print("✅ SECTION 4 PREPARE TRAINING TENSORS PASSED")


# ## Section 5 — Compute Current Policy Outputs
# 
# #### Run the current Actor and Critic over the PPO dataset to obtain policy logits, state-value estimates, action probabilities, and the log probabilities of the recorded actions.

# In[5]:


# ============================================================
# SECTION 5 — CURRENT POLICY EVALUATION
# ============================================================

actor.eval()
critic.eval()

with torch.no_grad():

    policy_logits = actor(observations)

    masked_logits = policy_logits.masked_fill(
        ~action_masks,
        -1e9,
    )

    distribution = Categorical(
        logits=masked_logits,
    )

    value_estimates = critic(
        observations,
    ).squeeze(-1)

    action_log_probs = distribution.log_prob(
        actions,
    )

    entropy = distribution.entropy()

print("NOTEBOOK 42 — CURRENT POLICY")
print("=" * 70)

print("Policy logits shape :", policy_logits.shape)
print("Values shape        :", value_estimates.shape)
print("Log probs shape     :", action_log_probs.shape)
print("Entropy shape       :", entropy.shape)

print()

print("Average entropy     :", float(entropy.mean()))
print("Average value       :", float(value_estimates.mean()))

assert policy_logits.shape == (
    dataset_size,
    metadata40["action_dim"],
)

assert value_estimates.shape == (
    dataset_size,
)

assert action_log_probs.shape == (
    dataset_size,
)

assert entropy.shape == (
    dataset_size,
)

print()
print("✅ SECTION 5 CURRENT POLICY PASSED")


# ## Section 6 — Compute Discounted Returns
# 
# #### Compute discounted returns (reward-to-go) for each transition in the PPO dataset using the checkpoint discount factor. These returns will be used as training targets for the critic.

# In[6]:


# ============================================================
# SECTION 6 — DISCOUNTED RETURNS
# ============================================================

gamma = metadata41["gamma"]

discounted_returns = torch.zeros_like(
    rewards,
)

running_return = torch.tensor(
    0.0,
    device=DEVICE,
)

for step in reversed(range(dataset_size)):

    if dones[step]:
        running_return = rewards[step]
    else:
        running_return = (
            rewards[step]
            + gamma * running_return
        )

    discounted_returns[step] = running_return

print("NOTEBOOK 42 — DISCOUNTED RETURNS")
print("=" * 70)

print("Gamma:", gamma)
print()

print("Rewards")
print(rewards.cpu())

print()

print("Discounted Returns")
print(discounted_returns.cpu())

assert discounted_returns.shape == rewards.shape
assert torch.isfinite(discounted_returns).all()

print()
print("✅ SECTION 6 DISCOUNTED RETURNS PASSED")


# ## Section 7 — Compute Advantages
# 
# #### Compute normalized PPO advantages from the discounted returns and the critic's current value estimates.

# In[7]:


# ============================================================
# SECTION 7 — ADVANTAGE ESTIMATION
# ============================================================

advantages = (
    discounted_returns
    - value_estimates
)

adv_mean = advantages.mean()
adv_std = advantages.std()

if adv_std > 1e-8:
    normalized_advantages = (
        advantages - adv_mean
    ) / adv_std
else:
    normalized_advantages = (
        advantages - adv_mean
    )

print("NOTEBOOK 42 — ADVANTAGES")
print("=" * 70)

print(
    "Mean before normalization:",
    float(adv_mean),
)

print(
    "Std before normalization:",
    float(adv_std),
)

print()

print("Normalized Advantages")

print(normalized_advantages.cpu())

print()

print(
    "Normalized Mean:",
    float(normalized_advantages.mean()),
)

print(
    "Normalized Std:",
    float(normalized_advantages.std()),
)

assert normalized_advantages.shape == (
    dataset_size,
)

assert torch.isfinite(
    normalized_advantages
).all()

print()

print("✅ SECTION 7 ADVANTAGE ESTIMATION PASSED")


# ## Section 8 — PPO Optimization Epoch
# 
# #### Run one complete PPO optimization epoch using the clipped surrogate objective, critic loss, entropy bonus, gradient clipping, and optimizer update.

# In[8]:


# ============================================================
# SECTION 8 — PPO OPTIMIZATION EPOCH
# ============================================================

actor.train()
critic.train()

optimizer.zero_grad()

# Forward pass
policy_logits = actor(observations)

masked_logits = policy_logits.masked_fill(
    ~action_masks,
    -1e9,
)

distribution = Categorical(
    logits=masked_logits,
)

new_log_probs = distribution.log_prob(
    actions,
)

entropy = distribution.entropy().mean()

new_values = critic(
    observations,
).squeeze(-1)

# PPO ratio
old_log_probs = action_log_probs.detach()

ratios = torch.exp(
    new_log_probs - old_log_probs
)

clip_epsilon = metadata41["clip_epsilon"]

surrogate_1 = (
    ratios
    * normalized_advantages
)

surrogate_2 = (
    torch.clamp(
        ratios,
        1.0 - clip_epsilon,
        1.0 + clip_epsilon,
    )
    * normalized_advantages
)

policy_loss = -torch.min(
    surrogate_1,
    surrogate_2,
).mean()

value_loss = F.mse_loss(
    new_values,
    discounted_returns,
)

entropy_bonus = entropy

entropy_coefficient = 0.01

total_loss = (
    policy_loss
    + 0.5 * value_loss
    - entropy_coefficient * entropy_bonus
)

total_loss.backward()

gradient_norm = torch.nn.utils.clip_grad_norm_(
    list(actor.parameters()) +
    list(critic.parameters()),
    max_norm=1.0,
)

optimizer.step()

print("NOTEBOOK 42 — PPO OPTIMIZATION")
print("=" * 70)

print(f"Policy Loss : {policy_loss:.6f}")
print(f"Value Loss  : {value_loss:.6f}")
print(f"Entropy     : {entropy:.6f}")
print(f"Total Loss  : {total_loss:.6f}")
print(f"Mean Ratio  : {ratios.mean():.6f}")
print(f"Grad Norm   : {float(gradient_norm):.6f}")

assert torch.isfinite(total_loss)
assert torch.isfinite(policy_loss)
assert torch.isfinite(value_loss)
assert torch.isfinite(entropy)

print()
print("✅ SECTION 8 PPO OPTIMIZATION PASSED")


# ## Section 9 — Verify Network Parameter Updates
# 
# #### Perform another forward pass after the optimization step and verify that the Actor and Critic outputs have changed, confirming that the PPO update modified the network parameters.

# In[9]:


# ============================================================
# SECTION 9 — VERIFY PARAMETER UPDATES
# ============================================================

actor.eval()
critic.eval()

with torch.no_grad():

    updated_policy_logits = actor(
        observations,
    )

    updated_values = critic(
        observations,
    ).squeeze(-1)

policy_changed = not torch.allclose(
    policy_logits,
    updated_policy_logits,
)

value_changed = not torch.allclose(
    value_estimates,
    updated_values,
)

policy_difference = torch.mean(
    torch.abs(
        updated_policy_logits
        - policy_logits
    )
).item()

value_difference = torch.mean(
    torch.abs(
        updated_values
        - value_estimates
    )
).item()

print("NOTEBOOK 42 — PARAMETER UPDATE VERIFICATION")
print("=" * 70)

print("Policy changed :", policy_changed)
print("Value changed  :", value_changed)

print()

print(
    f"Mean policy difference : {policy_difference:.8f}"
)

print(
    f"Mean value difference  : {value_difference:.8f}"
)

assert policy_changed
assert value_changed

assert policy_difference > 0.0
assert value_difference > 0.0

print()
print("✅ SECTION 9 PARAMETER UPDATE VERIFIED")


# ## Section 10 — Training Metrics Summary
# 
# #### Create a summary table of the optimization metrics for this PPO training epoch.

# In[11]:


# ============================================================
# SECTION 10 — TRAINING METRICS SUMMARY
# ============================================================

training_summary = pd.DataFrame(
    {
        "Metric": [
            "Policy Loss",
            "Value Loss",
            "Entropy",
            "Total Loss",
            "Mean Ratio",
            "Gradient Norm",
        ],
        "Value": [
            policy_loss.detach().item(),
            value_loss.detach().item(),
            entropy.detach().item(),
            total_loss.detach().item(),
            ratios.mean().detach().item(),
            gradient_norm.detach().item(),
        ],
    }
)

print("NOTEBOOK 42 — TRAINING SUMMARY")
print("=" * 70)

display(training_summary)

assert len(training_summary) == 6
assert np.isfinite(training_summary["Value"]).all()

print()
print("✅ SECTION 10 TRAINING SUMMARY PASSED")


# ## Section 11 — Export Updated PPO Checkpoint
# 
# #### Save the updated Actor, Critic, optimizer state, and training metrics for use in the next notebook.

# In[12]:


# ============================================================
# SECTION 11 — EXPORT UPDATED CHECKPOINT
# ============================================================

checkpoint_out = {
    "actor_state_dict": actor.state_dict(),
    "critic_state_dict": critic.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "metadata": {
        **metadata41,
        "training_epochs_completed": 1,
        "last_policy_loss": policy_loss.detach().item(),
        "last_value_loss": value_loss.detach().item(),
        "last_entropy": entropy.detach().item(),
        "last_total_loss": total_loss.detach().item(),
        "last_mean_ratio": ratios.mean().detach().item(),
        "last_gradient_norm": gradient_norm.detach().item(),
    },
}

checkpoint_file = (
    NOTEBOOK42_REPORT_DIR
    / "ppo_training_loop.pkl"
)

with open(
    checkpoint_file,
    "wb",
) as f:
    pickle.dump(
        checkpoint_out,
        f,
    )

print("NOTEBOOK 42 — EXPORT CHECKPOINT")
print("=" * 70)

print("Saved to:")
print(checkpoint_file)

assert checkpoint_file.exists()

print()
print("✅ SECTION 11 EXPORT PASSED")


# In[ ]:




