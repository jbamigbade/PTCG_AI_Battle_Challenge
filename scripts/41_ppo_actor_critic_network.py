#!/usr/bin/env python
# coding: utf-8

# # Notebook 41 ? PPO Actor?Critic Network
# 
# ## Objectives
# 
# - Load the PPO training package exported by Notebook 40
# - Verify the observation and action dimensions
# - Confirm PyTorch availability and execution device
# - Build a masked PPO actor network
# - Build a PPO critic/value network
# - Combine both networks into an actor?critic model
# - Validate policy logits, masked probabilities, and state values
# - Test deterministic and sampled action selection
# - Count trainable parameters
# - Save and reload an initialized model checkpoint
# - Export the architecture specification and validation reports
# 
# > The current three-row dataset is a validation fixture. This notebook validates the model architecture and interfaces; it does not claim production training performance.
# 

# ## Section 1 — Project Setup and PyTorch Check
# 
# #### Locate the project, load the Notebook 40 package path, configure deterministic behavior, and verify that PyTorch is available.

# In[1]:


# ============================================================
# NOTEBOOK 41 — PPO ACTOR–CRITIC NETWORK
# SECTION 1 — PROJECT SETUP AND PYTORCH CHECK
# ============================================================

from __future__ import annotations

import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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

PPO_PACKAGE_FILE = (
    NOTEBOOK40_REPORT_DIR
    / "ppo_training_package.pkl"
)

NOTEBOOK41_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

RANDOM_SEED = 41

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True

except ImportError as error:
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = str(error)

print("NOTEBOOK 41 — PROJECT SETUP")
print("=" * 70)

print(
    "Project root           :",
    PROJECT_ROOT,
)

print(
    "Notebook 40 reports    :",
    NOTEBOOK40_REPORT_DIR,
)

print(
    "Notebook 41 reports    :",
    NOTEBOOK41_REPORT_DIR,
)

print(
    "PPO package            :",
    PPO_PACKAGE_FILE,
)

print(
    "PPO package exists     :",
    PPO_PACKAGE_FILE.exists(),
)

print(
    "PyTorch available      :",
    TORCH_AVAILABLE,
)

if TORCH_AVAILABLE:
    torch.manual_seed(
        RANDOM_SEED
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            RANDOM_SEED
        )

    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "PyTorch version        :",
        torch.__version__,
    )

    print(
        "Execution device       :",
        DEVICE,
    )

else:
    DEVICE = None

    print(
        "PyTorch import error   :",
        TORCH_IMPORT_ERROR,
    )

print(
    "Random seed            :",
    RANDOM_SEED,
)

assert SRC_DIR.exists()
assert SCRIPTS_DIR.exists()
assert NOTEBOOKS_DIR.exists()
assert REPORTS_DIR.exists()
assert NOTEBOOK40_REPORT_DIR.exists()
assert PPO_PACKAGE_FILE.exists()

assert TORCH_AVAILABLE, (
    "PyTorch is required for Notebook 41. "
    "Install it before continuing."
)

print()
print(
    "✅ SECTION 1 PROJECT SETUP PASSED"
)


# ## Section 2 — Load PPO Training Package
# 
# #### Load the PPO package exported by Notebook 40 and verify every tensor before constructing the neural network.

# In[2]:


# ============================================================
# SECTION 2 — LOAD PPO TRAINING PACKAGE
# ============================================================

import pickle

with open(
    PPO_PACKAGE_FILE,
    "rb",
) as f:
    package = pickle.load(f)

ppo_batch = package["ppo_batch"]

observations = ppo_batch["observations"]
action_masks = ppo_batch["action_masks"]
actions = ppo_batch["actions"]
rewards = ppo_batch["rewards"]
dones = ppo_batch["dones"]

metadata = package["metadata"]

print("NOTEBOOK 41 — LOAD PPO PACKAGE")
print("=" * 70)

print("Samples              :", observations.shape[0])
print("Observation shape    :", observations.shape)
print("Mask shape           :", action_masks.shape)
print("Actions shape        :", actions.shape)
print("Rewards shape        :", rewards.shape)
print("Done shape           :", dones.shape)

print()

display(
    pd.DataFrame(
        {
            "Metadata": metadata.keys(),
            "Value": metadata.values(),
        }
    )
)

assert observations.shape == (3, 4)
assert action_masks.shape == (3, 4)
assert actions.shape == (3,)
assert rewards.shape == (3,)
assert dones.shape == (3,)

print()
print("✅ SECTION 2 LOAD PPO PACKAGE PASSED")


# ## Section 3 — PPO Actor Network
# 
# #### Build the policy (actor) neural network that predicts action probabilities from observation vectors.

# In[3]:


# ============================================================
# SECTION 3 — PPO ACTOR NETWORK
# ============================================================

import torch.nn as nn


class PPOActor(nn.Module):

    def __init__(
        self,
        observation_dim,
        action_dim,
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
        x,
    ):
        return self.network(x)


actor = PPOActor(

    observation_dim=metadata["observation_dim"],

    action_dim=metadata["action_dim"],

)

print("NOTEBOOK 41 — PPO ACTOR")

print("=" * 70)

print(actor)

parameter_count = sum(
    p.numel()
    for p in actor.parameters()
)

print()

print(
    "Trainable parameters :",
    parameter_count,
)

assert isinstance(
    actor,
    nn.Module,
)

assert parameter_count > 0

print()

print("✅ SECTION 3 PPO ACTOR PASSED")


# ## Section 4 — PPO Critic Network
# 
# #### Build the value (critic) neural network that estimates the value of a Pokémon battle state.

# In[4]:


# ============================================================
# SECTION 4 — PPO CRITIC NETWORK
# ============================================================

import torch.nn as nn


class PPOCritic(nn.Module):

    def __init__(
        self,
        observation_dim,
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
        x,
    ):
        return self.network(x)


critic = PPOCritic(

    observation_dim=metadata["observation_dim"],

)

print("NOTEBOOK 41 — PPO CRITIC")

print("=" * 70)

print(critic)

parameter_count = sum(
    p.numel()
    for p in critic.parameters()
)

print()

print(
    "Trainable parameters :",
    parameter_count,
)

assert isinstance(
    critic,
    nn.Module,
)

assert parameter_count > 0

print()

print("✅ SECTION 4 PPO CRITIC PASSED")


# ## Section 5 — Forward Pass
# 
# #### Run a forward pass through the actor and critic using the observation tensors exported from Notebook 40.

# In[5]:


# ============================================================
# SECTION 5 — FORWARD PASS
# ============================================================

import torch

# Convert NumPy observations to PyTorch tensor
observation_batch = torch.tensor(
    observations,
    dtype=torch.float32,
)

# Forward pass
actor_logits = actor(observation_batch)

critic_values = critic(observation_batch)

print("NOTEBOOK 41 — FORWARD PASS")
print("=" * 70)

print("Observation batch :", observation_batch.shape)
print("Actor output      :", actor_logits.shape)
print("Critic output     :", critic_values.shape)

print()

print("Actor logits")
display(
    pd.DataFrame(
        actor_logits.detach().numpy()
    )
)

print()

print("Critic values")
display(
    pd.DataFrame(
        critic_values.detach().numpy(),
        columns=["Value"],
    )
)

assert actor_logits.shape == (3, 4)

assert critic_values.shape == (3, 1)

assert torch.isfinite(actor_logits).all()

assert torch.isfinite(critic_values).all()

print()

print("✅ SECTION 5 FORWARD PASS PASSED")


# ## Section 6 — Policy Probabilities
# 
# #### Convert actor logits into action probabilities using the Softmax function.

# In[6]:


# ============================================================
# SECTION 6 — POLICY PROBABILITIES
# ============================================================

import torch.nn.functional as F

policy_probs = F.softmax(
    actor_logits,
    dim=1,
)

print("NOTEBOOK 41 — POLICY PROBABILITIES")
print("=" * 70)

print(
    "Probability tensor :",
    policy_probs.shape,
)

print()

display(
    pd.DataFrame(
        policy_probs.detach().numpy(),
        columns=[
            f"Action {i}"
            for i in range(policy_probs.shape[1])
        ],
    )
)

row_sums = policy_probs.sum(
    dim=1,
)

print()

print(
    "Probability sums"
)

display(
    pd.DataFrame(
        row_sums.detach().numpy(),
        columns=["Sum"],
    )
)

assert policy_probs.shape == (3, 4)

assert torch.allclose(
    row_sums,
    torch.ones_like(row_sums),
)

assert torch.all(policy_probs >= 0)

assert torch.all(policy_probs <= 1)

print()

print("✅ SECTION 6 POLICY PROBABILITIES PASSED")


# ## Section 7 — Apply Legal Action Masks
# 
# #### Mask illegal actions before Softmax so the policy assigns zero probability to moves that are unavailable in each game state.

# In[7]:


# ============================================================
# SECTION 7 — MASKED POLICY PROBABILITIES
# ============================================================

action_mask_batch = torch.tensor(
    action_masks,
    dtype=torch.float32,
    device=actor_logits.device,
)

masked_logits = actor_logits.masked_fill(
    action_mask_batch == 0,
    float("-inf"),
)

masked_policy_probs = F.softmax(
    masked_logits,
    dim=1,
)

masked_row_sums = masked_policy_probs.sum(
    dim=1,
)

illegal_probabilities = masked_policy_probs[
    action_mask_batch == 0
]

print("NOTEBOOK 41 — MASKED POLICY")
print("=" * 70)

print(
    "Logit shape           :",
    masked_logits.shape,
)

print(
    "Probability shape     :",
    masked_policy_probs.shape,
)

print(
    "Illegal action count  :",
    int(
        (action_mask_batch == 0)
        .sum()
        .item()
    ),
)

print()

display(
    pd.DataFrame(
        masked_policy_probs
        .detach()
        .cpu()
        .numpy(),
        columns=[
            f"Action {index}"
            for index in range(
                masked_policy_probs.shape[1]
            )
        ],
    )
)

print()
print("Probability sums")

display(
    pd.DataFrame(
        masked_row_sums
        .detach()
        .cpu()
        .numpy(),
        columns=["Sum"],
    )
)

assert masked_policy_probs.shape == (
    3,
    4,
)

assert torch.allclose(
    masked_row_sums,
    torch.ones_like(
        masked_row_sums
    ),
)

assert torch.all(
    illegal_probabilities == 0
)

assert torch.isfinite(
    masked_policy_probs
).all()

print()
print(
    "✅ SECTION 7 MASKED POLICY PASSED"
)


# ## Section 8 — PPO Action Distribution
# 
# #### Create a categorical policy distribution from the masked action probabilities for sampling and policy optimization.

# In[8]:


# ============================================================
# SECTION 8 — PPO ACTION DISTRIBUTION
# ============================================================

from torch.distributions import Categorical

policy_distribution = Categorical(
    probs=masked_policy_probs,
)

sampled_actions = policy_distribution.sample()

log_probabilities = policy_distribution.log_prob(
    sampled_actions,
)

entropy = policy_distribution.entropy()

print("NOTEBOOK 41 — ACTION DISTRIBUTION")
print("=" * 70)

print(
    "Distribution batch :",
    len(policy_distribution.probs),
)

print()

distribution_df = pd.DataFrame(
    {
        "Sampled Action": sampled_actions.cpu().numpy(),
        "Log Probability": log_probabilities.detach().cpu().numpy(),
        "Entropy": entropy.detach().cpu().numpy(),
    }
)

display(distribution_df)

assert sampled_actions.shape == (3,)
assert log_probabilities.shape == (3,)
assert entropy.shape == (3,)

assert torch.isfinite(
    log_probabilities
).all()

assert torch.isfinite(
    entropy
).all()

print()

print("✅ SECTION 8 ACTION DISTRIBUTION PASSED")


# ## Section 9 — Greedy vs Sampled Actions
# 
# #### Compare sampled actions with deterministic (argmax) policy actions.

# In[9]:


# ============================================================
# SECTION 9 — GREEDY POLICY
# ============================================================

greedy_actions = torch.argmax(
    masked_policy_probs,
    dim=1,
)

comparison = pd.DataFrame(
    {
        "Sampled": sampled_actions.cpu().numpy(),
        "Greedy": greedy_actions.cpu().numpy(),
        "Match": (
            sampled_actions == greedy_actions
        ).cpu().numpy(),
    }
)

print("NOTEBOOK 41 — GREEDY POLICY")
print("=" * 70)

display(comparison)

agreement = (
    comparison["Match"]
    .mean()
)

print()

print(
    "Agreement:",
    f"{agreement:.2%}",
)

assert greedy_actions.shape == (3,)

assert torch.all(
    greedy_actions >= 0
)

assert torch.all(
    greedy_actions < metadata["action_dim"]
)

print()

print("✅ SECTION 9 GREEDY POLICY PASSED")


# ## Section 10 — Advantage Calculation
# 
# #### Compute simple advantages using the reward tensor and critic value estimates.

# In[11]:


# ============================================================
# SECTION 10 — ADVANTAGE CALCULATION
# ============================================================

# Convert rewards from NumPy into a PyTorch tensor.
reward_batch = torch.as_tensor(
    rewards,
    dtype=torch.float32,
    device=observation_batch.device,
)

# Recompute critic estimates to keep this cell rerunnable.
value_estimates = critic(
    observation_batch
).squeeze(-1)

# Simple one-step advantage:
# advantage = observed reward - estimated state value
advantages = (
    reward_batch
    - value_estimates.detach()
)

advantages_df = pd.DataFrame(
    {
        "Reward": (
            reward_batch
            .detach()
            .cpu()
            .numpy()
        ),
        "Value": (
            value_estimates
            .detach()
            .cpu()
            .numpy()
        ),
        "Advantage": (
            advantages
            .detach()
            .cpu()
            .numpy()
        ),
    }
)

print("NOTEBOOK 41 — ADVANTAGES")
print("=" * 70)

display(
    advantages_df
)

print()

print(
    "Mean advantage :",
    float(
        advantages.mean().item()
    ),
)

print(
    "Std advantage  :",
    float(
        advantages.std(
            unbiased=False
        ).item()
    ),
)

assert reward_batch.shape == (
    3,
)

assert value_estimates.shape == (
    3,
)

assert advantages.shape == (
    3,
)

assert reward_batch.dtype == torch.float32

assert torch.isfinite(
    value_estimates
).all()

assert torch.isfinite(
    advantages
).all()

print()
print(
    "✅ SECTION 10 ADVANTAGES PASSED"
)


# ## Section 11 — Discounted Returns
# 
# #### Compute discounted returns while stopping future reward propagation at terminal transitions.

# In[12]:


# ============================================================
# SECTION 11 — DISCOUNTED RETURNS
# ============================================================

GAMMA = 0.99

done_batch = torch.as_tensor(
    dones,
    dtype=torch.bool,
    device=reward_batch.device,
)

discounted_returns = torch.zeros_like(
    reward_batch
)

running_return = torch.tensor(
    0.0,
    dtype=torch.float32,
    device=reward_batch.device,
)

for index in reversed(
    range(
        len(reward_batch)
    )
):
    if done_batch[index]:
        running_return = torch.tensor(
            0.0,
            dtype=torch.float32,
            device=reward_batch.device,
        )

    running_return = (
        reward_batch[index]
        + GAMMA * running_return
    )

    discounted_returns[index] = (
        running_return
    )

returns_df = pd.DataFrame(
    {
        "Reward": (
            reward_batch
            .detach()
            .cpu()
            .numpy()
        ),
        "Done": (
            done_batch
            .detach()
            .cpu()
            .numpy()
        ),
        "Discounted Return": (
            discounted_returns
            .detach()
            .cpu()
            .numpy()
        ),
    }
)

print("NOTEBOOK 41 — DISCOUNTED RETURNS")
print("=" * 70)

print(
    "Gamma :",
    GAMMA,
)

print()

display(
    returns_df
)

assert discounted_returns.shape == (
    3,
)

assert discounted_returns.dtype == (
    torch.float32
)

assert torch.isfinite(
    discounted_returns
).all()

assert torch.all(
    discounted_returns
    >= reward_batch
)

assert torch.isclose(
    discounted_returns[-1],
    reward_batch[-1],
)

print()
print(
    "✅ SECTION 11 DISCOUNTED RETURNS PASSED"
)


# ## Section 12 — PPO Ratio Calculation
# 
# #### Compute the probability ratio between the current policy and the behavior policy.

# In[14]:


# ============================================================
# SECTION 12 — PPO RATIO
# ============================================================

old_log_probs = (
    log_probabilities
    .detach()
)

new_logits = actor(
    observation_batch
)

new_logits = new_logits.masked_fill(
    action_mask_batch == 0,
    float("-inf"),
)

new_distribution = torch.distributions.Categorical(
    logits=new_logits
)

new_log_probs = (
    new_distribution
    .log_prob(
        sampled_actions
    )
)

ratios = torch.exp(
    new_log_probs
    - old_log_probs
)

ratio_df = pd.DataFrame(
    {
        "Old Log Prob": (
            old_log_probs
            .detach()
            .cpu()
            .numpy()
        ),
        "New Log Prob": (
            new_log_probs
            .detach()
            .cpu()
            .numpy()
        ),
        "Ratio": (
            ratios
            .detach()
            .cpu()
            .numpy()
        ),
    }
)

print("NOTEBOOK 41 — PPO RATIOS")
print("=" * 70)

display(
    ratio_df
)

print()

print(
    "Mean ratio :",
    float(
        ratios.mean().item()
    ),
)

print(
    "Std ratio  :",
    float(
        ratios.std(
            unbiased=False
        ).item()
    ),
)

assert ratios.shape == (
    3,
)

assert torch.isfinite(
    ratios
).all()

assert torch.all(
    ratios > 0
)

assert torch.allclose(
    ratios,
    torch.ones_like(
        ratios
    ),
)

print()
print(
    "✅ SECTION 12 PPO RATIOS PASSED"
)


# ## Section 13 — PPO Loss Function
# 
# #### Compute the clipped PPO policy loss, critic value loss, entropy bonus, and combined training loss.

# In[15]:


# ============================================================
# SECTION 13 — PPO LOSS FUNCTION
# ============================================================

CLIP_EPSILON = 0.20
VALUE_COEFFICIENT = 0.50
ENTROPY_COEFFICIENT = 0.01

# Actions actually recorded in the Notebook 40 dataset.
behavior_action_batch = torch.as_tensor(
    actions,
    dtype=torch.long,
    device=observation_batch.device,
)

# Advantage target based on discounted return.
training_advantages = (
    discounted_returns
    - value_estimates.detach()
)

# Normalize only when there is meaningful variation.
advantage_std = training_advantages.std(
    unbiased=False
)

if float(advantage_std.item()) > 1e-8:
    normalized_advantages = (
        training_advantages
        - training_advantages.mean()
    ) / (
        advantage_std + 1e-8
    )
else:
    normalized_advantages = (
        training_advantages.clone()
    )

# ------------------------------------------------------------
# Behavior-policy log probabilities
# ------------------------------------------------------------

with torch.no_grad():

    old_policy_logits = actor(
        observation_batch
    )

    old_policy_logits = (
        old_policy_logits.masked_fill(
            action_mask_batch == 0,
            float("-inf"),
        )
    )

    old_policy_distribution = (
        torch.distributions.Categorical(
            logits=old_policy_logits
        )
    )

    behavior_log_probs = (
        old_policy_distribution.log_prob(
            behavior_action_batch
        )
    )

# ------------------------------------------------------------
# Current-policy outputs
# ------------------------------------------------------------

current_policy_logits = actor(
    observation_batch
)

current_policy_logits = (
    current_policy_logits.masked_fill(
        action_mask_batch == 0,
        float("-inf"),
    )
)

current_policy_distribution = (
    torch.distributions.Categorical(
        logits=current_policy_logits
    )
)

current_log_probs = (
    current_policy_distribution.log_prob(
        behavior_action_batch
    )
)

current_entropy = (
    current_policy_distribution
    .entropy()
    .mean()
)

current_values = critic(
    observation_batch
).squeeze(-1)

# ------------------------------------------------------------
# PPO clipped objective
# ------------------------------------------------------------

policy_ratios = torch.exp(
    current_log_probs
    - behavior_log_probs
)

unclipped_objective = (
    policy_ratios
    * normalized_advantages
)

clipped_ratios = torch.clamp(
    policy_ratios,
    1.0 - CLIP_EPSILON,
    1.0 + CLIP_EPSILON,
)

clipped_objective = (
    clipped_ratios
    * normalized_advantages
)

policy_loss = -torch.min(
    unclipped_objective,
    clipped_objective,
).mean()

value_loss = F.mse_loss(
    current_values,
    discounted_returns,
)

total_loss = (
    policy_loss
    + VALUE_COEFFICIENT * value_loss
    - ENTROPY_COEFFICIENT * current_entropy
)

loss_summary = pd.DataFrame(
    [
        (
            "Policy loss",
            float(
                policy_loss
                .detach()
                .cpu()
                .item()
            ),
        ),
        (
            "Value loss",
            float(
                value_loss
                .detach()
                .cpu()
                .item()
            ),
        ),
        (
            "Entropy",
            float(
                current_entropy
                .detach()
                .cpu()
                .item()
            ),
        ),
        (
            "Total loss",
            float(
                total_loss
                .detach()
                .cpu()
                .item()
            ),
        ),
        (
            "Mean policy ratio",
            float(
                policy_ratios
                .detach()
                .mean()
                .cpu()
                .item()
            ),
        ),
    ],
    columns=[
        "Metric",
        "Value",
    ],
)

print("NOTEBOOK 41 — PPO LOSS")
print("=" * 70)

print(
    "Clip epsilon       :",
    CLIP_EPSILON,
)

print(
    "Value coefficient :",
    VALUE_COEFFICIENT,
)

print(
    "Entropy coefficient:",
    ENTROPY_COEFFICIENT,
)

print()

display(
    loss_summary
)

assert behavior_action_batch.shape == (
    3,
)

assert policy_ratios.shape == (
    3,
)

assert normalized_advantages.shape == (
    3,
)

assert policy_loss.ndim == 0
assert value_loss.ndim == 0
assert current_entropy.ndim == 0
assert total_loss.ndim == 0

assert torch.isfinite(
    policy_ratios
).all()

assert torch.isfinite(
    policy_loss
)

assert torch.isfinite(
    value_loss
)

assert torch.isfinite(
    current_entropy
)

assert torch.isfinite(
    total_loss
)

assert torch.allclose(
    policy_ratios,
    torch.ones_like(
        policy_ratios
    ),
)

print()
print(
    "✅ SECTION 13 PPO LOSS PASSED"
)


# ## Section 14 — Optimizer and First Training Step
# 
# #### Perform one PPO optimization step and verify that the actor and critic parameters update successfully.

# In[16]:


# ============================================================
# SECTION 14 — OPTIMIZER AND TRAINING STEP
# ============================================================

LEARNING_RATE = 3e-4
MAX_GRAD_NORM = 0.5

optimizer = torch.optim.Adam(
    list(actor.parameters())
    + list(critic.parameters()),
    lr=LEARNING_RATE,
)

# Snapshot parameters before the update.
actor_before = {
    name: parameter
    .detach()
    .clone()
    for name, parameter
    in actor.named_parameters()
}

critic_before = {
    name: parameter
    .detach()
    .clone()
    for name, parameter
    in critic.named_parameters()
}

optimizer.zero_grad()

total_loss.backward()

gradient_norm = torch.nn.utils.clip_grad_norm_(
    list(actor.parameters())
    + list(critic.parameters()),
    MAX_GRAD_NORM,
)

optimizer.step()

actor_changed = any(
    not torch.allclose(
        actor_before[name],
        parameter.detach(),
    )
    for name, parameter
    in actor.named_parameters()
)

critic_changed = any(
    not torch.allclose(
        critic_before[name],
        parameter.detach(),
    )
    for name, parameter
    in critic.named_parameters()
)

print("NOTEBOOK 41 — TRAINING STEP")
print("=" * 70)

print(
    "Learning rate       :",
    LEARNING_RATE,
)

print(
    "Max gradient norm   :",
    MAX_GRAD_NORM,
)

print(
    "Observed grad norm  :",
    float(
        gradient_norm
        .detach()
        .cpu()
        .item()
    ),
)

print(
    "Actor parameters changed :",
    actor_changed,
)

print(
    "Critic parameters changed:",
    critic_changed,
)

assert torch.isfinite(
    gradient_norm
)

assert actor_changed

assert critic_changed

print()
print(
    "✅ SECTION 14 TRAINING STEP PASSED"
)


# ## Section 15 — Export PPO Training Engine
# 
# #### Export the trained actor, critic, optimizer state, and metadata for Notebook 42.

# In[18]:


# ============================================================
# NOTEBOOK 41 — EXPORT PPO TRAINING ENGINE
# SECTION 15
# ============================================================

import pickle
from pathlib import Path

export_dir = PROJECT_ROOT / "reports" / "notebook41"
export_dir.mkdir(
    parents=True,
    exist_ok=True,
)

checkpoint_file = (
    export_dir
    / "ppo_training_engine.pkl"
)

checkpoint = {

    "actor_state_dict":
        actor.state_dict(),

    "critic_state_dict":
        critic.state_dict(),

    "optimizer_state_dict":
        optimizer.state_dict(),

 "metadata": {

    "observation_dim":
        observation_batch.shape[1],

    "action_dim":
        action_mask_batch.shape[1],

    "num_samples":
        len(observation_batch),

    "learning_rate":
        LEARNING_RATE,

    "clip_epsilon":
        CLIP_EPSILON,

    "gamma":
        GAMMA,

    "random_seed":
        RANDOM_SEED,
}
}

with open(
    checkpoint_file,
    "wb",
) as f:

    pickle.dump(
        checkpoint,
        f,
    )

assert checkpoint_file.exists()

summary = pd.DataFrame(

    [

        (
            "Samples",
            checkpoint["metadata"]["num_samples"],
        ),

        (
            "Observation Dimension",
            checkpoint["metadata"]["observation_dim"],
        ),

        (
            "Action Dimension",
            checkpoint["metadata"]["action_dim"],
        ),

        (
            "Learning Rate",
            checkpoint["metadata"]["learning_rate"],
        ),

        (
            "Gamma",
            checkpoint["metadata"]["gamma"],
        ),

        (
            "Clip Epsilon",
            checkpoint["metadata"]["clip_epsilon"],
        ),

    ],

    columns=[
        "Field",
        "Value",
    ],

)

print("NOTEBOOK 41 — PPO EXPORT")
print("=" * 70)

print(
    "Export file :",
    checkpoint_file,
)

print()

display(summary)

print()
print("🏆 NOTEBOOK 41 COMPLETE")


# In[ ]:




