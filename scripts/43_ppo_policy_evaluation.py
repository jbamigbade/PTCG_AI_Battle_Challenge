#!/usr/bin/env python
# coding: utf-8

# # Notebook 43 -  PPO Policy Evaluation
# 
# ## Objectives
# 
# - Load the PPO dataset from Notebook 40
# - Load the baseline checkpoint from Notebook 41
# - Load the trained checkpoint from Notebook 42
# - Reconstruct both Actor and Critic model pairs
# - Compare baseline and trained policy outputs
# - Validate legal-action masking
# - Evaluate recorded-action probabilities
# - Compare critic predictions and value errors
# - Measure policy changes after training
# - Export a structured PPO evaluation report
# 
# > The current dataset contains three validation samples. This notebook verifies the evaluation pipeline and checkpoint integrity rather than measuring production-level gameplay strength.
# 

# ## Section 1 — Project Setup and Artifact Discovery
# 
# #### Locate the project root and verify the dataset, baseline checkpoint, and trained checkpoint required for PPO policy evaluation.

# In[2]:


# ============================================================
# NOTEBOOK 43 — PPO POLICY EVALUATION
# SECTION 1 — PROJECT SETUP AND ARTIFACT DISCOVERY
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

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical


def find_project_root(
    start_path: Path | None = None,
) -> Path:
    """Locate the Pokémon project root."""

    current = (
        start_path or Path.cwd()
    ).resolve()

    for candidate in [
        current,
        *current.parents,
    ]:
        required_directories = [
            candidate / "src",
            candidate / "notebooks",
            candidate / "scripts",
            candidate / "reports",
        ]

        if all(
            directory.is_dir()
            for directory in required_directories
        ):
            return candidate

    raise FileNotFoundError(
        "Could not locate the project root. "
        "Run this notebook from inside the project."
    )


PROJECT_ROOT = find_project_root()

SRC_DIR = PROJECT_ROOT / "src"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
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

NOTEBOOK43_REPORT_DIR = (
    REPORTS_DIR / "notebook43"
)

PPO_DATASET_FILE = (
    NOTEBOOK40_REPORT_DIR
    / "ppo_training_package.pkl"
)

BASELINE_CHECKPOINT_FILE = (
    NOTEBOOK41_REPORT_DIR
    / "ppo_training_engine.pkl"
)

TRAINED_CHECKPOINT_FILE = (
    NOTEBOOK42_REPORT_DIR
    / "ppo_training_loop.pkl"
)

NOTEBOOK43_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

RANDOM_SEED = 43

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

artifact_summary = pd.DataFrame(
    {
        "Artifact": [
            "Notebook 40 PPO Dataset",
            "Notebook 41 Baseline Checkpoint",
            "Notebook 42 Trained Checkpoint",
            "Notebook 43 Report Directory",
        ],
        "Path": [
            str(PPO_DATASET_FILE),
            str(BASELINE_CHECKPOINT_FILE),
            str(TRAINED_CHECKPOINT_FILE),
            str(NOTEBOOK43_REPORT_DIR),
        ],
        "Exists": [
            PPO_DATASET_FILE.exists(),
            BASELINE_CHECKPOINT_FILE.exists(),
            TRAINED_CHECKPOINT_FILE.exists(),
            NOTEBOOK43_REPORT_DIR.exists(),
        ],
    }
)

print("NOTEBOOK 43 — PROJECT SETUP")
print("=" * 72)

print("Project root   :", PROJECT_ROOT)
print("Device         :", DEVICE)
print("PyTorch version:", torch.__version__)
print("Random seed    :", RANDOM_SEED)

print()
display(artifact_summary)

assert SRC_DIR.exists()
assert NOTEBOOKS_DIR.exists()
assert SCRIPTS_DIR.exists()
assert REPORTS_DIR.exists()

assert PPO_DATASET_FILE.exists(), (
    f"Missing Notebook 40 dataset: "
    f"{PPO_DATASET_FILE}"
)

assert BASELINE_CHECKPOINT_FILE.exists(), (
    f"Missing Notebook 41 checkpoint: "
    f"{BASELINE_CHECKPOINT_FILE}"
)

assert TRAINED_CHECKPOINT_FILE.exists(), (
    f"Missing Notebook 42 checkpoint: "
    f"{TRAINED_CHECKPOINT_FILE}"
)

assert artifact_summary["Exists"].all()

print()
print("✅ SECTION 1 PROJECT SETUP PASSED")


# ## Section 2 — Load Dataset and Checkpoints
# 
# #### Load the PPO dataset from Notebook 40 along with the baseline checkpoint from Notebook 41 and the trained checkpoint from Notebook 42. Verify that all required components are present before evaluation.

# In[3]:


# ============================================================
# SECTION 2 — LOAD DATASET & CHECKPOINTS
# ============================================================

with open(
    PPO_DATASET_FILE,
    "rb",
) as f:
    ppo_package = pickle.load(f)

with open(
    BASELINE_CHECKPOINT_FILE,
    "rb",
) as f:
    baseline_checkpoint = pickle.load(f)

with open(
    TRAINED_CHECKPOINT_FILE,
    "rb",
) as f:
    trained_checkpoint = pickle.load(f)

ppo_batch = ppo_package["ppo_batch"]
dataset_metadata = ppo_package["metadata"]

baseline_metadata = baseline_checkpoint["metadata"]
trained_metadata = trained_checkpoint["metadata"]

print("NOTEBOOK 43 — LOAD ARTIFACTS")
print("=" * 72)

print("Dataset Samples :", dataset_metadata["num_samples"])
print("Observation Dim :", dataset_metadata["observation_dim"])
print("Action Dim      :", dataset_metadata["action_dim"])

print()

print("Baseline Learning Rate :", baseline_metadata["learning_rate"])
print("Trained Learning Rate  :", trained_metadata["learning_rate"])

summary = pd.DataFrame(
    {
        "Component": [
            "Observations",
            "Action Masks",
            "Actions",
            "Rewards",
            "Dones",
        ],
        "Shape": [
            ppo_batch["observations"].shape,
            ppo_batch["action_masks"].shape,
            ppo_batch["actions"].shape,
            ppo_batch["rewards"].shape,
            ppo_batch["dones"].shape,
        ],
    }
)

display(summary)

assert "actor_state_dict" in baseline_checkpoint
assert "critic_state_dict" in baseline_checkpoint
assert "optimizer_state_dict" in baseline_checkpoint

assert "actor_state_dict" in trained_checkpoint
assert "critic_state_dict" in trained_checkpoint
assert "optimizer_state_dict" in trained_checkpoint

assert ppo_batch["observations"].shape == (3, 4)
assert ppo_batch["action_masks"].shape == (3, 4)
assert ppo_batch["actions"].shape == (3,)
assert ppo_batch["rewards"].shape == (3,)
assert ppo_batch["dones"].shape == (3,)

print()
print("✅ SECTION 2 LOAD ARTIFACTS PASSED")


# ## Section 3 — Rebuild Baseline and Trained Networks
# 
# #### Reconstruct the Actor and Critic architectures, then load both the baseline (Notebook 41) and trained (Notebook 42) checkpoints for side-by-side evaluation.

# In[4]:


# ============================================================
# SECTION 3 — REBUILD NETWORKS
# ============================================================

class PPOActor(nn.Module):

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.network(x)


class PPOCritic(nn.Module):

    def __init__(
        self,
        observation_dim: int,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x)


baseline_actor = PPOActor(
    dataset_metadata["observation_dim"],
    dataset_metadata["action_dim"],
).to(DEVICE)

baseline_critic = PPOCritic(
    dataset_metadata["observation_dim"],
).to(DEVICE)

trained_actor = PPOActor(
    dataset_metadata["observation_dim"],
    dataset_metadata["action_dim"],
).to(DEVICE)

trained_critic = PPOCritic(
    dataset_metadata["observation_dim"],
).to(DEVICE)

baseline_actor.load_state_dict(
    baseline_checkpoint["actor_state_dict"]
)

baseline_critic.load_state_dict(
    baseline_checkpoint["critic_state_dict"]
)

trained_actor.load_state_dict(
    trained_checkpoint["actor_state_dict"]
)

trained_critic.load_state_dict(
    trained_checkpoint["critic_state_dict"]
)

baseline_actor.eval()
baseline_critic.eval()
trained_actor.eval()
trained_critic.eval()

print("NOTEBOOK 43 — REBUILD NETWORKS")
print("=" * 72)

print("Baseline Actor Loaded :", True)
print("Baseline Critic Loaded:", True)
print("Trained Actor Loaded  :", True)
print("Trained Critic Loaded :", True)

assert isinstance(baseline_actor, PPOActor)
assert isinstance(trained_actor, PPOActor)
assert isinstance(baseline_critic, PPOCritic)
assert isinstance(trained_critic, PPOCritic)

print()
print("✅ SECTION 3 REBUILD NETWORKS PASSED")


# ## Section 4 — Prepare Evaluation Tensors
# 
# #### Convert the PPO dataset into PyTorch tensors and move them to the selected device for evaluation.

# In[5]:


# ============================================================
# SECTION 4 — PREPARE EVALUATION TENSORS
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

print("NOTEBOOK 43 — PREPARE EVALUATION TENSORS")
print("=" * 72)

print("Dataset Size       :", dataset_size)
print("Observation Shape  :", observations.shape)
print("Action Mask Shape  :", action_masks.shape)
print("Action Shape       :", actions.shape)
print("Reward Shape       :", rewards.shape)
print("Done Shape         :", dones.shape)

assert observations.shape == (3, 4)
assert action_masks.shape == (3, 4)
assert actions.shape == (3,)
assert rewards.shape == (3,)
assert dones.shape == (3,)

print()
print("✅ SECTION 4 PREPARE EVALUATION TENSORS PASSED")


# ## Section 5 — Evaluate Baseline and Trained Policies
# 
# #### Run both Actor–Critic models on the same observations and compare their policy logits, action probabilities, entropy, and value predictions.

# In[6]:


# ============================================================
# SECTION 5 — POLICY EVALUATION
# ============================================================

with torch.no_grad():

    # -------------------------
    # Baseline
    # -------------------------

    baseline_logits = baseline_actor(
        observations
    )

    baseline_logits = baseline_logits.masked_fill(
        ~action_masks,
        -1e9,
    )

    baseline_distribution = Categorical(
        logits=baseline_logits,
    )

    baseline_probabilities = (
        baseline_distribution.probs
    )

    baseline_entropy = (
        baseline_distribution.entropy()
    )

    baseline_values = baseline_critic(
        observations
    ).squeeze(-1)

    # -------------------------
    # Trained
    # -------------------------

    trained_logits = trained_actor(
        observations
    )

    trained_logits = trained_logits.masked_fill(
        ~action_masks,
        -1e9,
    )

    trained_distribution = Categorical(
        logits=trained_logits,
    )

    trained_probabilities = (
        trained_distribution.probs
    )

    trained_entropy = (
        trained_distribution.entropy()
    )

    trained_values = trained_critic(
        observations
    ).squeeze(-1)

print("NOTEBOOK 43 — POLICY EVALUATION")
print("=" * 72)

print(
    "Baseline Mean Entropy :",
    baseline_entropy.mean().item(),
)

print(
    "Trained Mean Entropy  :",
    trained_entropy.mean().item(),
)

print()

print(
    "Baseline Mean Value :",
    baseline_values.mean().item(),
)

print(
    "Trained Mean Value  :",
    trained_values.mean().item(),
)

assert baseline_logits.shape == trained_logits.shape
assert baseline_values.shape == trained_values.shape

print()
print("✅ SECTION 5 POLICY EVALUATION PASSED")


# ## Section 6 — Quantify Policy Changes
# 
# #### Measure how much the policy changed after PPO training by comparing action probabilities and critic value predictions between the baseline and trained models.

# In[7]:


# ============================================================
# SECTION 6 — POLICY DIFFERENCE ANALYSIS
# ============================================================

policy_probability_difference = torch.abs(
    trained_probabilities - baseline_probabilities
)

value_difference = torch.abs(
    trained_values - baseline_values
)

mean_probability_difference = (
    policy_probability_difference.mean().item()
)

max_probability_difference = (
    policy_probability_difference.max().item()
)

mean_value_difference = (
    value_difference.mean().item()
)

max_value_difference = (
    value_difference.max().item()
)

comparison_summary = pd.DataFrame(
    {
        "Metric": [
            "Mean Probability Difference",
            "Maximum Probability Difference",
            "Mean Value Difference",
            "Maximum Value Difference",
        ],
        "Value": [
            mean_probability_difference,
            max_probability_difference,
            mean_value_difference,
            max_value_difference,
        ],
    }
)

print("NOTEBOOK 43 — POLICY DIFFERENCE ANALYSIS")
print("=" * 72)

display(comparison_summary)

assert torch.isfinite(policy_probability_difference).all()
assert torch.isfinite(value_difference).all()

assert mean_probability_difference >= 0.0
assert mean_value_difference >= 0.0

print()
print("✅ SECTION 6 POLICY DIFFERENCE ANALYSIS PASSED")


# ## Section 7 — Compare Recorded Action Log Probabilities
# 
# #### Evaluate the probability assigned to each recorded action by the baseline and trained policies.

# In[8]:


# ============================================================
# SECTION 7 — LOG PROBABILITY COMPARISON
# ============================================================

with torch.no_grad():

    baseline_log_probs = baseline_distribution.log_prob(
        actions
    )

    trained_log_probs = trained_distribution.log_prob(
        actions
    )

log_probability_summary = pd.DataFrame(
    {
        "Sample": np.arange(dataset_size),
        "Baseline Log Prob": baseline_log_probs.cpu().numpy(),
        "Trained Log Prob": trained_log_probs.cpu().numpy(),
        "Difference": (
            trained_log_probs
            - baseline_log_probs
        ).cpu().numpy(),
    }
)

print("NOTEBOOK 43 — LOG PROBABILITY COMPARISON")
print("=" * 72)

display(log_probability_summary)

assert len(log_probability_summary) == dataset_size

print()
print("✅ SECTION 7 LOG PROBABILITY COMPARISON PASSED")


# ## Section 8 — Compare Greedy Action Predictions
# 
# #### ompare the greedy action selected by the baseline and trained policies for each observation in the evaluation dataset.

# In[9]:


# ============================================================
# SECTION 8 — GREEDY ACTION COMPARISON
# ============================================================

baseline_actions = baseline_probabilities.argmax(dim=1)

trained_actions = trained_probabilities.argmax(dim=1)

action_changed = (
    baseline_actions != trained_actions
)

comparison_table = pd.DataFrame(
    {
        "Sample": np.arange(dataset_size),
        "Baseline Action": baseline_actions.cpu().numpy(),
        "Trained Action": trained_actions.cpu().numpy(),
        "Changed": action_changed.cpu().numpy(),
    }
)

print("NOTEBOOK 43 — GREEDY ACTION COMPARISON")
print("=" * 72)

display(comparison_table)

num_changes = int(action_changed.sum().item())

print(f"Changed Actions: {num_changes}/{dataset_size}")

assert len(comparison_table) == dataset_size

print()
print("✅ SECTION 8 GREEDY ACTION COMPARISON PASSED")


# ## Section 9 — Critic Prediction Error
# 
# #### Compare the baseline and trained critic predictions against the observed discounted returns.

# In[10]:


# ============================================================
# SECTION 9 — CRITIC ERROR ANALYSIS
# ============================================================

# Recompute discounted returns

gamma = trained_metadata["gamma"]

discounted_returns = torch.zeros_like(rewards)

running_return = torch.tensor(
    0.0,
    device=DEVICE,
)

for step in reversed(range(dataset_size)):

    if dones[step]:
        running_return = rewards[step]
    else:
        running_return = rewards[step] + gamma * running_return

    discounted_returns[step] = running_return

baseline_mse = F.mse_loss(
    baseline_values,
    discounted_returns,
)

trained_mse = F.mse_loss(
    trained_values,
    discounted_returns,
)

critic_summary = pd.DataFrame(
    {
        "Metric": [
            "Baseline MSE",
            "Trained MSE",
            "Improvement",
        ],
        "Value": [
            baseline_mse.item(),
            trained_mse.item(),
            baseline_mse.item() - trained_mse.item(),
        ],
    }
)

print("NOTEBOOK 43 — CRITIC ERROR ANALYSIS")
print("=" * 72)

display(critic_summary)

assert baseline_mse >= 0
assert trained_mse >= 0

print()
print("✅ SECTION 9 CRITIC ERROR ANALYSIS PASSED")


# ## Section 10 — Export Evaluation Report
# 
# #### Save the PPO evaluation results, comparison metrics, and critic analysis for future benchmarking and visualization.

# In[11]:


# ============================================================
# SECTION 10 — EXPORT EVALUATION REPORT
# ============================================================

evaluation_report = {
    "baseline_entropy": baseline_entropy.mean().item(),
    "trained_entropy": trained_entropy.mean().item(),
    "baseline_value_mean": baseline_values.mean().item(),
    "trained_value_mean": trained_values.mean().item(),
    "mean_probability_difference": mean_probability_difference,
    "max_probability_difference": max_probability_difference,
    "mean_value_difference": mean_value_difference,
    "max_value_difference": max_value_difference,
    "baseline_mse": baseline_mse.item(),
    "trained_mse": trained_mse.item(),
    "critic_improvement": (
        baseline_mse.item()
        - trained_mse.item()
    ),
    "changed_actions": int(
        action_changed.sum().item()
    ),
    "dataset_size": dataset_size,
}

evaluation_file = (
    NOTEBOOK43_REPORT_DIR
    / "ppo_policy_evaluation.pkl"
)

with open(
    evaluation_file,
    "wb",
) as f:
    pickle.dump(
        evaluation_report,
        f,
    )

print("NOTEBOOK 43 — EXPORT EVALUATION")
print("=" * 72)

print("Saved evaluation report:")

print(evaluation_file)

assert evaluation_file.exists()

print()
print("✅ SECTION 10 EXPORT PASSED")


# ## Section 11 — Final Validation
# 
# #### Verify that all required Notebook 43 outputs have been generated successfully.

# In[12]:


# ============================================================
# SECTION 11 — FINAL VALIDATION
# ============================================================

summary = pd.DataFrame(
    {
        "Artifact": [
            "Notebook 40 Dataset",
            "Notebook 41 Baseline",
            "Notebook 42 Trained",
            "Notebook 43 Evaluation",
            "Comparison Summary",
            "Critic Analysis",
        ],
        "Status": [
            PPO_DATASET_FILE.exists(),
            BASELINE_CHECKPOINT_FILE.exists(),
            TRAINED_CHECKPOINT_FILE.exists(),
            evaluation_file.exists(),
            len(comparison_summary) > 0,
            len(critic_summary) > 0,
        ],
    }
)

print("NOTEBOOK 43 COMPLETE")
print("=" * 72)

display(summary)

assert summary["Status"].all()

print()
print("🎉 NOTEBOOK 43 COMPLETE")


# In[ ]:




