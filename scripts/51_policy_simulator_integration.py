#!/usr/bin/env python
# coding: utf-8

# # Notebook 51 — Policy Simulator Integration
# 
# ## Purpose
# 
# ### This notebook integrates the trained Notebook 50 policy into the Pokémon TCG simulator.
# 
# #### The integration pipeline will:
# 
# 1. Validate the Notebook 50 handoff.
# 2. Load the trained policy model and preprocessing artifacts.
# 3. reconstruct simulator states into policy features.
# 4. generate legal moves from the simulator.
# 5. score candidate actions with the trained policy.
# 6. enforce legal-action masking.
# 7. compare policy-selected moves with search-selected moves.
# 8. run integration smoke tests.
# 9. export a reusable policy-agent interface.
# 10. create the Notebook 51 handoff for tournament evaluation.
# 
# #### Notebook 50 handoff
# 
# #### Expected upstream status:
# 
# - `READY_FOR_SIMULATOR_INTEGRATION`
# - 6 policy classes
# - 18 encoded features
# - side-balanced training enabled
# - model reload validation passed
# - 14/14 final checks passed

# ## Section 1A — Imports and project paths

# In[1]:


# ======================================================================================
# SECTION 1A — IMPORTS AND PROJECT PATHS
# ======================================================================================

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# --------------------------------------------------------------------------------------
# Resolve project root
# --------------------------------------------------------------------------------------

CURRENT_DIRECTORY = Path.cwd().resolve()

if CURRENT_DIRECTORY.name.lower() == "notebooks":
    PROJECT_ROOT = CURRENT_DIRECTORY.parent
else:
    possible_root = CURRENT_DIRECTORY

    while (
        possible_root.parent != possible_root
        and not (possible_root / "notebooks").exists()
    ):
        possible_root = possible_root.parent

    PROJECT_ROOT = possible_root


NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"

NOTEBOOK50_REPORT_DIR = REPORTS_DIR / "notebook50"
NOTEBOOK50_MODEL_DIR = MODELS_DIR / "notebook50"

NOTEBOOK51_REPORT_DIR = REPORTS_DIR / "notebook51"
NOTEBOOK51_MODEL_DIR = MODELS_DIR / "notebook51"

NOTEBOOK51_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

NOTEBOOK51_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print("=" * 100)
print("SECTION 1A — IMPORTS AND PROJECT PATHS")
print("=" * 100)

print()
print("DIRECTORIES")
print("-" * 100)

print(f"Current directory       : {CURRENT_DIRECTORY}")
print(f"Project root            : {PROJECT_ROOT}")
print(f"Notebook 50 reports     : {NOTEBOOK50_REPORT_DIR}")
print(f"Notebook 50 models      : {NOTEBOOK50_MODEL_DIR}")
print(f"Notebook 51 reports     : {NOTEBOOK51_REPORT_DIR}")
print(f"Notebook 51 models      : {NOTEBOOK51_MODEL_DIR}")
print(f"Source directory        : {SRC_DIR}")

print()
print("PACKAGE VERSIONS")
print("-" * 100)

print(f"Python                   : {sys.version.split()[0]}")
print(f"NumPy                    : {np.__version__}")
print(f"Pandas                   : {pd.__version__}")

assert PROJECT_ROOT.exists()
assert NOTEBOOK50_REPORT_DIR.exists()
assert NOTEBOOK50_MODEL_DIR.exists()

print()
print("✅ SECTION 1A IMPORTS AND PROJECT PATHS PASSED")


# # Section 1B — Notebook 50 Handoff Validation

# In[2]:


# ======================================================================================
# SECTION 1B — NOTEBOOK 50 HANDOFF VALIDATION
# ======================================================================================

SECTION7_DIR = NOTEBOOK50_REPORT_DIR / "section7"

HANDOFF_MANIFEST = SECTION7_DIR / "section7a_handoff_manifest.json"
FINAL_SUMMARY = SECTION7_DIR / "section7a_final_summary.json"
VALIDATION_CHECKS = SECTION7_DIR / "section7a_validation_checks.csv"

MODEL_FILE = NOTEBOOK50_MODEL_DIR / "final_side_balanced_policy.joblib"
PREPROCESSOR_FILE = NOTEBOOK50_MODEL_DIR / "leakage_safe_preprocessor.joblib"
LABEL_ENCODER_FILE = NOTEBOOK50_MODEL_DIR / "policy_label_encoder.joblib"
FEATURE_NAMES_FILE = NOTEBOOK50_MODEL_DIR / "feature_names.json"
MODEL_METADATA_FILE = NOTEBOOK50_MODEL_DIR / "model_metadata.json"

required_artifacts = [
    ("handoff_manifest", HANDOFF_MANIFEST),
    ("final_summary", FINAL_SUMMARY),
    ("validation_checks", VALIDATION_CHECKS),
    ("policy_model", MODEL_FILE),
    ("preprocessor", PREPROCESSOR_FILE),
    ("label_encoder", LABEL_ENCODER_FILE),
    ("feature_names", FEATURE_NAMES_FILE),
    ("model_metadata", MODEL_METADATA_FILE),
]

inventory = []

for name, path in required_artifacts:

    inventory.append(
        {
            "artifact": name,
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "path": str(path),
        }
    )

inventory_df = pd.DataFrame(inventory)

print("=" * 100)
print("SECTION 1B — NOTEBOOK 50 HANDOFF VALIDATION")
print("=" * 100)

print()
print("REQUIRED ARTIFACTS")
print("-" * 100)

display(inventory_df)

assert inventory_df["exists"].all(), "Notebook 50 artifacts are missing."

with open(HANDOFF_MANIFEST, "r") as f:
    handoff = json.load(f)

print()
print("HANDOFF STATUS")
print("-" * 100)

for k, v in handoff.items():
    print(f"{k:30}: {v}")

status = handoff.get(
    "final_status",
    handoff.get(
        "status",
        None
    )
)

assert status == "READY_FOR_SIMULATOR_INTEGRATION"

print()
print("✅ SECTION 1B NOTEBOOK 50 HANDOFF VALIDATION PASSED")


# ## Section 2A — Load Final Policy Artifacts

# In[3]:


# ======================================================================================
# SECTION 2A — LOAD FINAL POLICY ARTIFACTS
# ======================================================================================

import json
import joblib

print("=" * 100)
print("SECTION 2A — LOAD FINAL POLICY ARTIFACTS")
print("=" * 100)

policy_model = joblib.load(
    NOTEBOOK50_MODEL_DIR / "final_side_balanced_policy.joblib"
)

preprocessor = joblib.load(
    NOTEBOOK50_MODEL_DIR / "leakage_safe_preprocessor.joblib"
)

label_encoder = joblib.load(
    NOTEBOOK50_MODEL_DIR / "policy_label_encoder.joblib"
)

with open(
    NOTEBOOK50_MODEL_DIR / "feature_names.json",
    "r",
) as f:
    feature_names = json.load(f)

with open(
    NOTEBOOK50_MODEL_DIR / "model_metadata.json",
    "r",
) as f:
    model_metadata = json.load(f)

print()
print("MODEL SUMMARY")
print("-" * 100)

print(f"Model type              : {type(policy_model).__name__}")
print(f"Policy classes          : {len(label_encoder.classes_)}")
print(f"Encoded features        : {len(feature_names)}")
print(f"Training examples       : {model_metadata['training_examples']}")
print(f"Validation examples     : {model_metadata['validation_examples']}")
print(f"Test examples           : {model_metadata['test_examples']}")

print()
print("Policy Classes")
print("-" * 100)

for i, cls in enumerate(label_encoder.classes_):
    print(f"{i:2d} : {cls}")

assert len(feature_names) == model_metadata["encoded_feature_count"]

print()
print("✅ SECTION 2A POLICY ARTIFACT LOADING PASSED")


# ## Section 2B — Policy Inference Wrapper

# In[4]:


# ======================================================================================
# SECTION 2B — POLICY INFERENCE WRAPPER
# ======================================================================================

print("=" * 100)
print("SECTION 2B — POLICY INFERENCE WRAPPER")
print("=" * 100)


def predict_policy_move(feature_dataframe):
    """
    Predict the expert policy move from a single battle state.

    Parameters
    ----------
    feature_dataframe : pandas.DataFrame
        Single-row dataframe containing raw battle features.

    Returns
    -------
    dict
        prediction information
    """

    encoded = preprocessor.transform(feature_dataframe)

    predicted_label = policy_model.predict(encoded)[0]

    probabilities = policy_model.predict_proba(encoded)[0]

    predicted_move = label_encoder.inverse_transform([predicted_label])[0]

    probability_lookup = {
        move: float(prob)
        for move, prob in zip(
            label_encoder.classes_,
            probabilities,
        )
    }

    return {
        "predicted_move": predicted_move,
        "predicted_label": int(predicted_label),
        "probabilities": probability_lookup,
        "confidence": float(np.max(probabilities)),
    }


print()
print("Inference function successfully created.")

print()
print("Supported policy classes")
print("-" * 100)

for move in label_encoder.classes_:
    print(move)

print()
print("✅ SECTION 2B POLICY INFERENCE WRAPPER PASSED")


# # Section 2C — Policy Inference Validation

# In[6]:


# ======================================================================================
# SECTION 2C — POLICY INFERENCE VALIDATION
# ======================================================================================

print("=" * 100)
print("SECTION 2C — POLICY INFERENCE VALIDATION")
print("=" * 100)

# Load the original training dataset produced in Notebook 49
policy_dataset = pd.read_csv(
    REPORTS_DIR / "notebook49"
    / "section8"
    / "policy_dataset"
    / "section8c_expert_policy_dataset.csv"
)

feature_columns = [
    "variant_name",
    "current_side",
    "side_mode",
    "player_card",
    "opponent_card",
    "turn_number",
    "player_energy",
    "opponent_energy",
    "player_damage",
    "opponent_damage",
    "prize_cards_remaining",
    "hand_size",
    "legal_move_count",
]

sample_size = 20

sample_df = policy_dataset.sample(
    n=sample_size,
    random_state=42,
).reset_index(drop=True)

predicted_moves = []

for _, row in sample_df.iterrows():

    features = row[feature_columns].to_frame().T

    prediction = predict_policy_move(features)

    predicted_moves.append(prediction["predicted_move"])

sample_df["predicted_move"] = predicted_moves
sample_df["prediction_match"] = (
    sample_df["expert_move"] == sample_df["predicted_move"]
)

print()
print("INFERENCE RESULTS")
print("-" * 100)

display(
    sample_df[
        [
            "expert_move",
            "predicted_move",
            "prediction_match",
        ]
    ]
)

accuracy = sample_df["prediction_match"].mean()

print()
print(f"Prediction agreement : {accuracy:.2%}")

assert accuracy == 1.0

print()
print("✅ SECTION 2C POLICY INFERENCE VALIDATION PASSED")


# # Section 3 — Simulator State Feature Extraction
# 
# ## Section 3A — Live Battle-State Feature Extractor
# 
# #### This section converts a live simulator battle state into the exact raw feature schema expected by the Notebook 50 preprocessing pipeline.
# 
# #### The extractor is designed to tolerate both:
# 
# - object-based simulator states, and
# - dictionary-based reconstructed states.
# 
# #### It also supports common naming variations used across earlier notebooks.

# In[7]:


# ======================================================================================
# SECTION 3A — LIVE BATTLE-STATE FEATURE EXTRACTOR
# ======================================================================================

print("=" * 100)
print("SECTION 3A — LIVE BATTLE-STATE FEATURE EXTRACTOR")
print("=" * 100)


# --------------------------------------------------------------------------------------
# Final raw feature schema used by the leakage-safe Notebook 50 policy
# --------------------------------------------------------------------------------------

POLICY_RAW_FEATURE_COLUMNS = list(
    model_metadata["input_feature_columns"]
)

print()
print("EXPECTED RAW POLICY FEATURES")
print("-" * 100)

for index, feature_name in enumerate(
    POLICY_RAW_FEATURE_COLUMNS,
    start=1,
):
    print(f"{index:>2}. {feature_name}")


# --------------------------------------------------------------------------------------
# Generic nested-value helpers
# --------------------------------------------------------------------------------------

_MISSING = object()


def read_member(
    source: Any,
    name: str,
    default: Any = _MISSING,
) -> Any:
    """
    Read one member from a dictionary-like or object-like source.
    """

    if source is None:
        return default

    if isinstance(source, Mapping):
        return source.get(
            name,
            default,
        )

    return getattr(
        source,
        name,
        default,
    )


def read_first(
    source: Any,
    candidate_names: Sequence[str],
    default: Any = None,
) -> Any:
    """
    Return the first available value among several possible field names.
    """

    for candidate_name in candidate_names:

        value = read_member(
            source,
            candidate_name,
            _MISSING,
        )

        if value is not _MISSING and value is not None:
            return value

    return default


def normalize_side_name(
    side_value: Any,
) -> str:
    """
    Normalize simulator side representations to Player or Opponent.
    """

    if side_value is None:
        return "Player"

    if hasattr(side_value, "value"):
        side_value = side_value.value

    normalized = str(
        side_value
    ).strip().lower()

    if normalized in {
        "opponent",
        "enemy",
        "second",
        "player_2",
        "player2",
        "p2",
        "1",
    }:
        return "Opponent"

    return "Player"


def to_numeric_value(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Convert simulator values to finite numeric values.
    """

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return float(default)

    if not np.isfinite(numeric_value):
        return float(default)

    return numeric_value


def extract_card_name(
    pokemon_state: Any,
    default: str = "Unknown",
) -> str:
    """
    Extract a card name from a PokemonState-like object or dictionary.
    """

    if pokemon_state is None:
        return default

    direct_name = read_first(
        pokemon_state,
        [
            "name",
            "card_name",
            "pokemon_name",
        ],
        default=None,
    )

    if direct_name is not None:
        return str(direct_name)

    card = read_first(
        pokemon_state,
        [
            "card",
            "card_record",
            "metadata",
        ],
        default=None,
    )

    card_name = read_first(
        card,
        [
            "name",
            "Name",
            "card_name",
            "Card Name",
            "pokemon_name",
        ],
        default=default,
    )

    return str(card_name)


def extract_attached_energy(
    pokemon_state: Any,
) -> float:
    """
    Extract attached Energy from a PokemonState-like value.
    """

    value = read_first(
        pokemon_state,
        [
            "attached_energy",
            "energy",
            "energy_count",
            "energies",
        ],
        default=0,
    )

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
            dict,
        ),
    ):
        return float(len(value))

    return to_numeric_value(
        value,
        default=0.0,
    )


def extract_damage(
    pokemon_state: Any,
) -> float:
    """
    Extract damage already placed on a Pokémon.
    """

    direct_damage = read_first(
        pokemon_state,
        [
            "damage",
            "damage_taken",
            "damage_counters",
        ],
        default=None,
    )

    if direct_damage is not None:
        return to_numeric_value(
            direct_damage,
            default=0.0,
        )

    current_hp = read_first(
        pokemon_state,
        [
            "current_hp",
            "hp_remaining",
        ],
        default=None,
    )

    maximum_hp = read_first(
        pokemon_state,
        [
            "max_hp",
            "maximum_hp",
            "base_hp",
            "hp",
        ],
        default=None,
    )

    if current_hp is not None and maximum_hp is not None:

        return max(
            0.0,
            to_numeric_value(
                maximum_hp,
            )
            - to_numeric_value(
                current_hp,
            ),
        )

    return 0.0


def extract_active_pokemon(
    side_state: Any,
) -> Any:
    """
    Extract the active Pokémon from one side of the battle state.
    """

    return read_first(
        side_state,
        [
            "active",
            "active_pokemon",
            "active_card",
            "pokemon",
        ],
        default=side_state,
    )


def extract_side_state(
    battle_state: Any,
    requested_side: str,
) -> Any:
    """
    Extract Player-side or Opponent-side state data.
    """

    requested_side = normalize_side_name(
        requested_side
    )

    if requested_side == "Opponent":

        return read_first(
            battle_state,
            [
                "opponent",
                "opponent_state",
                "enemy",
                "player_2",
                "player2",
                "p2",
            ],
            default=None,
        )

    return read_first(
        battle_state,
        [
            "player",
            "player_state",
            "self_player",
            "player_1",
            "player1",
            "p1",
        ],
        default=None,
    )


def get_live_legal_moves(
    battle_state: Any,
    legal_moves: Optional[Sequence[Any]] = None,
) -> List[Any]:
    """
    Resolve legal moves from an explicit argument or simulator state.
    """

    if legal_moves is not None:
        return list(legal_moves)

    state_legal_moves = read_first(
        battle_state,
        [
            "legal_moves",
            "available_moves",
            "current_legal_moves",
        ],
        default=None,
    )

    if state_legal_moves is not None:
        return list(state_legal_moves)

    for function_name in [
        "get_current_legal_moves",
        "get_legal_moves",
    ]:

        candidate_function = globals().get(
            function_name
        )

        if callable(candidate_function):

            try:
                resolved_moves = candidate_function(
                    battle_state
                )

                if resolved_moves is not None:
                    return list(resolved_moves)

            except Exception:
                pass

    return []


def extract_prize_cards_remaining(
    battle_state: Any,
    current_side: str,
    current_side_state: Any,
) -> float:
    """
    Extract prize cards remaining for the side whose turn it is.
    """

    direct_value = read_first(
        current_side_state,
        [
            "prize_cards_remaining",
            "prizes_remaining",
            "prize_count",
            "prizes",
        ],
        default=None,
    )

    if isinstance(
        direct_value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return float(len(direct_value))

    if direct_value is not None:
        return to_numeric_value(
            direct_value,
            default=0.0,
        )

    side_prefix = (
        "opponent"
        if current_side == "Opponent"
        else "player"
    )

    battle_value = read_first(
        battle_state,
        [
            f"{side_prefix}_prize_cards_remaining",
            f"{side_prefix}_prizes_remaining",
            "prize_cards_remaining",
            "prizes_remaining",
        ],
        default=0,
    )

    return to_numeric_value(
        battle_value,
        default=0.0,
    )


def extract_hand_size(
    battle_state: Any,
    current_side: str,
    current_side_state: Any,
) -> float:
    """
    Extract hand size for the side whose turn it is.
    """

    hand_value = read_first(
        current_side_state,
        [
            "hand_size",
            "cards_in_hand",
            "hand",
        ],
        default=None,
    )

    if isinstance(
        hand_value,
        (
            list,
            tuple,
            set,
            dict,
        ),
    ):
        return float(len(hand_value))

    if hand_value is not None:
        return to_numeric_value(
            hand_value,
            default=0.0,
        )

    side_prefix = (
        "opponent"
        if current_side == "Opponent"
        else "player"
    )

    battle_value = read_first(
        battle_state,
        [
            f"{side_prefix}_hand_size",
            "hand_size",
        ],
        default=0,
    )

    return to_numeric_value(
        battle_value,
        default=0.0,
    )


def battle_state_to_policy_features(
    battle_state: Any,
    legal_moves: Optional[Sequence[Any]] = None,
    side_mode: str = "PRESERVE",
) -> pd.DataFrame:
    """
    Convert one live simulator battle state into a one-row policy dataframe.
    """

    current_side = normalize_side_name(
        read_first(
            battle_state,
            [
                "current_player",
                "current_side",
                "active_side",
                "turn_player",
            ],
            default="Player",
        )
    )

    player_state = extract_side_state(
        battle_state,
        "Player",
    )

    opponent_state = extract_side_state(
        battle_state,
        "Opponent",
    )

    player_active = extract_active_pokemon(
        player_state
    )

    opponent_active = extract_active_pokemon(
        opponent_state
    )

    active_side_state = (
        opponent_state
        if current_side == "Opponent"
        else player_state
    )

    resolved_legal_moves = get_live_legal_moves(
        battle_state,
        legal_moves=legal_moves,
    )

    feature_record = {
        "current_side": current_side,
        "side_mode": str(
            side_mode
        ).strip().upper(),
        "player_card": extract_card_name(
            player_active
        ),
        "opponent_card": extract_card_name(
            opponent_active
        ),
        "turn_number": to_numeric_value(
            read_first(
                battle_state,
                [
                    "turn_number",
                    "turn",
                    "current_turn",
                ],
                default=0,
            ),
            default=0.0,
        ),
        "player_energy": extract_attached_energy(
            player_active
        ),
        "opponent_energy": extract_attached_energy(
            opponent_active
        ),
        "player_damage": extract_damage(
            player_active
        ),
        "opponent_damage": extract_damage(
            opponent_active
        ),
        "prize_cards_remaining":
            extract_prize_cards_remaining(
                battle_state,
                current_side,
                active_side_state,
            ),
        "hand_size": extract_hand_size(
            battle_state,
            current_side,
            active_side_state,
        ),
        "legal_move_count": float(
            len(resolved_legal_moves)
        ),
    }

    feature_dataframe = pd.DataFrame(
        [
            feature_record
        ]
    )

    missing_features = [
        feature_name
        for feature_name in POLICY_RAW_FEATURE_COLUMNS
        if feature_name not in feature_dataframe.columns
    ]

    assert not missing_features, (
        "Policy feature extraction missed required columns: "
        f"{missing_features}"
    )

    return feature_dataframe[
        POLICY_RAW_FEATURE_COLUMNS
    ].copy()


print()
print("EXTRACTOR STATUS")
print("-" * 100)

print(
    "battle_state_to_policy_features() created successfully."
)

print(
    f"Raw feature count       : "
    f"{len(POLICY_RAW_FEATURE_COLUMNS)}"
)

assert (
    "variant_name"
    not in POLICY_RAW_FEATURE_COLUMNS
), "The deployed policy should not depend on variant_name."

assert len(
    POLICY_RAW_FEATURE_COLUMNS
) == 12

print()
print("✅ SECTION 3A LIVE BATTLE-STATE FEATURE EXTRACTOR PASSED")


# ## Section 3B — End-to-End State Extraction Validation
# 
# #### This section reconstructs a representative simulator state from a validated Notebook 49 policy example.
# 
# ## It verifies the complete deployment path:
# 
# #### **policy example → reconstructed battle state → extracted raw features → preprocessing → policy prediction**
# 
# ## The extracted feature values must match the original policy dataset row.

# In[8]:


# ======================================================================================
# SECTION 3B — END-TO-END STATE EXTRACTION VALIDATION
# ======================================================================================

print("=" * 100)
print("SECTION 3B — END-TO-END STATE EXTRACTION VALIDATION")
print("=" * 100)


# --------------------------------------------------------------------------------------
# Select a representative example
#
# Prefer a state with at least two legal moves so the test includes a real
# policy decision rather than only a forced move.
# --------------------------------------------------------------------------------------

decision_examples_df = policy_dataset[
    pd.to_numeric(
        policy_dataset["legal_move_count"],
        errors="coerce",
    ) >= 2
].copy()

if not decision_examples_df.empty:
    validation_source_row = (
        decision_examples_df
        .sample(
            n=1,
            random_state=RANDOM_SEED,
        )
        .iloc[0]
    )
else:
    validation_source_row = (
        policy_dataset
        .sample(
            n=1,
            random_state=RANDOM_SEED,
        )
        .iloc[0]
    )


# --------------------------------------------------------------------------------------
# Build placeholder legal moves
#
# The extractor only needs their count in this subsection. Section 4 will
# integrate actual simulator move dictionaries.
# --------------------------------------------------------------------------------------

expected_legal_move_count = int(
    validation_source_row[
        "legal_move_count"
    ]
)

reconstructed_legal_moves = [
    {
        "name": f"Legal Move {move_number + 1}",
        "damage": 0.0,
        "energy_cost": 0,
    }
    for move_number in range(
        expected_legal_move_count
    )
]


# --------------------------------------------------------------------------------------
# Reconstruct a dictionary-based battle state
# --------------------------------------------------------------------------------------

reconstructed_battle_state = {
    "current_player":
        validation_source_row["current_side"],

    "turn_number":
        int(
            validation_source_row[
                "turn_number"
            ]
        ),

    "player": {
        "active": {
            "card": {
                "Card Name":
                    validation_source_row[
                        "player_card"
                    ],
            },
            "attached_energy":
                float(
                    validation_source_row[
                        "player_energy"
                    ]
                ),
            "damage":
                float(
                    validation_source_row[
                        "player_damage"
                    ]
                ),
        },
        "prize_cards_remaining":
            float(
                validation_source_row[
                    "prize_cards_remaining"
                ]
            ),
        "hand_size":
            float(
                validation_source_row[
                    "hand_size"
                ]
            ),
    },

    "opponent": {
        "active": {
            "card": {
                "Card Name":
                    validation_source_row[
                        "opponent_card"
                    ],
            },
            "attached_energy":
                float(
                    validation_source_row[
                        "opponent_energy"
                    ]
                ),
            "damage":
                float(
                    validation_source_row[
                        "opponent_damage"
                    ]
                ),
        },
        "prize_cards_remaining":
            float(
                validation_source_row[
                    "prize_cards_remaining"
                ]
            ),
        "hand_size":
            float(
                validation_source_row[
                    "hand_size"
                ]
            ),
    },

    "legal_moves":
        reconstructed_legal_moves,
}


# --------------------------------------------------------------------------------------
# Extract deployable policy features
# --------------------------------------------------------------------------------------

extracted_policy_features_df = (
    battle_state_to_policy_features(
        reconstructed_battle_state,
        legal_moves=reconstructed_legal_moves,
        side_mode=validation_source_row[
            "side_mode"
        ],
    )
)

expected_policy_features_df = pd.DataFrame(
    [
        {
            feature_name:
                validation_source_row[
                    feature_name
                ]
            for feature_name in (
                POLICY_RAW_FEATURE_COLUMNS
            )
        }
    ]
)


print()
print("SOURCE EXAMPLE")
print("-" * 100)

print(
    f"Variant ID       : "
    f"{validation_source_row.get('variant_id', 'Unknown')}"
)

print(
    f"Current side     : "
    f"{validation_source_row['current_side']}"
)

print(
    f"Player card      : "
    f"{validation_source_row['player_card']}"
)

print(
    f"Opponent card    : "
    f"{validation_source_row['opponent_card']}"
)

print(
    f"Legal moves      : "
    f"{expected_legal_move_count}"
)

print(
    f"Expert move      : "
    f"{validation_source_row['expert_move']}"
)


print()
print("EXTRACTED POLICY FEATURES")
print("-" * 100)

display(
    extracted_policy_features_df
)


# --------------------------------------------------------------------------------------
# Compare categorical features
# --------------------------------------------------------------------------------------

categorical_validation_columns = [
    "current_side",
    "side_mode",
    "player_card",
    "opponent_card",
]

numeric_validation_columns = [
    column_name
    for column_name in POLICY_RAW_FEATURE_COLUMNS
    if column_name
    not in categorical_validation_columns
]

comparison_rows = []

for column_name in (
    categorical_validation_columns
):

    expected_value = str(
        expected_policy_features_df.loc[
            0,
            column_name,
        ]
    )

    extracted_value = str(
        extracted_policy_features_df.loc[
            0,
            column_name,
        ]
    )

    comparison_rows.append(
        {
            "feature": column_name,
            "expected_value": expected_value,
            "extracted_value": extracted_value,
            "match": (
                expected_value
                == extracted_value
            ),
        }
    )


# --------------------------------------------------------------------------------------
# Compare numeric features
# --------------------------------------------------------------------------------------

for column_name in numeric_validation_columns:

    expected_value = float(
        expected_policy_features_df.loc[
            0,
            column_name,
        ]
    )

    extracted_value = float(
        extracted_policy_features_df.loc[
            0,
            column_name,
        ]
    )

    comparison_rows.append(
        {
            "feature": column_name,
            "expected_value": expected_value,
            "extracted_value": extracted_value,
            "match": bool(
                np.isclose(
                    expected_value,
                    extracted_value,
                )
            ),
        }
    )


feature_comparison_df = pd.DataFrame(
    comparison_rows
)

print()
print("FEATURE COMPARISON")
print("-" * 100)

display(
    feature_comparison_df
)

assert feature_comparison_df[
    "match"
].all(), (
    "The reconstructed battle-state features do not match "
    "the original policy example."
)


# --------------------------------------------------------------------------------------
# End-to-end policy prediction
# --------------------------------------------------------------------------------------

integration_prediction = predict_policy_move(
    extracted_policy_features_df
)

expected_expert_move = str(
    validation_source_row[
        "expert_move"
    ]
)

predicted_policy_move = str(
    integration_prediction[
        "predicted_move"
    ]
)

prediction_match = (
    expected_expert_move
    == predicted_policy_move
)

print()
print("END-TO-END POLICY PREDICTION")
print("-" * 100)

print(
    f"Expected expert move : "
    f"{expected_expert_move}"
)

print(
    f"Predicted policy move: "
    f"{predicted_policy_move}"
)

print(
    f"Prediction confidence: "
    f"{integration_prediction['confidence']:.4f}"
)

print(
    f"Prediction match     : "
    f"{prediction_match}"
)

print()
print("POLICY PROBABILITIES")
print("-" * 100)

probability_profile_df = (
    pd.DataFrame(
        [
            {
                "policy_move": move_name,
                "probability": probability,
            }
            for move_name, probability
            in integration_prediction[
                "probabilities"
            ].items()
        ]
    )
    .sort_values(
        "probability",
        ascending=False,
    )
    .reset_index(drop=True)
)

display(
    probability_profile_df
)


# --------------------------------------------------------------------------------------
# Assertions
# --------------------------------------------------------------------------------------

assert list(
    extracted_policy_features_df.columns
) == POLICY_RAW_FEATURE_COLUMNS

assert extracted_policy_features_df.shape == (
    1,
    len(POLICY_RAW_FEATURE_COLUMNS),
)

assert prediction_match, (
    "The reconstructed battle state did not reproduce "
    "the expected expert move."
)

assert np.isclose(
    probability_profile_df[
        "probability"
    ].sum(),
    1.0,
)


print()
print("✅ SECTION 3B END-TO-END STATE EXTRACTION VALIDATION PASSED")


# # Section 4 — Legal-Action Policy Integration
# 
# ## Section 4A — Legal-Move Masked Policy Selection
# 
# #### This section converts policy probabilities into a simulator-safe action.
# 
# #### The policy agent must never return an unavailable move. Therefore, predictions are filtered against the legal moves supplied by the battle engine.
# 
# ##### If the highest-probability policy action is illegal, the selector chooses the highest-probability legal alternative. If no trained action name matches, it falls back safely to the first legal move.

# In[9]:


# ======================================================================================
# SECTION 4A — LEGAL-MOVE MASKED POLICY SELECTION
# ======================================================================================

print("=" * 100)
print("SECTION 4A — LEGAL-MOVE MASKED POLICY SELECTION")
print("=" * 100)


def normalize_move_name(
    move_value: Any,
) -> str:
    """
    Normalize a move name for reliable matching.
    """

    if move_value is None:
        return ""

    if isinstance(move_value, Mapping):
        move_value = read_first(
            move_value,
            [
                "name",
                "move_name",
                "Move Name",
                "action",
            ],
            default="",
        )

    elif not isinstance(move_value, str):
        move_value = read_first(
            move_value,
            [
                "name",
                "move_name",
                "action",
            ],
            default=move_value,
        )

    return " ".join(
        str(move_value)
        .strip()
        .lower()
        .split()
    )


def move_display_name(
    move_value: Any,
) -> str:
    """
    Extract a human-readable move name while preserving capitalization.
    """

    if isinstance(move_value, Mapping):
        return str(
            read_first(
                move_value,
                [
                    "name",
                    "move_name",
                    "Move Name",
                    "action",
                ],
                default="Unknown Move",
            )
        )

    if isinstance(move_value, str):
        return move_value

    return str(
        read_first(
            move_value,
            [
                "name",
                "move_name",
                "action",
            ],
            default="Unknown Move",
        )
    )


def build_legal_move_lookup(
    legal_moves: Sequence[Any],
) -> Dict[str, Any]:
    """
    Map normalized move names to their original simulator move objects.
    """

    legal_move_lookup: Dict[str, Any] = {}

    for legal_move in legal_moves:

        normalized_name = normalize_move_name(
            legal_move
        )

        if normalized_name:
            legal_move_lookup[
                normalized_name
            ] = legal_move

    return legal_move_lookup


def select_legal_policy_move(
    battle_state: Any,
    legal_moves: Sequence[Any],
    side_mode: str = "PRESERVE",
) -> Dict[str, Any]:
    """
    Predict and return the highest-probability legal move.

    Returns a detailed record suitable for simulator logging.
    """

    resolved_legal_moves = list(
        legal_moves
    )

    if not resolved_legal_moves:
        raise ValueError(
            "select_legal_policy_move requires at least one legal move."
        )

    policy_features = (
        battle_state_to_policy_features(
            battle_state,
            legal_moves=resolved_legal_moves,
            side_mode=side_mode,
        )
    )

    prediction = predict_policy_move(
        policy_features
    )

    legal_move_lookup = build_legal_move_lookup(
        resolved_legal_moves
    )

    probability_rows = []

    for policy_move, probability in (
        prediction["probabilities"].items()
    ):

        normalized_policy_move = (
            normalize_move_name(
                policy_move
            )
        )

        is_legal = (
            normalized_policy_move
            in legal_move_lookup
        )

        probability_rows.append(
            {
                "policy_move": str(policy_move),
                "normalized_move":
                    normalized_policy_move,
                "probability": float(
                    probability
                ),
                "is_legal": bool(is_legal),
            }
        )

    masked_probability_df = pd.DataFrame(
        probability_rows
    )

    legal_probability_df = (
        masked_probability_df[
            masked_probability_df["is_legal"]
        ]
        .copy()
    )

    fallback_used = False
    fallback_reason = None

    if not legal_probability_df.empty:

        legal_probability_total = float(
            legal_probability_df[
                "probability"
            ].sum()
        )

        if legal_probability_total > 0:

            legal_probability_df[
                "masked_probability"
            ] = (
                legal_probability_df[
                    "probability"
                ]
                / legal_probability_total
            )

            selected_probability_row = (
                legal_probability_df
                .sort_values(
                    [
                        "masked_probability",
                        "policy_move",
                    ],
                    ascending=[
                        False,
                        True,
                    ],
                )
                .iloc[0]
            )

        else:

            fallback_used = True
            fallback_reason = (
                "All legal learned actions had zero probability."
            )

            selected_probability_row = (
                legal_probability_df
                .sort_values(
                    "policy_move"
                )
                .iloc[0]
            )

            legal_probability_df[
                "masked_probability"
            ] = 0.0

        selected_normalized_name = str(
            selected_probability_row[
                "normalized_move"
            ]
        )

        selected_move = legal_move_lookup[
            selected_normalized_name
        ]

        selected_policy_name = str(
            selected_probability_row[
                "policy_move"
            ]
        )

        selected_probability = float(
            selected_probability_row[
                "probability"
            ]
        )

        selected_masked_probability = float(
            selected_probability_row.get(
                "masked_probability",
                0.0,
            )
        )

    else:

        fallback_used = True
        fallback_reason = (
            "No legal move name matched a trained policy class."
        )

        selected_move = resolved_legal_moves[0]
        selected_policy_name = move_display_name(
            selected_move
        )
        selected_probability = 0.0
        selected_masked_probability = 1.0

    return {
        "selected_move": selected_move,
        "selected_move_name": move_display_name(
            selected_move
        ),
        "selected_policy_name":
            selected_policy_name,
        "raw_predicted_move":
            prediction["predicted_move"],
        "raw_confidence":
            prediction["confidence"],
        "selected_probability":
            selected_probability,
        "selected_masked_probability":
            selected_masked_probability,
        "fallback_used":
            fallback_used,
        "fallback_reason":
            fallback_reason,
        "policy_features":
            policy_features,
        "raw_probabilities":
            prediction["probabilities"],
        "masked_probability_table":
            masked_probability_df,
    }


print()
print("SELECTOR STATUS")
print("-" * 100)

print(
    "select_legal_policy_move() created successfully."
)

print(
    f"Supported trained actions: "
    f"{list(label_encoder.classes_)}"
)

print()
print(
    "✅ SECTION 4A LEGAL-MOVE MASKED POLICY SELECTION PASSED"
)


# # Section 4B — Legal-Move Selector Validation
# 
# #### This section validates the deployment wrapper under several realistic simulator situations.
# 
# #### The selector must:
# 
# - choose the highest-probability legal move,
# - ignore illegal policy predictions,
# - renormalize probabilities over legal moves,
# - safely fall back when no learned move names match.

# In[12]:


# =====================================================================================
# SECTION 4B — LEGAL-MOVE SELECTOR VALIDATION
# =====================================================================================

print("=" * 100)
print("SECTION 4B — LEGAL-MOVE SELECTOR VALIDATION")
print("=" * 100)

# -------------------------------------------------------------------------------------
# Reuse the reconstructed battle state from Section 3B
# -------------------------------------------------------------------------------------

validation_state = reconstructed_battle_state

# =====================================================================================
# TEST 1
# Predicted move is legal
# =====================================================================================

print()
print("TEST 1 — Predicted move is legal")
print("-" * 100)

legal_moves_case1 = [
    {"name": "Quick Attack"},
    {"name": "Ascension"},
]

result_case1 = select_legal_policy_move(
    validation_state,
    legal_moves_case1,
    side_mode="PRESERVE",
)

print("Selected move :", result_case1["selected_move_name"])
print("Fallback used :", result_case1["fallback_used"])

assert result_case1["selected_move_name"] == "Quick Attack"
assert result_case1["fallback_used"] is False

print("✓ Passed")


# =====================================================================================
# TEST 2
# Highest probability move is illegal
# =====================================================================================

print()
print("TEST 2 — Highest prediction illegal")
print("-" * 100)

legal_moves_case2 = [
    {"name": "Ascension"},
    {"name": "Bind Down"},
]

result_case2 = select_legal_policy_move(
    validation_state,
    legal_moves_case2,
    side_mode="PRESERVE",
)

print("Raw prediction :", result_case2["raw_predicted_move"])
print("Selected move  :", result_case2["selected_move_name"])
print("Fallback used  :", result_case2["fallback_used"])

assert result_case2["selected_move_name"] in [
    "Ascension",
    "Bind Down",
]

assert result_case2["fallback_used"] is True

print("✓ Passed")


# =====================================================================================
# TEST 3
# No learned move names match
# =====================================================================================

print()
print("TEST 3 — No learned move names match")
print("-" * 100)

legal_moves_case3 = [
    {"name": "Scratch"},
    {"name": "Growl"},
    {"name": "Tail Whip"},
]

result_case3 = select_legal_policy_move(
    validation_state,
    legal_moves_case3,
    side_mode="PRESERVE",
)

print("Selected move :", result_case3["selected_move_name"])
print("Fallback used :", result_case3["fallback_used"])
print("Reason        :", result_case3["fallback_reason"])

assert result_case3["fallback_used"] is True
assert result_case3["selected_move_name"] == "Scratch"

print("✓ Passed")


# =====================================================================================
# SUMMARY
# =====================================================================================

summary_df = pd.DataFrame(
    [
        {
            "test": "Prediction legal",
            "selected_move": result_case1["selected_move_name"],
            "fallback": result_case1["fallback_used"],
        },
        {
            "test": "Prediction illegal",
            "selected_move": result_case2["selected_move_name"],
            "fallback": result_case2["fallback_used"],
        },
        {
            "test": "No matching move",
            "selected_move": result_case3["selected_move_name"],
            "fallback": result_case3["fallback_used"],
        },
    ]
)

print()
print("VALIDATION SUMMARY")
print("-" * 100)

display(summary_df)

assert summary_df.shape[0] == 3

print()
print("✅ SECTION 4B LEGAL-MOVE SELECTOR VALIDATION PASSED")


# # Section 5 — Simulator Policy Agent
# 
# ## Section 5A — Policy Simulator Adapter
# 
# * This section packages the trained policy, feature extractor, and legal-action selector into a reusable agent.
# 
# * The adapter accepts a live battle state and a collection of legal moves, then returns a simulator-compatible move.
# 
# * It also records decision metadata for debugging, evaluation, and later tournament analysis.

# In[13]:


# ======================================================================================
# SECTION 5A — POLICY SIMULATOR ADAPTER
# ======================================================================================

print("=" * 100)
print("SECTION 5A — POLICY SIMULATOR ADAPTER")
print("=" * 100)


@dataclass
class PolicyDecision:
    """
    Structured record describing one policy-agent decision.
    """

    selected_move: Any
    selected_move_name: str
    raw_predicted_move: str
    confidence: float
    selected_probability: float
    selected_masked_probability: float
    fallback_used: bool
    fallback_reason: Optional[str]
    current_side: str
    turn_number: float
    legal_move_count: int
    policy_features: Dict[str, Any]
    raw_probabilities: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the decision to a JSON-friendly dictionary.
        """

        return {
            "selected_move_name":
                self.selected_move_name,

            "raw_predicted_move":
                self.raw_predicted_move,

            "confidence":
                float(self.confidence),

            "selected_probability":
                float(self.selected_probability),

            "selected_masked_probability":
                float(
                    self.selected_masked_probability
                ),

            "fallback_used":
                bool(self.fallback_used),

            "fallback_reason":
                self.fallback_reason,

            "current_side":
                self.current_side,

            "turn_number":
                float(self.turn_number),

            "legal_move_count":
                int(self.legal_move_count),

            "policy_features":
                self.policy_features,

            "raw_probabilities":
                {
                    str(move_name): float(probability)
                    for move_name, probability
                    in self.raw_probabilities.items()
                },
        }


class PolicyAgent:
    """
    Simulator-facing adapter for the Notebook 50 policy.

    Parameters
    ----------
    name : str
        Agent display name.

    side_mode : str
        Side-mode value supplied to the feature extractor.

    retain_history : bool
        Whether to retain decision records in memory.
    """

    def __init__(
        self,
        name: str = "Notebook50PolicyAgent",
        side_mode: str = "PRESERVE",
        retain_history: bool = True,
    ) -> None:

        self.name = str(name)

        self.side_mode = str(
            side_mode
        ).strip().upper()

        self.retain_history = bool(
            retain_history
        )

        self.decision_history: List[
            Dict[str, Any]
        ] = []

        self.total_decisions = 0
        self.fallback_decisions = 0

    def reset(
        self,
    ) -> None:
        """
        Reset counters and decision history.
        """

        self.decision_history = []
        self.total_decisions = 0
        self.fallback_decisions = 0

    def resolve_legal_moves(
        self,
        battle_state: Any,
        legal_moves: Optional[
            Sequence[Any]
        ] = None,
    ) -> List[Any]:
        """
        Resolve legal moves from an explicit argument or state.
        """

        resolved_moves = get_live_legal_moves(
            battle_state,
            legal_moves=legal_moves,
        )

        return list(
            resolved_moves
        )

    def decide(
        self,
        battle_state: Any,
        legal_moves: Optional[
            Sequence[Any]
        ] = None,
    ) -> PolicyDecision:
        """
        Produce one detailed legal policy decision.
        """

        resolved_legal_moves = (
            self.resolve_legal_moves(
                battle_state,
                legal_moves=legal_moves,
            )
        )

        if not resolved_legal_moves:
            raise ValueError(
                f"{self.name} received no legal moves."
            )

        selection = select_legal_policy_move(
            battle_state=battle_state,
            legal_moves=resolved_legal_moves,
            side_mode=self.side_mode,
        )

        feature_row = (
            selection[
                "policy_features"
            ]
            .iloc[0]
            .to_dict()
        )

        decision = PolicyDecision(
            selected_move=
                selection["selected_move"],

            selected_move_name=
                selection[
                    "selected_move_name"
                ],

            raw_predicted_move=
                selection[
                    "raw_predicted_move"
                ],

            confidence=float(
                selection[
                    "raw_confidence"
                ]
            ),

            selected_probability=float(
                selection[
                    "selected_probability"
                ]
            ),

            selected_masked_probability=float(
                selection[
                    "selected_masked_probability"
                ]
            ),

            fallback_used=bool(
                selection[
                    "fallback_used"
                ]
            ),

            fallback_reason=
                selection[
                    "fallback_reason"
                ],

            current_side=str(
                feature_row[
                    "current_side"
                ]
            ),

            turn_number=float(
                feature_row[
                    "turn_number"
                ]
            ),

            legal_move_count=int(
                feature_row[
                    "legal_move_count"
                ]
            ),

            policy_features=
                feature_row,

            raw_probabilities=dict(
                selection[
                    "raw_probabilities"
                ]
            ),
        )

        self.total_decisions += 1

        if decision.fallback_used:
            self.fallback_decisions += 1

        if self.retain_history:
            self.decision_history.append(
                decision.to_dict()
            )

        return decision

    def choose_move(
        self,
        battle_state: Any,
        legal_moves: Optional[
            Sequence[Any]
        ] = None,
    ) -> Any:
        """
        Return only the simulator-compatible move object.
        """

        decision = self.decide(
            battle_state=battle_state,
            legal_moves=legal_moves,
        )

        return decision.selected_move

    def select_action(
        self,
        battle_state: Any,
        legal_moves: Optional[
            Sequence[Any]
        ] = None,
    ) -> Any:
        """
        Alias commonly used by simulator interfaces.
        """

        return self.choose_move(
            battle_state=battle_state,
            legal_moves=legal_moves,
        )

    def act(
        self,
        battle_state: Any,
        legal_moves: Optional[
            Sequence[Any]
        ] = None,
    ) -> Any:
        """
        Additional compatibility alias.
        """

        return self.choose_move(
            battle_state=battle_state,
            legal_moves=legal_moves,
        )

    def get_statistics(
        self,
    ) -> Dict[str, Any]:
        """
        Return current adapter statistics.
        """

        fallback_rate = (
            self.fallback_decisions
            / self.total_decisions
            if self.total_decisions > 0
            else 0.0
        )

        return {
            "agent_name":
                self.name,

            "side_mode":
                self.side_mode,

            "total_decisions":
                int(
                    self.total_decisions
                ),

            "fallback_decisions":
                int(
                    self.fallback_decisions
                ),

            "fallback_rate":
                float(
                    fallback_rate
                ),

            "history_rows":
                int(
                    len(
                        self.decision_history
                    )
                ),
        }

    def history_dataframe(
        self,
    ) -> pd.DataFrame:
        """
        Return recorded decisions as a dataframe.
        """

        return pd.DataFrame(
            self.decision_history
        )


# --------------------------------------------------------------------------------------
# Instantiate the integration agent
# --------------------------------------------------------------------------------------

policy_agent = PolicyAgent(
    name="Notebook50SideBalancedPolicy",
    side_mode="PRESERVE",
    retain_history=True,
)


print()
print("POLICY AGENT PROFILE")
print("-" * 100)

print(
    f"Agent name       : "
    f"{policy_agent.name}"
)

print(
    f"Side mode        : "
    f"{policy_agent.side_mode}"
)

print(
    f"Retain history   : "
    f"{policy_agent.retain_history}"
)

print()
print("SUPPORTED ENTRY POINTS")
print("-" * 100)

print("policy_agent.decide(...)")
print("policy_agent.choose_move(...)")
print("policy_agent.select_action(...)")
print("policy_agent.act(...)")

assert callable(
    policy_agent.decide
)

assert callable(
    policy_agent.choose_move
)

assert callable(
    policy_agent.select_action
)

assert callable(
    policy_agent.act
)

assert policy_agent.total_decisions == 0
assert policy_agent.fallback_decisions == 0

print()
print("✅ SECTION 5A POLICY SIMULATOR ADAPTER PASSED")


# ## Section 5B — Policy Agent Integration Validation
# 
# #### This section validates the complete simulator-facing agent.
# 
# #### It checks:
# 
# - detailed decisions through `decide()`,
# - simulator-compatible moves through `choose_move()`,
# - compatibility aliases through `select_action()` and `act()`,
# - legal-action enforcement,
# - fallback behavior,
# - decision-history recording,
# - statistics tracking,
# - JSON-safe decision logging,
# - agent reset behavior.

# In[14]:


# ======================================================================================
# SECTION 5B — POLICY AGENT INTEGRATION VALIDATION
# ======================================================================================

print("=" * 100)
print("SECTION 5B — POLICY AGENT INTEGRATION VALIDATION")
print("=" * 100)


# --------------------------------------------------------------------------------------
# Reset the integration agent before validation
# --------------------------------------------------------------------------------------

policy_agent.reset()

assert policy_agent.total_decisions == 0
assert policy_agent.fallback_decisions == 0
assert len(policy_agent.decision_history) == 0


# --------------------------------------------------------------------------------------
# Validation move sets
# --------------------------------------------------------------------------------------

known_legal_moves = [
    {
        "name": "Quick Attack",
        "damage": 20.0,
        "energy_cost": 3,
    },
    {
        "name": "Ascension",
        "damage": 0.0,
        "energy_cost": 1,
    },
]

unknown_legal_moves = [
    {
        "name": "Scratch",
        "damage": 10.0,
        "energy_cost": 1,
    },
    {
        "name": "Growl",
        "damage": 0.0,
        "energy_cost": 0,
    },
]


# ======================================================================================
# TEST 1 — Detailed decision
# ======================================================================================

print()
print("TEST 1 — DETAILED POLICY DECISION")
print("-" * 100)

detailed_decision = policy_agent.decide(
    battle_state=reconstructed_battle_state,
    legal_moves=known_legal_moves,
)

print(
    f"Selected move           : "
    f"{detailed_decision.selected_move_name}"
)

print(
    f"Raw policy prediction   : "
    f"{detailed_decision.raw_predicted_move}"
)

print(
    f"Confidence              : "
    f"{detailed_decision.confidence:.4f}"
)

print(
    f"Fallback used           : "
    f"{detailed_decision.fallback_used}"
)

print(
    f"Legal move count        : "
    f"{detailed_decision.legal_move_count}"
)

assert isinstance(
    detailed_decision,
    PolicyDecision,
)

assert detailed_decision.selected_move in (
    known_legal_moves
)

assert detailed_decision.selected_move_name in {
    "Quick Attack",
    "Ascension",
}

assert detailed_decision.legal_move_count == 2

assert 0.0 <= detailed_decision.confidence <= 1.0

assert isinstance(
    detailed_decision.policy_features,
    dict,
)

assert set(
    POLICY_RAW_FEATURE_COLUMNS
).issubset(
    detailed_decision.policy_features.keys()
)

print("✓ Passed")


# ======================================================================================
# TEST 2 — choose_move() returns the simulator move
# ======================================================================================

print()
print("TEST 2 — SIMULATOR MOVE RETURN")
print("-" * 100)

chosen_move = policy_agent.choose_move(
    battle_state=reconstructed_battle_state,
    legal_moves=known_legal_moves,
)

print(
    f"Returned object         : "
    f"{chosen_move}"
)

print(
    f"Returned move name      : "
    f"{move_display_name(chosen_move)}"
)

assert chosen_move in known_legal_moves

assert move_display_name(
    chosen_move
) in {
    "Quick Attack",
    "Ascension",
}

print("✓ Passed")


# ======================================================================================
# TEST 3 — select_action() compatibility alias
# ======================================================================================

print()
print("TEST 3 — SELECT_ACTION COMPATIBILITY")
print("-" * 100)

selected_action = policy_agent.select_action(
    battle_state=reconstructed_battle_state,
    legal_moves=known_legal_moves,
)

print(
    f"Selected action         : "
    f"{move_display_name(selected_action)}"
)

assert selected_action in known_legal_moves

print("✓ Passed")


# ======================================================================================
# TEST 4 — act() and unmatched-action fallback
# ======================================================================================

print()
print("TEST 4 — ACT ALIAS AND FALLBACK")
print("-" * 100)

fallback_action = policy_agent.act(
    battle_state=reconstructed_battle_state,
    legal_moves=unknown_legal_moves,
)

print(
    f"Fallback action         : "
    f"{move_display_name(fallback_action)}"
)

assert fallback_action == unknown_legal_moves[0]

assert (
    move_display_name(
        fallback_action
    )
    == "Scratch"
)

print("✓ Passed")


# ======================================================================================
# TEST 5 — Decision history
# ======================================================================================

print()
print("TEST 5 — DECISION HISTORY")
print("-" * 100)

history_df = policy_agent.history_dataframe()

history_preview_columns = [
    column_name
    for column_name in [
        "selected_move_name",
        "raw_predicted_move",
        "confidence",
        "fallback_used",
        "fallback_reason",
        "current_side",
        "turn_number",
        "legal_move_count",
    ]
    if column_name in history_df.columns
]

display(
    history_df[
        history_preview_columns
    ]
)

assert len(history_df) == 4

assert history_df[
    "selected_move_name"
].notna().all()

assert history_df[
    "raw_predicted_move"
].notna().all()

assert history_df[
    "fallback_used"
].astype(bool).any(), (
    "The unmatched-action fallback was not recorded."
)

print("✓ Passed")


# ======================================================================================
# TEST 6 — Agent statistics
# ======================================================================================

print()
print("TEST 6 — AGENT STATISTICS")
print("-" * 100)

agent_statistics = policy_agent.get_statistics()

for statistic_name, statistic_value in (
    agent_statistics.items()
):
    print(
        f"{statistic_name:25}: "
        f"{statistic_value}"
    )

assert (
    agent_statistics[
        "total_decisions"
    ]
    == 4
)

assert (
    agent_statistics[
        "history_rows"
    ]
    == 4
)

assert (
    agent_statistics[
        "fallback_decisions"
    ]
    >= 1
)

assert (
    0.0
    <= agent_statistics[
        "fallback_rate"
    ]
    <= 1.0
)

print("✓ Passed")


# ======================================================================================
# TEST 7 — JSON-safe decision logging
# ======================================================================================

print()
print("TEST 7 — JSON-SAFE DECISION LOG")
print("-" * 100)

decision_dictionary = (
    detailed_decision.to_dict()
)

serialized_decision = json.dumps(
    decision_dictionary,
    indent=2,
    ensure_ascii=False,
    default=str,
)

print(
    serialized_decision[
        :1200
    ]
)

assert isinstance(
    serialized_decision,
    str,
)

assert (
    decision_dictionary[
        "selected_move_name"
    ]
    == detailed_decision.selected_move_name
)

assert (
    decision_dictionary[
        "legal_move_count"
    ]
    == 2
)

print("✓ Passed")


# ======================================================================================
# TEST 8 — Reset behavior
# ======================================================================================

print()
print("TEST 8 — RESET BEHAVIOR")
print("-" * 100)

policy_agent.reset()

statistics_after_reset = (
    policy_agent.get_statistics()
)

print(
    f"Total decisions after reset    : "
    f"{statistics_after_reset['total_decisions']}"
)

print(
    f"Fallback decisions after reset : "
    f"{statistics_after_reset['fallback_decisions']}"
)

print(
    f"History rows after reset       : "
    f"{statistics_after_reset['history_rows']}"
)

assert (
    statistics_after_reset[
        "total_decisions"
    ]
    == 0
)

assert (
    statistics_after_reset[
        "fallback_decisions"
    ]
    == 0
)

assert (
    statistics_after_reset[
        "history_rows"
    ]
    == 0
)

print("✓ Passed")


# ======================================================================================
# VALIDATION SUMMARY
# ======================================================================================

section5b_validation_df = pd.DataFrame(
    [
        {
            "validation_test":
                "Detailed decision",
            "passed": True,
        },
        {
            "validation_test":
                "Simulator move return",
            "passed": True,
        },
        {
            "validation_test":
                "select_action compatibility",
            "passed": True,
        },
        {
            "validation_test":
                "act fallback behavior",
            "passed": True,
        },
        {
            "validation_test":
                "Decision history",
            "passed": True,
        },
        {
            "validation_test":
                "Agent statistics",
            "passed": True,
        },
        {
            "validation_test":
                "JSON-safe decision log",
            "passed": True,
        },
        {
            "validation_test":
                "Reset behavior",
            "passed": True,
        },
    ]
)

print()
print("SECTION 5B VALIDATION SUMMARY")
print("-" * 100)

display(
    section5b_validation_df
)

assert (
    section5b_validation_df[
        "passed"
    ].all()
)

print()
print("✅ SECTION 5B POLICY AGENT INTEGRATION VALIDATION PASSED")


# # Section 6 — Live Simulator Binding
# 
# ## Section 6A — Simulator Runtime Discovery
# 
# #### Before connecting the policy agent to a live battle, this section discovers the simulator objects and callable entry points currently available.
# 
# #### The diagnostic searches:
# 
# - the active notebook namespace,
# - project Python scripts,
# - simulator-related classes and functions,
# - existing baseline agents,
# - legal-move and state-transition functions.
# 
# #### No battle is executed in this section.

# In[15]:


# ======================================================================================
# SECTION 6A — SIMULATOR RUNTIME DISCOVERY
# ======================================================================================

import inspect
import importlib.util

print("=" * 100)
print("SECTION 6A — SIMULATOR RUNTIME DISCOVERY")
print("=" * 100)


# --------------------------------------------------------------------------------------
# Search configuration
# --------------------------------------------------------------------------------------

SIMULATOR_NAME_KEYWORDS = [
    "battle",
    "simulator",
    "simulate",
    "game",
    "match",
    "episode",
    "tournament",
    "state",
    "legal",
    "move",
    "transition",
    "step",
    "agent",
    "policy",
    "random",
    "greedy",
    "search",
]

SIMULATOR_FILE_KEYWORDS = [
    "simulator",
    "battle",
    "integration",
    "search",
    "policy",
    "agent",
    "engine",
]


# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------

def safe_signature(
    callable_object: Any,
) -> str:
    """
    Return a callable signature without failing on unsupported objects.
    """

    try:
        return str(
            inspect.signature(
                callable_object
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        return "Unavailable"


def safe_source_file(
    object_value: Any,
) -> Optional[str]:
    """
    Return the source-file path associated with an object.
    """

    try:
        source_file = inspect.getsourcefile(
            object_value
        )

        if source_file is None:
            return None

        return str(
            Path(source_file).resolve()
        )

    except (
        TypeError,
        OSError,
    ):
        return None


def matches_simulator_keywords(
    name: str,
) -> List[str]:
    """
    Return simulator-related keywords found in an object name.
    """

    normalized_name = str(
        name
    ).lower()

    return [
        keyword
        for keyword in SIMULATOR_NAME_KEYWORDS
        if keyword in normalized_name
    ]


# --------------------------------------------------------------------------------------
# Search active notebook namespace
# --------------------------------------------------------------------------------------

namespace_records = []

for object_name, object_value in list(
    globals().items()
):

    matched_keywords = (
        matches_simulator_keywords(
            object_name
        )
    )

    if not matched_keywords:
        continue

    is_class = inspect.isclass(
        object_value
    )

    is_function = inspect.isfunction(
        object_value
    )

    is_method = inspect.ismethod(
        object_value
    )

    is_callable = callable(
        object_value
    )

    if is_class:
        object_type = "class"
    elif is_function:
        object_type = "function"
    elif is_method:
        object_type = "method"
    elif is_callable:
        object_type = "callable"
    else:
        object_type = type(
            object_value
        ).__name__

    namespace_records.append(
        {
            "object_name":
                object_name,

            "object_type":
                object_type,

            "callable":
                is_callable,

            "matched_keywords":
                ", ".join(
                    matched_keywords
                ),

            "signature":
                (
                    safe_signature(
                        object_value
                    )
                    if is_callable
                    else ""
                ),

            "source_file":
                safe_source_file(
                    object_value
                ),
        }
    )


namespace_discovery_df = (
    pd.DataFrame(
        namespace_records
    )
)

if not namespace_discovery_df.empty:

    namespace_discovery_df = (
        namespace_discovery_df
        .sort_values(
            [
                "callable",
                "object_type",
                "object_name",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


print()
print("CURRENT NOTEBOOK NAMESPACE")
print("-" * 100)

if namespace_discovery_df.empty:
    print(
        "No simulator-related namespace objects were found."
    )
else:
    display(
        namespace_discovery_df
    )


# --------------------------------------------------------------------------------------
# Search project Python scripts by filename
# --------------------------------------------------------------------------------------

script_file_records = []

script_search_roots = [
    SCRIPTS_DIR,
    SRC_DIR,
]

for search_root in script_search_roots:

    if not search_root.exists():
        continue

    for script_path in search_root.rglob(
        "*.py"
    ):

        normalized_filename = (
            script_path.name.lower()
        )

        matched_file_keywords = [
            keyword
            for keyword in SIMULATOR_FILE_KEYWORDS
            if keyword
            in normalized_filename
        ]

        if not matched_file_keywords:
            continue

        script_file_records.append(
            {
                "file_name":
                    script_path.name,

                "path":
                    str(
                        script_path.resolve()
                    ),

                "size_bytes":
                    script_path.stat().st_size,

                "last_modified":
                    datetime.fromtimestamp(
                        script_path.stat().st_mtime
                    ).isoformat(),

                "matched_keywords":
                    ", ".join(
                        matched_file_keywords
                    ),
            }
        )


script_discovery_df = (
    pd.DataFrame(
        script_file_records
    )
)

if not script_discovery_df.empty:

    script_discovery_df = (
        script_discovery_df
        .sort_values(
            [
                "last_modified",
                "file_name",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


print()
print("SIMULATOR-RELATED PROJECT SCRIPTS")
print("-" * 100)

if script_discovery_df.empty:
    print(
        "No simulator-related Python scripts were found."
    )
else:
    display(
        script_discovery_df.head(50)
    )


# --------------------------------------------------------------------------------------
# Identify likely binding candidates from the namespace
# --------------------------------------------------------------------------------------

likely_function_patterns = [
    "run_battle",
    "simulate_battle",
    "play_battle",
    "run_match",
    "simulate_match",
    "battle",
    "step",
    "transition",
    "apply_move",
    "execute_move",
    "get_legal_moves",
    "get_current_legal_moves",
]

likely_class_patterns = [
    "simulator",
    "battleengine",
    "battle_engine",
    "gameengine",
    "game_engine",
    "battle",
    "environment",
    "env",
]

binding_candidate_records = []

for record in namespace_records:

    normalized_object_name = (
        record[
            "object_name"
        ].lower()
    )

    binding_role = None

    if record["object_type"] in {
        "function",
        "method",
        "callable",
    }:

        if any(
            pattern
            in normalized_object_name
            for pattern
            in likely_function_patterns
        ):
            binding_role = (
                "candidate_function"
            )

    elif record["object_type"] == "class":

        if any(
            pattern
            in normalized_object_name
            for pattern
            in likely_class_patterns
        ):
            binding_role = (
                "candidate_class"
            )

    if binding_role is not None:

        binding_candidate_records.append(
            {
                **record,
                "binding_role":
                    binding_role,
            }
        )


binding_candidates_df = (
    pd.DataFrame(
        binding_candidate_records
    )
)

print()
print("LIKELY SIMULATOR BINDING CANDIDATES")
print("-" * 100)

if binding_candidates_df.empty:
    print(
        "No direct simulator binding candidates are currently loaded."
    )
else:
    display(
        binding_candidates_df
    )


# --------------------------------------------------------------------------------------
# Save diagnostic reports
# --------------------------------------------------------------------------------------

SECTION6_REPORT_DIR = (
    NOTEBOOK51_REPORT_DIR
    / "section6"
)

SECTION6_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SECTION6A_NAMESPACE_FILE = (
    SECTION6_REPORT_DIR
    / "section6a_namespace_discovery.csv"
)

SECTION6A_SCRIPT_FILE = (
    SECTION6_REPORT_DIR
    / "section6a_script_discovery.csv"
)

SECTION6A_BINDING_CANDIDATES_FILE = (
    SECTION6_REPORT_DIR
    / "section6a_binding_candidates.csv"
)

namespace_discovery_df.to_csv(
    SECTION6A_NAMESPACE_FILE,
    index=False,
)

script_discovery_df.to_csv(
    SECTION6A_SCRIPT_FILE,
    index=False,
)

binding_candidates_df.to_csv(
    SECTION6A_BINDING_CANDIDATES_FILE,
    index=False,
)


# --------------------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------------------

section6a_summary = {
    "namespace_objects_found":
        int(
            len(
                namespace_discovery_df
            )
        ),

    "project_scripts_found":
        int(
            len(
                script_discovery_df
            )
        ),

    "binding_candidates_found":
        int(
            len(
                binding_candidates_df
            )
        ),

    "policy_agent_ready":
        True,

    "next_stage":
        "SIMULATOR_INTERFACE_BINDING",
}

print()
print("DISCOVERY SUMMARY")
print("-" * 100)

for summary_name, summary_value in (
    section6a_summary.items()
):
    print(
        f"{summary_name:30}: "
        f"{summary_value}"
    )


assert (
    section6a_summary[
        "policy_agent_ready"
    ]
    is True
)

assert SECTION6A_NAMESPACE_FILE.exists()
assert SECTION6A_SCRIPT_FILE.exists()
assert (
    SECTION6A_BINDING_CANDIDATES_FILE
    .exists()
)

print()
print("SAVED SECTION 6A REPORTS")
print("-" * 100)

print(SECTION6A_NAMESPACE_FILE)
print(SECTION6A_SCRIPT_FILE)
print(
    SECTION6A_BINDING_CANDIDATES_FILE
)

print()
print("✅ SECTION 6A SIMULATOR RUNTIME DISCOVERY PASSED")


# ## Section 6B — Simulator Interface Source Inspection
# 
# #### The simulator implementation exists in project Python files but is not currently loaded into the notebook.
# 
# #### This section safely parses the most relevant source files without executing them. It identifies:
# 
# - classes,
# - functions,
# - method names,
# - constructor arguments,
# - likely battle entry points,
# - legal-move and state-transition interfaces.
# 
# #### The resulting inventory will determine the exact binding used by the policy agent.

# In[16]:


# ======================================================================================
# SECTION 6B — SIMULATOR INTERFACE SOURCE INSPECTION
# ======================================================================================

import ast
import textwrap

print("=" * 100)
print("SECTION 6B — SIMULATOR INTERFACE SOURCE INSPECTION")
print("=" * 100)


# --------------------------------------------------------------------------------------
# Locate the most relevant simulator files discovered in Section 6A
# --------------------------------------------------------------------------------------

TARGET_SIMULATOR_FILES = [
    "simulator.py",
    "battle_simulation.py",
    "battle_agent.py",
    "agent_decision.py",
    "battle_state.py",
    "advanced_search.py",
]

target_script_paths = {}

for target_filename in TARGET_SIMULATOR_FILES:

    matching_rows = script_discovery_df[
        script_discovery_df[
            "file_name"
        ].eq(target_filename)
    ]

    if not matching_rows.empty:

        selected_path = Path(
            matching_rows.iloc[0]["path"]
        ).resolve()

        target_script_paths[
            target_filename
        ] = selected_path


print()
print("TARGET SOURCE FILES")
print("-" * 100)

target_file_df = pd.DataFrame(
    [
        {
            "file_name": file_name,
            "path": str(file_path),
            "exists": file_path.exists(),
            "size_bytes": (
                file_path.stat().st_size
                if file_path.exists()
                else 0
            ),
        }
        for file_name, file_path
        in target_script_paths.items()
    ]
)

display(target_file_df)

assert not target_file_df.empty, (
    "No simulator source files were located."
)

assert target_file_df["exists"].all()


# --------------------------------------------------------------------------------------
# AST helpers
# --------------------------------------------------------------------------------------

def annotation_to_text(
    annotation_node: Optional[ast.AST],
) -> str:
    """
    Convert an AST annotation node into readable source text.
    """

    if annotation_node is None:
        return ""

    try:
        return ast.unparse(
            annotation_node
        )
    except Exception:
        return ""


def default_to_text(
    default_node: Optional[ast.AST],
) -> str:
    """
    Convert a default-value AST node into readable text.
    """

    if default_node is None:
        return ""

    try:
        return ast.unparse(
            default_node
        )
    except Exception:
        return ""


def build_ast_signature(
    function_node: ast.FunctionDef
    | ast.AsyncFunctionDef,
) -> str:
    """
    Build a readable function signature from an AST function node.
    """

    arguments = function_node.args

    positional_arguments = (
        list(arguments.posonlyargs)
        + list(arguments.args)
    )

    defaults = list(arguments.defaults)

    missing_default_count = (
        len(positional_arguments)
        - len(defaults)
    )

    padded_defaults = (
        [None] * missing_default_count
        + defaults
    )

    parameter_parts = []

    for argument_node, default_node in zip(
        positional_arguments,
        padded_defaults,
    ):

        parameter_text = argument_node.arg

        annotation_text = annotation_to_text(
            argument_node.annotation
        )

        if annotation_text:
            parameter_text += (
                f": {annotation_text}"
            )

        default_text = default_to_text(
            default_node
        )

        if default_text:
            parameter_text += (
                f" = {default_text}"
            )

        parameter_parts.append(
            parameter_text
        )

    if arguments.vararg is not None:

        vararg_text = (
            f"*{arguments.vararg.arg}"
        )

        vararg_annotation = annotation_to_text(
            arguments.vararg.annotation
        )

        if vararg_annotation:
            vararg_text += (
                f": {vararg_annotation}"
            )

        parameter_parts.append(
            vararg_text
        )

    elif arguments.kwonlyargs:

        parameter_parts.append("*")

    for keyword_argument, default_node in zip(
        arguments.kwonlyargs,
        arguments.kw_defaults,
    ):

        parameter_text = keyword_argument.arg

        annotation_text = annotation_to_text(
            keyword_argument.annotation
        )

        if annotation_text:
            parameter_text += (
                f": {annotation_text}"
            )

        default_text = default_to_text(
            default_node
        )

        if default_text:
            parameter_text += (
                f" = {default_text}"
            )

        parameter_parts.append(
            parameter_text
        )

    if arguments.kwarg is not None:

        kwarg_text = (
            f"**{arguments.kwarg.arg}"
        )

        kwarg_annotation = annotation_to_text(
            arguments.kwarg.annotation
        )

        if kwarg_annotation:
            kwarg_text += (
                f": {kwarg_annotation}"
            )

        parameter_parts.append(
            kwarg_text
        )

    return_annotation = annotation_to_text(
        function_node.returns
    )

    signature_text = (
        "("
        + ", ".join(parameter_parts)
        + ")"
    )

    if return_annotation:
        signature_text += (
            f" -> {return_annotation}"
        )

    return signature_text


def extract_docstring_summary(
    node: ast.AST,
) -> str:
    """
    Return the first nonempty line of an AST node's docstring.
    """

    docstring = ast.get_docstring(
        node
    )

    if not docstring:
        return ""

    for line in docstring.splitlines():

        cleaned_line = line.strip()

        if cleaned_line:
            return cleaned_line

    return ""


# --------------------------------------------------------------------------------------
# Parse files without importing or executing them
# --------------------------------------------------------------------------------------

source_inventory_records = []
class_method_records = []
import_records = []
parse_failure_records = []
source_text_lookup = {}

for file_name, file_path in (
    target_script_paths.items()
):

    try:

        source_text = file_path.read_text(
            encoding="utf-8"
        )

    except UnicodeDecodeError:

        source_text = file_path.read_text(
            encoding="utf-8-sig"
        )

    source_text_lookup[
        file_name
    ] = source_text

    try:

        syntax_tree = ast.parse(
            source_text,
            filename=str(file_path),
        )

    except SyntaxError as error:

        parse_failure_records.append(
            {
                "file_name": file_name,
                "path": str(file_path),
                "error_type": type(
                    error
                ).__name__,
                "error_message": str(
                    error
                ),
            }
        )

        continue

    for node in syntax_tree.body:

        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):

            source_inventory_records.append(
                {
                    "file_name": file_name,
                    "symbol_name": node.name,
                    "symbol_type": "function",
                    "signature": build_ast_signature(
                        node
                    ),
                    "line_number": int(
                        node.lineno
                    ),
                    "docstring_summary":
                        extract_docstring_summary(
                            node
                        ),
                }
            )

        elif isinstance(
            node,
            ast.ClassDef,
        ):

            source_inventory_records.append(
                {
                    "file_name": file_name,
                    "symbol_name": node.name,
                    "symbol_type": "class",
                    "signature": "",
                    "line_number": int(
                        node.lineno
                    ),
                    "docstring_summary":
                        extract_docstring_summary(
                            node
                        ),
                }
            )

            for class_member in node.body:

                if isinstance(
                    class_member,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                    ),
                ):

                    class_method_records.append(
                        {
                            "file_name":
                                file_name,

                            "class_name":
                                node.name,

                            "method_name":
                                class_member.name,

                            "signature":
                                build_ast_signature(
                                    class_member
                                ),

                            "line_number":
                                int(
                                    class_member.lineno
                                ),

                            "docstring_summary":
                                extract_docstring_summary(
                                    class_member
                                ),
                        }
                    )

        elif isinstance(
            node,
            ast.Import,
        ):

            for alias in node.names:

                import_records.append(
                    {
                        "file_name": file_name,
                        "import_type": "import",
                        "module": alias.name,
                        "imported_name": "",
                    }
                )

        elif isinstance(
            node,
            ast.ImportFrom,
        ):

            module_name = (
                node.module or ""
            )

            for alias in node.names:

                import_records.append(
                    {
                        "file_name":
                            file_name,

                        "import_type":
                            "from_import",

                        "module":
                            module_name,

                        "imported_name":
                            alias.name,
                    }
                )


source_inventory_df = pd.DataFrame(
    source_inventory_records
)

class_methods_df = pd.DataFrame(
    class_method_records
)

source_imports_df = pd.DataFrame(
    import_records
)

parse_failures_df = pd.DataFrame(
    parse_failure_records
)


# --------------------------------------------------------------------------------------
# Display top-level symbols
# --------------------------------------------------------------------------------------

print()
print("TOP-LEVEL CLASSES AND FUNCTIONS")
print("-" * 100)

if source_inventory_df.empty:

    print(
        "No top-level classes or functions were identified."
    )

else:

    source_inventory_df = (
        source_inventory_df
        .sort_values(
            [
                "file_name",
                "line_number",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    display(
        source_inventory_df
    )


# --------------------------------------------------------------------------------------
# Display class methods
# --------------------------------------------------------------------------------------

print()
print("CLASS METHODS")
print("-" * 100)

if class_methods_df.empty:

    print(
        "No class methods were identified."
    )

else:

    class_methods_df = (
        class_methods_df
        .sort_values(
            [
                "file_name",
                "class_name",
                "line_number",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    display(
        class_methods_df
    )


# --------------------------------------------------------------------------------------
# Identify likely runtime interfaces
# --------------------------------------------------------------------------------------

RUNTIME_INTERFACE_KEYWORDS = [
    "run",
    "battle",
    "simulate",
    "play",
    "step",
    "reset",
    "legal",
    "move",
    "action",
    "transition",
    "winner",
    "terminal",
    "done",
    "agent",
]

likely_interface_records = []

for _, row in source_inventory_df.iterrows():

    normalized_symbol_name = str(
        row["symbol_name"]
    ).lower()

    matched_keywords = [
        keyword
        for keyword in RUNTIME_INTERFACE_KEYWORDS
        if keyword in normalized_symbol_name
    ]

    if matched_keywords:

        likely_interface_records.append(
            {
                **row.to_dict(),
                "container": "module",
                "matched_keywords": ", ".join(
                    matched_keywords
                ),
            }
        )


for _, row in class_methods_df.iterrows():

    normalized_method_name = str(
        row["method_name"]
    ).lower()

    matched_keywords = [
        keyword
        for keyword in RUNTIME_INTERFACE_KEYWORDS
        if keyword in normalized_method_name
    ]

    if matched_keywords:

        likely_interface_records.append(
            {
                "file_name":
                    row["file_name"],

                "symbol_name":
                    row["method_name"],

                "symbol_type":
                    "method",

                "signature":
                    row["signature"],

                "line_number":
                    row["line_number"],

                "docstring_summary":
                    row[
                        "docstring_summary"
                    ],

                "container":
                    row["class_name"],

                "matched_keywords":
                    ", ".join(
                        matched_keywords
                    ),
            }
        )


likely_runtime_interfaces_df = (
    pd.DataFrame(
        likely_interface_records
    )
)

print()
print("LIKELY RUNTIME INTERFACES")
print("-" * 100)

if likely_runtime_interfaces_df.empty:

    print(
        "No likely runtime interfaces were identified."
    )

else:

    likely_runtime_interfaces_df = (
        likely_runtime_interfaces_df
        .sort_values(
            [
                "file_name",
                "container",
                "line_number",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    display(
        likely_runtime_interfaces_df
    )


# --------------------------------------------------------------------------------------
# Print short source excerpts for likely runtime interfaces
# --------------------------------------------------------------------------------------

print()
print("RUNTIME INTERFACE SOURCE EXCERPTS")
print("-" * 100)

MAX_EXCERPTS = 20
EXCERPT_LINE_COUNT = 18

for excerpt_number, (
    _,
    interface_row,
) in enumerate(
    likely_runtime_interfaces_df.head(
        MAX_EXCERPTS
    ).iterrows(),
    start=1,
):

    file_name = str(
        interface_row[
            "file_name"
        ]
    )

    source_text = source_text_lookup[
        file_name
    ]

    source_lines = (
        source_text.splitlines()
    )

    start_index = max(
        0,
        int(
            interface_row[
                "line_number"
            ]
        )
        - 1,
    )

    end_index = min(
        len(source_lines),
        start_index
        + EXCERPT_LINE_COUNT,
    )

    excerpt_text = "\n".join(
        source_lines[
            start_index:end_index
        ]
    )

    print()
    print(
        f"[{excerpt_number}] "
        f"{file_name} :: "
        f"{interface_row['container']} :: "
        f"{interface_row['symbol_name']}"
    )

    print("-" * 100)
    print(excerpt_text)


# --------------------------------------------------------------------------------------
# Save reports
# --------------------------------------------------------------------------------------

SECTION6B_SOURCE_INVENTORY_FILE = (
    SECTION6_REPORT_DIR
    / "section6b_source_inventory.csv"
)

SECTION6B_CLASS_METHODS_FILE = (
    SECTION6_REPORT_DIR
    / "section6b_class_methods.csv"
)

SECTION6B_RUNTIME_INTERFACES_FILE = (
    SECTION6_REPORT_DIR
    / "section6b_runtime_interfaces.csv"
)

SECTION6B_IMPORTS_FILE = (
    SECTION6_REPORT_DIR
    / "section6b_source_imports.csv"
)

SECTION6B_PARSE_FAILURES_FILE = (
    SECTION6_REPORT_DIR
    / "section6b_parse_failures.csv"
)

source_inventory_df.to_csv(
    SECTION6B_SOURCE_INVENTORY_FILE,
    index=False,
)

class_methods_df.to_csv(
    SECTION6B_CLASS_METHODS_FILE,
    index=False,
)

likely_runtime_interfaces_df.to_csv(
    SECTION6B_RUNTIME_INTERFACES_FILE,
    index=False,
)

source_imports_df.to_csv(
    SECTION6B_IMPORTS_FILE,
    index=False,
)

parse_failures_df.to_csv(
    SECTION6B_PARSE_FAILURES_FILE,
    index=False,
)


# --------------------------------------------------------------------------------------
# Summary and validation
# --------------------------------------------------------------------------------------

section6b_summary = {
    "target_files_found":
        int(
            len(
                target_script_paths
            )
        ),

    "top_level_symbols_found":
        int(
            len(
                source_inventory_df
            )
        ),

    "class_methods_found":
        int(
            len(
                class_methods_df
            )
        ),

    "runtime_interfaces_found":
        int(
            len(
                likely_runtime_interfaces_df
            )
        ),

    "parse_failures":
        int(
            len(
                parse_failures_df
            )
        ),

    "next_stage":
        "SAFE_SIMULATOR_MODULE_BINDING",
}

print()
print("SOURCE INSPECTION SUMMARY")
print("-" * 100)

for summary_name, summary_value in (
    section6b_summary.items()
):
    print(
        f"{summary_name:30}: "
        f"{summary_value}"
    )

assert (
    section6b_summary[
        "target_files_found"
    ] > 0
)

assert (
    section6b_summary[
        "runtime_interfaces_found"
    ] > 0
), (
    "No simulator runtime interface was discovered."
)

assert (
    section6b_summary[
        "parse_failures"
    ] == 0
), (
    "One or more simulator source files could not be parsed."
)

print()
print("SAVED SECTION 6B REPORTS")
print("-" * 100)

print(SECTION6B_SOURCE_INVENTORY_FILE)
print(SECTION6B_CLASS_METHODS_FILE)
print(SECTION6B_RUNTIME_INTERFACES_FILE)
print(SECTION6B_IMPORTS_FILE)
print(SECTION6B_PARSE_FAILURES_FILE)

print()
print("✅ SECTION 6B SIMULATOR INTERFACE SOURCE INSPECTION PASSED")


# ## Section 6C — Safe Simulator Module Binding
# 
# #### This section safely imports the production simulator modules discovered in Section 6B.
# 
# #### It validates:
# 
# - battle-state classes,
# - move application,
# - winner detection,
# - battle simulation entry points,
# - source-file locations,
# - callable signatures.
# 
# ##### No complete battle is executed yet.

# In[19]:


# ======================================================================================
# SECTION 6C — SAFE PACKAGE-AWARE SIMULATOR MODULE BINDING
# ======================================================================================

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import types
from pathlib import Path
from typing import Any


print("=" * 100)
print("SECTION 6C — SAFE PACKAGE-AWARE SIMULATOR MODULE BINDING")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Resolve required simulator files
# --------------------------------------------------------------------------------------

required_simulator_files = {
    "battle_state":
        target_script_paths.get("battle_state.py"),

    "agent_decision":
        target_script_paths.get("agent_decision.py"),

    "simulator":
        target_script_paths.get("simulator.py"),

    "battle_agent":
        target_script_paths.get("battle_agent.py"),

    "battle_simulation":
        target_script_paths.get("battle_simulation.py"),
}


missing_source_files = [
    module_name
    for module_name, module_path
    in required_simulator_files.items()
    if (
        module_path is None
        or not Path(module_path).exists()
    )
]

assert not missing_source_files, (
    "Required simulator source files are missing: "
    f"{missing_source_files}"
)


print()
print("SIMULATOR SOURCE FILES")
print("-" * 100)

for module_name, module_path in required_simulator_files.items():

    print(
        f"{module_name:24}: "
        f"{Path(module_path).resolve()}"
    )


# --------------------------------------------------------------------------------------
# 2. Resolve the common package directory
# --------------------------------------------------------------------------------------

simulator_source_directories = {
    Path(module_path).resolve().parent
    for module_path
    in required_simulator_files.values()
}

assert len(simulator_source_directories) == 1, (
    "The required simulator modules are not all stored "
    "in the same source directory."
)

SIMULATOR_PACKAGE_DIRECTORY = next(
    iter(simulator_source_directories)
)

SIMULATOR_RUNTIME_PACKAGE = (
    "notebook51_simulator_runtime"
)


print()
print("PACKAGE CONFIGURATION")
print("-" * 100)

print(
    f"Package name       : "
    f"{SIMULATOR_RUNTIME_PACKAGE}"
)

print(
    f"Package directory  : "
    f"{SIMULATOR_PACKAGE_DIRECTORY}"
)


# --------------------------------------------------------------------------------------
# 3. Remove any partial modules left behind by the failed run
# --------------------------------------------------------------------------------------

for loaded_module_name in list(sys.modules):

    if (
        loaded_module_name
        == SIMULATOR_RUNTIME_PACKAGE
        or loaded_module_name.startswith(
            SIMULATOR_RUNTIME_PACKAGE + "."
        )
    ):
        del sys.modules[
            loaded_module_name
        ]


# --------------------------------------------------------------------------------------
# 4. Create a synthetic package
#
# This gives modules such as simulator.py a valid package parent, allowing:
#
#     from .battle_state import BattleState
#
# to work correctly.
# --------------------------------------------------------------------------------------

runtime_package = types.ModuleType(
    SIMULATOR_RUNTIME_PACKAGE
)

runtime_package.__path__ = [
    str(SIMULATOR_PACKAGE_DIRECTORY)
]

runtime_package.__package__ = (
    SIMULATOR_RUNTIME_PACKAGE
)

runtime_package.__file__ = str(
    SIMULATOR_PACKAGE_DIRECTORY
    / "__init__.py"
)

sys.modules[
    SIMULATOR_RUNTIME_PACKAGE
] = runtime_package


# --------------------------------------------------------------------------------------
# 5. Package-aware module loader
# --------------------------------------------------------------------------------------

def load_package_submodule(
    short_module_name: str,
    module_path: Path,
) -> Any:
    """
    Load one source file as a submodule of the synthetic simulator package.
    """

    resolved_path = Path(
        module_path
    ).resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Module file does not exist: {resolved_path}"
        )

    qualified_module_name = (
        f"{SIMULATOR_RUNTIME_PACKAGE}."
        f"{short_module_name}"
    )

    module_spec = (
        importlib.util.spec_from_file_location(
            qualified_module_name,
            resolved_path,
        )
    )

    if (
        module_spec is None
        or module_spec.loader is None
    ):
        raise ImportError(
            "Could not build an import specification for "
            f"{resolved_path}"
        )

    module_object = (
        importlib.util.module_from_spec(
            module_spec
        )
    )

    # Critical for dataclasses and relative imports.
    sys.modules[
        qualified_module_name
    ] = module_object

    try:

        module_spec.loader.exec_module(
            module_object
        )

    except Exception:

        # Remove a partially initialized module if loading fails.
        sys.modules.pop(
            qualified_module_name,
            None,
        )

        raise

    setattr(
        runtime_package,
        short_module_name,
        module_object,
    )

    return module_object


# --------------------------------------------------------------------------------------
# 6. Load modules in dependency order
# --------------------------------------------------------------------------------------

battle_state_module = load_package_submodule(
    "battle_state",
    required_simulator_files[
        "battle_state"
    ],
)

agent_decision_module = load_package_submodule(
    "agent_decision",
    required_simulator_files[
        "agent_decision"
    ],
)

simulator_module = load_package_submodule(
    "simulator",
    required_simulator_files[
        "simulator"
    ],
)

battle_agent_module = load_package_submodule(
    "battle_agent",
    required_simulator_files[
        "battle_agent"
    ],
)

battle_simulation_module = (
    load_package_submodule(
        "battle_simulation",
        required_simulator_files[
            "battle_simulation"
        ],
    )
)


# --------------------------------------------------------------------------------------
# 7. Resolve runtime symbols
# --------------------------------------------------------------------------------------

SimulatorPokemonState = getattr(
    battle_state_module,
    "PokemonState",
    None,
)

SimulatorPlayerState = getattr(
    battle_state_module,
    "PlayerState",
    None,
)

SimulatorBattleState = getattr(
    battle_state_module,
    "BattleState",
    None,
)

simulator_apply_move = getattr(
    simulator_module,
    "apply_move",
    None,
)

SimulatorAgentDecision = getattr(
    agent_decision_module,
    "AgentDecision",
    None,
)

SimulatorPokemonBattleAgent = getattr(
    battle_agent_module,
    "PokemonBattleAgent",
    None,
)

simulator_determine_winner = getattr(
    battle_simulation_module,
    "determine_battle_winner",
    None,
)

simulator_run_battle = getattr(
    battle_simulation_module,
    "simulate_ai_battle",
    None,
)

simulator_create_transcript = getattr(
    battle_simulation_module,
    "create_battle_transcript",
    None,
)

SimulatorBattleTurnRecord = getattr(
    battle_simulation_module,
    "BattleTurnRecord",
    None,
)

SimulatorBattleSimulationResult = getattr(
    battle_simulation_module,
    "BattleSimulationResult",
    None,
)


required_runtime_symbols = {
    "PokemonState":
        SimulatorPokemonState,

    "PlayerState":
        SimulatorPlayerState,

    "BattleState":
        SimulatorBattleState,

    "apply_move":
        simulator_apply_move,

    "AgentDecision":
        SimulatorAgentDecision,

    "PokemonBattleAgent":
        SimulatorPokemonBattleAgent,

    "determine_battle_winner":
        simulator_determine_winner,

    "simulate_ai_battle":
        simulator_run_battle,

    "create_battle_transcript":
        simulator_create_transcript,

    "BattleTurnRecord":
        SimulatorBattleTurnRecord,

    "BattleSimulationResult":
        SimulatorBattleSimulationResult,
}


# --------------------------------------------------------------------------------------
# 8. Create binding inventory
# --------------------------------------------------------------------------------------

runtime_binding_rows = []

for symbol_name, symbol_object in (
    required_runtime_symbols.items()
):

    symbol_exists = (
        symbol_object is not None
    )

    symbol_callable = (
        callable(symbol_object)
        if symbol_exists
        else False
    )

    try:
        symbol_signature = (
            str(
                inspect.signature(
                    symbol_object
                )
            )
            if symbol_callable
            else ""
        )
    except (
        TypeError,
        ValueError,
    ):
        symbol_signature = "Unavailable"

    try:
        symbol_source_file = (
            inspect.getsourcefile(
                symbol_object
            )
            if symbol_exists
            else None
        )
    except (
        TypeError,
        OSError,
    ):
        symbol_source_file = None

    runtime_binding_rows.append(
        {
            "symbol_name":
                symbol_name,

            "exists":
                symbol_exists,

            "callable":
                symbol_callable,

            "object_type":
                (
                    type(
                        symbol_object
                    ).__name__
                    if symbol_exists
                    else None
                ),

            "signature":
                symbol_signature,

            "module":
                (
                    getattr(
                        symbol_object,
                        "__module__",
                        None,
                    )
                    if symbol_exists
                    else None
                ),

            "source_file":
                symbol_source_file,
        }
    )


runtime_binding_df = pd.DataFrame(
    runtime_binding_rows
)


print()
print("SIMULATOR RUNTIME BINDINGS")
print("-" * 100)

display(
    runtime_binding_df
)


# --------------------------------------------------------------------------------------
# 9. Validate bindings
# --------------------------------------------------------------------------------------

assert runtime_binding_df[
    "exists"
].all(), (
    "One or more required simulator symbols could not be loaded."
)

assert inspect.isclass(
    SimulatorPokemonState
)

assert inspect.isclass(
    SimulatorPlayerState
)

assert inspect.isclass(
    SimulatorBattleState
)

assert inspect.isclass(
    SimulatorPokemonBattleAgent
)

assert callable(
    simulator_apply_move
)

assert callable(
    simulator_determine_winner
)

assert callable(
    simulator_run_battle
)

assert callable(
    simulator_create_transcript
)


# Confirm relative imports resolved to the same BattleState class.
simulator_battle_state_reference = getattr(
    simulator_module,
    "BattleState",
    None,
)

battle_simulation_state_reference = getattr(
    battle_simulation_module,
    "BattleState",
    None,
)

assert (
    simulator_battle_state_reference
    is SimulatorBattleState
), (
    "simulator.py did not resolve the expected BattleState class."
)

assert (
    battle_simulation_state_reference
    is SimulatorBattleState
), (
    "battle_simulation.py did not resolve the expected BattleState class."
)


# --------------------------------------------------------------------------------------
# 10. Save reports
# --------------------------------------------------------------------------------------

SECTION6C_BINDING_REPORT_FILE = (
    SECTION6_REPORT_DIR
    / "section6c_simulator_runtime_bindings.csv"
)

SECTION6C_BINDING_SUMMARY_FILE = (
    SECTION6_REPORT_DIR
    / "section6c_simulator_binding_summary.json"
)

runtime_binding_df.to_csv(
    SECTION6C_BINDING_REPORT_FILE,
    index=False,
)


section6c_binding_summary = {
    "status":
        "SIMULATOR_MODULES_BOUND",

    "package_name":
        SIMULATOR_RUNTIME_PACKAGE,

    "package_directory":
        str(
            SIMULATOR_PACKAGE_DIRECTORY
        ),

    "source_files_loaded":
        int(
            len(
                required_simulator_files
            )
        ),

    "runtime_symbols_loaded":
        int(
            runtime_binding_df[
                "exists"
            ].sum()
        ),

    "runtime_symbols_required":
        int(
            len(
                runtime_binding_df
            )
        ),

    "relative_imports_resolved":
        True,

    "battle_state_class":
        SimulatorBattleState.__name__,

    "player_state_class":
        SimulatorPlayerState.__name__,

    "pokemon_state_class":
        SimulatorPokemonState.__name__,

    "apply_move_callable":
        bool(
            callable(
                simulator_apply_move
            )
        ),

    "battle_runner_callable":
        bool(
            callable(
                simulator_run_battle
            )
        ),

    "next_stage":
        "SIMULATOR_STATE_CONSTRUCTION",
}


with open(
    SECTION6C_BINDING_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section6c_binding_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


print()
print("SIMULATOR BINDING SUMMARY")
print("-" * 100)

for key, value in (
    section6c_binding_summary.items()
):

    print(
        f"{key:30}: {value}"
    )


print()
print("SAVED SECTION 6C REPORTS")
print("-" * 100)

print(
    SECTION6C_BINDING_REPORT_FILE
)

print(
    SECTION6C_BINDING_SUMMARY_FILE
)

print()
print(
    "✅ SECTION 6C SAFE PACKAGE-AWARE SIMULATOR MODULE BINDING PASSED"
)


# ## Section 6D — Real Simulator State Construction and Transition Test
# 
# #### This section constructs a genuine simulator `BattleState` using the production classes loaded in Section 6C.
# 
# #### It validates the complete runtime path:
# 
# #### **real BattleState → feature extraction → legal policy decision → apply_move → next BattleState**
# 
# #### The test also confirms that the simulator returns a new state rather than mutating the original position.

# In[20]:


# ======================================================================================
# SECTION 6D — REAL SIMULATOR STATE CONSTRUCTION AND TRANSITION TEST
# ======================================================================================

from copy import deepcopy

print("=" * 100)
print("SECTION 6D — REAL SIMULATOR STATE CONSTRUCTION AND TRANSITION TEST")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Inspect production constructor signatures
# --------------------------------------------------------------------------------------

print()
print("PRODUCTION STATE CONSTRUCTORS")
print("-" * 100)

print(
    f"PokemonState : "
    f"{inspect.signature(SimulatorPokemonState)}"
)

print(
    f"PlayerState  : "
    f"{inspect.signature(SimulatorPlayerState)}"
)

print(
    f"BattleState  : "
    f"{inspect.signature(SimulatorBattleState)}"
)


# --------------------------------------------------------------------------------------
# 2. Build simulator-compatible card records
#
# The loaded simulator expects each active Pokémon's card field to be a dictionary.
# Attack dictionaries use the same schema validated in Notebook 49.
# --------------------------------------------------------------------------------------

player_card_record = {
    "Card ID": 650,
    "Card Name": "Bulbasaur",
    "HP": 80.0,
    "attacks": [
        {
            "Move Name": "Bind Down",
            "damage_numeric": 10.0,
            "energy_cost": 1,
            "Effect Explanation":
                "During your opponent’s next turn, "
                "the Defending Pokémon can’t retreat.",
        }
    ],
}

opponent_card_record = {
    "Card ID": 43,
    "Card Name": "Eevee",
    "HP": 50.0,
    "attacks": [
        {
            "Move Name": "Ascension",
            "damage_numeric": 0.0,
            "energy_cost": 1,
            "Effect Explanation":
                "Search your deck for a card that evolves "
                "from this Pokémon and put it onto this Pokémon.",
        },
        {
            "Move Name": "Quick Attack",
            "damage_numeric": 20.0,
            "energy_cost": 3,
            "Effect Explanation":
                "Flip a coin. If heads, this attack does "
                "20 more damage.",
        },
    ],
}


# --------------------------------------------------------------------------------------
# 3. Construct real production state objects
# --------------------------------------------------------------------------------------

simulator_player_pokemon = SimulatorPokemonState(
    card=player_card_record,
    current_hp=48.0,
    attached_energy=3,
    status=None,
    damage=32.0,
    is_active=True,
)

simulator_opponent_pokemon = SimulatorPokemonState(
    card=opponent_card_record,
    current_hp=30.0,
    attached_energy=3,
    status=None,
    damage=20.0,
    is_active=True,
)

simulator_player_state = SimulatorPlayerState(
    active=simulator_player_pokemon,
    bench=[],
    prize_cards_remaining=2,
    hand_size=3,
)

simulator_opponent_state = SimulatorPlayerState(
    active=simulator_opponent_pokemon,
    bench=[],
    prize_cards_remaining=2,
    hand_size=3,
)

real_simulator_state = SimulatorBattleState(
    player=simulator_player_state,
    opponent=simulator_opponent_state,
    turn_number=9,
    current_player="Opponent",
)


# --------------------------------------------------------------------------------------
# 4. Define the current side's genuine legal moves
#
# The current side is Opponent, whose active Pokémon is Eevee with 3 Energy.
# Both Ascension and Quick Attack are affordable.
# --------------------------------------------------------------------------------------

real_legal_moves = [
    {
        "name": "Ascension",
        "damage": 0.0,
        "energy_cost": 1,
        "effect":
            opponent_card_record[
                "attacks"
            ][0][
                "Effect Explanation"
            ],
    },
    {
        "name": "Quick Attack",
        "damage": 20.0,
        "energy_cost": 3,
        "effect":
            opponent_card_record[
                "attacks"
            ][1][
                "Effect Explanation"
            ],
    },
]


# --------------------------------------------------------------------------------------
# 5. Validate constructed state
# --------------------------------------------------------------------------------------

print()
print("INITIAL REAL SIMULATOR STATE")
print("-" * 100)

print(
    f"Object type          : "
    f"{type(real_simulator_state).__name__}"
)

print(
    f"Current player       : "
    f"{real_simulator_state.current_player}"
)

print(
    f"Turn number          : "
    f"{real_simulator_state.turn_number}"
)

print(
    f"Player card          : "
    f"{real_simulator_state.player.active.card['Card Name']}"
)

print(
    f"Player HP            : "
    f"{real_simulator_state.player.active.current_hp}"
)

print(
    f"Opponent card        : "
    f"{real_simulator_state.opponent.active.card['Card Name']}"
)

print(
    f"Opponent HP          : "
    f"{real_simulator_state.opponent.active.current_hp}"
)

print(
    f"Legal moves          : "
    f"{[move['name'] for move in real_legal_moves]}"
)

assert isinstance(
    real_simulator_state,
    SimulatorBattleState,
)

assert isinstance(
    real_simulator_state.player,
    SimulatorPlayerState,
)

assert isinstance(
    real_simulator_state.opponent,
    SimulatorPlayerState,
)

assert isinstance(
    real_simulator_state.player.active,
    SimulatorPokemonState,
)

assert isinstance(
    real_simulator_state.opponent.active,
    SimulatorPokemonState,
)


# --------------------------------------------------------------------------------------
# 6. Extract policy features from the real simulator object
# --------------------------------------------------------------------------------------

real_policy_features_df = (
    battle_state_to_policy_features(
        battle_state=real_simulator_state,
        legal_moves=real_legal_moves,
        side_mode="PRESERVE",
    )
)

print()
print("EXTRACTED REAL-STATE POLICY FEATURES")
print("-" * 100)

display(
    real_policy_features_df
)

assert list(
    real_policy_features_df.columns
) == POLICY_RAW_FEATURE_COLUMNS

assert (
    real_policy_features_df.loc[
        0,
        "current_side",
    ]
    == "Opponent"
)

assert (
    real_policy_features_df.loc[
        0,
        "player_card",
    ]
    == "Bulbasaur"
)

assert (
    real_policy_features_df.loc[
        0,
        "opponent_card",
    ]
    == "Eevee"
)

assert float(
    real_policy_features_df.loc[
        0,
        "legal_move_count",
    ]
) == 2.0


# --------------------------------------------------------------------------------------
# 7. Run the policy agent against the real simulator state
# --------------------------------------------------------------------------------------

policy_agent.reset()

real_policy_decision = policy_agent.decide(
    battle_state=real_simulator_state,
    legal_moves=real_legal_moves,
)

print()
print("REAL-STATE POLICY DECISION")
print("-" * 100)

print(
    f"Raw prediction        : "
    f"{real_policy_decision.raw_predicted_move}"
)

print(
    f"Selected legal move   : "
    f"{real_policy_decision.selected_move_name}"
)

print(
    f"Confidence            : "
    f"{real_policy_decision.confidence:.4f}"
)

print(
    f"Fallback used         : "
    f"{real_policy_decision.fallback_used}"
)

assert (
    real_policy_decision.selected_move
    in real_legal_moves
)

assert (
    real_policy_decision.selected_move_name
    in {
        "Ascension",
        "Quick Attack",
    }
)


# --------------------------------------------------------------------------------------
# 8. Preserve original values before applying the move
# --------------------------------------------------------------------------------------

original_state_snapshot = deepcopy(
    real_simulator_state
)

original_player_hp = float(
    real_simulator_state.player.active.current_hp
)

original_opponent_hp = float(
    real_simulator_state.opponent.active.current_hp
)

original_turn_number = int(
    real_simulator_state.turn_number
)

original_current_player = str(
    real_simulator_state.current_player
)


# --------------------------------------------------------------------------------------
# 9. Apply the selected move through the production simulator
# --------------------------------------------------------------------------------------

next_simulator_state = simulator_apply_move(
    real_simulator_state,
    real_policy_decision.selected_move,
)


print()
print("POST-MOVE SIMULATOR STATE")
print("-" * 100)

print(
    f"Returned state type   : "
    f"{type(next_simulator_state).__name__}"
)

print(
    f"Selected move         : "
    f"{real_policy_decision.selected_move_name}"
)

print(
    f"Player HP before      : "
    f"{original_player_hp}"
)

print(
    f"Player HP after       : "
    f"{next_simulator_state.player.active.current_hp}"
)

print(
    f"Opponent HP before    : "
    f"{original_opponent_hp}"
)

print(
    f"Opponent HP after     : "
    f"{next_simulator_state.opponent.active.current_hp}"
)

print(
    f"Turn before           : "
    f"{original_turn_number}"
)

print(
    f"Turn after            : "
    f"{next_simulator_state.turn_number}"
)

print(
    f"Current side before   : "
    f"{original_current_player}"
)

print(
    f"Current side after    : "
    f"{next_simulator_state.current_player}"
)


# --------------------------------------------------------------------------------------
# 10. Determine expected transition values
# --------------------------------------------------------------------------------------

selected_move_damage = float(
    real_policy_decision.selected_move.get(
        "damage",
        0.0,
    )
)

expected_player_hp_after = max(
    0.0,
    original_player_hp
    - selected_move_damage,
)

expected_opponent_hp_after = (
    original_opponent_hp
)

expected_next_side = (
    "Player"
    if original_current_player == "Opponent"
    else "Opponent"
)


# --------------------------------------------------------------------------------------
# 11. Validate transition behavior
# --------------------------------------------------------------------------------------

assert isinstance(
    next_simulator_state,
    SimulatorBattleState,
)

assert (
    next_simulator_state
    is not real_simulator_state
), (
    "apply_move() should return a new BattleState object."
)

assert np.isclose(
    next_simulator_state.player.active.current_hp,
    expected_player_hp_after,
)

assert np.isclose(
    next_simulator_state.opponent.active.current_hp,
    expected_opponent_hp_after,
)

assert (
    next_simulator_state.current_player
    == expected_next_side
)

assert (
    next_simulator_state.turn_number
    == original_turn_number + 1
)


# --------------------------------------------------------------------------------------
# 12. Confirm original state was not mutated
# --------------------------------------------------------------------------------------

assert np.isclose(
    real_simulator_state.player.active.current_hp,
    original_state_snapshot.player.active.current_hp,
)

assert np.isclose(
    real_simulator_state.opponent.active.current_hp,
    original_state_snapshot.opponent.active.current_hp,
)

assert (
    real_simulator_state.turn_number
    == original_state_snapshot.turn_number
)

assert (
    real_simulator_state.current_player
    == original_state_snapshot.current_player
)


# --------------------------------------------------------------------------------------
# 13. Winner detection smoke test
# --------------------------------------------------------------------------------------

winner_before = simulator_determine_winner(
    real_simulator_state
)

winner_after = simulator_determine_winner(
    next_simulator_state
)

print()
print("WINNER DETECTION")
print("-" * 100)

print(
    f"Winner before move    : "
    f"{winner_before}"
)

print(
    f"Winner after move     : "
    f"{winner_after}"
)

assert winner_before is None

assert winner_after in {
    None,
    "Player",
    "Opponent",
    "Draw",
}


# --------------------------------------------------------------------------------------
# 14. Save transition report
# --------------------------------------------------------------------------------------

SECTION6D_TRANSITION_REPORT_FILE = (
    SECTION6_REPORT_DIR
    / "section6d_real_state_transition.csv"
)

SECTION6D_SUMMARY_FILE = (
    SECTION6_REPORT_DIR
    / "section6d_real_state_summary.json"
)


section6d_transition_df = pd.DataFrame(
    [
        {
            "initial_current_player":
                original_current_player,

            "next_current_player":
                next_simulator_state.current_player,

            "initial_turn_number":
                original_turn_number,

            "next_turn_number":
                next_simulator_state.turn_number,

            "player_card":
                real_simulator_state.player.active.card[
                    "Card Name"
                ],

            "opponent_card":
                real_simulator_state.opponent.active.card[
                    "Card Name"
                ],

            "selected_move":
                real_policy_decision.selected_move_name,

            "move_damage":
                selected_move_damage,

            "player_hp_before":
                original_player_hp,

            "player_hp_after":
                next_simulator_state.player.active.current_hp,

            "opponent_hp_before":
                original_opponent_hp,

            "opponent_hp_after":
                next_simulator_state.opponent.active.current_hp,

            "fallback_used":
                real_policy_decision.fallback_used,

            "winner_before":
                winner_before,

            "winner_after":
                winner_after,

            "original_state_unchanged":
                True,
        }
    ]
)

section6d_transition_df.to_csv(
    SECTION6D_TRANSITION_REPORT_FILE,
    index=False,
)


section6d_summary = {
    "status":
        "REAL_SIMULATOR_TRANSITION_VALIDATED",

    "state_type":
        type(
            real_simulator_state
        ).__name__,

    "selected_move":
        real_policy_decision.selected_move_name,

    "selected_move_is_legal":
        True,

    "policy_confidence":
        float(
            real_policy_decision.confidence
        ),

    "fallback_used":
        bool(
            real_policy_decision.fallback_used
        ),

    "apply_move_returned_new_state":
        True,

    "original_state_unchanged":
        True,

    "turn_advanced":
        bool(
            next_simulator_state.turn_number
            == original_turn_number + 1
        ),

    "side_switched":
        bool(
            next_simulator_state.current_player
            == expected_next_side
        ),

    "winner_before":
        winner_before,

    "winner_after":
        winner_after,

    "next_stage":
        "POLICY_BATTLE_RUNNER_ADAPTER",
}

with open(
    SECTION6D_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section6d_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


print()
print("TRANSITION REPORT")
print("-" * 100)

display(
    section6d_transition_df
)

print()
print("SECTION 6D SUMMARY")
print("-" * 100)

for key, value in (
    section6d_summary.items()
):

    print(
        f"{key:32}: {value}"
    )


print()
print("SAVED SECTION 6D REPORTS")
print("-" * 100)

print(
    SECTION6D_TRANSITION_REPORT_FILE
)

print(
    SECTION6D_SUMMARY_FILE
)

print()
print(
    "✅ SECTION 6D REAL SIMULATOR STATE AND TRANSITION TEST PASSED"
)


# ## Section 6E — Policy-Compatible Battle Runner Adapter
# 
# #### The production battle runner expects an agent whose `choose_move()` method returns an `AgentDecision`.
# 
# #### This section wraps the Notebook 50 `PolicyAgent` in a compatibility adapter that:
# 
# - resolves legal moves from the active Pokémon,
# - asks the trained policy for a legal action,
# - returns the production `AgentDecision` structure,
# - preserves policy confidence and decision history,
# - supports the existing `simulate_ai_battle()` entry point.

# In[21]:


# ======================================================================================
# SECTION 6E — POLICY-COMPATIBLE BATTLE RUNNER ADAPTER
# ======================================================================================

print("=" * 100)
print("SECTION 6E — POLICY-COMPATIBLE BATTLE RUNNER ADAPTER")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Resolve the active Pokémon for the current side
# --------------------------------------------------------------------------------------

def simulator_current_active_pokemon(
    state: Any,
) -> Any:
    """
    Return the active Pokémon belonging to the current side.
    """

    current_side = normalize_side_name(
        read_first(
            state,
            [
                "current_player",
                "current_side",
            ],
            default="Player",
        )
    )

    if current_side == "Opponent":
        return state.opponent.active

    return state.player.active


# --------------------------------------------------------------------------------------
# 2. Generate legal moves directly from the production state
# --------------------------------------------------------------------------------------

def simulator_generate_legal_moves(
    state: Any,
) -> List[Dict[str, Any]]:
    """
    Generate all attacks affordable by the current active Pokémon.

    If no attack is affordable, return a Pass action.
    """

    active_pokemon = simulator_current_active_pokemon(
        state
    )

    attached_energy = int(
        read_first(
            active_pokemon,
            [
                "attached_energy",
                "energy",
            ],
            default=0,
        )
        or 0
    )

    card_record = read_first(
        active_pokemon,
        [
            "card",
            "card_record",
        ],
        default={},
    )

    attacks = read_first(
        card_record,
        [
            "attacks",
            "Attacks",
        ],
        default=[],
    )

    if not isinstance(
        attacks,
        list,
    ):
        attacks = []

    legal_moves: List[
        Dict[str, Any]
    ] = []

    for attack in attacks:

        if not isinstance(
            attack,
            Mapping,
        ):
            continue

        move_name = read_first(
            attack,
            [
                "Move Name",
                "name",
                "move_name",
            ],
            default="Unknown Move",
        )

        try:
            required_energy = int(
                read_first(
                    attack,
                    [
                        "energy_cost",
                        "Energy Cost",
                        "cost",
                    ],
                    default=0,
                )
                or 0
            )
        except (
            TypeError,
            ValueError,
        ):
            required_energy = 0

        try:
            move_damage = float(
                read_first(
                    attack,
                    [
                        "damage_numeric",
                        "damage",
                        "Damage",
                    ],
                    default=0.0,
                )
                or 0.0
            )
        except (
            TypeError,
            ValueError,
        ):
            move_damage = 0.0

        if attached_energy < required_energy:
            continue

        legal_moves.append(
            {
                "name":
                    str(
                        move_name
                    ),

                "damage":
                    float(
                        move_damage
                    ),

                "energy_cost":
                    int(
                        required_energy
                    ),

                "effect":
                    read_first(
                        attack,
                        [
                            "Effect Explanation",
                            "effect",
                        ],
                        default=None,
                    ),
            }
        )

    if not legal_moves:

        legal_moves.append(
            {
                "name":
                    "Pass",

                "damage":
                    0.0,

                "energy_cost":
                    0,

                "effect":
                    "No legal attack was available.",
            }
        )

    return legal_moves


# --------------------------------------------------------------------------------------
# 3. Production runner compatibility adapter
# --------------------------------------------------------------------------------------

class PolicyBattleRunnerAgent:
    """
    Adapt PolicyAgent to the production PokemonBattleAgent interface.

    The production battle runner expects:

        choose_move(state, depth=...) -> AgentDecision
    """

    def __init__(
        self,
        policy: PolicyAgent,
        default_search_depth: int = 0,
    ) -> None:

        self.policy = policy

        self.default_search_depth = int(
            default_search_depth
        )

        self.total_decisions = 0

        self.decision_records: List[
            Dict[str, Any]
        ] = []

    def reset(
        self,
    ) -> None:
        """
        Reset both the wrapper and underlying policy agent.
        """

        self.total_decisions = 0

        self.decision_records = []

        self.policy.reset()

    def choose_move(
        self,
        state: Any,
        depth: int = 0,
    ) -> Any:
        """
        Return a production AgentDecision for one live state.
        """

        legal_moves = (
            simulator_generate_legal_moves(
                state
            )
        )

        policy_decision = (
            self.policy.decide(
                battle_state=state,
                legal_moves=legal_moves,
            )
        )

        selected_move = (
            policy_decision.selected_move
        )

        decision_depth = int(
            depth
            if depth is not None
            else self.default_search_depth
        )

        agent_decision = (
            SimulatorAgentDecision(
                move=selected_move,
                score=float(
                    policy_decision.confidence
                ),
                search_depth=
                    decision_depth,
                nodes=1,
                principal_variation=[
                    selected_move
                ],
            )
        )

        self.total_decisions += 1

        self.decision_records.append(
            {
                "turn_number":
                    int(
                        read_first(
                            state,
                            [
                                "turn_number",
                                "turn",
                            ],
                            default=0,
                        )
                    ),

                "current_player":
                    normalize_side_name(
                        read_first(
                            state,
                            [
                                "current_player",
                                "current_side",
                            ],
                            default="Player",
                        )
                    ),

                "legal_moves":
                    [
                        move_display_name(
                            move
                        )
                        for move in legal_moves
                    ],

                "selected_move":
                    policy_decision.selected_move_name,

                "raw_predicted_move":
                    policy_decision.raw_predicted_move,

                "confidence":
                    float(
                        policy_decision.confidence
                    ),

                "fallback_used":
                    bool(
                        policy_decision.fallback_used
                    ),

                "fallback_reason":
                    policy_decision.fallback_reason,
            }
        )

        return agent_decision

    def history_dataframe(
        self,
    ) -> pd.DataFrame:
        """
        Return runner-level decisions as a dataframe.
        """

        return pd.DataFrame(
            self.decision_records
        )

    def get_statistics(
        self,
    ) -> Dict[str, Any]:
        """
        Return runner-adapter statistics.
        """

        policy_statistics = (
            self.policy.get_statistics()
        )

        return {
            "wrapper_total_decisions":
                int(
                    self.total_decisions
                ),

            "policy_total_decisions":
                int(
                    policy_statistics[
                        "total_decisions"
                    ]
                ),

            "fallback_decisions":
                int(
                    policy_statistics[
                        "fallback_decisions"
                    ]
                ),

            "fallback_rate":
                float(
                    policy_statistics[
                        "fallback_rate"
                    ]
                ),

            "runner_history_rows":
                int(
                    len(
                        self.decision_records
                    )
                ),
        }


# --------------------------------------------------------------------------------------
# 4. Create the production-compatible policy agent
# --------------------------------------------------------------------------------------

policy_battle_agent = (
    PolicyBattleRunnerAgent(
        policy=policy_agent,
        default_search_depth=0,
    )
)


print()
print("BATTLE RUNNER ADAPTER PROFILE")
print("-" * 100)

print(
    f"Wrapper type          : "
    f"{type(policy_battle_agent).__name__}"
)

print(
    f"Underlying policy     : "
    f"{policy_battle_agent.policy.name}"
)

print(
    f"Default search depth  : "
    f"{policy_battle_agent.default_search_depth}"
)


# --------------------------------------------------------------------------------------
# 5. Single-decision compatibility test
# --------------------------------------------------------------------------------------

policy_battle_agent.reset()

runner_decision = (
    policy_battle_agent.choose_move(
        state=real_simulator_state,
        depth=0,
    )
)


print()
print("RUNNER COMPATIBILITY DECISION")
print("-" * 100)

print(
    f"Decision type         : "
    f"{type(runner_decision).__name__}"
)

print(
    f"Move                  : "
    f"{move_display_name(runner_decision.move)}"
)

print(
    f"Score                 : "
    f"{runner_decision.score}"
)

print(
    f"Search depth          : "
    f"{runner_decision.search_depth}"
)

print(
    f"Nodes                 : "
    f"{runner_decision.nodes}"
)

print(
    f"Principal variation   : "
    f"{[
        move_display_name(move)
        for move
        in runner_decision.principal_variation
    ]}"
)


# --------------------------------------------------------------------------------------
# 6. Validate production contract
# --------------------------------------------------------------------------------------

generated_real_moves = (
    simulator_generate_legal_moves(
        real_simulator_state
    )
)

generated_real_move_names = {
    move_display_name(
        move
    )
    for move in generated_real_moves
}

assert isinstance(
    runner_decision,
    SimulatorAgentDecision,
)

assert isinstance(
    runner_decision.move,
    dict,
)

assert (
    move_display_name(
        runner_decision.move
    )
    in generated_real_move_names
)

assert np.isfinite(
    float(
        runner_decision.score
    )
)

assert (
    runner_decision.nodes
    == 1
)

assert (
    len(
        runner_decision.principal_variation
    )
    == 1
)

assert (
    runner_decision.principal_variation[
        0
    ]
    == runner_decision.move
)


# --------------------------------------------------------------------------------------
# 7. Inspect wrapper history and statistics
# --------------------------------------------------------------------------------------

runner_history_df = (
    policy_battle_agent
    .history_dataframe()
)

runner_statistics = (
    policy_battle_agent
    .get_statistics()
)


print()
print("RUNNER DECISION HISTORY")
print("-" * 100)

display(
    runner_history_df
)


print()
print("RUNNER STATISTICS")
print("-" * 100)

for key, value in (
    runner_statistics.items()
):

    print(
        f"{key:30}: {value}"
    )


assert len(
    runner_history_df
) == 1

assert (
    runner_statistics[
        "wrapper_total_decisions"
    ]
    == 1
)

assert (
    runner_statistics[
        "policy_total_decisions"
    ]
    == 1
)

assert (
    runner_statistics[
        "runner_history_rows"
    ]
    == 1
)


# --------------------------------------------------------------------------------------
# 8. Save adapter reports
# --------------------------------------------------------------------------------------

SECTION6E_ADAPTER_HISTORY_FILE = (
    SECTION6_REPORT_DIR
    / "section6e_policy_runner_adapter_history.csv"
)

SECTION6E_ADAPTER_SUMMARY_FILE = (
    SECTION6_REPORT_DIR
    / "section6e_policy_runner_adapter_summary.json"
)

runner_history_df.to_csv(
    SECTION6E_ADAPTER_HISTORY_FILE,
    index=False,
)


section6e_summary = {
    "status":
        "POLICY_BATTLE_RUNNER_ADAPTER_VALIDATED",

    "adapter_type":
        type(
            policy_battle_agent
        ).__name__,

    "production_decision_type":
        type(
            runner_decision
        ).__name__,

    "selected_move":
        move_display_name(
            runner_decision.move
        ),

    "selected_move_legal":
        True,

    "score_finite":
        bool(
            np.isfinite(
                float(
                    runner_decision.score
                )
            )
        ),

    "principal_variation_valid":
        True,

    "wrapper_total_decisions":
        int(
            runner_statistics[
                "wrapper_total_decisions"
            ]
        ),

    "fallback_decisions":
        int(
            runner_statistics[
                "fallback_decisions"
            ]
        ),

    "next_stage":
        "FULL_POLICY_BATTLE_SIMULATION",
}

with open(
    SECTION6E_ADAPTER_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section6e_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


print()
print("SECTION 6E SUMMARY")
print("-" * 100)

for key, value in (
    section6e_summary.items()
):

    print(
        f"{key:32}: {value}"
    )


print()
print("SAVED SECTION 6E REPORTS")
print("-" * 100)

print(
    SECTION6E_ADAPTER_HISTORY_FILE
)

print(
    SECTION6E_ADAPTER_SUMMARY_FILE
)

print()
print(
    "✅ SECTION 6E POLICY BATTLE RUNNER ADAPTER PASSED"
)


# ## Section 6F — First Full Policy-Controlled Battle
# 
# #### This section runs the first complete production-simulator battle controlled by the trained Notebook 50 policy.
# 
# #### The same policy agent controls whichever side is active. The test validates:
# 
# - full battle-runner compatibility,
# - repeated legal-action selection,
# - production state transitions,
# - terminal-state or turn-limit handling,
# - turn records,
# - battle transcript creation,
# - policy decision history.

# In[24]:


# ======================================================================================
# SECTION 6F — FIRST FULL POLICY-CONTROLLED BATTLE
# ======================================================================================

from copy import deepcopy

print("=" * 100)
print("SECTION 6F — FIRST FULL POLICY-CONTROLLED BATTLE")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Inspect the exact production battle-runner contract
# --------------------------------------------------------------------------------------

battle_runner_signature = inspect.signature(
    simulator_run_battle
)

battle_runner_parameters = list(
    battle_runner_signature.parameters.keys()
)

print()
print("BATTLE RUNNER CONTRACT")
print("-" * 100)

print(
    f"Function              : "
    f"{simulator_run_battle.__name__}"
)

print(
    f"Signature             : "
    f"{battle_runner_signature}"
)

print(
    f"Parameters            : "
    f"{battle_runner_parameters}"
)


# --------------------------------------------------------------------------------------
# 2. Construct a fresh battle state
#
# Both active Pokémon have enough Energy to attack immediately.
# --------------------------------------------------------------------------------------

battle_player_pokemon = SimulatorPokemonState(
    card=deepcopy(
        player_card_record
    ),
    current_hp=80.0,
    attached_energy=3,
    status=None,
    damage=0.0,
    is_active=True,
)

battle_opponent_pokemon = SimulatorPokemonState(
    card=deepcopy(
        opponent_card_record
    ),
    current_hp=50.0,
    attached_energy=3,
    status=None,
    damage=0.0,
    is_active=True,
)

battle_player_state = SimulatorPlayerState(
    active=battle_player_pokemon,
    bench=[],
    prize_cards_remaining=1,
    hand_size=5,
)

battle_opponent_state = SimulatorPlayerState(
    active=battle_opponent_pokemon,
    bench=[],
    prize_cards_remaining=1,
    hand_size=5,
)

full_battle_initial_state = SimulatorBattleState(
    player=battle_player_state,
    opponent=battle_opponent_state,
    turn_number=1,
    current_player="Player",
)


print()
print("INITIAL BATTLE POSITION")
print("-" * 100)

print(
    f"Player Pokémon        : "
    f"{full_battle_initial_state.player.active.card['Card Name']}"
)

print(
    f"Player HP             : "
    f"{full_battle_initial_state.player.active.current_hp}"
)

print(
    f"Player Energy         : "
    f"{full_battle_initial_state.player.active.attached_energy}"
)

print(
    f"Opponent Pokémon      : "
    f"{full_battle_initial_state.opponent.active.card['Card Name']}"
)

print(
    f"Opponent HP           : "
    f"{full_battle_initial_state.opponent.active.current_hp}"
)

print(
    f"Opponent Energy       : "
    f"{full_battle_initial_state.opponent.active.attached_energy}"
)

print(
    f"Starting side         : "
    f"{full_battle_initial_state.current_player}"
)


# --------------------------------------------------------------------------------------
# 3. Reset the policy battle adapter
# --------------------------------------------------------------------------------------

policy_battle_agent.reset()

assert (
    policy_battle_agent.total_decisions
    == 0
)


# --------------------------------------------------------------------------------------
# 4. Build a signature-aware invocation
# --------------------------------------------------------------------------------------

battle_runner_kwargs = {}

if "initial_state" in battle_runner_parameters:
    battle_runner_kwargs[
        "initial_state"
    ] = full_battle_initial_state

elif "state" in battle_runner_parameters:
    battle_runner_kwargs[
        "state"
    ] = full_battle_initial_state

else:
    raise TypeError(
        "The battle runner has no recognized state parameter."
    )


if "agent" in battle_runner_parameters:
    battle_runner_kwargs[
        "agent"
    ] = policy_battle_agent

elif "battle_agent" in battle_runner_parameters:
    battle_runner_kwargs[
        "battle_agent"
    ] = policy_battle_agent

else:
    raise TypeError(
        "The battle runner has no recognized agent parameter."
    )


if "max_turns" in battle_runner_parameters:
    battle_runner_kwargs[
        "max_turns"
    ] = 30

if "search_depth" in battle_runner_parameters:
    battle_runner_kwargs[
        "search_depth"
    ] = 0

if "depth" in battle_runner_parameters:
    battle_runner_kwargs[
        "depth"
    ] = 0


print()
print("BATTLE RUNNER ARGUMENTS")
print("-" * 100)

for argument_name, argument_value in (
    battle_runner_kwargs.items()
):

    if argument_name in {
        "initial_state",
        "state",
    }:
        printable_value = (
            type(
                argument_value
            ).__name__
        )

    elif argument_name in {
        "agent",
        "battle_agent",
    }:
        printable_value = (
            type(
                argument_value
            ).__name__
        )

    else:
        printable_value = argument_value

    print(
        f"{argument_name:24}: "
        f"{printable_value}"
    )


# --------------------------------------------------------------------------------------
# 5. Run the complete production battle
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# Ensure production battle-runner card schema compatibility
# --------------------------------------------------------------------------------------

for pokemon_state in [
    full_battle_initial_state.player.active,
    full_battle_initial_state.opponent.active,
]:

    card_record = pokemon_state.card

    if "name" not in card_record:
        card_record["name"] = card_record.get(
            "Card Name",
            "Unknown Pokémon",
        )

    if "hp" not in card_record:
        card_record["hp"] = card_record.get(
            "HP",
            pokemon_state.current_hp,
        )

full_battle_result = simulator_run_battle(
    **battle_runner_kwargs
)


print()
print("BATTLE RESULT OBJECT")
print("-" * 100)

print(
    f"Result type           : "
    f"{type(full_battle_result).__name__}"
)

assert isinstance(
    full_battle_result,
    SimulatorBattleSimulationResult,
)


# --------------------------------------------------------------------------------------
# 6. Resolve result attributes defensively
# --------------------------------------------------------------------------------------

battle_final_state = read_first(
    full_battle_result,
    [
        "final_state",
        "state",
    ],
    default=None,
)

battle_turn_records = read_first(
    full_battle_result,
    [
        "turns",
        "turn_records",
        "history",
    ],
    default=[],
)

battle_winner = read_first(
    full_battle_result,
    [
        "winner",
        "winning_side",
    ],
    default=None,
)

battle_termination_reason = read_first(
    full_battle_result,
    [
        "stop_reason",
        "termination_reason",
        "reason",
        "status",
    ],
    default=None,
)

battle_turn_records = list(
    battle_turn_records or []
)

if (
    battle_winner is None
    and battle_final_state is not None
):
    battle_winner = simulator_determine_winner(
        battle_final_state
    )


print()
print("FULL BATTLE SUMMARY")
print("-" * 100)

print(
    f"Winner                : "
    f"{battle_winner}"
)

print(
    f"Recorded turns        : "
    f"{len(battle_turn_records)}"
)

print(
    f"Termination reason    : "
    f"{battle_termination_reason}"
)

if battle_final_state is not None:

    print(
        f"Final turn number     : "
        f"{battle_final_state.turn_number}"
    )

    print(
        f"Final current side    : "
        f"{battle_final_state.current_player}"
    )

    print(
        f"Final Player HP       : "
        f"{battle_final_state.player.active.current_hp}"
    )

    print(
        f"Final Opponent HP     : "
        f"{battle_final_state.opponent.active.current_hp}"
    )


# --------------------------------------------------------------------------------------
# 7. Convert turn records to a dataframe
# --------------------------------------------------------------------------------------

battle_turn_rows = []

for turn_index, turn_record in enumerate(
    battle_turn_records,
    start=1,
):

    if is_dataclass(
        turn_record
    ):
        turn_dictionary = asdict(
            turn_record
        )

    elif isinstance(
        turn_record,
        Mapping,
    ):
        turn_dictionary = dict(
            turn_record
        )

    else:
        turn_dictionary = {
            attribute_name:
                getattr(
                    turn_record,
                    attribute_name,
                )
            for attribute_name in dir(
                turn_record
            )
            if (
                not attribute_name.startswith("_")
                and not callable(
                    getattr(
                        turn_record,
                        attribute_name,
                    )
                )
            )
        }

    move_value = read_first(
        turn_dictionary,
        [
            "move",
            "selected_move",
            "action",
        ],
        default=None,
    )

    battle_turn_rows.append(
        {
            "record_number":
                turn_index,

            "turn_number":
                read_first(
                    turn_dictionary,
                    [
                        "turn_number",
                        "turn",
                    ],
                    default=turn_index,
                ),

            "acting_side":
                read_first(
                    turn_dictionary,
                    [
                        "acting_side",
                        "current_player",
                        "side",
                    ],
                    default=None,
                ),

            "pokemon_name":
                read_first(
                    turn_dictionary,
                    [
                        "pokemon_name",
                        "active_pokemon",
                    ],
                    default=None,
                ),

            "move_name":
                move_display_name(
                    move_value
                )
                if move_value is not None
                else read_first(
                    turn_dictionary,
                    [
                        "move_name",
                        "action_name",
                    ],
                    default=None,
                ),

            "score":
                read_first(
                    turn_dictionary,
                    [
                        "score",
                        "decision_score",
                    ],
                    default=None,
                ),

            "player_hp":
                read_first(
                    turn_dictionary,
                    [
                        "player_hp",
                        "player_current_hp",
                    ],
                    default=None,
                ),

            "opponent_hp":
                read_first(
                    turn_dictionary,
                    [
                        "opponent_hp",
                        "opponent_current_hp",
                    ],
                    default=None,
                ),
        }
    )


battle_turns_df = pd.DataFrame(
    battle_turn_rows
)


print()
print("BATTLE TURN RECORDS")
print("-" * 100)

if battle_turns_df.empty:
    print(
        "The battle runner returned no explicit turn records."
    )
else:
    display(
        battle_turns_df
    )


# --------------------------------------------------------------------------------------
# 8. Inspect policy decisions made during the complete battle
# --------------------------------------------------------------------------------------

full_battle_policy_history_df = (
    policy_battle_agent
    .history_dataframe()
)

full_battle_agent_statistics = (
    policy_battle_agent
    .get_statistics()
)


print()
print("POLICY DECISION HISTORY")
print("-" * 100)

display(
    full_battle_policy_history_df
)


print()
print("POLICY BATTLE STATISTICS")
print("-" * 100)

for statistic_name, statistic_value in (
    full_battle_agent_statistics.items()
):

    print(
        f"{statistic_name:30}: "
        f"{statistic_value}"
    )


# --------------------------------------------------------------------------------------
# 9. Generate the production transcript
# --------------------------------------------------------------------------------------

battle_transcript = simulator_create_transcript(
    full_battle_result
)

print()
print("BATTLE TRANSCRIPT")
print("-" * 100)

print(
    battle_transcript[
        :5000
    ]
)


# --------------------------------------------------------------------------------------
# 10. Validate full battle execution
# --------------------------------------------------------------------------------------

assert battle_final_state is not None

assert isinstance(
    battle_final_state,
    SimulatorBattleState,
)

assert (
    full_battle_agent_statistics[
        "wrapper_total_decisions"
    ] > 0
)

assert (
    full_battle_agent_statistics[
        "wrapper_total_decisions"
    ]
    == full_battle_agent_statistics[
        "policy_total_decisions"
    ]
)

assert (
    len(
        full_battle_policy_history_df
    )
    == full_battle_agent_statistics[
        "wrapper_total_decisions"
    ]
)

assert isinstance(
    battle_transcript,
    str,
)

assert len(
    battle_transcript
) > 0

assert (
    battle_final_state.turn_number
    >= full_battle_initial_state.turn_number
)

assert (
    battle_final_state.player.active.current_hp
    <= full_battle_initial_state.player.active.current_hp
)

assert (
    battle_final_state.opponent.active.current_hp
    <= full_battle_initial_state.opponent.active.current_hp
)


# --------------------------------------------------------------------------------------
# 11. Save full battle reports
# --------------------------------------------------------------------------------------

SECTION6F_BATTLE_TURNS_FILE = (
    SECTION6_REPORT_DIR
    / "section6f_first_full_battle_turns.csv"
)

SECTION6F_POLICY_HISTORY_FILE = (
    SECTION6_REPORT_DIR
    / "section6f_first_full_battle_policy_history.csv"
)

SECTION6F_TRANSCRIPT_FILE = (
    SECTION6_REPORT_DIR
    / "section6f_first_full_battle_transcript.txt"
)

SECTION6F_SUMMARY_FILE = (
    SECTION6_REPORT_DIR
    / "section6f_first_full_battle_summary.json"
)


battle_turns_df.to_csv(
    SECTION6F_BATTLE_TURNS_FILE,
    index=False,
)

full_battle_policy_history_df.to_csv(
    SECTION6F_POLICY_HISTORY_FILE,
    index=False,
)

SECTION6F_TRANSCRIPT_FILE.write_text(
    battle_transcript,
    encoding="utf-8",
)


section6f_summary = {
    "status":
        "FULL_POLICY_BATTLE_COMPLETED",

    "result_type":
        type(
            full_battle_result
        ).__name__,

    "winner":
        battle_winner,

    "termination_reason":
        battle_termination_reason,

    "recorded_turns":
        int(
            len(
                battle_turn_records
            )
        ),

    "policy_decisions":
        int(
            full_battle_agent_statistics[
                "wrapper_total_decisions"
            ]
        ),

    "fallback_decisions":
        int(
            full_battle_agent_statistics[
                "fallback_decisions"
            ]
        ),

    "fallback_rate":
        float(
            full_battle_agent_statistics[
                "fallback_rate"
            ]
        ),

    "final_turn_number":
        int(
            battle_final_state.turn_number
        ),

    "final_player_hp":
        float(
            battle_final_state.player.active.current_hp
        ),

    "final_opponent_hp":
        float(
            battle_final_state.opponent.active.current_hp
        ),

    "transcript_created":
        True,

    "next_stage":
        "NOTEBOOK51_FINAL_VALIDATION",
}


with open(
    SECTION6F_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section6f_summary,
        file,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


print()
print("SECTION 6F SUMMARY")
print("-" * 100)

for key, value in (
    section6f_summary.items()
):

    print(
        f"{key:32}: {value}"
    )


print()
print("SAVED SECTION 6F REPORTS")
print("-" * 100)

print(SECTION6F_BATTLE_TURNS_FILE)
print(SECTION6F_POLICY_HISTORY_FILE)
print(SECTION6F_TRANSCRIPT_FILE)
print(SECTION6F_SUMMARY_FILE)

print()
print(
    "✅ SECTION 6F FIRST FULL POLICY-CONTROLLED BATTLE PASSED"
)


# # Section 7 — Final Validation and Notebook 51 Handoff
# 
# ## Section 7A — Final Integration Validation
# 
# ### This section performs the final validation for Notebook 51.
# 
# ##### It verifies:
# 
# - Notebook 50 policy artifacts loaded successfully,
# - policy inference reproduced validated expert moves,
# - live simulator features matched the deployment schema,
# - legal-action masking worked,
# - the PolicyAgent adapter passed integration tests,
# - simulator modules loaded correctly,
# - production state transitions succeeded,
# - the full policy-controlled battle completed,
# - all required Notebook 51 reports exist and are nonempty.
# 
# #### If every validation check passes, Notebook 51 is marked ready for tournament evaluation.

# In[25]:


# ======================================================================================
# SECTION 7A — FINAL VALIDATION AND NOTEBOOK 51 HANDOFF
# ======================================================================================

print("=" * 100)
print("SECTION 7A — FINAL VALIDATION AND NOTEBOOK 51 HANDOFF")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Final report directory and output paths
# --------------------------------------------------------------------------------------

SECTION7_REPORT_DIR = (
    NOTEBOOK51_REPORT_DIR
    / "section7"
)

SECTION7_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SECTION7_ARTIFACT_INVENTORY_FILE = (
    SECTION7_REPORT_DIR
    / "section7a_artifact_inventory.csv"
)

SECTION7_VALIDATION_CHECKS_FILE = (
    SECTION7_REPORT_DIR
    / "section7a_validation_checks.csv"
)

SECTION7_FINAL_PROFILE_FILE = (
    SECTION7_REPORT_DIR
    / "section7a_final_integration_profile.csv"
)

SECTION7_HANDOFF_MANIFEST_FILE = (
    SECTION7_REPORT_DIR
    / "section7a_handoff_manifest.json"
)

SECTION7_FINAL_SUMMARY_FILE = (
    SECTION7_REPORT_DIR
    / "section7a_final_summary.json"
)


# --------------------------------------------------------------------------------------
# 2. Required Notebook 51 artifacts
# --------------------------------------------------------------------------------------

required_notebook51_artifacts = {
    "section6a_namespace_discovery":
        SECTION6A_NAMESPACE_FILE,

    "section6a_script_discovery":
        SECTION6A_SCRIPT_FILE,

    "section6a_binding_candidates":
        SECTION6A_BINDING_CANDIDATES_FILE,

    "section6b_source_inventory":
        SECTION6B_SOURCE_INVENTORY_FILE,

    "section6b_class_methods":
        SECTION6B_CLASS_METHODS_FILE,

    "section6b_runtime_interfaces":
        SECTION6B_RUNTIME_INTERFACES_FILE,

    "section6b_source_imports":
        SECTION6B_IMPORTS_FILE,

    "section6b_parse_failures":
        SECTION6B_PARSE_FAILURES_FILE,

    "section6c_runtime_bindings":
        SECTION6C_BINDING_REPORT_FILE,

    "section6c_binding_summary":
        SECTION6C_BINDING_SUMMARY_FILE,

    "section6d_transition_report":
        SECTION6D_TRANSITION_REPORT_FILE,

    "section6d_state_summary":
        SECTION6D_SUMMARY_FILE,

    "section6e_adapter_history":
        SECTION6E_ADAPTER_HISTORY_FILE,

    "section6e_adapter_summary":
        SECTION6E_ADAPTER_SUMMARY_FILE,

    "section6f_battle_turns":
        SECTION6F_BATTLE_TURNS_FILE,

    "section6f_policy_history":
        SECTION6F_POLICY_HISTORY_FILE,

    "section6f_battle_transcript":
        SECTION6F_TRANSCRIPT_FILE,

    "section6f_battle_summary":
        SECTION6F_SUMMARY_FILE,
}


artifact_inventory_rows = []

for artifact_name, artifact_path in (
    required_notebook51_artifacts.items()
):

    artifact_path = Path(
        artifact_path
    )

    artifact_inventory_rows.append(
        {
            "artifact_name":
                artifact_name,

            "path":
                str(
                    artifact_path
                ),

            "exists":
                artifact_path.exists(),

            "is_file":
                artifact_path.is_file(),

            "size_bytes":
                (
                    artifact_path.stat().st_size
                    if artifact_path.exists()
                    else 0
                ),
        }
    )


section7_artifact_inventory_df = pd.DataFrame(
    artifact_inventory_rows
)

section7_artifact_inventory_df[
    "nonempty"
] = (
    section7_artifact_inventory_df[
        "size_bytes"
    ] > 0
)


print()
print("ARTIFACT INVENTORY")
print("-" * 100)

display(
    section7_artifact_inventory_df
)


# --------------------------------------------------------------------------------------
# 3. Validation helper
# --------------------------------------------------------------------------------------

validation_checks = []


def add_validation_check(
    check_name: str,
    passed: Any,
    value: Any = "",
    expected: Any = "",
    details: Any = "",
) -> None:
    """
    Append one final validation check.
    """

    validation_checks.append(
        {
            "check":
                check_name,

            "passed":
                bool(
                    passed
                ),

            "value":
                value,

            "expected":
                expected,

            "details":
                details,
        }
    )


# --------------------------------------------------------------------------------------
# 4. Artifact checks
# --------------------------------------------------------------------------------------

add_validation_check(
    "all_required_artifacts_exist",
    section7_artifact_inventory_df[
        "exists"
    ].all(),
    int(
        (
            ~section7_artifact_inventory_df[
                "exists"
            ]
        ).sum()
    ),
    0,
)

add_validation_check(
    "all_required_artifacts_are_files",
    section7_artifact_inventory_df[
        "is_file"
    ].all(),
    int(
        (
            ~section7_artifact_inventory_df[
                "is_file"
            ]
        ).sum()
    ),
    0,
)

add_validation_check(
    "all_required_artifacts_nonempty",
    section7_artifact_inventory_df[
        "nonempty"
    ].all(),
    int(
        (
            ~section7_artifact_inventory_df[
                "nonempty"
            ]
        ).sum()
    ),
    0,
)


# --------------------------------------------------------------------------------------
# 5. Notebook 50 handoff and policy checks
# --------------------------------------------------------------------------------------

add_validation_check(
    "notebook50_handoff_ready",
    status
    == "READY_FOR_SIMULATOR_INTEGRATION",
    status,
    "READY_FOR_SIMULATOR_INTEGRATION",
)

add_validation_check(
    "policy_class_count_valid",
    len(
        label_encoder.classes_
    ) == 6,
    len(
        label_encoder.classes_
    ),
    6,
)

add_validation_check(
    "encoded_feature_count_valid",
    len(
        feature_names
    ) == int(
        model_metadata[
            "encoded_feature_count"
        ]
    ),
    len(
        feature_names
    ),
    int(
        model_metadata[
            "encoded_feature_count"
        ]
    ),
)

add_validation_check(
    "policy_inference_validation_passed",
    bool(
        accuracy == 1.0
    ),
    float(
        accuracy
    ),
    1.0,
)

add_validation_check(
    "deployment_feature_schema_valid",
    list(
        extracted_policy_features_df.columns
    )
    == POLICY_RAW_FEATURE_COLUMNS,
    len(
        extracted_policy_features_df.columns
    ),
    len(
        POLICY_RAW_FEATURE_COLUMNS
    ),
)

add_validation_check(
    "variant_name_removed_from_deployment",
    "variant_name"
    not in POLICY_RAW_FEATURE_COLUMNS,
    (
        "variant_name"
        not in POLICY_RAW_FEATURE_COLUMNS
    ),
    True,
)


# --------------------------------------------------------------------------------------
# 6. Legal-action and adapter checks
# --------------------------------------------------------------------------------------

add_validation_check(
    "legal_selector_validation_passed",
    bool(
        summary_df.shape[0] == 3
        and summary_df[
            "fallback"
        ].notna().all()
    ),
    int(
        summary_df.shape[0]
    ),
    3,
)

add_validation_check(
    "policy_agent_validation_passed",
    section5b_validation_df[
        "passed"
    ].all(),
    int(
        section5b_validation_df[
            "passed"
        ].sum()
    ),
    int(
        len(
            section5b_validation_df
        )
    ),
)

add_validation_check(
    "policy_agent_entry_points_available",
    all(
        [
            callable(
                policy_agent.decide
            ),
            callable(
                policy_agent.choose_move
            ),
            callable(
                policy_agent.select_action
            ),
            callable(
                policy_agent.act
            ),
        ]
    ),
    4,
    4,
)


# --------------------------------------------------------------------------------------
# 7. Simulator binding checks
# --------------------------------------------------------------------------------------

add_validation_check(
    "simulator_runtime_symbols_loaded",
    runtime_binding_df[
        "exists"
    ].all(),
    int(
        runtime_binding_df[
            "exists"
        ].sum()
    ),
    int(
        len(
            runtime_binding_df
        )
    ),
)

add_validation_check(
    "relative_imports_resolved",
    bool(
        section6c_binding_summary[
            "relative_imports_resolved"
        ]
    ),
    section6c_binding_summary[
        "relative_imports_resolved"
    ],
    True,
)

add_validation_check(
    "simulator_state_transition_validated",
    section6d_summary[
        "status"
    ]
    == "REAL_SIMULATOR_TRANSITION_VALIDATED",
    section6d_summary[
        "status"
    ],
    "REAL_SIMULATOR_TRANSITION_VALIDATED",
)

add_validation_check(
    "apply_move_returns_new_state",
    bool(
        section6d_summary[
            "apply_move_returned_new_state"
        ]
    ),
    section6d_summary[
        "apply_move_returned_new_state"
    ],
    True,
)

add_validation_check(
    "original_state_preserved",
    bool(
        section6d_summary[
            "original_state_unchanged"
        ]
    ),
    section6d_summary[
        "original_state_unchanged"
    ],
    True,
)

add_validation_check(
    "turn_transition_valid",
    bool(
        section6d_summary[
            "turn_advanced"
        ]
        and section6d_summary[
            "side_switched"
        ]
    ),
    {
        "turn_advanced":
            section6d_summary[
                "turn_advanced"
            ],

        "side_switched":
            section6d_summary[
                "side_switched"
            ],
    },
    {
        "turn_advanced":
            True,

        "side_switched":
            True,
    },
)


# --------------------------------------------------------------------------------------
# 8. Production battle adapter checks
# --------------------------------------------------------------------------------------

add_validation_check(
    "battle_runner_adapter_validated",
    section6e_summary[
        "status"
    ]
    == "POLICY_BATTLE_RUNNER_ADAPTER_VALIDATED",
    section6e_summary[
        "status"
    ],
    "POLICY_BATTLE_RUNNER_ADAPTER_VALIDATED",
)

add_validation_check(
    "production_agent_decision_valid",
    (
        section6e_summary[
            "production_decision_type"
        ]
        == "AgentDecision"
        and section6e_summary[
            "selected_move_legal"
        ]
        is True
    ),
    {
        "decision_type":
            section6e_summary[
                "production_decision_type"
            ],

        "selected_move_legal":
            section6e_summary[
                "selected_move_legal"
            ],
    },
    {
        "decision_type":
            "AgentDecision",

        "selected_move_legal":
            True,
    },
)


# --------------------------------------------------------------------------------------
# 9. Full battle checks
# --------------------------------------------------------------------------------------

add_validation_check(
    "full_policy_battle_completed",
    section6f_summary[
        "status"
    ]
    == "FULL_POLICY_BATTLE_COMPLETED",
    section6f_summary[
        "status"
    ],
    "FULL_POLICY_BATTLE_COMPLETED",
)

add_validation_check(
    "full_battle_has_policy_decisions",
    int(
        section6f_summary[
            "policy_decisions"
        ]
    ) > 0,
    int(
        section6f_summary[
            "policy_decisions"
        ]
    ),
    "> 0",
)

add_validation_check(
    "full_battle_fallback_rate_valid",
    (
        0.0
        <= float(
            section6f_summary[
                "fallback_rate"
            ]
        )
        <= 1.0
    ),
    float(
        section6f_summary[
            "fallback_rate"
        ]
    ),
    "0.0 to 1.0",
)

add_validation_check(
    "full_battle_final_state_valid",
    (
        float(
            section6f_summary[
                "final_player_hp"
            ]
        )
        >= 0.0
        and float(
            section6f_summary[
                "final_opponent_hp"
            ]
        )
        >= 0.0
    ),
    {
        "player_hp":
            section6f_summary[
                "final_player_hp"
            ],

        "opponent_hp":
            section6f_summary[
                "final_opponent_hp"
            ],
    },
    "Both >= 0",
)

add_validation_check(
    "full_battle_termination_reason_present",
    bool(
        str(
            section6f_summary[
                "termination_reason"
            ]
        ).strip()
    ),
    section6f_summary[
        "termination_reason"
    ],
    "Nonempty",
)

add_validation_check(
    "battle_transcript_created",
    bool(
        section6f_summary[
            "transcript_created"
        ]
    ),
    section6f_summary[
        "transcript_created"
    ],
    True,
)

add_validation_check(
    "battle_winner_valid",
    section6f_summary[
        "winner"
    ]
    in {
        "Player",
        "Opponent",
        "Draw",
        None,
    },
    section6f_summary[
        "winner"
    ],
    "Player, Opponent, Draw, or None",
)


# --------------------------------------------------------------------------------------
# 10. Build validation dataframe and stop on failure
# --------------------------------------------------------------------------------------

section7_validation_checks_df = pd.DataFrame(
    validation_checks
)


print()
print("FINAL VALIDATION CHECKS")
print("-" * 100)

display(
    section7_validation_checks_df
)


failed_checks_df = (
    section7_validation_checks_df.loc[
        ~section7_validation_checks_df[
            "passed"
        ]
    ]
    .copy()
)


assert failed_checks_df.empty, (
    "Notebook 51 final validation failed: "
    f"{failed_checks_df['check'].tolist()}"
)


# --------------------------------------------------------------------------------------
# 11. Final integration profile
# --------------------------------------------------------------------------------------

section7_final_profile_df = pd.DataFrame(
    {
        "metric": [
            "policy_class_count",
            "encoded_feature_count",
            "raw_feature_count",
            "inference_validation_accuracy",
            "legal_selector_tests",
            "policy_agent_validation_tests",
            "simulator_source_files_loaded",
            "simulator_runtime_symbols_loaded",
            "real_transition_selected_move",
            "real_transition_fallback_used",
            "full_battle_turns",
            "full_battle_policy_decisions",
            "full_battle_fallback_decisions",
            "full_battle_fallback_rate",
            "full_battle_final_player_hp",
            "full_battle_final_opponent_hp",
            "validation_checks_passed",
            "validation_checks_failed",
        ],
        "value": [
            int(
                len(
                    label_encoder.classes_
                )
            ),

            int(
                len(
                    feature_names
                )
            ),

            int(
                len(
                    POLICY_RAW_FEATURE_COLUMNS
                )
            ),

            float(
                accuracy
            ),

            int(
                len(
                    summary_df
                )
            ),

            int(
                len(
                    section5b_validation_df
                )
            ),

            int(
                section6c_binding_summary[
                    "source_files_loaded"
                ]
            ),

            int(
                section6c_binding_summary[
                    "runtime_symbols_loaded"
                ]
            ),

            section6d_summary[
                "selected_move"
            ],

            bool(
                section6d_summary[
                    "fallback_used"
                ]
            ),

            int(
                section6f_summary[
                    "recorded_turns"
                ]
            ),

            int(
                section6f_summary[
                    "policy_decisions"
                ]
            ),

            int(
                section6f_summary[
                    "fallback_decisions"
                ]
            ),

            float(
                section6f_summary[
                    "fallback_rate"
                ]
            ),

            float(
                section6f_summary[
                    "final_player_hp"
                ]
            ),

            float(
                section6f_summary[
                    "final_opponent_hp"
                ]
            ),

            int(
                section7_validation_checks_df[
                    "passed"
                ].sum()
            ),

            int(
                (
                    ~section7_validation_checks_df[
                        "passed"
                    ]
                ).sum()
            ),
        ],
    }
)


print()
print("FINAL INTEGRATION PROFILE")
print("-" * 100)

display(
    section7_final_profile_df
)


# --------------------------------------------------------------------------------------
# 12. Handoff manifest
# --------------------------------------------------------------------------------------

notebook51_handoff_manifest = {
    "notebook":
        "51_policy_simulator_integration",

    "status":
        "READY_FOR_TOURNAMENT_EVALUATION",

    "completed_sections": [
        "1A",
        "1B",
        "2A",
        "2B",
        "2C",
        "3A",
        "3B",
        "4A",
        "4B",
        "5A",
        "5B",
        "6A",
        "6B",
        "6C",
        "6D",
        "6E",
        "6F",
        "7A",
    ],

    "source_policy_notebook":
        "50_side_balanced_policy_fine_tuning",

    "policy_class_count":
        int(
            len(
                label_encoder.classes_
            )
        ),

    "policy_classes":
        list(
            label_encoder.classes_
        ),

    "raw_feature_count":
        int(
            len(
                POLICY_RAW_FEATURE_COLUMNS
            )
        ),

    "encoded_feature_count":
        int(
            len(
                feature_names
            )
        ),

    "policy_agent_type":
        type(
            policy_agent
        ).__name__,

    "battle_runner_adapter_type":
        type(
            policy_battle_agent
        ).__name__,

    "simulator_package":
        SIMULATOR_RUNTIME_PACKAGE,

    "simulator_source_directory":
        str(
            SIMULATOR_PACKAGE_DIRECTORY
        ),

    "simulator_symbols_loaded":
        int(
            section6c_binding_summary[
                "runtime_symbols_loaded"
            ]
        ),

    "real_transition_validated":
        True,

    "full_battle_completed":
        True,

    "full_battle_winner":
        section6f_summary[
            "winner"
        ],

    "full_battle_turns":
        int(
            section6f_summary[
                "recorded_turns"
            ]
        ),

    "full_battle_policy_decisions":
        int(
            section6f_summary[
                "policy_decisions"
            ]
        ),

    "full_battle_fallback_decisions":
        int(
            section6f_summary[
                "fallback_decisions"
            ]
        ),

    "full_battle_fallback_rate":
        float(
            section6f_summary[
                "fallback_rate"
            ]
        ),

    "validation_checks_passed":
        int(
            section7_validation_checks_df[
                "passed"
            ].sum()
        ),

    "validation_checks_failed":
        int(
            (
                ~section7_validation_checks_df[
                    "passed"
                ]
            ).sum()
        ),

    "policy_model_file":
        str(
            MODEL_FILE
        ),

    "policy_preprocessor_file":
        str(
            PREPROCESSOR_FILE
        ),

    "policy_label_encoder_file":
        str(
            LABEL_ENCODER_FILE
        ),

    "policy_feature_names_file":
        str(
            FEATURE_NAMES_FILE
        ),

    "first_battle_transcript":
        str(
            SECTION6F_TRANSCRIPT_FILE
        ),

    "next_notebook":
        "Notebook 52 — Tournament Policy Evaluation",

    "next_stage":
        "TOURNAMENT_POLICY_EVALUATION",
}


notebook51_final_summary = {
    "final_status":
        notebook51_handoff_manifest[
            "status"
        ],

    "policy_class_count":
        notebook51_handoff_manifest[
            "policy_class_count"
        ],

    "raw_feature_count":
        notebook51_handoff_manifest[
            "raw_feature_count"
        ],

    "encoded_feature_count":
        notebook51_handoff_manifest[
            "encoded_feature_count"
        ],

    "simulator_symbols_loaded":
        notebook51_handoff_manifest[
            "simulator_symbols_loaded"
        ],

    "full_battle_completed":
        notebook51_handoff_manifest[
            "full_battle_completed"
        ],

    "full_battle_winner":
        notebook51_handoff_manifest[
            "full_battle_winner"
        ],

    "full_battle_turns":
        notebook51_handoff_manifest[
            "full_battle_turns"
        ],

    "validation_checks_passed":
        notebook51_handoff_manifest[
            "validation_checks_passed"
        ],

    "validation_checks_failed":
        notebook51_handoff_manifest[
            "validation_checks_failed"
        ],

    "next_notebook":
        notebook51_handoff_manifest[
            "next_notebook"
        ],

    "next_stage":
        notebook51_handoff_manifest[
            "next_stage"
        ],
}


# --------------------------------------------------------------------------------------
# 13. Save final reports
# --------------------------------------------------------------------------------------

section7_artifact_inventory_df.to_csv(
    SECTION7_ARTIFACT_INVENTORY_FILE,
    index=False,
)

section7_validation_checks_df.to_csv(
    SECTION7_VALIDATION_CHECKS_FILE,
    index=False,
)

section7_final_profile_df.to_csv(
    SECTION7_FINAL_PROFILE_FILE,
    index=False,
)


with open(
    SECTION7_HANDOFF_MANIFEST_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        notebook51_handoff_manifest,
        file,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


with open(
    SECTION7_FINAL_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        notebook51_final_summary,
        file,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


# --------------------------------------------------------------------------------------
# 14. Final handoff summary
# --------------------------------------------------------------------------------------

print()
print("NOTEBOOK 51 HANDOFF SUMMARY")
print("-" * 100)

print(
    f"Final status               : "
    f"{notebook51_handoff_manifest['status']}"
)

print(
    f"Policy classes             : "
    f"{notebook51_handoff_manifest['policy_class_count']}"
)

print(
    f"Raw policy features        : "
    f"{notebook51_handoff_manifest['raw_feature_count']}"
)

print(
    f"Encoded features           : "
    f"{notebook51_handoff_manifest['encoded_feature_count']}"
)

print(
    f"Simulator symbols loaded   : "
    f"{notebook51_handoff_manifest['simulator_symbols_loaded']}"
)

print(
    f"Full battle completed      : "
    f"{notebook51_handoff_manifest['full_battle_completed']}"
)

print(
    f"Full battle winner         : "
    f"{notebook51_handoff_manifest['full_battle_winner']}"
)

print(
    f"Full battle turns          : "
    f"{notebook51_handoff_manifest['full_battle_turns']}"
)

print(
    f"Validation checks passed   : "
    f"{notebook51_handoff_manifest['validation_checks_passed']}"
)

print(
    f"Validation checks failed   : "
    f"{notebook51_handoff_manifest['validation_checks_failed']}"
)

print(
    f"Next notebook              : "
    f"{notebook51_handoff_manifest['next_notebook']}"
)

print(
    f"Next stage                 : "
    f"{notebook51_handoff_manifest['next_stage']}"
)


print()
print("SAVED SECTION 7A REPORTS")
print("-" * 100)

print(
    SECTION7_ARTIFACT_INVENTORY_FILE
)

print(
    SECTION7_VALIDATION_CHECKS_FILE
)

print(
    SECTION7_FINAL_PROFILE_FILE
)

print(
    SECTION7_HANDOFF_MANIFEST_FILE
)

print(
    SECTION7_FINAL_SUMMARY_FILE
)


print()
print(
    "✅ SECTION 7A FINAL VALIDATION AND HANDOFF PASSED"
)

print()
print(
    "🎉 NOTEBOOK 51 COMPLETE — READY FOR NOTEBOOK 52"
)


# In[ ]:




