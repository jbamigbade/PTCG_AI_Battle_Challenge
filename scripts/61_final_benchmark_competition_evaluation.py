#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================================
# NOTEBOOK 61 — FINAL BENCHMARK / COMPETITION EVALUATION
# Cell 1: Environment + Certified Notebook 60 Handoff Validation
# ============================================================

from pathlib import Path
import hashlib
import json
import sys
import platform
from datetime import datetime

# ------------------------------------------------------------
# 1. Project identity
# ------------------------------------------------------------

PROJECT_ROOT = Path(
    r"D:\02_AI_and_Data\Kaggle-AI-Agents\PTCG_AI_Battle_Challenge"
)

NOTEBOOK61_NAME = "61_final_benchmark_competition_evaluation.ipynb"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

NB60_OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "notebook60"
NB60_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "notebook60"
NB60_MODELS_DIR = PROJECT_ROOT / "models" / "notebook60"

NB61_OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "notebook61"
NB61_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "notebook61"

# ------------------------------------------------------------
# 2. Certified Notebook 60 model contract
# ------------------------------------------------------------

CERTIFIED_MODEL_PATH = (
    NB60_MODELS_DIR
    / "notebook60_certified_61_feature_random_forest.joblib"
)

EXPECTED_MODEL_SHA256 = (
    "615BAF3D5725CCCA0C1C94E37F62FB24A68613248E167BB566BEF4A6D50F2581"
)

EXPECTED_MODEL_SIZE = 22_396_321

EXPECTED_RUNTIME_FEATURE_COUNT = 61
EXPECTED_TARGET_CLASS_COUNT = 11

# ------------------------------------------------------------
# 3. Required Notebook 60 handoff evidence
# ------------------------------------------------------------

REQUIRED_HANDOFF_FILES = [
    "section4d6as_final_benchmark_handoff.json",
    "section4d6as_changed_matches.csv",
    "section4d6as_decision_audit.csv",
    "section4d6as_handoff_manifest.csv",
    "notebook60_certified_model_contract.json",
]

# ------------------------------------------------------------
# 4. Helper functions
# ------------------------------------------------------------

def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest().upper()


def locate_handoff_file(filename: str):
    candidates = [
        NB60_OUTPUTS_DIR / filename,
        NB60_ARTIFACTS_DIR / filename,
    ]

    matches = [path for path in candidates if path.exists()]

    return matches


# ------------------------------------------------------------
# 5. Basic environment validation
# ------------------------------------------------------------

print("=" * 78)
print("NOTEBOOK 61 — ENVIRONMENT AND CERTIFIED HANDOFF VALIDATION")
print("=" * 78)

print(f"Validation timestamp : {datetime.now().isoformat(timespec='seconds')}")
print(f"Python executable    : {sys.executable}")
print(f"Python version       : {sys.version.split()[0]}")
print(f"Platform             : {platform.platform()}")
print(f"Project root         : {PROJECT_ROOT}")

assert PROJECT_ROOT.exists(), (
    f"PROJECT ROOT NOT FOUND:\n{PROJECT_ROOT}"
)

assert NOTEBOOKS_DIR.exists(), (
    f"NOTEBOOK DIRECTORY NOT FOUND:\n{NOTEBOOKS_DIR}"
)

assert NB60_OUTPUTS_DIR.exists(), (
    f"NOTEBOOK 60 OUTPUT DIRECTORY NOT FOUND:\n{NB60_OUTPUTS_DIR}"
)

assert NB60_ARTIFACTS_DIR.exists(), (
    f"NOTEBOOK 60 ARTIFACT DIRECTORY NOT FOUND:\n{NB60_ARTIFACTS_DIR}"
)

assert NB60_MODELS_DIR.exists(), (
    f"NOTEBOOK 60 MODEL DIRECTORY NOT FOUND:\n{NB60_MODELS_DIR}"
)

assert NB61_OUTPUTS_DIR.exists(), (
    f"NOTEBOOK 61 OUTPUT DIRECTORY NOT FOUND:\n{NB61_OUTPUTS_DIR}"
)

assert NB61_ARTIFACTS_DIR.exists(), (
    f"NOTEBOOK 61 ARTIFACT DIRECTORY NOT FOUND:\n{NB61_ARTIFACTS_DIR}"
)

print("\n[OK] Project and Notebook 60/61 directories validated.")

# ------------------------------------------------------------
# 6. Certified model integrity validation
# ------------------------------------------------------------

assert CERTIFIED_MODEL_PATH.exists(), (
    f"CERTIFIED MODEL NOT FOUND:\n{CERTIFIED_MODEL_PATH}"
)

actual_model_size = CERTIFIED_MODEL_PATH.stat().st_size
actual_model_hash = sha256_file(CERTIFIED_MODEL_PATH)

print("\nCertified model:")
print(f"  Path       : {CERTIFIED_MODEL_PATH}")
print(f"  Size       : {actual_model_size:,} bytes")
print(f"  SHA256     : {actual_model_hash}")

assert actual_model_size == EXPECTED_MODEL_SIZE, (
    "CERTIFIED MODEL SIZE MISMATCH\n"
    f"Expected: {EXPECTED_MODEL_SIZE:,}\n"
    f"Actual:   {actual_model_size:,}"
)

assert actual_model_hash == EXPECTED_MODEL_SHA256, (
    "CERTIFIED MODEL SHA256 MISMATCH\n"
    f"Expected: {EXPECTED_MODEL_SHA256}\n"
    f"Actual:   {actual_model_hash}"
)

print("[OK] Certified model size and SHA256 match Notebook 60.")

# ------------------------------------------------------------
# 7. Notebook 60 handoff-artifact validation
# ------------------------------------------------------------

handoff_locations = {}

print("\nNotebook 60 required handoff evidence:")

for filename in REQUIRED_HANDOFF_FILES:
    matches = locate_handoff_file(filename)

    assert matches, (
        f"REQUIRED NOTEBOOK 60 HANDOFF FILE NOT FOUND:\n{filename}"
    )

    handoff_locations[filename] = matches

    print(f"\n  [OK] {filename}")
    for match in matches:
        print(f"       {match}")

# ------------------------------------------------------------
# 8. Final Cell 1 certification
# ------------------------------------------------------------

CELL1_STATUS = {
    "notebook": NOTEBOOK61_NAME,
    "project_root": str(PROJECT_ROOT),
    "certified_model": str(CERTIFIED_MODEL_PATH),
    "certified_model_sha256": actual_model_hash,
    "certified_model_size_bytes": actual_model_size,
    "expected_runtime_feature_count": EXPECTED_RUNTIME_FEATURE_COUNT,
    "expected_target_class_count": EXPECTED_TARGET_CLASS_COUNT,
    "required_handoff_files_found": len(handoff_locations),
    "retraining_performed": False,
    "model_modified": False,
    "validation_status": "PASS",
}

print("\n" + "=" * 78)
print("CELL 1 VALIDATION STATUS: PASS")
print("=" * 78)

print(json.dumps(CELL1_STATUS, indent=2))

print("\nNotebook 61 prerequisite state is valid.")
print("No retraining performed.")
print("No model modification performed.")


# In[5]:


# ============================================================
# NOTEBOOK 61 — FINAL BENCHMARK / COMPETITION EVALUATION
# Cell 2: Notebook 60 Handoff Consistency + Content Validation
# CORRECTED SCHEMA VERSION
# ============================================================

import pandas as pd

print("=" * 78)
print("NOTEBOOK 61 — NOTEBOOK 60 HANDOFF CONSISTENCY VALIDATION")
print("=" * 78)

# ------------------------------------------------------------
# 1. Verify duplicate output/artifact copies are byte-identical
# ------------------------------------------------------------

duplicate_consistency = {}

print("\n1. Duplicate-copy integrity check")

for filename, matches in handoff_locations.items():

    hashes = [sha256_file(path) for path in matches]
    identical = len(set(hashes)) == 1

    duplicate_consistency[filename] = {
        "copies": len(matches),
        "identical": identical,
        "sha256": hashes[0] if identical else hashes,
    }

    print(f"\n{'[OK]' if identical else '[FAIL]'} {filename}")
    print(f"     Copies found: {len(matches)}")

    for path, file_hash in zip(matches, hashes):
        print(f"     {path}")
        print(f"       SHA256: {file_hash}")

    if len(matches) > 1:
        assert identical, (
            f"OUTPUT/ARTIFACT COPY MISMATCH DETECTED:\n{filename}"
        )

print("\n[OK] Notebook 60 duplicate handoff copies are consistent.")

# ------------------------------------------------------------
# 2. Select canonical read copies
# ------------------------------------------------------------

canonical_handoff_paths = {
    filename: matches[0]
    for filename, matches in handoff_locations.items()
}

# ------------------------------------------------------------
# 3. Load Notebook 60 final benchmark handoff
# ------------------------------------------------------------

handoff_json_path = canonical_handoff_paths[
    "section4d6as_final_benchmark_handoff.json"
]

with handoff_json_path.open("r", encoding="utf-8-sig") as file:
    nb60_handoff = json.load(file)

print("\n" + "-" * 78)
print("2. Final benchmark handoff JSON")
print("-" * 78)

print(json.dumps(nb60_handoff, indent=2))

# ------------------------------------------------------------
# 4. Load certified model contract
# ------------------------------------------------------------

model_contract_path = canonical_handoff_paths[
    "notebook60_certified_model_contract.json"
]

with model_contract_path.open("r", encoding="utf-8-sig") as file:
    nb60_model_contract = json.load(file)

print("\n" + "-" * 78)
print("3. Certified model contract")
print("-" * 78)

print(json.dumps(nb60_model_contract, indent=2))

# ------------------------------------------------------------
# 5. Load tabular evidence
# ------------------------------------------------------------

changed_matches_df = pd.read_csv(
    canonical_handoff_paths[
        "section4d6as_changed_matches.csv"
    ]
)

decision_audit_df = pd.read_csv(
    canonical_handoff_paths[
        "section4d6as_decision_audit.csv"
    ]
)

handoff_manifest_df = pd.read_csv(
    canonical_handoff_paths[
        "section4d6as_handoff_manifest.csv"
    ]
)

print("\n" + "-" * 78)
print("4. Handoff table dimensions")
print("-" * 78)

print(f"Changed matches rows : {len(changed_matches_df)}")
print(f"Decision audit rows  : {len(decision_audit_df)}")
print(f"Handoff manifest rows: {len(handoff_manifest_df)}")

assert len(changed_matches_df) == 11
assert len(decision_audit_df) == 2

# ------------------------------------------------------------
# 6. Authoritative Notebook 60 certified benchmark facts
# ------------------------------------------------------------

EXPECTED_CONCLUSION = (
    "PRESERVE_CONTROLLED_61_FEATURE_MODEL_AND_REPORT_POLICY_PATH_DIFFERENCE"
)

NB60_CERTIFIED_BENCHMARK = {
    "historical_matches": 300,
    "controlled_matches": 300,
    "historical_wins": 194,
    "controlled_wins": 183,
    "historical_win_rate": 194 / 300,
    "controlled_win_rate": 183 / 300,
    "benchmark_delta_percentage_points":
        ((183 / 300) - (194 / 300)) * 100,
    "identical_match_outcomes": 289,
    "changed_matches": 11,
    "historical_win_to_controlled_loss": 11,
    "historical_loss_to_controlled_win": 0,
    "diagnostic_states_audited": 2,
    "historical_controlled_move_differences": 2,
    "historical_fallback_decisions": 2,
    "controlled_fallback_decisions": 0,
    "scientific_conclusion": EXPECTED_CONCLUSION,
    "retraining_recommended": False,
    "model_modification_recommended": False,
}

# ------------------------------------------------------------
# 7. Mathematical consistency checks
# ------------------------------------------------------------

assert (
    NB60_CERTIFIED_BENCHMARK["identical_match_outcomes"]
    + NB60_CERTIFIED_BENCHMARK["changed_matches"]
    == 300
)

assert (
    NB60_CERTIFIED_BENCHMARK["historical_win_to_controlled_loss"]
    + NB60_CERTIFIED_BENCHMARK["historical_loss_to_controlled_win"]
    == 11
)

print("\n[OK] Certified benchmark arithmetic validated.")

# ------------------------------------------------------------
# 8. Validate FINAL HANDOFF JSON using its actual schema
# ------------------------------------------------------------

assert nb60_handoff["source_notebook"] == 60
assert nb60_handoff["historical_matches"] == 300
assert nb60_handoff["historical_wins"] == 194
assert nb60_handoff["controlled_matches"] == 300
assert nb60_handoff["controlled_wins"] == 183

assert abs(
    float(nb60_handoff["historical_win_rate"])
    - NB60_CERTIFIED_BENCHMARK["historical_win_rate"]
) < 1e-12

assert abs(
    float(nb60_handoff["controlled_win_rate"])
    - NB60_CERTIFIED_BENCHMARK["controlled_win_rate"]
) < 1e-12

assert abs(
    float(nb60_handoff["benchmark_delta_percentage_points"])
    - NB60_CERTIFIED_BENCHMARK["benchmark_delta_percentage_points"]
) < 1e-10

assert nb60_handoff["identical_match_outcomes"] == 289
assert nb60_handoff["changed_match_count"] == 11

assert (
    nb60_handoff["historical_decision_fallbacks_observed"]
    == 2
)

assert (
    nb60_handoff["controlled_decision_fallbacks_observed"]
    == 0
)

assert (
    nb60_handoff["different_selected_moves_observed"]
    == 2
)

assert (
    nb60_handoff["scientific_conclusion"]
    == EXPECTED_CONCLUSION
)

assert nb60_handoff["retraining_recommended"] is False
assert nb60_handoff["model_modification_recommended"] is False
assert nb60_handoff["frozen_model_preserved"] is True
assert nb60_handoff["fit_calls_executed"] == 0
assert nb60_handoff["retraining_executed"] is False
assert nb60_handoff["model_modified"] is False
assert nb60_handoff["additional_tournament_matches_executed"] == 0

print("[OK] Final benchmark handoff matches Notebook 60 certified state.")

# ------------------------------------------------------------
# 9. Validate MODEL CONTRACT using its actual schema
# ------------------------------------------------------------

assert nb60_model_contract["notebook"] == 60
assert nb60_model_contract["model_type"] == "RandomForestClassifier"
assert nb60_model_contract["feature_count"] == 61
assert nb60_model_contract["target_class_count"] == 11

assert abs(
    float(nb60_model_contract["historical_benchmark"])
    - (194 / 300)
) < 1e-12

assert abs(
    float(nb60_model_contract["controlled_benchmark"])
    - (183 / 300)
) < 1e-12

assert abs(
    float(nb60_model_contract["benchmark_delta_percentage_points"])
    - NB60_CERTIFIED_BENCHMARK["benchmark_delta_percentage_points"]
) < 1e-10

assert (
    nb60_model_contract["scientific_conclusion"]
    == EXPECTED_CONCLUSION
)

assert nb60_model_contract["retraining_required"] is False
assert nb60_model_contract["model_modification_required"] is False
assert nb60_model_contract["notebook60_certified"] is True

assert len(nb60_model_contract["feature_names"]) == 61
assert len(nb60_model_contract["target_classes"]) == 11

print("[OK] Certified model contract semantics validated.")

# ------------------------------------------------------------
# 10. Validate certified target classes
# ------------------------------------------------------------

expected_target_classes = {
    "Bite",
    "Flame Tail",
    "Fury Swipes",
    "Quick Attack",
    "Razor Leaf",
    "Scratch",
    "Tackle",
    "Tail Whap",
    "Thunderbolt",
    "Vine Whip",
    "Water Pulse",
}

actual_target_classes = set(
    nb60_model_contract["target_classes"]
)

assert actual_target_classes == expected_target_classes

print(f"[OK] Certified target classes validated: {len(actual_target_classes)}")

# ------------------------------------------------------------
# 11. Inherit physical-model integrity from Cell 1
# ------------------------------------------------------------

assert CELL1_STATUS["validation_status"] == "PASS"

assert (
    CELL1_STATUS["certified_model_sha256"]
    == EXPECTED_MODEL_SHA256
)

assert (
    CELL1_STATUS["certified_model_size_bytes"]
    == EXPECTED_MODEL_SIZE
)

print("[OK] Physical certified-model integrity inherited from Cell 1.")
print(f"     SHA256: {EXPECTED_MODEL_SHA256}")

# ------------------------------------------------------------
# 12. Cell 2 certification
# ------------------------------------------------------------

CELL2_STATUS = {
    "duplicate_handoff_copies_consistent": all(
        item["identical"]
        for item in duplicate_consistency.values()
    ),
    "changed_matches_rows": len(changed_matches_df),
    "decision_audit_rows": len(decision_audit_df),
    "handoff_manifest_rows": len(handoff_manifest_df),
    "historical_matches": 300,
    "controlled_matches": 300,
    "historical_wins": 194,
    "controlled_wins": 183,
    "historical_win_rate_pct": (194 / 300) * 100,
    "controlled_win_rate_pct": (183 / 300) * 100,
    "benchmark_delta_percentage_points":
        ((183 / 300) - (194 / 300)) * 100,
    "certified_feature_count": 61,
    "certified_target_class_count": 11,
    "scientific_conclusion": EXPECTED_CONCLUSION,
    "physical_model_hash_validated_in_cell1": True,
    "benchmark_handoff_validated": True,
    "model_contract_semantics_validated": True,
    "tournament_rerun_performed": False,
    "fit_calls_executed": 0,
    "retraining_performed": False,
    "model_modified": False,
    "validation_status": "PASS",
}

print("\n" + "=" * 78)
print("CELL 2 VALIDATION STATUS: PASS")
print("=" * 78)

print(json.dumps(CELL2_STATUS, indent=2))

print("\nNotebook 60 handoff evidence is internally consistent.")
print("Certified model integrity remains established by Cell 1.")
print("Certified 300-match evidence reused without tournament rerun.")
print("No fit calls executed.")
print("No retraining performed.")
print("No model modification performed.")


# In[6]:


# ============================================================
# NOTEBOOK 61 — FINAL BENCHMARK / COMPETITION EVALUATION
# Cell 3: Changed-Match + Decision-Audit Evidence Inspection
# ============================================================

print("=" * 78)
print("NOTEBOOK 61 — CHANGED-MATCH AND DECISION-AUDIT EVIDENCE")
print("=" * 78)

# ------------------------------------------------------------
# 1. Prerequisite certification
# ------------------------------------------------------------

assert CELL1_STATUS["validation_status"] == "PASS"
assert CELL2_STATUS["validation_status"] == "PASS"

print("\n[OK] Cell 1 prerequisite certification confirmed.")
print("[OK] Cell 2 handoff certification confirmed.")

# ------------------------------------------------------------
# 2. Changed-match schema
# ------------------------------------------------------------

print("\n" + "-" * 78)
print("1. CHANGED MATCHES — SCHEMA")
print("-" * 78)

print(f"Rows    : {len(changed_matches_df)}")
print(f"Columns : {len(changed_matches_df.columns)}")

for index, column in enumerate(changed_matches_df.columns, start=1):
    print(f"{index:>3}. {column}")

assert len(changed_matches_df) == 11

# ------------------------------------------------------------
# 3. Changed-match complete evidence
# ------------------------------------------------------------

print("\n" + "-" * 78)
print("2. CHANGED MATCHES — COMPLETE EVIDENCE")
print("-" * 78)

with pd.option_context(
    "display.max_columns", None,
    "display.max_rows", None,
    "display.width", 240,
    "display.max_colwidth", 120,
):
    print(changed_matches_df.to_string(index=False))

# ------------------------------------------------------------
# 4. Changed-match unique-value profile
# ------------------------------------------------------------

print("\n" + "-" * 78)
print("3. CHANGED MATCHES — UNIQUE VALUE PROFILE")
print("-" * 78)

for column in changed_matches_df.columns:
    unique_values = changed_matches_df[column].dropna().unique()

    print(f"\n{column}")
    print(f"  dtype        : {changed_matches_df[column].dtype}")
    print(f"  unique count : {len(unique_values)}")

    # Print all unique values when reasonably small.
    if len(unique_values) <= 20:
        print(f"  values       : {unique_values.tolist()}")
    else:
        print(f"  first values : {unique_values[:10].tolist()}")

# ------------------------------------------------------------
# 5. Decision-audit schema
# ------------------------------------------------------------

print("\n" + "-" * 78)
print("4. DECISION AUDIT — SCHEMA")
print("-" * 78)

print(f"Rows    : {len(decision_audit_df)}")
print(f"Columns : {len(decision_audit_df.columns)}")

for index, column in enumerate(decision_audit_df.columns, start=1):
    print(f"{index:>3}. {column}")

assert len(decision_audit_df) == 2

# ------------------------------------------------------------
# 6. Decision-audit complete evidence
# ------------------------------------------------------------

print("\n" + "-" * 78)
print("5. DECISION AUDIT — COMPLETE EVIDENCE")
print("-" * 78)

with pd.option_context(
    "display.max_columns", None,
    "display.max_rows", None,
    "display.width", 260,
    "display.max_colwidth", 160,
):
    print(decision_audit_df.to_string(index=False))

# ------------------------------------------------------------
# 7. Decision-audit unique-value profile
# ------------------------------------------------------------

print("\n" + "-" * 78)
print("6. DECISION AUDIT — UNIQUE VALUE PROFILE")
print("-" * 78)

for column in decision_audit_df.columns:
    unique_values = decision_audit_df[column].dropna().unique()

    print(f"\n{column}")
    print(f"  dtype        : {decision_audit_df[column].dtype}")
    print(f"  unique count : {len(unique_values)}")

    if len(unique_values) <= 20:
        print(f"  values       : {unique_values.tolist()}")
    else:
        print(f"  first values : {unique_values[:10].tolist()}")

# ------------------------------------------------------------
# 8. Handoff manifest schema and contents
# ------------------------------------------------------------

print("\n" + "-" * 78)
print("7. HANDOFF MANIFEST — COMPLETE EVIDENCE")
print("-" * 78)

print(f"Rows    : {len(handoff_manifest_df)}")
print(f"Columns : {len(handoff_manifest_df.columns)}")

with pd.option_context(
    "display.max_columns", None,
    "display.max_rows", None,
    "display.width", 220,
    "display.max_colwidth", 160,
):
    print(handoff_manifest_df.to_string(index=False))

# ------------------------------------------------------------
# 9. Schema-independent integrity checks
# ------------------------------------------------------------

assert not changed_matches_df.empty
assert not decision_audit_df.empty
assert not handoff_manifest_df.empty

assert len(changed_matches_df) == nb60_handoff["changed_match_count"]
assert len(decision_audit_df) == nb60_handoff["different_selected_moves_observed"]

CELL3_STATUS = {
    "cell1_prerequisite_pass": True,
    "cell2_prerequisite_pass": True,
    "changed_matches_rows": len(changed_matches_df),
    "changed_matches_columns": len(changed_matches_df.columns),
    "decision_audit_rows": len(decision_audit_df),
    "decision_audit_columns": len(decision_audit_df.columns),
    "handoff_manifest_rows": len(handoff_manifest_df),
    "schema_inspection_complete": True,
    "content_inspection_complete": True,
    "tournament_rerun_performed": False,
    "fit_calls_executed": 0,
    "retraining_performed": False,
    "model_modified": False,
    "validation_status": "PASS",
}

print("\n" + "=" * 78)
print("CELL 3 INSPECTION STATUS: PASS")
print("=" * 78)

print(json.dumps(CELL3_STATUS, indent=2))

print("\nExact Notebook 60 evidence schema is now exposed.")
print("No assumptions were made about CSV column naming.")
print("No tournament rerun performed.")
print("No fit calls executed.")
print("No retraining performed.")
print("No model modification performed.")


# In[7]:


# ============================================================
# NOTEBOOK 61 — FINAL BENCHMARK / COMPETITION EVALUATION
# Cell 4: Formal Policy-Path Divergence Certification
# ============================================================

import ast

print("=" * 78)
print("NOTEBOOK 61 — POLICY-PATH DIVERGENCE CERTIFICATION")
print("=" * 78)

# ------------------------------------------------------------
# 1. Prerequisites
# ------------------------------------------------------------

assert CELL1_STATUS["validation_status"] == "PASS"
assert CELL2_STATUS["validation_status"] == "PASS"
assert CELL3_STATUS["validation_status"] == "PASS"

print("\n[OK] Cells 1–3 prerequisite certifications confirmed.")

# ------------------------------------------------------------
# 2. Match-level scope certification
# ------------------------------------------------------------

assert len(changed_matches_df) == 11

assert changed_matches_df["match_id"].nunique() == 11

assert changed_matches_df["_merge"].eq("both").all()

assert changed_matches_df["winner_changed"].eq(True).all()

assert changed_matches_df[
    "evaluated_outcome_changed"
].eq(True).all()

print("[OK] Exactly 11 uniquely matched benchmark outcomes changed.")

# ------------------------------------------------------------
# 3. Side and matchup localization
# ------------------------------------------------------------

assert changed_matches_df[
    "evaluated_agent_side"
].eq("Player").all()

assert changed_matches_df[
    "player_card_name"
].eq("Eevee").all()

assert changed_matches_df[
    "opponent_card_name"
].eq("Charmander").all()

assert changed_matches_df[
    "player_card_name_controlled"
].eq("Eevee").all()

assert changed_matches_df[
    "opponent_card_name_controlled"
].eq("Charmander").all()

print("[OK] All 11 changes localize to Player-side Eevee vs Charmander.")

# ------------------------------------------------------------
# 4. Outcome-direction certification
# ------------------------------------------------------------

assert changed_matches_df[
    "historical_winner"
].eq("Player").all()

assert changed_matches_df[
    "controlled_winner"
].eq("Opponent").all()

assert changed_matches_df[
    "historical_evaluated_win"
].eq(1).all()

assert changed_matches_df[
    "historical_evaluated_loss"
].eq(0).all()

assert changed_matches_df[
    "controlled_evaluated_win"
].eq(0).all()

assert changed_matches_df[
    "controlled_evaluated_loss"
].eq(1).all()

assert changed_matches_df[
    "historical_win_to_controlled_loss"
].eq(True).all()

assert changed_matches_df[
    "historical_loss_to_controlled_win"
].eq(False).all()

historical_to_controlled_losses = int(
    changed_matches_df[
        "historical_win_to_controlled_loss"
    ].sum()
)

controlled_gain_count = int(
    changed_matches_df[
        "historical_loss_to_controlled_win"
    ].sum()
)

assert historical_to_controlled_losses == 11
assert controlled_gain_count == 0

print("[OK] Directionality certified: 11 historical wins -> controlled losses.")
print("[OK] Reverse changes certified: 0.")

# ------------------------------------------------------------
# 5. Baseline distribution certification
# ------------------------------------------------------------

baseline_counts = (
    changed_matches_df["baseline_name"]
    .value_counts()
    .to_dict()
)

expected_baseline_counts = {
    "Greedy Damage Baseline": 8,
    "Random Baseline": 3,
}

assert baseline_counts == expected_baseline_counts

# Notebook 60 explicitly reported zero Depth-6 changes.
assert (
    nb60_handoff["changed_by_baseline"]["Depth-6 Search Baseline"]
    == 0
)

assert (
    nb60_handoff["changed_by_baseline"]["Greedy Damage Baseline"]
    == 8
)

assert (
    nb60_handoff["changed_by_baseline"]["Random Baseline"]
    == 3
)

print("[OK] Baseline distribution certified:")
print("     Greedy Damage Baseline : 8")
print("     Random Baseline        : 3")
print("     Depth-6 Search Baseline: 0")

# ------------------------------------------------------------
# 6. Starting-side characterization
# ------------------------------------------------------------

starting_side_counts = (
    changed_matches_df["starting_side"]
    .value_counts()
    .to_dict()
)

print("\nChanged-match starting-side distribution:")

for side, count in starting_side_counts.items():
    print(f"     {side}: {count}")

# Important distinction:
# evaluated_agent_side is always Player,
# but starting_side is not required to always be Player.

assert set(starting_side_counts).issubset(
    {"Player", "Opponent"}
)

# ------------------------------------------------------------
# 7. Decision-audit move divergence
# ------------------------------------------------------------

assert len(decision_audit_df) == 2

assert decision_audit_df["turn_number"].eq(1).all()

assert decision_audit_df["same_move"].eq(False).all()

different_move_count = int(
    (~decision_audit_df["same_move"]).sum()
)

assert different_move_count == 2

print("\n[OK] Both audited turn-1 decisions select different moves.")

# ------------------------------------------------------------
# 8. Fallback-path certification
# ------------------------------------------------------------

assert decision_audit_df[
    "historical_fallback"
].eq(True).all()

assert decision_audit_df[
    "controlled_fallback"
].eq(False).all()

historical_fallback_count = int(
    decision_audit_df["historical_fallback"].sum()
)

controlled_fallback_count = int(
    decision_audit_df["controlled_fallback"].sum()
)

assert historical_fallback_count == 2
assert controlled_fallback_count == 0

print("[OK] Historical fallback decisions: 2 / 2")
print("[OK] Controlled fallback decisions: 0 / 2")

# ------------------------------------------------------------
# 9. Legal-move certification
# ------------------------------------------------------------

parsed_legal_moves = decision_audit_df[
    "legal_moves"
].apply(ast.literal_eval)

for row_index, row in decision_audit_df.iterrows():

    legal_moves = parsed_legal_moves.loc[row_index]

    historical_move = row["historical_move"]
    controlled_move = row["controlled_move"]

    assert historical_move in legal_moves
    assert controlled_move in legal_moves

print("[OK] Historical and controlled selected moves are legal")
print("     within both audited simulator states.")

# ------------------------------------------------------------
# 10. Controlled direct-model-path evidence
# ------------------------------------------------------------

expected_controlled_reason = (
    "Certified 11-class model selected highest-probability "
    "simulator-legal move."
)

assert decision_audit_df[
    "controlled_reason"
].eq(expected_controlled_reason).all()

expected_historical_reason = (
    "R3 legality fallback used because no model probability "
    "mass covered legal actions."
)

assert decision_audit_df[
    "historical_reason"
].eq(expected_historical_reason).all()

assert decision_audit_df[
    "controlled_confidence"
].between(0.0, 1.0).all()

print("[OK] Controlled decisions use certified direct-model legal selection.")
print("[OK] Historical decisions explicitly document R3 fallback behavior.")

# ------------------------------------------------------------
# 11. Cross-check against Notebook 60 handoff
# ------------------------------------------------------------

assert (
    nb60_handoff["changed_match_scope"]
    == "PLAYER_SIDE_EEVEE_VS_CHARMANDER_ONLY"
)

assert nb60_handoff[
    "historical_decision_fallbacks_observed"
] == historical_fallback_count

assert nb60_handoff[
    "controlled_decision_fallbacks_observed"
] == controlled_fallback_count

assert nb60_handoff[
    "different_selected_moves_observed"
] == different_move_count

print("[OK] Match-level and decision-level evidence agrees with")
print("     the certified Notebook 60 handoff.")

# ------------------------------------------------------------
# 12. Scientific interpretation
# ------------------------------------------------------------

POLICY_PATH_FINDING = {
    "changed_matches": 11,
    "changed_match_scope":
        "PLAYER_SIDE_EEVEE_VS_CHARMANDER_ONLY",
    "historical_win_to_controlled_loss": 11,
    "historical_loss_to_controlled_win": 0,
    "changed_by_baseline": {
        "Greedy Damage Baseline": 8,
        "Random Baseline": 3,
        "Depth-6 Search Baseline": 0,
    },
    "audited_decision_states": 2,
    "different_selected_moves": 2,
    "historical_fallbacks": 2,
    "controlled_fallbacks": 0,
    "controlled_moves_legal": True,
    "controlled_direct_model_selection": True,
    "policy_paths_behaviorally_identical": False,
    "benchmark_delta_is_clean_model_degradation_signal": False,
    "retraining_evidence_from_delta": False,
    "model_modification_evidence_from_delta": False,
    "scientific_conclusion": EXPECTED_CONCLUSION,
}

# ------------------------------------------------------------
# 13. Cell 4 certification
# ------------------------------------------------------------

CELL4_STATUS = {
    "match_level_divergence_certified": True,
    "decision_level_divergence_certified": True,
    "changed_matches": 11,
    "all_changed_evaluated_agent_side_player": True,
    "all_changed_matchup_eevee_vs_charmander": True,
    "historical_win_to_controlled_loss": 11,
    "historical_loss_to_controlled_win": 0,
    "greedy_changed_matches": 8,
    "random_changed_matches": 3,
    "depth6_changed_matches": 0,
    "audited_states": 2,
    "different_selected_moves": 2,
    "historical_fallbacks": 2,
    "controlled_fallbacks": 0,
    "controlled_selected_moves_legal": True,
    "policy_paths_behaviorally_identical": False,
    "benchmark_delta_interpreted_as_clean_model_degradation": False,
    "retraining_indicated": False,
    "model_modification_indicated": False,
    "tournament_rerun_performed": False,
    "fit_calls_executed": 0,
    "validation_status": "PASS",
}

print("\n" + "=" * 78)
print("CELL 4 VALIDATION STATUS: PASS")
print("=" * 78)

print(json.dumps(CELL4_STATUS, indent=2))

print("\nSCIENTIFIC FINDING:")
print(EXPECTED_CONCLUSION)

print(
    "\nThe 3.6667 percentage-point benchmark difference is not "
    "treated as a clean estimate of certified-model degradation "
    "because the historical and controlled evaluation paths are "
    "behaviorally different in the audited states."
)

print("\nNo retraining indicated.")
print("No model modification indicated.")
print("No tournament rerun performed.")
print("No fit calls executed.")


# In[8]:


# ============================================================
# NOTEBOOK 61 — FINAL BENCHMARK / COMPETITION EVALUATION
# Cell 5: Certified Model Read-Only Runtime Contract Validation
# ============================================================

import joblib
from sklearn.ensemble import RandomForestClassifier

print("=" * 78)
print("NOTEBOOK 61 — CERTIFIED MODEL RUNTIME CONTRACT VALIDATION")
print("=" * 78)

# ------------------------------------------------------------
# 1. Prerequisite certification
# ------------------------------------------------------------

assert CELL1_STATUS["validation_status"] == "PASS"
assert CELL2_STATUS["validation_status"] == "PASS"
assert CELL3_STATUS["validation_status"] == "PASS"
assert CELL4_STATUS["validation_status"] == "PASS"

print("\n[OK] Cells 1–4 prerequisite certifications confirmed.")

# ------------------------------------------------------------
# 2. Reconfirm physical artifact BEFORE deserialization
# ------------------------------------------------------------

preload_size = CERTIFIED_MODEL_PATH.stat().st_size
preload_hash = sha256_file(CERTIFIED_MODEL_PATH)

assert preload_size == EXPECTED_MODEL_SIZE
assert preload_hash == EXPECTED_MODEL_SHA256

print("\n[OK] Certified artifact integrity reconfirmed before load.")
print(f"     Size   : {preload_size:,} bytes")
print(f"     SHA256 : {preload_hash}")

# ------------------------------------------------------------
# 3. Load certified model
#
# READ-ONLY evaluation:
# - no fit()
# - no partial_fit()
# - no parameter mutation
# - no serialization overwrite
# ------------------------------------------------------------

certified_model = joblib.load(CERTIFIED_MODEL_PATH)

print("\nCertified runtime object loaded.")
print(f"     Python type : {type(certified_model).__name__}")
print(f"     Module      : {type(certified_model).__module__}")

assert isinstance(certified_model, RandomForestClassifier)

print("[OK] Runtime model type = RandomForestClassifier.")

# ------------------------------------------------------------
# 4. Validate fitted-state interface
# ------------------------------------------------------------

required_runtime_attributes = [
    "classes_",
    "n_features_in_",
    "estimators_",
    "predict",
    "predict_proba",
]

missing_runtime_attributes = [
    attribute
    for attribute in required_runtime_attributes
    if not hasattr(certified_model, attribute)
]

assert not missing_runtime_attributes, (
    "Certified model is missing required fitted/runtime attributes: "
    f"{missing_runtime_attributes}"
)

print("[OK] Required fitted inference interface is present.")

# ------------------------------------------------------------
# 5. Feature-count contract
# ------------------------------------------------------------

runtime_feature_count = int(certified_model.n_features_in_)
contract_feature_count = int(nb60_model_contract["feature_count"])

assert runtime_feature_count == 61
assert contract_feature_count == 61
assert runtime_feature_count == contract_feature_count
assert len(nb60_model_contract["feature_names"]) == 61

print("\n[OK] Runtime feature contract validated.")
print(f"     Runtime n_features_in_ : {runtime_feature_count}")
print(f"     Contract feature_count : {contract_feature_count}")
print(
    f"     Contract feature names : "
    f"{len(nb60_model_contract['feature_names'])}"
)

# ------------------------------------------------------------
# 6. Target-class contract
# ------------------------------------------------------------

runtime_classes = [
    str(value)
    for value in certified_model.classes_.tolist()
]

contract_classes = [
    str(value)
    for value in nb60_model_contract["target_classes"]
]

assert len(runtime_classes) == 11
assert len(contract_classes) == 11
assert runtime_classes == contract_classes

print("\n[OK] Runtime target-class contract validated.")
print(f"     Runtime class count : {len(runtime_classes)}")

for index, class_name in enumerate(runtime_classes, start=1):
    print(f"     {index:>2}. {class_name}")

# ------------------------------------------------------------
# 7. Random Forest structural inspection
# ------------------------------------------------------------

estimator_count = len(certified_model.estimators_)

assert estimator_count > 0

print("\nRandom Forest structural metadata:")
print(f"     Number of trees : {estimator_count}")
print(f"     Criterion       : {certified_model.criterion}")
print(f"     Max depth       : {certified_model.max_depth}")
print(f"     Random state    : {certified_model.random_state}")

# ------------------------------------------------------------
# 8. Verify model feature-name metadata if available
# ------------------------------------------------------------

runtime_feature_names_available = hasattr(
    certified_model,
    "feature_names_in_"
)

runtime_feature_names_match = None

if runtime_feature_names_available:

    runtime_feature_names = [
        str(value)
        for value in certified_model.feature_names_in_.tolist()
    ]

    contract_feature_names = [
        str(value)
        for value in nb60_model_contract["feature_names"]
    ]

    runtime_feature_names_match = (
        runtime_feature_names == contract_feature_names
    )

    assert runtime_feature_names_match

    print("\n[OK] Runtime feature_names_in_ exactly matches")
    print("     the certified 61-feature contract.")

else:

    print("\n[INFO] Runtime model does not expose feature_names_in_.")
    print("       Feature dimensionality remains certified through")
    print("       n_features_in_ + Notebook 60 feature contract.")

# ------------------------------------------------------------
# 9. Reconfirm artifact AFTER load
#
# Loading must not modify the certified model file.
# ------------------------------------------------------------

postload_size = CERTIFIED_MODEL_PATH.stat().st_size
postload_hash = sha256_file(CERTIFIED_MODEL_PATH)

assert postload_size == preload_size
assert postload_hash == preload_hash
assert postload_hash == EXPECTED_MODEL_SHA256

print("\n[OK] Certified artifact unchanged after deserialization.")
print(f"     SHA256 remains: {postload_hash}")

# ------------------------------------------------------------
# 10. Explicitly certify that this cell did not train
# ------------------------------------------------------------

MODEL_RUNTIME_CONTRACT = {
    "model_type": type(certified_model).__name__,
    "feature_count": runtime_feature_count,
    "target_class_count": len(runtime_classes),
    "target_classes": runtime_classes,
    "tree_count": estimator_count,
    "predict_available": callable(certified_model.predict),
    "predict_proba_available": callable(
        certified_model.predict_proba
    ),
    "runtime_feature_names_available":
        runtime_feature_names_available,
    "runtime_feature_names_match_contract":
        runtime_feature_names_match,
    "model_sha256_before_load": preload_hash,
    "model_sha256_after_load": postload_hash,
}

CELL5_STATUS = {
    "prerequisite_cells_1_to_4_pass": True,
    "certified_model_loaded": True,
    "runtime_model_type_validated": True,
    "runtime_feature_count": runtime_feature_count,
    "contract_feature_count": contract_feature_count,
    "runtime_target_class_count": len(runtime_classes),
    "contract_target_class_count": len(contract_classes),
    "runtime_classes_match_contract": True,
    "predict_interface_available": True,
    "predict_proba_interface_available": True,
    "certified_artifact_unchanged_after_load": True,
    "fit_calls_executed": 0,
    "retraining_performed": False,
    "model_modified": False,
    "tournament_rerun_performed": False,
    "validation_status": "PASS",
}

print("\n" + "=" * 78)
print("CELL 5 VALIDATION STATUS: PASS")
print("=" * 78)

print(json.dumps(CELL5_STATUS, indent=2))

print("\nCertified model runtime contract is valid.")
print("Model was loaded for read-only structural validation.")
print("No predictions were required in this cell.")
print("No fit calls executed.")
print("No retraining performed.")
print("No model modification performed.")
print("No tournament rerun performed.")


# In[9]:


# ============================================================
# NOTEBOOK 61 — FINAL BENCHMARK / COMPETITION EVALUATION
# Cell 6: Final Benchmark Statistical Characterization
# ============================================================

import math
from statistics import NormalDist

print("=" * 78)
print("NOTEBOOK 61 — FINAL BENCHMARK STATISTICAL CHARACTERIZATION")
print("=" * 78)

# ------------------------------------------------------------
# 1. Prerequisites
# ------------------------------------------------------------

for status_name, status in [
    ("Cell 1", CELL1_STATUS),
    ("Cell 2", CELL2_STATUS),
    ("Cell 3", CELL3_STATUS),
    ("Cell 4", CELL4_STATUS),
    ("Cell 5", CELL5_STATUS),
]:
    assert status["validation_status"] == "PASS", (
        f"{status_name} prerequisite did not pass."
    )

print("\n[OK] Cells 1–5 prerequisite certifications confirmed.")

# ------------------------------------------------------------
# 2. Certified benchmark counts
# ------------------------------------------------------------

historical_n = int(nb60_handoff["historical_matches"])
historical_wins = int(nb60_handoff["historical_wins"])

controlled_n = int(nb60_handoff["controlled_matches"])
controlled_wins = int(nb60_handoff["controlled_wins"])

historical_p = historical_wins / historical_n
controlled_p = controlled_wins / controlled_n

observed_delta_pp = (
    controlled_p - historical_p
) * 100

assert historical_n == 300
assert controlled_n == 300
assert historical_wins == 194
assert controlled_wins == 183

print("\nCertified benchmark counts:")
print(
    f"     Historical : {historical_wins}/{historical_n} "
    f"= {historical_p * 100:.4f}%"
)
print(
    f"     Controlled : {controlled_wins}/{controlled_n} "
    f"= {controlled_p * 100:.4f}%"
)
print(
    f"     Observed delta : {observed_delta_pp:.4f} "
    "percentage points"
)

# ------------------------------------------------------------
# 3. Wilson confidence interval
#
# Appropriate for binomial win-rate characterization.
# No new simulation is performed.
# ------------------------------------------------------------

def wilson_interval(successes, trials, confidence=0.95):

    if trials <= 0:
        raise ValueError("trials must be positive")

    p_hat = successes / trials

    z = NormalDist().inv_cdf(
        1 - (1 - confidence) / 2
    )

    denominator = 1 + (z**2 / trials)

    center = (
        p_hat
        + (z**2 / (2 * trials))
    ) / denominator

    margin = (
        z
        * math.sqrt(
            (p_hat * (1 - p_hat) / trials)
            + (z**2 / (4 * trials**2))
        )
        / denominator
    )

    return center - margin, center + margin


historical_ci_low, historical_ci_high = wilson_interval(
    historical_wins,
    historical_n,
)

controlled_ci_low, controlled_ci_high = wilson_interval(
    controlled_wins,
    controlled_n,
)

print("\n95% Wilson confidence intervals:")

print(
    "     Historical : "
    f"[{historical_ci_low * 100:.3f}%, "
    f"{historical_ci_high * 100:.3f}%]"
)

print(
    "     Controlled : "
    f"[{controlled_ci_low * 100:.3f}%, "
    f"{controlled_ci_high * 100:.3f}%]"
)

# ------------------------------------------------------------
# 4. Standard-error characterization
# ------------------------------------------------------------

historical_se = math.sqrt(
    historical_p * (1 - historical_p)
    / historical_n
)

controlled_se = math.sqrt(
    controlled_p * (1 - controlled_p)
    / controlled_n
)

print("\nBinomial standard errors:")

print(
    f"     Historical : {historical_se * 100:.3f} "
    "percentage points"
)

print(
    f"     Controlled : {controlled_se * 100:.3f} "
    "percentage points"
)

# ------------------------------------------------------------
# 5. Outcome stability characterization
# ------------------------------------------------------------

identical_outcomes = int(
    nb60_handoff["identical_match_outcomes"]
)

changed_outcomes = int(
    nb60_handoff["changed_match_count"]
)

outcome_agreement_rate = (
    identical_outcomes / controlled_n
)

outcome_change_rate = (
    changed_outcomes / controlled_n
)

assert identical_outcomes == 289
assert changed_outcomes == 11
assert identical_outcomes + changed_outcomes == 300

print("\nHistorical/controlled match-outcome agreement:")

print(
    f"     Identical outcomes : "
    f"{identical_outcomes}/300 "
    f"= {outcome_agreement_rate * 100:.3f}%"
)

print(
    f"     Changed outcomes   : "
    f"{changed_outcomes}/300 "
    f"= {outcome_change_rate * 100:.3f}%"
)

# ------------------------------------------------------------
# 6. Delta interpretation guardrail
#
# IMPORTANT:
# Do not interpret this as a conventional independent-policy
# model-performance significance comparison.
#
# Cell 4 established that historical and controlled adapters
# are behaviorally non-identical.
# ------------------------------------------------------------

assert (
    CELL4_STATUS[
        "policy_paths_behaviorally_identical"
    ]
    is False
)

assert (
    CELL4_STATUS[
        "benchmark_delta_interpreted_as_clean_model_degradation"
    ]
    is False
)

STATISTICAL_INTERPRETATION = (
    "The certified controlled benchmark is 183/300 = 61.00%. "
    "Its binomial uncertainty can be characterized directly from "
    "the frozen 300-match evidence. The historical 194/300 = "
    "64.67% result is retained as contextual benchmark evidence, "
    "but the observed -3.67 percentage-point difference is not "
    "treated as a clean model-degradation estimate because "
    "Notebook 60/61 established behaviorally different historical "
    "and controlled policy paths."
)

print("\nInterpretation guardrail:")
print(STATISTICAL_INTERPRETATION)

# ------------------------------------------------------------
# 7. Competition-facing benchmark record
# ------------------------------------------------------------

FINAL_BENCHMARK_STATISTICS = {
    "controlled_matches": controlled_n,
    "controlled_wins": controlled_wins,
    "controlled_win_rate": controlled_p,
    "controlled_win_rate_pct":
        controlled_p * 100,
    "controlled_standard_error_pct_points":
        controlled_se * 100,
    "controlled_wilson_95_ci_low_pct":
        controlled_ci_low * 100,
    "controlled_wilson_95_ci_high_pct":
        controlled_ci_high * 100,

    "historical_matches": historical_n,
    "historical_wins": historical_wins,
    "historical_win_rate": historical_p,
    "historical_win_rate_pct":
        historical_p * 100,
    "historical_standard_error_pct_points":
        historical_se * 100,
    "historical_wilson_95_ci_low_pct":
        historical_ci_low * 100,
    "historical_wilson_95_ci_high_pct":
        historical_ci_high * 100,

    "observed_delta_percentage_points":
        observed_delta_pp,

    "identical_match_outcomes":
        identical_outcomes,
    "changed_match_outcomes":
        changed_outcomes,
    "outcome_agreement_rate_pct":
        outcome_agreement_rate * 100,
    "outcome_change_rate_pct":
        outcome_change_rate * 100,

    "policy_paths_behaviorally_identical":
        False,

    "delta_is_clean_model_degradation_estimate":
        False,

    "independent_policy_significance_test_performed":
        False,

    "additional_matches_executed":
        0,
}

# ------------------------------------------------------------
# 8. Cell 6 certification
# ------------------------------------------------------------

CELL6_STATUS = {
    "prerequisite_cells_1_to_5_pass": True,

    "controlled_benchmark_statistically_characterized":
        True,

    "historical_benchmark_statistically_characterized":
        True,

    "wilson_intervals_computed":
        True,

    "outcome_agreement_characterized":
        True,

    "policy_path_guardrail_preserved":
        True,

    "inappropriate_independent_policy_test_avoided":
        True,

    "additional_tournament_matches_executed":
        0,

    "fit_calls_executed":
        0,

    "retraining_performed":
        False,

    "model_modified":
        False,

    "validation_status":
        "PASS",
}

print("\n" + "=" * 78)
print("CELL 6 VALIDATION STATUS: PASS")
print("=" * 78)

print(json.dumps(CELL6_STATUS, indent=2))

print("\nFINAL CONTROLLED BENCHMARK:")
print(
    f"183 / 300 = {controlled_p * 100:.2f}%"
)

print(
    "95% Wilson CI: "
    f"{controlled_ci_low * 100:.2f}% "
    "to "
    f"{controlled_ci_high * 100:.2f}%"
)

print(
    f"Historical/controlled outcome agreement: "
    f"{outcome_agreement_rate * 100:.2f}%"
)

print("\nNo additional matches executed.")
print("No fit calls executed.")
print("No retraining performed.")
print("No model modification performed.")


# In[10]:


# ============================================================
# NOTEBOOK 61 — FINAL BENCHMARK / COMPETITION EVALUATION
# Cell 6: Final Benchmark Statistical Characterization
# ============================================================

import math
from statistics import NormalDist

print("=" * 78)
print("NOTEBOOK 61 — FINAL BENCHMARK STATISTICAL CHARACTERIZATION")
print("=" * 78)

# ------------------------------------------------------------
# 1. Prerequisites
# ------------------------------------------------------------

for status_name, status in [
    ("Cell 1", CELL1_STATUS),
    ("Cell 2", CELL2_STATUS),
    ("Cell 3", CELL3_STATUS),
    ("Cell 4", CELL4_STATUS),
    ("Cell 5", CELL5_STATUS),
]:
    assert status["validation_status"] == "PASS", (
        f"{status_name} prerequisite did not pass."
    )

print("\n[OK] Cells 1–5 prerequisite certifications confirmed.")

# ------------------------------------------------------------
# 2. Certified benchmark counts
# ------------------------------------------------------------

historical_n = int(nb60_handoff["historical_matches"])
historical_wins = int(nb60_handoff["historical_wins"])

controlled_n = int(nb60_handoff["controlled_matches"])
controlled_wins = int(nb60_handoff["controlled_wins"])

historical_p = historical_wins / historical_n
controlled_p = controlled_wins / controlled_n

observed_delta_pp = (
    controlled_p - historical_p
) * 100

assert historical_n == 300
assert controlled_n == 300
assert historical_wins == 194
assert controlled_wins == 183

print("\nCertified benchmark counts:")
print(
    f"     Historical : {historical_wins}/{historical_n} "
    f"= {historical_p * 100:.4f}%"
)
print(
    f"     Controlled : {controlled_wins}/{controlled_n} "
    f"= {controlled_p * 100:.4f}%"
)
print(
    f"     Observed delta : {observed_delta_pp:.4f} "
    "percentage points"
)

# ------------------------------------------------------------
# 3. Wilson confidence interval
#
# Appropriate for binomial win-rate characterization.
# No new simulation is performed.
# ------------------------------------------------------------

def wilson_interval(successes, trials, confidence=0.95):

    if trials <= 0:
        raise ValueError("trials must be positive")

    p_hat = successes / trials

    z = NormalDist().inv_cdf(
        1 - (1 - confidence) / 2
    )

    denominator = 1 + (z**2 / trials)

    center = (
        p_hat
        + (z**2 / (2 * trials))
    ) / denominator

    margin = (
        z
        * math.sqrt(
            (p_hat * (1 - p_hat) / trials)
            + (z**2 / (4 * trials**2))
        )
        / denominator
    )

    return center - margin, center + margin


historical_ci_low, historical_ci_high = wilson_interval(
    historical_wins,
    historical_n,
)

controlled_ci_low, controlled_ci_high = wilson_interval(
    controlled_wins,
    controlled_n,
)

print("\n95% Wilson confidence intervals:")

print(
    "     Historical : "
    f"[{historical_ci_low * 100:.3f}%, "
    f"{historical_ci_high * 100:.3f}%]"
)

print(
    "     Controlled : "
    f"[{controlled_ci_low * 100:.3f}%, "
    f"{controlled_ci_high * 100:.3f}%]"
)

# ------------------------------------------------------------
# 4. Standard-error characterization
# ------------------------------------------------------------

historical_se = math.sqrt(
    historical_p * (1 - historical_p)
    / historical_n
)

controlled_se = math.sqrt(
    controlled_p * (1 - controlled_p)
    / controlled_n
)

print("\nBinomial standard errors:")

print(
    f"     Historical : {historical_se * 100:.3f} "
    "percentage points"
)

print(
    f"     Controlled : {controlled_se * 100:.3f} "
    "percentage points"
)

# ------------------------------------------------------------
# 5. Outcome stability characterization
# ------------------------------------------------------------

identical_outcomes = int(
    nb60_handoff["identical_match_outcomes"]
)

changed_outcomes = int(
    nb60_handoff["changed_match_count"]
)

outcome_agreement_rate = (
    identical_outcomes / controlled_n
)

outcome_change_rate = (
    changed_outcomes / controlled_n
)

assert identical_outcomes == 289
assert changed_outcomes == 11
assert identical_outcomes + changed_outcomes == 300

print("\nHistorical/controlled match-outcome agreement:")

print(
    f"     Identical outcomes : "
    f"{identical_outcomes}/300 "
    f"= {outcome_agreement_rate * 100:.3f}%"
)

print(
    f"     Changed outcomes   : "
    f"{changed_outcomes}/300 "
    f"= {outcome_change_rate * 100:.3f}%"
)

# ------------------------------------------------------------
# 6. Delta interpretation guardrail
#
# IMPORTANT:
# Do not interpret this as a conventional independent-policy
# model-performance significance comparison.
#
# Cell 4 established that historical and controlled adapters
# are behaviorally non-identical.
# ------------------------------------------------------------

assert (
    CELL4_STATUS[
        "policy_paths_behaviorally_identical"
    ]
    is False
)

assert (
    CELL4_STATUS[
        "benchmark_delta_interpreted_as_clean_model_degradation"
    ]
    is False
)

STATISTICAL_INTERPRETATION = (
    "The certified controlled benchmark is 183/300 = 61.00%. "
    "Its binomial uncertainty can be characterized directly from "
    "the frozen 300-match evidence. The historical 194/300 = "
    "64.67% result is retained as contextual benchmark evidence, "
    "but the observed -3.67 percentage-point difference is not "
    "treated as a clean model-degradation estimate because "
    "Notebook 60/61 established behaviorally different historical "
    "and controlled policy paths."
)

print("\nInterpretation guardrail:")
print(STATISTICAL_INTERPRETATION)

# ------------------------------------------------------------
# 7. Competition-facing benchmark record
# ------------------------------------------------------------

FINAL_BENCHMARK_STATISTICS = {
    "controlled_matches": controlled_n,
    "controlled_wins": controlled_wins,
    "controlled_win_rate": controlled_p,
    "controlled_win_rate_pct":
        controlled_p * 100,
    "controlled_standard_error_pct_points":
        controlled_se * 100,
    "controlled_wilson_95_ci_low_pct":
        controlled_ci_low * 100,
    "controlled_wilson_95_ci_high_pct":
        controlled_ci_high * 100,

    "historical_matches": historical_n,
    "historical_wins": historical_wins,
    "historical_win_rate": historical_p,
    "historical_win_rate_pct":
        historical_p * 100,
    "historical_standard_error_pct_points":
        historical_se * 100,
    "historical_wilson_95_ci_low_pct":
        historical_ci_low * 100,
    "historical_wilson_95_ci_high_pct":
        historical_ci_high * 100,

    "observed_delta_percentage_points":
        observed_delta_pp,

    "identical_match_outcomes":
        identical_outcomes,
    "changed_match_outcomes":
        changed_outcomes,
    "outcome_agreement_rate_pct":
        outcome_agreement_rate * 100,
    "outcome_change_rate_pct":
        outcome_change_rate * 100,

    "policy_paths_behaviorally_identical":
        False,

    "delta_is_clean_model_degradation_estimate":
        False,

    "independent_policy_significance_test_performed":
        False,

    "additional_matches_executed":
        0,
}

# ------------------------------------------------------------
# 8. Cell 6 certification
# ------------------------------------------------------------

CELL6_STATUS = {
    "prerequisite_cells_1_to_5_pass": True,

    "controlled_benchmark_statistically_characterized":
        True,

    "historical_benchmark_statistically_characterized":
        True,

    "wilson_intervals_computed":
        True,

    "outcome_agreement_characterized":
        True,

    "policy_path_guardrail_preserved":
        True,

    "inappropriate_independent_policy_test_avoided":
        True,

    "additional_tournament_matches_executed":
        0,

    "fit_calls_executed":
        0,

    "retraining_performed":
        False,

    "model_modified":
        False,

    "validation_status":
        "PASS",
}

print("\n" + "=" * 78)
print("CELL 6 VALIDATION STATUS: PASS")
print("=" * 78)

print(json.dumps(CELL6_STATUS, indent=2))

print("\nFINAL CONTROLLED BENCHMARK:")
print(
    f"183 / 300 = {controlled_p * 100:.2f}%"
)

print(
    "95% Wilson CI: "
    f"{controlled_ci_low * 100:.2f}% "
    "to "
    f"{controlled_ci_high * 100:.2f}%"
)

print(
    f"Historical/controlled outcome agreement: "
    f"{outcome_agreement_rate * 100:.2f}%"
)

print("\nNo additional matches executed.")
print("No fit calls executed.")
print("No retraining performed.")
print("No model modification performed.")


# In[11]:


# ============================================================
# NOTEBOOK 61 — FINAL BENCHMARK / COMPETITION EVALUATION
# Cell 6: Final Benchmark Statistical Characterization
# ============================================================

import math
from statistics import NormalDist

print("=" * 78)
print("NOTEBOOK 61 — FINAL BENCHMARK STATISTICAL CHARACTERIZATION")
print("=" * 78)

# ------------------------------------------------------------
# 1. Prerequisites
# ------------------------------------------------------------

for status_name, status in [
    ("Cell 1", CELL1_STATUS),
    ("Cell 2", CELL2_STATUS),
    ("Cell 3", CELL3_STATUS),
    ("Cell 4", CELL4_STATUS),
    ("Cell 5", CELL5_STATUS),
]:
    assert status["validation_status"] == "PASS", (
        f"{status_name} prerequisite did not pass."
    )

print("\n[OK] Cells 1–5 prerequisite certifications confirmed.")

# ------------------------------------------------------------
# 2. Certified benchmark counts
# ------------------------------------------------------------

historical_n = int(nb60_handoff["historical_matches"])
historical_wins = int(nb60_handoff["historical_wins"])

controlled_n = int(nb60_handoff["controlled_matches"])
controlled_wins = int(nb60_handoff["controlled_wins"])

historical_p = historical_wins / historical_n
controlled_p = controlled_wins / controlled_n

observed_delta_pp = (
    controlled_p - historical_p
) * 100

assert historical_n == 300
assert controlled_n == 300
assert historical_wins == 194
assert controlled_wins == 183

print("\nCertified benchmark counts:")
print(
    f"     Historical : {historical_wins}/{historical_n} "
    f"= {historical_p * 100:.4f}%"
)
print(
    f"     Controlled : {controlled_wins}/{controlled_n} "
    f"= {controlled_p * 100:.4f}%"
)
print(
    f"     Observed delta : {observed_delta_pp:.4f} "
    "percentage points"
)

# ------------------------------------------------------------
# 3. Wilson confidence interval
#
# Appropriate for binomial win-rate characterization.
# No new simulation is performed.
# ------------------------------------------------------------

def wilson_interval(successes, trials, confidence=0.95):

    if trials <= 0:
        raise ValueError("trials must be positive")

    p_hat = successes / trials

    z = NormalDist().inv_cdf(
        1 - (1 - confidence) / 2
    )

    denominator = 1 + (z**2 / trials)

    center = (
        p_hat
        + (z**2 / (2 * trials))
    ) / denominator

    margin = (
        z
        * math.sqrt(
            (p_hat * (1 - p_hat) / trials)
            + (z**2 / (4 * trials**2))
        )
        / denominator
    )

    return center - margin, center + margin


historical_ci_low, historical_ci_high = wilson_interval(
    historical_wins,
    historical_n,
)

controlled_ci_low, controlled_ci_high = wilson_interval(
    controlled_wins,
    controlled_n,
)

print("\n95% Wilson confidence intervals:")

print(
    "     Historical : "
    f"[{historical_ci_low * 100:.3f}%, "
    f"{historical_ci_high * 100:.3f}%]"
)

print(
    "     Controlled : "
    f"[{controlled_ci_low * 100:.3f}%, "
    f"{controlled_ci_high * 100:.3f}%]"
)

# ------------------------------------------------------------
# 4. Standard-error characterization
# ------------------------------------------------------------

historical_se = math.sqrt(
    historical_p * (1 - historical_p)
    / historical_n
)

controlled_se = math.sqrt(
    controlled_p * (1 - controlled_p)
    / controlled_n
)

print("\nBinomial standard errors:")

print(
    f"     Historical : {historical_se * 100:.3f} "
    "percentage points"
)

print(
    f"     Controlled : {controlled_se * 100:.3f} "
    "percentage points"
)

# ------------------------------------------------------------
# 5. Outcome stability characterization
# ------------------------------------------------------------

identical_outcomes = int(
    nb60_handoff["identical_match_outcomes"]
)

changed_outcomes = int(
    nb60_handoff["changed_match_count"]
)

outcome_agreement_rate = (
    identical_outcomes / controlled_n
)

outcome_change_rate = (
    changed_outcomes / controlled_n
)

assert identical_outcomes == 289
assert changed_outcomes == 11
assert identical_outcomes + changed_outcomes == 300

print("\nHistorical/controlled match-outcome agreement:")

print(
    f"     Identical outcomes : "
    f"{identical_outcomes}/300 "
    f"= {outcome_agreement_rate * 100:.3f}%"
)

print(
    f"     Changed outcomes   : "
    f"{changed_outcomes}/300 "
    f"= {outcome_change_rate * 100:.3f}%"
)

# ------------------------------------------------------------
# 6. Delta interpretation guardrail
#
# IMPORTANT:
# Do not interpret this as a conventional independent-policy
# model-performance significance comparison.
#
# Cell 4 established that historical and controlled adapters
# are behaviorally non-identical.
# ------------------------------------------------------------

assert (
    CELL4_STATUS[
        "policy_paths_behaviorally_identical"
    ]
    is False
)

assert (
    CELL4_STATUS[
        "benchmark_delta_interpreted_as_clean_model_degradation"
    ]
    is False
)

STATISTICAL_INTERPRETATION = (
    "The certified controlled benchmark is 183/300 = 61.00%. "
    "Its binomial uncertainty can be characterized directly from "
    "the frozen 300-match evidence. The historical 194/300 = "
    "64.67% result is retained as contextual benchmark evidence, "
    "but the observed -3.67 percentage-point difference is not "
    "treated as a clean model-degradation estimate because "
    "Notebook 60/61 established behaviorally different historical "
    "and controlled policy paths."
)

print("\nInterpretation guardrail:")
print(STATISTICAL_INTERPRETATION)

# ------------------------------------------------------------
# 7. Competition-facing benchmark record
# ------------------------------------------------------------

FINAL_BENCHMARK_STATISTICS = {
    "controlled_matches": controlled_n,
    "controlled_wins": controlled_wins,
    "controlled_win_rate": controlled_p,
    "controlled_win_rate_pct":
        controlled_p * 100,
    "controlled_standard_error_pct_points":
        controlled_se * 100,
    "controlled_wilson_95_ci_low_pct":
        controlled_ci_low * 100,
    "controlled_wilson_95_ci_high_pct":
        controlled_ci_high * 100,

    "historical_matches": historical_n,
    "historical_wins": historical_wins,
    "historical_win_rate": historical_p,
    "historical_win_rate_pct":
        historical_p * 100,
    "historical_standard_error_pct_points":
        historical_se * 100,
    "historical_wilson_95_ci_low_pct":
        historical_ci_low * 100,
    "historical_wilson_95_ci_high_pct":
        historical_ci_high * 100,

    "observed_delta_percentage_points":
        observed_delta_pp,

    "identical_match_outcomes":
        identical_outcomes,
    "changed_match_outcomes":
        changed_outcomes,
    "outcome_agreement_rate_pct":
        outcome_agreement_rate * 100,
    "outcome_change_rate_pct":
        outcome_change_rate * 100,

    "policy_paths_behaviorally_identical":
        False,

    "delta_is_clean_model_degradation_estimate":
        False,

    "independent_policy_significance_test_performed":
        False,

    "additional_matches_executed":
        0,
}

# ------------------------------------------------------------
# 8. Cell 6 certification
# ------------------------------------------------------------

CELL6_STATUS = {
    "prerequisite_cells_1_to_5_pass": True,

    "controlled_benchmark_statistically_characterized":
        True,

    "historical_benchmark_statistically_characterized":
        True,

    "wilson_intervals_computed":
        True,

    "outcome_agreement_characterized":
        True,

    "policy_path_guardrail_preserved":
        True,

    "inappropriate_independent_policy_test_avoided":
        True,

    "additional_tournament_matches_executed":
        0,

    "fit_calls_executed":
        0,

    "retraining_performed":
        False,

    "model_modified":
        False,

    "validation_status":
        "PASS",
}

print("\n" + "=" * 78)
print("CELL 6 VALIDATION STATUS: PASS")
print("=" * 78)

print(json.dumps(CELL6_STATUS, indent=2))

print("\nFINAL CONTROLLED BENCHMARK:")
print(
    f"183 / 300 = {controlled_p * 100:.2f}%"
)

print(
    "95% Wilson CI: "
    f"{controlled_ci_low * 100:.2f}% "
    "to "
    f"{controlled_ci_high * 100:.2f}%"
)

print(
    f"Historical/controlled outcome agreement: "
    f"{outcome_agreement_rate * 100:.2f}%"
)

print("\nNo additional matches executed.")
print("No fit calls executed.")
print("No retraining performed.")
print("No model modification performed.")


# In[12]:


# ============================================================
# NOTEBOOK 61 — FINAL BENCHMARK / COMPETITION EVALUATION
# Cell 7: Upstream Evidence Inventory — Notebooks 57–59
# ============================================================

print("=" * 78)
print("NOTEBOOK 61 — UPSTREAM EVIDENCE INVENTORY")
print("=" * 78)

# ------------------------------------------------------------
# 1. Prerequisites
# ------------------------------------------------------------

for status_name, status in [
    ("Cell 1", CELL1_STATUS),
    ("Cell 2", CELL2_STATUS),
    ("Cell 3", CELL3_STATUS),
    ("Cell 4", CELL4_STATUS),
    ("Cell 5", CELL5_STATUS),
    ("Cell 6", CELL6_STATUS),
]:
    assert status["validation_status"] == "PASS", (
        f"{status_name} prerequisite did not pass."
    )

print("\n[OK] Cells 1–6 prerequisite certifications confirmed.")

# ------------------------------------------------------------
# 2. Define upstream notebooks
# ------------------------------------------------------------

UPSTREAM_NOTEBOOKS = {
    57: {
        "purpose": "SHAP / explainability foundation",
    },
    58: {
        "purpose": "SHAP / local explainability",
    },
    59: {
        "purpose": "robustness / calibration / stress testing",
    },
}

SEARCH_ROOT_NAMES = [
    "outputs",
    "artifacts",
    "reports",
    "models",
]

# ------------------------------------------------------------
# 3. Inventory helper
# ------------------------------------------------------------

def inventory_directory(path: Path):

    if not path.exists():
        return {
            "exists": False,
            "files": [],
        }

    files = sorted(
        [
            item
            for item in path.rglob("*")
            if item.is_file()
        ],
        key=lambda p: str(p).lower(),
    )

    return {
        "exists": True,
        "files": files,
    }


# ------------------------------------------------------------
# 4. Inventory Notebook 57–59 standard locations
# ------------------------------------------------------------

UPSTREAM_EVIDENCE_INVENTORY = {}

for notebook_number, metadata in UPSTREAM_NOTEBOOKS.items():

    print("\n" + "=" * 78)
    print(
        f"NOTEBOOK {notebook_number} — "
        f"{metadata['purpose'].upper()}"
    )
    print("=" * 78)

    notebook_inventory = {
        "purpose": metadata["purpose"],
        "locations": {},
        "total_files": 0,
    }

    for root_name in SEARCH_ROOT_NAMES:

        directory = (
            PROJECT_ROOT
            / root_name
            / f"notebook{notebook_number}"
        )

        result = inventory_directory(directory)

        notebook_inventory["locations"][root_name] = {
            "path": str(directory),
            "exists": result["exists"],
            "file_count": len(result["files"]),
            "files": [
                str(file)
                for file in result["files"]
            ],
        }

        notebook_inventory["total_files"] += len(
            result["files"]
        )

        print(f"\n{root_name.upper()}")
        print(f"  Path   : {directory}")
        print(f"  Exists : {result['exists']}")
        print(f"  Files  : {len(result['files'])}")

        for file in result["files"]:
            print(
                f"    - {file.name} "
                f"({file.stat().st_size:,} bytes)"
            )

    UPSTREAM_EVIDENCE_INVENTORY[
        notebook_number
    ] = notebook_inventory

# ------------------------------------------------------------
# 5. Locate notebook and script source files
# ------------------------------------------------------------

print("\n" + "=" * 78)
print("NOTEBOOK / SCRIPT SOURCE INVENTORY")
print("=" * 78)

UPSTREAM_SOURCE_FILES = {}

for notebook_number in [57, 58, 59]:

    notebook_matches = sorted(
        NOTEBOOKS_DIR.glob(
            f"{notebook_number}_*.ipynb"
        )
    )

    script_matches = sorted(
        SCRIPTS_DIR.glob(
            f"{notebook_number}_*.py"
        )
    )

    UPSTREAM_SOURCE_FILES[notebook_number] = {
        "notebooks": [
            str(path)
            for path in notebook_matches
        ],
        "scripts": [
            str(path)
            for path in script_matches
        ],
    }

    print(f"\nNotebook {notebook_number}")

    print("  .ipynb files:")
    if notebook_matches:
        for path in notebook_matches:
            print(
                f"    - {path.name} "
                f"({path.stat().st_size:,} bytes)"
            )
    else:
        print("    - NONE FOUND")

    print("  .py files:")
    if script_matches:
        for path in script_matches:
            print(
                f"    - {path.name} "
                f"({path.stat().st_size:,} bytes)"
            )
    else:
        print("    - NONE FOUND")

# ------------------------------------------------------------
# 6. Summarize evidence availability
# ------------------------------------------------------------

print("\n" + "=" * 78)
print("UPSTREAM EVIDENCE SUMMARY")
print("=" * 78)

for notebook_number in [57, 58, 59]:

    inventory = UPSTREAM_EVIDENCE_INVENTORY[
        notebook_number
    ]

    source_info = UPSTREAM_SOURCE_FILES[
        notebook_number
    ]

    print(
        f"\nNotebook {notebook_number}: "
        f"{inventory['purpose']}"
    )

    print(
        f"  Persisted evidence files : "
        f"{inventory['total_files']}"
    )

    print(
        f"  Notebook files           : "
        f"{len(source_info['notebooks'])}"
    )

    print(
        f"  Script files             : "
        f"{len(source_info['scripts'])}"
    )

# ------------------------------------------------------------
# 7. Inspection status
# ------------------------------------------------------------

CELL7_STATUS = {
    "prerequisite_cells_1_to_6_pass": True,
    "notebooks_inventoried": [57, 58, 59],
    "standard_locations_checked": SEARCH_ROOT_NAMES,
    "source_notebooks_checked": True,
    "source_scripts_checked": True,
    "files_modified": 0,
    "models_loaded": 0,
    "predictions_executed": 0,
    "additional_tournament_matches_executed": 0,
    "fit_calls_executed": 0,
    "retraining_performed": False,
    "model_modified": False,
    "inspection_status": "PASS",
}

print("\n" + "=" * 78)
print("CELL 7 INSPECTION STATUS: PASS")
print("=" * 78)

print(json.dumps(CELL7_STATUS, indent=2))

print(
    "\nNotebook 57–59 evidence locations have been "
    "inventoried without modifying project state."
)


# In[13]:


# ============================================================
# NOTEBOOK 61 — FINAL BENCHMARK / COMPETITION EVALUATION
# Cell 8: Upstream Summary + Gate Evidence Inspection
# ============================================================

print("=" * 78)
print("NOTEBOOK 61 — UPSTREAM SUMMARY AND GATE EVIDENCE")
print("=" * 78)

# ------------------------------------------------------------
# 1. Prerequisites
# ------------------------------------------------------------

for status_name, status in [
    ("Cell 1", CELL1_STATUS),
    ("Cell 2", CELL2_STATUS),
    ("Cell 3", CELL3_STATUS),
    ("Cell 4", CELL4_STATUS),
    ("Cell 5", CELL5_STATUS),
    ("Cell 6", CELL6_STATUS),
]:
    assert status["validation_status"] == "PASS"

assert CELL7_STATUS["inspection_status"] == "PASS"

print("\n[OK] Cells 1–7 prerequisite state confirmed.")

# ------------------------------------------------------------
# 2. Define exact persisted summary evidence
# ------------------------------------------------------------

NB57_SUMMARY = (
    PROJECT_ROOT
    / "artifacts"
    / "notebook57"
    / "notebook57_summary.json"
)

NB57_EXECUTIVE_FINDINGS = (
    PROJECT_ROOT
    / "reports"
    / "notebook57"
    / "section7_executive_findings.csv"
)

NB57_POLICY_RECOMMENDATION = (
    PROJECT_ROOT
    / "reports"
    / "notebook57"
    / "section7_policy_recommendation.csv"
)

NB58_SUMMARY = (
    PROJECT_ROOT
    / "artifacts"
    / "notebook58"
    / "notebook58_summary.json"
)

NB58_GATE_CHECKLIST = (
    PROJECT_ROOT
    / "reports"
    / "notebook58"
    / "notebook58_gate_checklist.csv"
)

NB58_RELEASE_SUMMARY = (
    PROJECT_ROOT
    / "reports"
    / "notebook58"
    / "notebook58_release_summary.csv"
)

NB58_RISK_ASSESSMENT = (
    PROJECT_ROOT
    / "reports"
    / "notebook58"
    / "notebook58_risk_assessment.csv"
)

NB58_VALIDATION_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "notebook58"
    / "notebook58_validation_report.csv"
)

NB59_SUMMARY = (
    PROJECT_ROOT
    / "artifacts"
    / "notebook59"
    / "notebook59_r3_live_integration_summary.json"
)

NB59_PAIRWISE = (
    PROJECT_ROOT
    / "artifacts"
    / "notebook59"
    / "r3_pairwise_matchup_summary.csv"
)

NB59_STARTING_SIDE = (
    PROJECT_ROOT
    / "artifacts"
    / "notebook59"
    / "r3_starting_side_summary.csv"
)

NB59_ACTION_USAGE = (
    PROJECT_ROOT
    / "artifacts"
    / "notebook59"
    / "r3_action_usage.csv"
)

# ------------------------------------------------------------
# 3. Verify all expected summary files exist
# ------------------------------------------------------------

UPSTREAM_SUMMARY_FILES = {
    "NB57_SUMMARY": NB57_SUMMARY,
    "NB57_EXECUTIVE_FINDINGS": NB57_EXECUTIVE_FINDINGS,
    "NB57_POLICY_RECOMMENDATION": NB57_POLICY_RECOMMENDATION,

    "NB58_SUMMARY": NB58_SUMMARY,
    "NB58_GATE_CHECKLIST": NB58_GATE_CHECKLIST,
    "NB58_RELEASE_SUMMARY": NB58_RELEASE_SUMMARY,
    "NB58_RISK_ASSESSMENT": NB58_RISK_ASSESSMENT,
    "NB58_VALIDATION_REPORT": NB58_VALIDATION_REPORT,

    "NB59_SUMMARY": NB59_SUMMARY,
    "NB59_PAIRWISE": NB59_PAIRWISE,
    "NB59_STARTING_SIDE": NB59_STARTING_SIDE,
    "NB59_ACTION_USAGE": NB59_ACTION_USAGE,
}

print("\nSummary-file existence check:")

for label, path in UPSTREAM_SUMMARY_FILES.items():

    exists = path.exists()

    print(f"  {'[OK]' if exists else '[MISSING]'} {label}")
    print(f"       {path}")

    assert exists, f"Required upstream summary file missing: {path}"

print("\n[OK] All selected upstream summary files exist.")

# ------------------------------------------------------------
# 4. Load Notebook 57 summary evidence
# ------------------------------------------------------------

with NB57_SUMMARY.open("r", encoding="utf-8-sig") as file:
    nb57_summary = json.load(file)

nb57_executive_findings_df = pd.read_csv(
    NB57_EXECUTIVE_FINDINGS
)

nb57_policy_recommendation_df = pd.read_csv(
    NB57_POLICY_RECOMMENDATION
)

print("\n" + "=" * 78)
print("NOTEBOOK 57 — SUMMARY JSON")
print("=" * 78)
print(json.dumps(nb57_summary, indent=2))

print("\nNotebook 57 — Executive findings")
print("Columns:")
print(nb57_executive_findings_df.columns.tolist())

with pd.option_context(
    "display.max_columns", None,
    "display.max_rows", None,
    "display.width", 220,
    "display.max_colwidth", 160,
):
    print(nb57_executive_findings_df.to_string(index=False))

print("\nNotebook 57 — Policy recommendation")
print("Columns:")
print(nb57_policy_recommendation_df.columns.tolist())

with pd.option_context(
    "display.max_columns", None,
    "display.max_rows", None,
    "display.width", 220,
    "display.max_colwidth", 160,
):
    print(nb57_policy_recommendation_df.to_string(index=False))

# ------------------------------------------------------------
# 5. Load Notebook 58 summary/gate evidence
# ------------------------------------------------------------

with NB58_SUMMARY.open("r", encoding="utf-8-sig") as file:
    nb58_summary = json.load(file)

nb58_gate_checklist_df = pd.read_csv(
    NB58_GATE_CHECKLIST
)

nb58_release_summary_df = pd.read_csv(
    NB58_RELEASE_SUMMARY
)

nb58_risk_assessment_df = pd.read_csv(
    NB58_RISK_ASSESSMENT
)

nb58_validation_report_df = pd.read_csv(
    NB58_VALIDATION_REPORT
)

print("\n" + "=" * 78)
print("NOTEBOOK 58 — SUMMARY JSON")
print("=" * 78)
print(json.dumps(nb58_summary, indent=2))

for title, dataframe in [
    ("Gate checklist", nb58_gate_checklist_df),
    ("Release summary", nb58_release_summary_df),
    ("Risk assessment", nb58_risk_assessment_df),
    ("Validation report", nb58_validation_report_df),
]:
    print(f"\nNotebook 58 — {title}")
    print("Columns:")
    print(dataframe.columns.tolist())

    with pd.option_context(
        "display.max_columns", None,
        "display.max_rows", None,
        "display.width", 240,
        "display.max_colwidth", 180,
    ):
        print(dataframe.to_string(index=False))

# ------------------------------------------------------------
# 6. Load Notebook 59 summary/live-evaluation evidence
# ------------------------------------------------------------

with NB59_SUMMARY.open("r", encoding="utf-8-sig") as file:
    nb59_summary = json.load(file)

nb59_pairwise_df = pd.read_csv(
    NB59_PAIRWISE
)

nb59_starting_side_df = pd.read_csv(
    NB59_STARTING_SIDE
)

nb59_action_usage_df = pd.read_csv(
    NB59_ACTION_USAGE
)

print("\n" + "=" * 78)
print("NOTEBOOK 59 — SUMMARY JSON")
print("=" * 78)
print(json.dumps(nb59_summary, indent=2))

for title, dataframe in [
    ("Pairwise matchup summary", nb59_pairwise_df),
    ("Starting-side summary", nb59_starting_side_df),
    ("Action usage", nb59_action_usage_df),
]:
    print(f"\nNotebook 59 — {title}")
    print("Columns:")
    print(dataframe.columns.tolist())

    with pd.option_context(
        "display.max_columns", None,
        "display.max_rows", None,
        "display.width", 240,
        "display.max_colwidth", 180,
    ):
        print(dataframe.to_string(index=False))

# ------------------------------------------------------------
# 7. Record inspection state
# ------------------------------------------------------------

CELL8_STATUS = {
    "prerequisite_cells_1_to_7_confirmed": True,

    "notebook57_summary_loaded": True,
    "notebook57_executive_findings_loaded": True,
    "notebook57_policy_recommendation_loaded": True,

    "notebook58_summary_loaded": True,
    "notebook58_gate_checklist_loaded": True,
    "notebook58_release_summary_loaded": True,
    "notebook58_risk_assessment_loaded": True,
    "notebook58_validation_report_loaded": True,

    "notebook59_summary_loaded": True,
    "notebook59_pairwise_summary_loaded": True,
    "notebook59_starting_side_summary_loaded": True,
    "notebook59_action_usage_loaded": True,

    "upstream_schema_inspection_complete": True,

    "competition_go_no_go_decision_made": False,

    "files_modified": 0,
    "models_modified": 0,
    "fit_calls_executed": 0,
    "additional_tournament_matches_executed": 0,
    "retraining_performed": False,

    "inspection_status": "PASS",
}

print("\n" + "=" * 78)
print("CELL 8 INSPECTION STATUS: PASS")
print("=" * 78)

print(json.dumps(CELL8_STATUS, indent=2))

print("\nUpstream Notebook 57–59 summary evidence is now loaded.")
print("No GO/NO-GO conclusion has been imposed yet.")
print("No files modified.")
print("No fitting or retraining performed.")
print("No tournament rerun performed.")


# In[14]:


# ============================================================
# NOTEBOOK 61 — FINAL BENCHMARK / COMPETITION EVALUATION
# Cell 9: Competition-Readiness Evidence Synthesis
#         + Model-Lineage Guardrail
# ============================================================

print("=" * 78)
print("NOTEBOOK 61 — COMPETITION-READINESS EVIDENCE SYNTHESIS")
print("=" * 78)

# ------------------------------------------------------------
# 1. Prerequisites
# ------------------------------------------------------------

for status_name, status, status_key in [
    ("Cell 1", CELL1_STATUS, "validation_status"),
    ("Cell 2", CELL2_STATUS, "validation_status"),
    ("Cell 3", CELL3_STATUS, "validation_status"),
    ("Cell 4", CELL4_STATUS, "validation_status"),
    ("Cell 5", CELL5_STATUS, "validation_status"),
    ("Cell 6", CELL6_STATUS, "validation_status"),
    ("Cell 7", CELL7_STATUS, "inspection_status"),
    ("Cell 8", CELL8_STATUS, "inspection_status"),
]:
    assert status[status_key] == "PASS", (
        f"{status_name} prerequisite did not pass."
    )

print("\n[OK] Cells 1–8 prerequisite state confirmed.")

# ------------------------------------------------------------
# 2. Notebook 57 developmental evidence
# ------------------------------------------------------------

assert nb57_summary["notebook"] == 57
assert nb57_summary["evaluation_cases"] == 184
assert nb57_summary["strategy_count"] == 4

assert (
    nb57_summary["primary_candidate"]
    == "R3_STATE_CENTRIC_POLICY"
)

assert (
    nb57_summary["status"]
    == "PROBABILITY_ATTRIBUTION_AND_FEATURE_IMPORTANCE_COMPLETE"
)

assert len(nb57_executive_findings_df) == 12

assert (
    nb57_executive_findings_df[
        "finding_id"
    ].nunique()
    == 12
)

print("\n[OK] Notebook 57 explainability evidence validated.")
print("     Evaluation cases : 184")
print("     Strategies       : 4")
print("     Primary candidate: R3_STATE_CENTRIC_POLICY")

# ------------------------------------------------------------
# 3. Notebook 58 controlled validation evidence
# ------------------------------------------------------------

assert nb58_summary["notebook"] == 58

assert (
    nb58_summary["release_decision"]
    == "GO"
)

assert (
    nb58_summary["production_policy"]
    == "R3_STATE_CENTRIC_POLICY"
)

assert nb58_summary["feature_count"] == 46
assert nb58_summary["controlled_validation_risk"] == "LOW"
assert nb58_summary["action_agreement_with_r0"] == 1.0
assert nb58_summary["maximum_reproduction_error"] == 0.0
assert nb58_summary["ready_for_next_stage"] is True

assert nb58_gate_checklist_df["Passed"].all()

print("\n[OK] Notebook 58 controlled-validation gate validated.")
print("     Release decision : GO")
print("     Controlled risk  : LOW")
print("     Feature count    : 46")
print("     Agreement with R0: 100%")
print("     Reproduction err : 0.0")

# ------------------------------------------------------------
# 4. Notebook 59 live integration evidence
# ------------------------------------------------------------

assert nb59_summary["notebook"] == 59

assert (
    nb59_summary["production_strategy"]
    == "R3_STATE_CENTRIC_POLICY"
)

assert (
    nb59_summary["production_model_type"]
    == "RandomForestClassifier"
)

assert nb59_summary["retained_feature_count"] == 46
assert nb59_summary["model_class_count"] == 6

assert nb59_summary["controlled_battles"] == 12
assert nb59_summary["total_live_decisions"] == 86
assert nb59_summary["model_driven_decisions"] == 86
assert nb59_summary["fallback_decisions"] == 0

assert nb59_summary["model_driven_rate"] == 1.0
assert nb59_summary["fallback_rate"] == 0.0
assert nb59_summary["terminal_battle_rate"] == 1.0

assert (
    nb59_summary["pairwise_winner_side_consistency"]
    == 1.0
)

assert nb59_summary["integration_status"] == "PASSED"

assert (
    nb59_summary["benchmark_status"]
    == "READY_FOR_NOTEBOOK_60"
)

print("\n[OK] Notebook 59 live integration evidence validated.")
print("     Controlled battles : 12")
print("     Live decisions      : 86")
print("     Model-driven        : 86")
print("     Fallback decisions  : 0")
print("     Integration status  : PASSED")

# ------------------------------------------------------------
# 5. Explicit model-lineage guardrail
#
# IMPORTANT:
# NB57–59 evidence belongs to the earlier R3 development stage.
# The final Notebook 60 certified model has a different
# runtime contract: 61 features / 11 classes.
# ------------------------------------------------------------

upstream_feature_count = int(
    nb59_summary["retained_feature_count"]
)

upstream_class_count = int(
    nb59_summary["model_class_count"]
)

final_feature_count = int(
    CELL5_STATUS["runtime_feature_count"]
)

final_class_count = int(
    CELL5_STATUS["runtime_target_class_count"]
)

assert upstream_feature_count == 46
assert upstream_class_count == 6

assert final_feature_count == 61
assert final_class_count == 11

same_runtime_contract = (
    upstream_feature_count == final_feature_count
    and
    upstream_class_count == final_class_count
)

assert same_runtime_contract is False

print("\n" + "-" * 78)
print("MODEL-LINEAGE GUARDRAIL")
print("-" * 78)

print(
    f"Notebook 59 developmental runtime : "
    f"{upstream_feature_count} features / "
    f"{upstream_class_count} classes"
)

print(
    f"Notebook 60 certified runtime     : "
    f"{final_feature_count} features / "
    f"{final_class_count} classes"
)

print(
    "\n[OK] Runtime-contract evolution detected and explicitly "
    "recorded."
)

print(
    "[OK] Notebook 57–59 metrics will be treated as upstream "
    "developmental evidence, not falsely attributed directly "
    "to the final 61-feature / 11-class certified model."
)

# ------------------------------------------------------------
# 6. Final-model authoritative evidence
# ------------------------------------------------------------

assert CELL1_STATUS["validation_status"] == "PASS"
assert CELL2_STATUS["validation_status"] == "PASS"
assert CELL4_STATUS["validation_status"] == "PASS"
assert CELL5_STATUS["validation_status"] == "PASS"
assert CELL6_STATUS["validation_status"] == "PASS"

assert (
    CELL5_STATUS["runtime_feature_count"]
    == 61
)

assert (
    CELL5_STATUS["runtime_target_class_count"]
    == 11
)

assert (
    CELL5_STATUS[
        "certified_artifact_unchanged_after_load"
    ]
    is True
)

assert (
    CELL4_STATUS[
        "controlled_selected_moves_legal"
    ]
    is True
)

assert (
    CELL4_STATUS["controlled_fallbacks"]
    == 0
)

assert (
    CELL4_STATUS["retraining_indicated"]
    is False
)

assert (
    CELL4_STATUS["model_modification_indicated"]
    is False
)

assert (
    FINAL_BENCHMARK_STATISTICS["controlled_matches"]
    == 300
)

assert (
    FINAL_BENCHMARK_STATISTICS["controlled_wins"]
    == 183
)

assert abs(
    FINAL_BENCHMARK_STATISTICS[
        "controlled_win_rate"
    ]
    - 0.61
) < 1e-12

print("\n[OK] Final-model authoritative evidence validated.")
print("     Certified features : 61")
print("     Certified classes  : 11")
print("     Controlled record  : 183 / 300")
print("     Controlled win rate: 61.00%")
print("     Controlled fallback: 0 in audited divergence states")

# ------------------------------------------------------------
# 7. Scientific interpretation of benchmark delta
# ------------------------------------------------------------

assert (
    CELL4_STATUS[
        "policy_paths_behaviorally_identical"
    ]
    is False
)

assert (
    CELL4_STATUS[
        "benchmark_delta_interpreted_as_clean_model_degradation"
    ]
    is False
)

assert (
    FINAL_BENCHMARK_STATISTICS[
        "independent_policy_significance_test_performed"
    ]
    is False
)

assert (
    nb60_handoff["scientific_conclusion"]
    == EXPECTED_CONCLUSION
)

print("\n[OK] Benchmark interpretation guardrail preserved.")
print(
    "     The -3.6667 percentage-point delta is not treated "
    "as a clean certified-model degradation estimate."
)

# ------------------------------------------------------------
# 8. Evidence-pillar synthesis
# ------------------------------------------------------------

COMPETITION_EVIDENCE_PILLARS = {
    "developmental_explainability": {
        "source": "Notebook 57",
        "status": "PASS",
        "scope":
            "Upstream R3 strategy development evidence",
        "direct_final_model_validation": False,
    },

    "developmental_controlled_validation": {
        "source": "Notebook 58",
        "status": "PASS",
        "scope":
            "46-feature R3 controlled validation",
        "direct_final_model_validation": False,
    },

    "developmental_live_integration": {
        "source": "Notebook 59",
        "status": "PASS",
        "scope":
            "46-feature / 6-class R3 live simulator integration",
        "direct_final_model_validation": False,
    },

    "final_tournament_benchmark": {
        "source": "Notebook 60",
        "status": "PASS",
        "scope":
            "61-feature / 11-class certified controlled benchmark",
        "direct_final_model_validation": True,
    },

    "final_runtime_integrity": {
        "source": "Notebook 61 Cells 1 and 5",
        "status": "PASS",
        "scope":
            "Certified model hash, type, feature and class contract",
        "direct_final_model_validation": True,
    },

    "final_policy_path_diagnostics": {
        "source": "Notebook 61 Cells 2–4",
        "status": "PASS",
        "scope":
            "Changed-match and fallback-path interpretation",
        "direct_final_model_validation": True,
    },

    "final_statistical_characterization": {
        "source": "Notebook 61 Cell 6",
        "status": "PASS",
        "scope":
            "Frozen 300-match benchmark uncertainty characterization",
        "direct_final_model_validation": True,
    },
}

for pillar in COMPETITION_EVIDENCE_PILLARS.values():
    assert pillar["status"] == "PASS"

# ------------------------------------------------------------
# 9. Readiness gate
#
# This is a Notebook 61 evaluation gate.
# It does NOT package or submit anything.
# Packaging remains Notebook 62.
# ------------------------------------------------------------

blocking_conditions = []

if CELL5_STATUS["runtime_feature_count"] != 61:
    blocking_conditions.append(
        "Certified runtime feature count mismatch."
    )

if CELL5_STATUS["runtime_target_class_count"] != 11:
    blocking_conditions.append(
        "Certified runtime class count mismatch."
    )

if not CELL5_STATUS[
    "certified_artifact_unchanged_after_load"
]:
    blocking_conditions.append(
        "Certified artifact integrity failure."
    )

if CELL4_STATUS["controlled_fallbacks"] != 0:
    blocking_conditions.append(
        "Controlled audited decisions contain fallback."
    )

if CELL4_STATUS["retraining_indicated"]:
    blocking_conditions.append(
        "Evidence indicates retraining is required."
    )

if CELL4_STATUS["model_modification_indicated"]:
    blocking_conditions.append(
        "Evidence indicates model modification is required."
    )

competition_evaluation_gate = (
    "GO_TO_NOTEBOOK_62"
    if len(blocking_conditions) == 0
    else "NO_GO_REQUIRES_REVIEW"
)

assert competition_evaluation_gate == "GO_TO_NOTEBOOK_62"

# ------------------------------------------------------------
# 10. Notebook 61 scientific conclusion
# ------------------------------------------------------------

NOTEBOOK61_COMPETITION_CONCLUSION = {
    "evaluation_gate": competition_evaluation_gate,

    "certified_model":
        str(CERTIFIED_MODEL_PATH),

    "certified_model_sha256":
        EXPECTED_MODEL_SHA256,

    "certified_feature_count":
        61,

    "certified_target_class_count":
        11,

    "controlled_benchmark_matches":
        300,

    "controlled_benchmark_wins":
        183,

    "controlled_benchmark_win_rate":
        0.61,

    "controlled_wilson_95_ci_pct": [
        FINAL_BENCHMARK_STATISTICS[
            "controlled_wilson_95_ci_low_pct"
        ],
        FINAL_BENCHMARK_STATISTICS[
            "controlled_wilson_95_ci_high_pct"
        ],
    ],

    "historical_context_win_rate":
        194 / 300,

    "benchmark_delta_percentage_points":
        nb60_handoff[
            "benchmark_delta_percentage_points"
        ],

    "policy_paths_behaviorally_identical":
        False,

    "benchmark_delta_is_clean_model_degradation":
        False,

    "retraining_required":
        False,

    "model_modification_required":
        False,

    "additional_tournament_rerun_required":
        False,

    "upstream_57_59_directly_same_runtime_contract":
        False,

    "upstream_57_59_role":
        "DEVELOPMENTAL_SUPPORTING_EVIDENCE",

    "notebook60_61_role":
        "AUTHORITATIVE_FINAL_MODEL_EVIDENCE",

    "blocking_conditions":
        blocking_conditions,

    "next_notebook":
        62,

    "notebook62_scope":
        "FINAL_PACKAGING_DOCUMENTATION_AND_SUBMISSION",
}

# ------------------------------------------------------------
# 11. Cell 9 certification
# ------------------------------------------------------------

CELL9_STATUS = {
    "prerequisite_cells_1_to_8_pass": True,

    "notebook57_evidence_validated": True,
    "notebook58_evidence_validated": True,
    "notebook59_evidence_validated": True,

    "model_lineage_difference_explicitly_recorded": True,

    "upstream_metrics_not_misattributed_to_final_model": True,

    "final_model_authoritative_evidence_validated": True,

    "competition_evidence_pillars_synthesized": True,

    "blocking_condition_count":
        len(blocking_conditions),

    "competition_evaluation_gate":
        competition_evaluation_gate,

    "retraining_required":
        False,

    "model_modification_required":
        False,

    "additional_tournament_rerun_required":
        False,

    "fit_calls_executed":
        0,

    "next_notebook":
        62,

    "validation_status":
        "PASS",
}

print("\n" + "=" * 78)
print("CELL 9 VALIDATION STATUS: PASS")
print("=" * 78)

print(json.dumps(CELL9_STATUS, indent=2))

print("\n" + "=" * 78)
print("NOTEBOOK 61 COMPETITION EVALUATION GATE")
print("=" * 78)

print(f"\nDecision: {competition_evaluation_gate}")

print("\nCertified final model:")
print(f"  Type     : {type(certified_model).__name__}")
print("  Features : 61")
print("  Classes  : 11")
print(f"  SHA256   : {EXPECTED_MODEL_SHA256}")

print("\nCertified controlled benchmark:")
print("  Wins     : 183 / 300")
print("  Win rate : 61.00%")

print(
    "  95% CI   : "
    f"{FINAL_BENCHMARK_STATISTICS['controlled_wilson_95_ci_low_pct']:.2f}%"
    " to "
    f"{FINAL_BENCHMARK_STATISTICS['controlled_wilson_95_ci_high_pct']:.2f}%"
)

print("\nScientific conclusion:")
print(EXPECTED_CONCLUSION)

print(
    "\nNotebook 57–59 evidence is retained as developmental "
    "supporting evidence."
)

print(
    "Notebook 60–61 evidence is authoritative for the final "
    "61-feature / 11-class certified model."
)

print("\nNo blocking condition detected.")
print("No retraining required.")
print("No model modification required.")
print("No additional tournament rerun required.")
print("No fit calls executed.")

print(
    "\nIf this cell remains PASS, Notebook 61 supports "
    "progression to Notebook 62 for final packaging, "
    "documentation, and submission."
)


# In[16]:


# ============================================================
# NOTEBOOK 61 — FINAL BENCHMARK / COMPETITION EVALUATION
# Cell 10: Persist Final Competition-Evaluation Handoff
# ============================================================

import csv

print("=" * 78)
print("NOTEBOOK 61 — FINAL EVALUATION HANDOFF PERSISTENCE")
print("=" * 78)

# ------------------------------------------------------------
# 1. Prerequisites
# ------------------------------------------------------------

for status_name, status, status_key in [
    ("Cell 1", CELL1_STATUS, "validation_status"),
    ("Cell 2", CELL2_STATUS, "validation_status"),
    ("Cell 3", CELL3_STATUS, "validation_status"),
    ("Cell 4", CELL4_STATUS, "validation_status"),
    ("Cell 5", CELL5_STATUS, "validation_status"),
    ("Cell 6", CELL6_STATUS, "validation_status"),
    ("Cell 7", CELL7_STATUS, "inspection_status"),
    ("Cell 8", CELL8_STATUS, "inspection_status"),
    ("Cell 9", CELL9_STATUS, "validation_status"),
]:
    assert status[status_key] == "PASS", (
        f"{status_name} prerequisite did not pass."
    )

assert (
    CELL9_STATUS["competition_evaluation_gate"]
    == "GO_TO_NOTEBOOK_62"
)

assert CELL9_STATUS["blocking_condition_count"] == 0

print("\n[OK] Cells 1–9 certification state confirmed.")
print("[OK] Notebook 61 evaluation gate = GO_TO_NOTEBOOK_62.")

# ------------------------------------------------------------
# 2. Ensure Notebook 61 persistence directories exist
# ------------------------------------------------------------

NB61_OUTPUTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

NB61_ARTIFACTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("\n[OK] Notebook 61 persistence directories available.")
print(f"     Outputs   : {NB61_OUTPUTS_DIR}")
print(f"     Artifacts : {NB61_ARTIFACTS_DIR}")

# ------------------------------------------------------------
# 3. Build final Notebook 61 evaluation record
# ------------------------------------------------------------

NOTEBOOK61_FINAL_EVALUATION = {
    "notebook": 61,

    "title":
        "Final Benchmark / Competition Evaluation",

    "evaluation_status":
        "COMPLETE",

    "competition_evaluation_gate":
        "GO_TO_NOTEBOOK_62",

    "blocking_condition_count":
        0,

    "blocking_conditions":
        [],

    "certified_model": {
        "model_type":
            "RandomForestClassifier",

        "runtime_feature_count":
            61,

        "runtime_target_class_count":
            11,

        "tree_count":
            MODEL_RUNTIME_CONTRACT["tree_count"],

        "source_path":
            str(CERTIFIED_MODEL_PATH),

        "size_bytes":
            EXPECTED_MODEL_SIZE,

        "sha256":
            EXPECTED_MODEL_SHA256,

        "model_copied_in_notebook61":
            False,

        "model_modified":
            False,

        "retraining_performed":
            False,

        "fit_calls_executed":
            0,
    },

    "controlled_benchmark": {
        "matches":
            300,

        "wins":
            183,

        "win_rate":
            183 / 300,

        "win_rate_pct":
            61.0,

        "wilson_95_ci_low_pct":
            FINAL_BENCHMARK_STATISTICS[
                "controlled_wilson_95_ci_low_pct"
            ],

        "wilson_95_ci_high_pct":
            FINAL_BENCHMARK_STATISTICS[
                "controlled_wilson_95_ci_high_pct"
            ],
    },

    "historical_context": {
        "matches":
            300,

        "wins":
            194,

        "win_rate":
            194 / 300,

        "win_rate_pct":
            (194 / 300) * 100,

        "benchmark_delta_percentage_points":
            nb60_handoff[
                "benchmark_delta_percentage_points"
            ],

        "identical_match_outcomes":
            289,

        "changed_match_outcomes":
            11,

        "outcome_agreement_rate_pct":
            (289 / 300) * 100,
    },

    "policy_path_diagnostics": {
        "changed_match_scope":
            "PLAYER_SIDE_EEVEE_VS_CHARMANDER_ONLY",

        "historical_win_to_controlled_loss":
            11,

        "historical_loss_to_controlled_win":
            0,

        "greedy_damage_changed_matches":
            8,

        "random_baseline_changed_matches":
            3,

        "depth6_changed_matches":
            0,

        "audited_states":
            2,

        "different_selected_moves":
            2,

        "historical_fallbacks":
            2,

        "controlled_fallbacks":
            0,

        "controlled_selected_moves_legal":
            True,

        "policy_paths_behaviorally_identical":
            False,
    },

    "scientific_interpretation": {
        "benchmark_delta_is_clean_model_degradation":
            False,

        "independent_policy_significance_test_performed":
            False,

        "retraining_required":
            False,

        "model_modification_required":
            False,

        "additional_tournament_rerun_required":
            False,

        "scientific_conclusion":
            EXPECTED_CONCLUSION,
    },

    "model_lineage": {
        "notebook57_59_role":
            "DEVELOPMENTAL_SUPPORTING_EVIDENCE",

        "notebook59_runtime_feature_count":
            46,

        "notebook59_runtime_class_count":
            6,

        "notebook60_61_role":
            "AUTHORITATIVE_FINAL_MODEL_EVIDENCE",

        "notebook60_runtime_feature_count":
            61,

        "notebook60_runtime_class_count":
            11,

        "same_runtime_contract":
            False,
    },

    "execution_controls": {
        "additional_matches_executed_in_notebook61":
            0,

        "fit_calls_executed":
            0,

        "retraining_performed":
            False,

        "model_modified":
            False,
    },

    "next_notebook":
        62,

    "next_notebook_scope":
        "FINAL_PACKAGING_DOCUMENTATION_AND_SUBMISSION",
}

# ------------------------------------------------------------
# 4. Build certified model reference
#
# NOTE:
# We reference the frozen Notebook 60 model.
# We DO NOT copy the 22 MB model into Notebook 61.
# ------------------------------------------------------------

NOTEBOOK61_MODEL_REFERENCE = {
    "reference_type":
        "FROZEN_CERTIFIED_MODEL_REFERENCE",

    "source_notebook":
        60,

    "referenced_by_notebook":
        61,

    "model_path":
        str(CERTIFIED_MODEL_PATH),

    "model_type":
        "RandomForestClassifier",

    "feature_count":
        61,

    "target_class_count":
        11,

    "tree_count":
        MODEL_RUNTIME_CONTRACT["tree_count"],

    "size_bytes":
        EXPECTED_MODEL_SIZE,

    "sha256":
        EXPECTED_MODEL_SHA256,

    "physical_model_integrity_verified":
        True,

    "runtime_contract_verified":
        True,

    "model_duplicated":
        False,

    "model_modified":
        False,

    "retraining_performed":
        False,

    "fit_calls_executed":
        0,
}

# ------------------------------------------------------------
# 5. Build evidence-pillar table
# ------------------------------------------------------------

evidence_pillar_rows = []

for pillar_name, pillar in COMPETITION_EVIDENCE_PILLARS.items():

    evidence_pillar_rows.append({
        "pillar":
            pillar_name,

        "source":
            pillar["source"],

        "status":
            pillar["status"],

        "scope":
            pillar["scope"],

        "direct_final_model_validation":
            pillar[
                "direct_final_model_validation"
            ],
    })

evidence_pillars_df = pd.DataFrame(
    evidence_pillar_rows
)

# ------------------------------------------------------------
# 6. Persistence helper
# ------------------------------------------------------------

def write_json(path: Path, payload):

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:

        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )

        file.write("\n")


# ------------------------------------------------------------
# 7. Define output filenames
# ------------------------------------------------------------

FINAL_EVALUATION_FILENAME = (
    "notebook61_final_competition_evaluation.json"
)

MODEL_REFERENCE_FILENAME = (
    "notebook61_certified_model_reference.json"
)

EVIDENCE_PILLARS_FILENAME = (
    "notebook61_competition_evidence_pillars.csv"
)

# ------------------------------------------------------------
# 8. Write evidence to outputs + artifacts
# ------------------------------------------------------------

PERSISTED_FILES = []

for destination_dir in [
    NB61_OUTPUTS_DIR,
    NB61_ARTIFACTS_DIR,
]:

    final_evaluation_path = (
        destination_dir
        / FINAL_EVALUATION_FILENAME
    )

    model_reference_path = (
        destination_dir
        / MODEL_REFERENCE_FILENAME
    )

    evidence_pillars_path = (
        destination_dir
        / EVIDENCE_PILLARS_FILENAME
    )

    write_json(
        final_evaluation_path,
        NOTEBOOK61_FINAL_EVALUATION,
    )

    write_json(
        model_reference_path,
        NOTEBOOK61_MODEL_REFERENCE,
    )

    evidence_pillars_df.to_csv(
        evidence_pillars_path,
        index=False,
    )

    PERSISTED_FILES.extend([
        final_evaluation_path,
        model_reference_path,
        evidence_pillars_path,
    ])

print("\n[OK] Core Notebook 61 evidence persisted.")

# ------------------------------------------------------------
# 9. Verify paired output/artifact files are identical
# ------------------------------------------------------------

paired_filenames = [
    FINAL_EVALUATION_FILENAME,
    MODEL_REFERENCE_FILENAME,
    EVIDENCE_PILLARS_FILENAME,
]

pair_hash_records = []

for filename in paired_filenames:

    output_path = (
        NB61_OUTPUTS_DIR
        / filename
    )

    artifact_path = (
        NB61_ARTIFACTS_DIR
        / filename
    )

    assert output_path.exists()
    assert artifact_path.exists()

    output_hash = sha256_file(
        output_path
    )

    artifact_hash = sha256_file(
        artifact_path
    )

    assert output_hash == artifact_hash

    pair_hash_records.append({
        "filename":
            filename,

        "output_path":
            str(output_path),

        "artifact_path":
            str(artifact_path),

        "sha256":
            output_hash,

        "copies_identical":
            True,
    })

    print(f"\n[OK] {filename}")
    print(f"     SHA256: {output_hash}")
    print("     Output/artifact copies identical.")

# ------------------------------------------------------------
# 10. Create handoff manifest
# ------------------------------------------------------------

manifest_rows = []

for path in PERSISTED_FILES:

    manifest_rows.append({
        "filename":
            path.name,

        "location":
            str(path.parent),

        "full_path":
            str(path),

        "exists":
            path.exists(),

        "size_bytes":
            path.stat().st_size,

        "sha256":
            sha256_file(path),
    })

# Add the referenced certified model without duplicating it.

manifest_rows.append({
    "filename":
        CERTIFIED_MODEL_PATH.name,

    "location":
        "models/notebook60 (REFERENCE ONLY)",

    "full_path":
        str(CERTIFIED_MODEL_PATH),

    "exists":
        CERTIFIED_MODEL_PATH.exists(),

    "size_bytes":
        CERTIFIED_MODEL_PATH.stat().st_size,

    "sha256":
        sha256_file(CERTIFIED_MODEL_PATH),
})

handoff_manifest_df_61 = pd.DataFrame(
    manifest_rows
)

MANIFEST_FILENAME = (
    "notebook61_handoff_manifest.csv"
)

for destination_dir in [
    NB61_OUTPUTS_DIR,
    NB61_ARTIFACTS_DIR,
]:

    manifest_path = (
        destination_dir
        / MANIFEST_FILENAME
    )

    handoff_manifest_df_61.to_csv(
        manifest_path,
        index=False,
    )

# ------------------------------------------------------------
# 11. Verify manifest copies
# ------------------------------------------------------------

output_manifest = (
    NB61_OUTPUTS_DIR
    / MANIFEST_FILENAME
)

artifact_manifest = (
    NB61_ARTIFACTS_DIR
    / MANIFEST_FILENAME
)

assert output_manifest.exists()
assert artifact_manifest.exists()

assert (
    sha256_file(output_manifest)
    ==
    sha256_file(artifact_manifest)
)

print("\n[OK] Notebook 61 handoff manifests created.")
print("[OK] Manifest output/artifact copies are identical.")

# ------------------------------------------------------------
# 12. Final persisted-state validation
# ------------------------------------------------------------

assert (
    NOTEBOOK61_FINAL_EVALUATION[
        "competition_evaluation_gate"
    ]
    == "GO_TO_NOTEBOOK_62"
)

assert (
    NOTEBOOK61_FINAL_EVALUATION[
        "scientific_interpretation"
    ]["retraining_required"]
    is False
)

assert (
    NOTEBOOK61_FINAL_EVALUATION[
        "scientific_interpretation"
    ]["model_modification_required"]
    is False
)

assert (
    NOTEBOOK61_MODEL_REFERENCE[
        "sha256"
    ]
    == EXPECTED_MODEL_SHA256
)

assert (
    NOTEBOOK61_MODEL_REFERENCE[
        "model_duplicated"
    ]
    is False
)

# Reconfirm original model was not changed.

final_model_hash_check = sha256_file(
    CERTIFIED_MODEL_PATH
)

assert (
    final_model_hash_check
    == EXPECTED_MODEL_SHA256
)

# ------------------------------------------------------------
# 13. Cell 10 certification
# ------------------------------------------------------------

CELL10_STATUS = {
    "prerequisite_cells_1_to_9_pass":
        True,

    "final_evaluation_persisted":
        True,

    "model_reference_persisted":
        True,

    "evidence_pillars_persisted":
        True,

    "handoff_manifest_persisted":
        True,

    "output_artifact_pairs_identical":
        True,

    "certified_model_referenced_not_duplicated":
        True,

    "certified_model_sha256_unchanged":
        True,

    "competition_evaluation_gate":
        "GO_TO_NOTEBOOK_62",

    "blocking_conditions":
        0,

    "files_written":
        8,

    "fit_calls_executed":
        0,

    "retraining_performed":
        False,

    "model_modified":
        False,

    "additional_tournament_matches_executed":
        0,

    "validation_status":
        "PASS",
}

print("\n" + "=" * 78)
print("CELL 10 VALIDATION STATUS: PASS")
print("=" * 78)

print(json.dumps(CELL10_STATUS, indent=2))

print("\nPersisted Notebook 61 handoff files:")

for filename in [
    FINAL_EVALUATION_FILENAME,
    MODEL_REFERENCE_FILENAME,
    EVIDENCE_PILLARS_FILENAME,
    MANIFEST_FILENAME,
]:
    print(f"  outputs/notebook61/{filename}")
    print(f"  artifacts/notebook61/{filename}")

print("\nCertified model remains referenced at:")
print(f"  {CERTIFIED_MODEL_PATH}")

print("\nCertified model SHA256:")
print(f"  {final_model_hash_check}")

print("\nNo model copy created.")
print("No model modification performed.")
print("No retraining performed.")
print("No tournament rerun performed.")

print(
    "\nNotebook 61 final evaluation evidence is "
    "persisted and ready for repository-level verification."
)


# In[ ]:




