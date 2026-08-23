#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =============================================================================
# NOTEBOOK 64 — CELL 1
# NB63 HANDOFF + STRATEGY INTERPRETATION INITIALIZATION
# =============================================================================

from __future__ import annotations

import hashlib
import json
from pathlib import Path


print("=" * 92)
print("NOTEBOOK 64 — MATCHUP / DECISION / STRATEGY INTERPRETATION")
print("CELL 1 — NB63 HANDOFF + EVIDENCE INITIALIZATION")
print("=" * 92)


# =============================================================================
# 1. PROJECT ROOT RESOLUTION
# =============================================================================

PROJECT_ROOT = Path(
    r"D:\02_AI_and_Data\Kaggle-AI-Agents\PTCG_AI_Battle_Challenge"
).resolve()


SUBMISSION_DIR = (
    PROJECT_ROOT
    /
    "submission"
)


NB64_ARTIFACT_DIR = (
    PROJECT_ROOT
    /
    "artifacts"
    /
    "notebook64"
)


NB64_OUTPUT_DIR = (
    PROJECT_ROOT
    /
    "outputs"
    /
    "notebook64"
)


NB64_REPORT_DIR = (
    PROJECT_ROOT
    /
    "reports"
    /
    "notebook64"
)


for directory in [
    NB64_ARTIFACT_DIR,
    NB64_OUTPUT_DIR,
    NB64_REPORT_DIR,
]:

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


print()
print("Project root :", PROJECT_ROOT)
print("Submission   :", SUBMISSION_DIR)
print("NB64 artifacts:", NB64_ARTIFACT_DIR)
print("NB64 outputs  :", NB64_OUTPUT_DIR)
print("NB64 reports  :", NB64_REPORT_DIR)


assert PROJECT_ROOT.is_dir()
assert SUBMISSION_DIR.is_dir()


print()
print("[OK] Project root resolved.")
print("[OK] NB64 evidence directories ready.")


# =============================================================================
# 2. HASH HELPER
# =============================================================================

def sha256_file_nb64(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):

            digest.update(
                chunk
            )

    return digest.hexdigest()


# =============================================================================
# 3. AUTHORITATIVE CERTIFIED ASSETS
# =============================================================================

FINAL_MAIN = (
    SUBMISSION_DIR
    /
    "main.py"
)


CERTIFIED_MODEL = (
    PROJECT_ROOT
    /
    "models"
    /
    "notebook60"
    /
    "notebook60_certified_61_feature_random_forest.joblib"
)


DEPLOYMENT_PPO_CHECKPOINT = (
    SUBMISSION_DIR
    /
    "final_agent"
    /
    "models"
    /
    "ppo_training_loop.pkl"
)


EXPECTED_MAIN_SHA256 = (
    "e328508c39b6c48411f46ad43f5ec1ffe34f54f740aed24389f88d8166dc97cd"
)


EXPECTED_MODEL_SHA256 = (
    "615baf3d5725ccca0c1c94e37f62fb24a68613248e167bb566bef4a6d50f2581"
)


EXPECTED_PPO_SHA256 = (
    "4e6a19861eb5be5f61bc86508dff07dd838a42df9350355d95e8f2dc436ae787"
)


assert FINAL_MAIN.is_file()
assert CERTIFIED_MODEL.is_file()
assert DEPLOYMENT_PPO_CHECKPOINT.is_file()


main_hash_nb64 = (
    sha256_file_nb64(
        FINAL_MAIN
    )
)


model_hash_nb64 = (
    sha256_file_nb64(
        CERTIFIED_MODEL
    )
)


ppo_hash_nb64 = (
    sha256_file_nb64(
        DEPLOYMENT_PPO_CHECKPOINT
    )
)


assert (
    main_hash_nb64
    ==
    EXPECTED_MAIN_SHA256
)


assert (
    model_hash_nb64
    ==
    EXPECTED_MODEL_SHA256
)


assert (
    ppo_hash_nb64
    ==
    EXPECTED_PPO_SHA256
)


print()
print("=" * 92)
print("CERTIFIED ASSET FREEZE")
print("=" * 92)

print(
    "main.py SHA256 :",
    main_hash_nb64,
)

print(
    "RF model SHA256:",
    model_hash_nb64,
)

print(
    "PPO SHA256     :",
    ppo_hash_nb64,
)


print()
print("[OK] Certified NB62 main.py preserved.")
print("[OK] Certified NB60 RF model preserved.")
print("[OK] Deployment PPO checkpoint preserved.")


# =============================================================================
# 4. NB63 -> NB64 HANDOFF FILES
# =============================================================================

NB63_HANDOFF_PATH = (
    PROJECT_ROOT
    /
    "artifacts"
    /
    "notebook63"
    /
    "notebook63_to_notebook64_handoff.json"
)


NB63_FINAL_PACKAGE_PATH = (
    PROJECT_ROOT
    /
    "artifacts"
    /
    "notebook63"
    /
    "notebook63_final_evidence_package.json"
)


NB63_ANALYSIS_PATH = (
    PROJECT_ROOT
    /
    "reports"
    /
    "notebook63"
    /
    "notebook63_full_campaign_analysis.json"
)


NB63_RECORDS_PATH = (
    PROJECT_ROOT
    /
    "outputs"
    /
    "notebook63"
    /
    "cell12b_full_campaign_records.json"
)


NB63_SCENARIOS_PATH = (
    PROJECT_ROOT
    /
    "artifacts"
    /
    "notebook63"
    /
    "cell10_schema_correct_scenarios.json"
)


required_handoff_files_nb64 = [
    NB63_HANDOFF_PATH,
    NB63_FINAL_PACKAGE_PATH,
    NB63_ANALYSIS_PATH,
    NB63_RECORDS_PATH,
    NB63_SCENARIOS_PATH,
]


print()
print("=" * 92)
print("NB63 -> NB64 HANDOFF FILES")
print("=" * 92)


for path in required_handoff_files_nb64:

    assert path.is_file(), (
        f"Missing NB63 handoff file: {path}"
    )

    print(
        "[OK]",
        path.relative_to(
            PROJECT_ROOT
        ),
    )


# =============================================================================
# 5. LOAD NB63 HANDOFF / ANALYSIS
# =============================================================================

NB63_HANDOFF = json.loads(
    NB63_HANDOFF_PATH.read_text(
        encoding="utf-8"
    )
)


NB63_FINAL_PACKAGE = json.loads(
    NB63_FINAL_PACKAGE_PATH.read_text(
        encoding="utf-8"
    )
)


NB63_ANALYSIS = json.loads(
    NB63_ANALYSIS_PATH.read_text(
        encoding="utf-8"
    )
)


NB63_RECORDS = json.loads(
    NB63_RECORDS_PATH.read_text(
        encoding="utf-8"
    )
)


NB63_SCENARIOS = json.loads(
    NB63_SCENARIOS_PATH.read_text(
        encoding="utf-8"
    )
)


assert (
    len(
        NB63_RECORDS
    )
    ==
    240
)


assert (
    len(
        NB63_SCENARIOS
    )
    ==
    4
)


print()
print("[OK] NB63 final package loaded.")
print("[OK] NB63 campaign analysis loaded.")
print("[OK] 240 battle records loaded.")
print("[OK] 4 corrected scenarios loaded.")


# =============================================================================
# 6. NB63 CORE HANDOFF SUMMARY
# =============================================================================

NB63_CORE_HANDOFF = {

    "campaign_configuration_sha256":
        NB63_HANDOFF[
            "campaign_configuration_sha256"
        ],

    "key_runtime_result":
        NB63_HANDOFF[
            "key_runtime_result"
        ],

    "key_semantic_result":
        NB63_HANDOFF[
            "key_semantic_result"
        ],

    "key_behavioral_result":
        NB63_HANDOFF[
            "key_behavioral_result"
        ],

    "important_limitation":
        NB63_HANDOFF[
            "important_limitation"
        ],
}


print()
print("=" * 92)
print("NB63 CORE HANDOFF")
print("=" * 92)

print(
    json.dumps(
        NB63_CORE_HANDOFF,
        indent=2,
    )
)


# =============================================================================
# 7. NOTEBOOK 64 INTERPRETATION POLICY
# =============================================================================

NB64_INTERPRETATION_POLICY = {

    "purpose":
        "MATCHUP_DECISION_STRATEGY_INTERPRETATION",

    "simulation_execution_allowed":
        False,

    "training_allowed":
        False,

    "retraining_allowed":
        False,

    "submission_writes_allowed":
        False,

    "certified_model_mutation_allowed":
        False,

    "analysis_only":
        True,

    "primary_evidence_sources": [
        "NB57",
        "NB58",
        "NB59",
        "NB60",
        "NB61",
        "NB62",
        "NB63",
    ],

    "nb63_role":
        (
            "Controlled runtime, semantic, side-neutrality, "
            "and behavioral evidence."
        ),

    "nb63_non_claim":
        (
            "Do not claim policy superiority from NB63."
        ),

    "required_analysis_dimensions": [

        "scenario_matchups",

        "normal_vs_swapped orientation",

        "damage efficiency",

        "turn efficiency",

        "deployment vs deterministic baselines",

        "decision-pattern equivalence",

        "controlled-environment limitations",

        "integration with certified upstream evidence",
    ],
}


print()
print("=" * 92)
print("NOTEBOOK 64 INTERPRETATION POLICY")
print("=" * 92)

print(
    json.dumps(
        NB64_INTERPRETATION_POLICY,
        indent=2,
    )
)


# =============================================================================
# 8. CELL 1 EVIDENCE SNAPSHOT
# =============================================================================

cell1_evidence_nb64 = {

    "notebook":
        64,

    "cell":
        1,

    "purpose":
        "NB63_HANDOFF_AND_STRATEGY_INTERPRETATION_INITIALIZATION",

    "project_root":
        str(
            PROJECT_ROOT
        ),

    "nb63_battle_records":
        len(
            NB63_RECORDS
        ),

    "nb63_scenarios":
        len(
            NB63_SCENARIOS
        ),

    "nb63_handoff":
        NB63_CORE_HANDOFF,

    "interpretation_policy":
        NB64_INTERPRETATION_POLICY,

    "main_sha256":
        main_hash_nb64,

    "certified_rf_sha256":
        model_hash_nb64,

    "deployment_ppo_sha256":
        ppo_hash_nb64,

    "simulation_executed":
        False,

    "training_executed":
        False,

    "submission_modified":
        False,

    "model_modified":
        False,
}


cell1_artifact_nb64 = (
    NB64_ARTIFACT_DIR
    /
    "cell1_nb63_handoff_initialization.json"
)


cell1_output_nb64 = (
    NB64_OUTPUT_DIR
    /
    "cell1_nb63_handoff_initialization.json"
)


for path in [
    cell1_artifact_nb64,
    cell1_output_nb64,
]:

    path.write_text(
        json.dumps(
            cell1_evidence_nb64,
            indent=2,
        ),
        encoding="utf-8",
    )


print()
print("=" * 92)
print("CELL 1 EVIDENCE PERSISTENCE")
print("=" * 92)

print(
    "[OK]",
    cell1_artifact_nb64,
)

print(
    "[OK]",
    cell1_output_nb64,
)


# =============================================================================
# 9. FINAL STATUS
# =============================================================================

CELL1_NB64_STATUS = {

    "project_root_resolved":
        True,

    "nb63_handoff_loaded":
        True,

    "nb63_final_package_loaded":
        True,

    "nb63_analysis_loaded":
        True,

    "nb63_records_loaded":
        True,

    "nb63_record_count":
        len(
            NB63_RECORDS
        ),

    "nb63_scenario_count":
        len(
            NB63_SCENARIOS
        ),

    "analysis_only":
        True,

    "simulation_prohibited":
        True,

    "training_prohibited":
        True,

    "submission_writes_prohibited":
        True,

    "main_integrity_preserved":
        True,

    "certified_model_integrity_preserved":
        True,

    "deployment_checkpoint_integrity_preserved":
        True,

    "validation_status":
        "PASS",
}


print()
print("=" * 92)
print("NOTEBOOK 64 — CELL 1 STATUS: PASS")
print("=" * 92)

print(
    json.dumps(
        CELL1_NB64_STATUS,
        indent=2,
    )
)


print()
print(
    "Notebook 64 interpretation environment initialized."
)

print(
    "NEXT STEP: Cell 2 — scenario / matchup performance matrix."
)


# In[2]:


# =============================================================================
# NOTEBOOK 64 — CELL 2
# SCENARIO / MATCHUP PERFORMANCE MATRIX
# =============================================================================

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


print("=" * 92)
print("NOTEBOOK 64 — CELL 2")
print("SCENARIO / MATCHUP PERFORMANCE MATRIX")
print("=" * 92)


# =============================================================================
# 0. PREREQUISITES
# =============================================================================

assert (
    CELL1_NB64_STATUS[
        "validation_status"
    ]
    ==
    "PASS"
)

assert (
    CELL1_NB64_STATUS[
        "nb63_record_count"
    ]
    ==
    240
)

assert (
    CELL1_NB64_STATUS[
        "analysis_only"
    ]
    is True
)

assert (
    CELL1_NB64_STATUS[
        "simulation_prohibited"
    ]
    is True
)

assert (
    CELL1_NB64_STATUS[
        "training_prohibited"
    ]
    is True
)


print()
print("[OK] Cell 1 PASS confirmed.")
print("[OK] 240 NB63 battle records available.")
print("[OK] Analysis-only restrictions preserved.")
print("[OK] No simulation or training will execute.")


# =============================================================================
# 1. EXPECTED CAMPAIGN STRUCTURE
# =============================================================================

AGENT_IDS_NB64 = [

    "DEPLOYMENT_POLICY",

    "FIRST_LEGAL_MOVE",

    "MAX_DAMAGE_LEGAL_MOVE",
]


SCENARIO_IDS_NB64 = [

    "S1_BALANCED_LOW_HP",

    "S2_BALANCED_HIGH_HP",

    "S3_PLAYER_DAMAGE_ADVANTAGE",

    "S4_OPPONENT_DAMAGE_ADVANTAGE",
]


SIDE_MODES_NB64 = [

    "NORMAL",

    "SWAPPED",
]


SEEDS_NB64 = sorted(
    {
        int(
            row[
                "seed"
            ]
        )
        for row
        in NB63_RECORDS
    }
)


assert (
    len(
        SEEDS_NB64
    )
    ==
    10
)


print()
print("=" * 92)
print("1. CAMPAIGN STRUCTURE")
print("=" * 92)

print(
    "Agents:",
    AGENT_IDS_NB64,
)

print(
    "Scenarios:",
    SCENARIO_IDS_NB64,
)

print(
    "Side modes:",
    SIDE_MODES_NB64,
)

print(
    "Seed count:",
    len(
        SEEDS_NB64
    ),
)


# =============================================================================
# 2. RECORD INTEGRITY AUDIT
# =============================================================================

record_count_by_agent_nb64 = defaultdict(
    int
)

record_count_by_scenario_nb64 = defaultdict(
    int
)

record_count_by_side_nb64 = defaultdict(
    int
)


for row in NB63_RECORDS:

    record_count_by_agent_nb64[
        row[
            "agent_variant"
        ]
    ] += 1

    record_count_by_scenario_nb64[
        row[
            "scenario_id"
        ]
    ] += 1

    record_count_by_side_nb64[
        row[
            "side_mode"
        ]
    ] += 1


assert (
    dict(
        record_count_by_agent_nb64
    )
    ==
    {
        "DEPLOYMENT_POLICY":
            80,

        "FIRST_LEGAL_MOVE":
            80,

        "MAX_DAMAGE_LEGAL_MOVE":
            80,
    }
)


assert (
    dict(
        record_count_by_scenario_nb64
    )
    ==
    {
        "S1_BALANCED_LOW_HP":
            60,

        "S2_BALANCED_HIGH_HP":
            60,

        "S3_PLAYER_DAMAGE_ADVANTAGE":
            60,

        "S4_OPPONENT_DAMAGE_ADVANTAGE":
            60,
    }
)


assert (
    dict(
        record_count_by_side_nb64
    )
    ==
    {
        "NORMAL":
            120,

        "SWAPPED":
            120,
    }
)


print()
print("=" * 92)
print("2. RECORD INTEGRITY AUDIT")
print("=" * 92)

print(
    "By agent:",
    json.dumps(
        dict(
            record_count_by_agent_nb64
        ),
        indent=2,
    ),
)

print(
    "By scenario:",
    json.dumps(
        dict(
            record_count_by_scenario_nb64
        ),
        indent=2,
    ),
)

print(
    "By side:",
    json.dumps(
        dict(
            record_count_by_side_nb64
        ),
        indent=2,
    ),
)


print()
print(
    "[OK] Full 240-record campaign structure confirmed."
)


# =============================================================================
# 3. SCENARIO DESCRIPTIONS
# =============================================================================

SCENARIO_DESCRIPTION_NB64 = {}


for scenario in NB63_SCENARIOS:

    SCENARIO_DESCRIPTION_NB64[
        scenario[
            "scenario_id"
        ]
    ] = {

        "description":
            scenario[
                "description"
            ],

        "player_hp":
            float(
                scenario[
                    "player"
                ][
                    "hp"
                ]
            ),

        "opponent_hp":
            float(
                scenario[
                    "opponent"
                ][
                    "hp"
                ]
            ),

        "player_energy":
            int(
                scenario[
                    "player"
                ][
                    "energy"
                ]
            ),

        "opponent_energy":
            int(
                scenario[
                    "opponent"
                ][
                    "energy"
                ]
            ),

        "player_attacks":
            [
                attack[
                    "Move Name"
                ]
                for attack
                in scenario[
                    "player"
                ][
                    "attacks"
                ]
            ],

        "opponent_attacks":
            [
                attack[
                    "Move Name"
                ]
                for attack
                in scenario[
                    "opponent"
                ][
                    "attacks"
                ]
            ],
    }


print()
print("=" * 92)
print("3. SCENARIO DEFINITIONS")
print("=" * 92)


for scenario_id in SCENARIO_IDS_NB64:

    print()
    print(
        scenario_id
    )

    print(
        json.dumps(
            SCENARIO_DESCRIPTION_NB64[
                scenario_id
            ],
            indent=2,
        )
    )


# =============================================================================
# 4. HELPERS
# =============================================================================

def mean_or_none_nb64(
    values,
):

    cleaned = [
        float(
            value
        )
        for value
        in values
        if value is not None
    ]

    if not cleaned:

        return None

    return float(
        statistics.mean(
            cleaned
        )
    )


def median_or_none_nb64(
    values,
):

    cleaned = [
        float(
            value
        )
        for value
        in values
        if value is not None
    ]

    if not cleaned:

        return None

    return float(
        statistics.median(
            cleaned
        )
    )


def proportion_nb64(
    numerator,
    denominator,
):

    if denominator <= 0:

        return None

    return float(
        numerator
        /
        denominator
    )


# =============================================================================
# 5. BUILD SCENARIO × AGENT PERFORMANCE MATRIX
# =============================================================================

SCENARIO_AGENT_MATRIX_NB64 = []


for scenario_id in SCENARIO_IDS_NB64:

    for agent_id in AGENT_IDS_NB64:

        rows = [

            row
            for row
            in NB63_RECORDS

            if (
                row[
                    "scenario_id"
                ]
                ==
                scenario_id

                and

                row[
                    "agent_variant"
                ]
                ==
                agent_id
            )
        ]


        assert (
            len(
                rows
            )
            ==
            20
        ), (
            f"Expected 20 records for "
            f"{scenario_id} / {agent_id}; "
            f"found {len(rows)}."
        )


        player_wins = sum(

            1
            for row
            in rows

            if (
                row[
                    "normalized_outcome"
                ]
                ==
                "PLAYER_WIN"
            )
        )


        opponent_wins = sum(

            1
            for row
            in rows

            if (
                row[
                    "normalized_outcome"
                ]
                ==
                "OPPONENT_WIN"
            )
        )


        timeouts = sum(

            1
            for row
            in rows

            if (
                row[
                    "normalized_outcome"
                ]
                ==
                "TIMEOUT_DRAW"
            )
        )


        runtime_failures = sum(

            1
            for row
            in rows

            if row[
                "runtime_exception"
            ]
        )


        total_moves = sum(

            int(
                row[
                    "total_moves"
                ]
                or
                0
            )

            for row
            in rows
        )


        total_damage = sum(

            float(
                row[
                    "total_damage"
                ]
                or
                0.0
            )

            for row
            in rows
        )


        unknown_moves = sum(

            int(
                row[
                    "unknown_move_count"
                ]
                or
                0
            )

            for row
            in rows
        )


        zero_damage_moves = sum(

            int(
                row[
                    "zero_damage_move_count"
                ]
                or
                0
            )

            for row
            in rows
        )


        positive_damage_moves = sum(

            int(
                row[
                    "positive_damage_move_count"
                ]
                or
                0
            )

            for row
            in rows
        )


        mean_turn_count = (
            mean_or_none_nb64(
                [
                    row[
                        "turn_count"
                    ]
                    for row
                    in rows
                ]
            )
        )


        median_turn_count = (
            median_or_none_nb64(
                [
                    row[
                        "turn_count"
                    ]
                    for row
                    in rows
                ]
            )
        )


        mean_player_final_hp = (
            mean_or_none_nb64(
                [
                    row[
                        "player_final_hp"
                    ]
                    for row
                    in rows
                ]
            )
        )


        mean_opponent_final_hp = (
            mean_or_none_nb64(
                [
                    row[
                        "opponent_final_hp"
                    ]
                    for row
                    in rows
                ]
            )
        )


        hp_margin = (

            mean_player_final_hp
            -
            mean_opponent_final_hp

            if (
                mean_player_final_hp
                is not None

                and

                mean_opponent_final_hp
                is not None
            )

            else None
        )


        SCENARIO_AGENT_MATRIX_NB64.append({

            "scenario_id":
                scenario_id,

            "agent_variant":
                agent_id,

            "battle_count":
                len(
                    rows
                ),

            "player_wins":
                player_wins,

            "opponent_wins":
                opponent_wins,

            "timeouts":
                timeouts,

            "runtime_failures":
                runtime_failures,

            "player_win_rate":
                proportion_nb64(
                    player_wins,
                    len(
                        rows
                    ),
                ),

            "opponent_win_rate":
                proportion_nb64(
                    opponent_wins,
                    len(
                        rows
                    ),
                ),

            "timeout_rate":
                proportion_nb64(
                    timeouts,
                    len(
                        rows
                    ),
                ),

            "mean_turn_count":
                mean_turn_count,

            "median_turn_count":
                median_turn_count,

            "total_moves":
                total_moves,

            "mean_moves_per_battle":
                proportion_nb64(
                    total_moves,
                    len(
                        rows
                    ),
                ),

            "total_damage":
                total_damage,

            "mean_damage_per_battle":
                proportion_nb64(
                    total_damage,
                    len(
                        rows
                    ),
                ),

            "mean_damage_per_move":
                proportion_nb64(
                    total_damage,
                    total_moves,
                ),

            "unknown_move_count":
                unknown_moves,

            "unknown_move_rate":
                proportion_nb64(
                    unknown_moves,
                    total_moves,
                ),

            "zero_damage_move_count":
                zero_damage_moves,

            "zero_damage_move_rate":
                proportion_nb64(
                    zero_damage_moves,
                    total_moves,
                ),

            "positive_damage_move_count":
                positive_damage_moves,

            "positive_damage_move_rate":
                proportion_nb64(
                    positive_damage_moves,
                    total_moves,
                ),

            "mean_player_final_hp":
                mean_player_final_hp,

            "mean_opponent_final_hp":
                mean_opponent_final_hp,

            "mean_final_hp_margin":
                hp_margin,
        })


assert (
    len(
        SCENARIO_AGENT_MATRIX_NB64
    )
    ==
    12
)


print()
print("=" * 92)
print("4. SCENARIO × AGENT PERFORMANCE MATRIX")
print("=" * 92)


for record in SCENARIO_AGENT_MATRIX_NB64:

    print()

    print(
        json.dumps(
            record,
            indent=2,
        )
    )


# =============================================================================
# 6. COMPACT MATCHUP TABLE
# =============================================================================

MATCHUP_MATRIX_COMPACT_NB64 = []


for scenario_id in SCENARIO_IDS_NB64:

    scenario_rows = [

        row
        for row
        in SCENARIO_AGENT_MATRIX_NB64

        if (
            row[
                "scenario_id"
            ]
            ==
            scenario_id
        )
    ]


    row_lookup = {

        row[
            "agent_variant"
        ]:
            row

        for row
        in scenario_rows
    }


    MATCHUP_MATRIX_COMPACT_NB64.append({

        "scenario_id":
            scenario_id,

        "deployment_win_rate":
            row_lookup[
                "DEPLOYMENT_POLICY"
            ][
                "player_win_rate"
            ],

        "first_legal_win_rate":
            row_lookup[
                "FIRST_LEGAL_MOVE"
            ][
                "player_win_rate"
            ],

        "max_damage_win_rate":
            row_lookup[
                "MAX_DAMAGE_LEGAL_MOVE"
            ][
                "player_win_rate"
            ],

        "deployment_mean_turns":
            row_lookup[
                "DEPLOYMENT_POLICY"
            ][
                "mean_turn_count"
            ],

        "first_legal_mean_turns":
            row_lookup[
                "FIRST_LEGAL_MOVE"
            ][
                "mean_turn_count"
            ],

        "max_damage_mean_turns":
            row_lookup[
                "MAX_DAMAGE_LEGAL_MOVE"
            ][
                "mean_turn_count"
            ],

        "deployment_damage_per_move":
            row_lookup[
                "DEPLOYMENT_POLICY"
            ][
                "mean_damage_per_move"
            ],

        "first_legal_damage_per_move":
            row_lookup[
                "FIRST_LEGAL_MOVE"
            ][
                "mean_damage_per_move"
            ],

        "max_damage_damage_per_move":
            row_lookup[
                "MAX_DAMAGE_LEGAL_MOVE"
            ][
                "mean_damage_per_move"
            ],

        "deployment_hp_margin":
            row_lookup[
                "DEPLOYMENT_POLICY"
            ][
                "mean_final_hp_margin"
            ],

        "first_legal_hp_margin":
            row_lookup[
                "FIRST_LEGAL_MOVE"
            ][
                "mean_final_hp_margin"
            ],

        "max_damage_hp_margin":
            row_lookup[
                "MAX_DAMAGE_LEGAL_MOVE"
            ][
                "mean_final_hp_margin"
            ],
    })


print()
print("=" * 92)
print("5. COMPACT MATCHUP MATRIX")
print("=" * 92)

print(
    json.dumps(
        MATCHUP_MATRIX_COMPACT_NB64,
        indent=2,
    )
)


# =============================================================================
# 7. DEPLOYMENT POLICY SCENARIO PROFILE
# =============================================================================

DEPLOYMENT_SCENARIO_PROFILE_NB64 = []


for scenario_id in SCENARIO_IDS_NB64:

    deployment_row = next(

        row
        for row
        in SCENARIO_AGENT_MATRIX_NB64

        if (
            row[
                "scenario_id"
            ]
            ==
            scenario_id

            and

            row[
                "agent_variant"
            ]
            ==
            "DEPLOYMENT_POLICY"
        )
    )


    first_row = next(

        row
        for row
        in SCENARIO_AGENT_MATRIX_NB64

        if (
            row[
                "scenario_id"
            ]
            ==
            scenario_id

            and

            row[
                "agent_variant"
            ]
            ==
            "FIRST_LEGAL_MOVE"
        )
    )


    max_row = next(

        row
        for row
        in SCENARIO_AGENT_MATRIX_NB64

        if (
            row[
                "scenario_id"
            ]
            ==
            scenario_id

            and

            row[
                "agent_variant"
            ]
            ==
            "MAX_DAMAGE_LEGAL_MOVE"
        )
    )


    deployment_vs_first_win_delta = (

        deployment_row[
            "player_win_rate"
        ]
        -
        first_row[
            "player_win_rate"
        ]
    )


    deployment_vs_max_win_delta = (

        deployment_row[
            "player_win_rate"
        ]
        -
        max_row[
            "player_win_rate"
        ]
    )


    deployment_vs_first_damage_delta = (

        deployment_row[
            "mean_damage_per_move"
        ]
        -
        first_row[
            "mean_damage_per_move"
        ]
    )


    deployment_vs_max_damage_delta = (

        deployment_row[
            "mean_damage_per_move"
        ]
        -
        max_row[
            "mean_damage_per_move"
        ]
    )


    DEPLOYMENT_SCENARIO_PROFILE_NB64.append({

        "scenario_id":
            scenario_id,

        "deployment_win_rate":
            deployment_row[
                "player_win_rate"
            ],

        "deployment_mean_turns":
            deployment_row[
                "mean_turn_count"
            ],

        "deployment_damage_per_move":
            deployment_row[
                "mean_damage_per_move"
            ],

        "deployment_final_hp_margin":
            deployment_row[
                "mean_final_hp_margin"
            ],

        "deployment_vs_first_win_delta":
            deployment_vs_first_win_delta,

        "deployment_vs_max_win_delta":
            deployment_vs_max_win_delta,

        "deployment_vs_first_damage_per_move_delta":
            deployment_vs_first_damage_delta,

        "deployment_vs_max_damage_per_move_delta":
            deployment_vs_max_damage_delta,

        "deployment_first_legal_win_equivalent":
            (
                deployment_vs_first_win_delta
                ==
                0.0
            ),

        "deployment_max_damage_win_equivalent":
            (
                deployment_vs_max_win_delta
                ==
                0.0
            ),
    })


print()
print("=" * 92)
print("6. DEPLOYMENT POLICY SCENARIO PROFILE")
print("=" * 92)

print(
    json.dumps(
        DEPLOYMENT_SCENARIO_PROFILE_NB64,
        indent=2,
    )
)


# =============================================================================
# 8. SCENARIO INTERPRETATION FLAGS
# =============================================================================

SCENARIO_INTERPRETATION_FLAGS_NB64 = {}


for profile in DEPLOYMENT_SCENARIO_PROFILE_NB64:

    scenario_id = profile[
        "scenario_id"
    ]


    SCENARIO_INTERPRETATION_FLAGS_NB64[
        scenario_id
    ] = {

        "deployment_wins_majority":
            (
                profile[
                    "deployment_win_rate"
                ]
                >
                0.5
            ),

        "deployment_exactly_matches_first_legal_win_rate":
            (
                profile[
                    "deployment_vs_first_win_delta"
                ]
                ==
                0.0
            ),

        "deployment_exactly_matches_max_damage_win_rate":
            (
                profile[
                    "deployment_vs_max_win_delta"
                ]
                ==
                0.0
            ),

        "deployment_less_damage_efficient_than_max_damage":
            (
                profile[
                    "deployment_vs_max_damage_per_move_delta"
                ]
                <
                0.0
            ),

        "deployment_positive_final_hp_margin":
            (
                profile[
                    "deployment_final_hp_margin"
                ]
                >
                0.0
            ),

        "deployment_negative_final_hp_margin":
            (
                profile[
                    "deployment_final_hp_margin"
                ]
                <
                0.0
            ),
    }


print()
print("=" * 92)
print("7. SCENARIO INTERPRETATION FLAGS")
print("=" * 92)

print(
    json.dumps(
        SCENARIO_INTERPRETATION_FLAGS_NB64,
        indent=2,
    )
)


# =============================================================================
# 9. MATCHUP CLASSIFICATION
# =============================================================================
#
# These labels describe the observed controlled scenario outcomes.
# They are NOT claims about full official Pokémon TCG matchups.
# =============================================================================

MATCHUP_CLASSIFICATION_NB64 = {}


for profile in DEPLOYMENT_SCENARIO_PROFILE_NB64:

    scenario_id = profile[
        "scenario_id"
    ]

    win_rate = profile[
        "deployment_win_rate"
    ]


    if win_rate > 0.5:

        observed_classification = (
            "CONTROLLED_PLAYER_FAVORED"
        )


    elif win_rate < 0.5:

        observed_classification = (
            "CONTROLLED_PLAYER_DISFAVORED"
        )


    else:

        observed_classification = (
            "CONTROLLED_BALANCED"
        )


    MATCHUP_CLASSIFICATION_NB64[
        scenario_id
    ] = {

        "classification":
            observed_classification,

        "deployment_win_rate":
            win_rate,

        "scope":
            (
                "NB63_CONTROLLED_ATTACK_ONLY_SCENARIO"
            ),
    }


print()
print("=" * 92)
print("8. CONTROLLED MATCHUP CLASSIFICATION")
print("=" * 92)

print(
    json.dumps(
        MATCHUP_CLASSIFICATION_NB64,
        indent=2,
    )
)


# =============================================================================
# 10. PERSIST MATCHUP MATRIX
# =============================================================================

cell2_payload_nb64 = {

    "notebook":
        64,

    "cell":
        2,

    "purpose":
        "SCENARIO_MATCHUP_PERFORMANCE_MATRIX",

    "campaign_configuration_sha256":
        NB63_CORE_HANDOFF[
            "campaign_configuration_sha256"
        ],

    "battle_record_count":
        len(
            NB63_RECORDS
        ),

    "scenario_descriptions":
        SCENARIO_DESCRIPTION_NB64,

    "scenario_agent_matrix":
        SCENARIO_AGENT_MATRIX_NB64,

    "compact_matchup_matrix":
        MATCHUP_MATRIX_COMPACT_NB64,

    "deployment_scenario_profile":
        DEPLOYMENT_SCENARIO_PROFILE_NB64,

    "scenario_interpretation_flags":
        SCENARIO_INTERPRETATION_FLAGS_NB64,

    "controlled_matchup_classification":
        MATCHUP_CLASSIFICATION_NB64,

    "analysis_scope":
        (
            "CONTROLLED_ATTACK_ONLY_SYNTHETIC_BATTLE_STATES"
        ),

    "simulation_executed":
        False,

    "training_executed":
        False,

    "submission_modified":
        False,

    "model_modified":
        False,
}


cell2_artifact_nb64 = (
    NB64_ARTIFACT_DIR
    /
    "cell2_scenario_matchup_matrix.json"
)


cell2_output_nb64 = (
    NB64_OUTPUT_DIR
    /
    "cell2_scenario_matchup_matrix.json"
)


cell2_report_nb64 = (
    NB64_REPORT_DIR
    /
    "notebook64_scenario_matchup_matrix.json"
)


for path in [
    cell2_artifact_nb64,
    cell2_output_nb64,
    cell2_report_nb64,
]:

    path.write_text(
        json.dumps(
            cell2_payload_nb64,
            indent=2,
        ),
        encoding="utf-8",
    )


# =============================================================================
# 11. CSV EXPORT
# =============================================================================

cell2_csv_nb64 = (
    NB64_OUTPUT_DIR
    /
    "cell2_scenario_agent_matrix.csv"
)


csv_fields_nb64 = [

    "scenario_id",

    "agent_variant",

    "battle_count",

    "player_wins",

    "opponent_wins",

    "timeouts",

    "runtime_failures",

    "player_win_rate",

    "opponent_win_rate",

    "timeout_rate",

    "mean_turn_count",

    "median_turn_count",

    "total_moves",

    "mean_moves_per_battle",

    "total_damage",

    "mean_damage_per_battle",

    "mean_damage_per_move",

    "unknown_move_count",

    "unknown_move_rate",

    "zero_damage_move_count",

    "zero_damage_move_rate",

    "positive_damage_move_count",

    "positive_damage_move_rate",

    "mean_player_final_hp",

    "mean_opponent_final_hp",

    "mean_final_hp_margin",
]


with cell2_csv_nb64.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=csv_fields_nb64,
    )

    writer.writeheader()

    writer.writerows(
        SCENARIO_AGENT_MATRIX_NB64
    )


print()
print("=" * 92)
print("9. EVIDENCE PERSISTENCE")
print("=" * 92)

print(
    "[OK]",
    cell2_artifact_nb64,
)

print(
    "[OK]",
    cell2_output_nb64,
)

print(
    "[OK]",
    cell2_report_nb64,
)

print(
    "[OK]",
    cell2_csv_nb64,
)


# =============================================================================
# 12. CERTIFIED-ASSET INTEGRITY
# =============================================================================

assert (
    sha256_file_nb64(
        FINAL_MAIN
    )
    ==
    EXPECTED_MAIN_SHA256
)


assert (
    sha256_file_nb64(
        CERTIFIED_MODEL
    )
    ==
    EXPECTED_MODEL_SHA256
)


assert (
    sha256_file_nb64(
        DEPLOYMENT_PPO_CHECKPOINT
    )
    ==
    EXPECTED_PPO_SHA256
)


print()
print("=" * 92)
print("10. CERTIFIED-ASSET INTEGRITY")
print("=" * 92)

print(
    "[OK] main.py unchanged."
)

print(
    "[OK] Certified RF model unchanged."
)

print(
    "[OK] Deployment PPO checkpoint unchanged."
)

print(
    "[OK] No simulation executed."
)

print(
    "[OK] No training or retraining executed."
)


# =============================================================================
# 13. FINAL STATUS
# =============================================================================

CELL2_NB64_STATUS = {

    "cell1_pass":
        True,

    "scenario_matchup_matrix_complete":
        True,

    "scenario_count":
        len(
            SCENARIO_IDS_NB64
        ),

    "agent_count":
        len(
            AGENT_IDS_NB64
        ),

    "matrix_rows":
        len(
            SCENARIO_AGENT_MATRIX_NB64
        ),

    "battle_records_analyzed":
        len(
            NB63_RECORDS
        ),

    "deployment_scenario_profiles":
        len(
            DEPLOYMENT_SCENARIO_PROFILE_NB64
        ),

    "controlled_matchup_classifications_created":
        True,

    "analysis_only":
        True,

    "simulation_executed":
        False,

    "training_executed":
        False,

    "submission_modified":
        False,

    "model_modified":
        False,

    "main_integrity_preserved":
        True,

    "certified_model_integrity_preserved":
        True,

    "deployment_checkpoint_integrity_preserved":
        True,

    "validation_status":
        "PASS",
}


print()
print("=" * 92)
print("NOTEBOOK 64 — CELL 2 STATUS: PASS")
print("=" * 92)

print(
    json.dumps(
        CELL2_NB64_STATUS,
        indent=2,
    )
)


print()
print(
    "Scenario / matchup performance matrix complete."
)

print(
    "NEXT STEP: Cell 3 — side-orientation and side-neutrality interpretation."
)


# In[3]:


# =============================================================================
# NOTEBOOK 64 — CELL 3
# SIDE-ORIENTATION + SIDE-NEUTRALITY INTERPRETATION
# =============================================================================

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict


print("=" * 92)
print("NOTEBOOK 64 — CELL 3")
print("SIDE-ORIENTATION + SIDE-NEUTRALITY INTERPRETATION")
print("=" * 92)


# =============================================================================
# 0. PREREQUISITES
# =============================================================================

assert CELL2_NB64_STATUS["validation_status"] == "PASS"
assert CELL2_NB64_STATUS["battle_records_analyzed"] == 240
assert CELL2_NB64_STATUS["analysis_only"] is True
assert CELL2_NB64_STATUS["simulation_executed"] is False
assert CELL2_NB64_STATUS["training_executed"] is False

print()
print("[OK] Cell 2 PASS confirmed.")
print("[OK] 240 NB63 battle records available.")
print("[OK] Analysis-only restrictions preserved.")
print("[OK] No simulation or training will execute.")


# =============================================================================
# 1. HELPERS
# =============================================================================

def mean_or_none_cell3_nb64(values):
    cleaned = [
        float(value)
        for value in values
        if value is not None
    ]

    if not cleaned:
        return None

    return float(statistics.mean(cleaned))


def rate_cell3_nb64(numerator, denominator):
    if denominator <= 0:
        return None

    return float(numerator / denominator)


# =============================================================================
# 2. SIDE-ORIENTATION RECORD AUDIT
# =============================================================================

side_counts_cell3_nb64 = defaultdict(int)
agent_side_counts_cell3_nb64 = defaultdict(int)

for row in NB63_RECORDS:

    side_counts_cell3_nb64[
        row["side_mode"]
    ] += 1

    agent_side_counts_cell3_nb64[
        (
            row["agent_variant"],
            row["side_mode"],
        )
    ] += 1


assert dict(side_counts_cell3_nb64) == {
    "NORMAL": 120,
    "SWAPPED": 120,
}


for agent_id in AGENT_IDS_NB64:

    assert (
        agent_side_counts_cell3_nb64[
            (agent_id, "NORMAL")
        ]
        == 40
    )

    assert (
        agent_side_counts_cell3_nb64[
            (agent_id, "SWAPPED")
        ]
        == 40
    )


print()
print("=" * 92)
print("1. SIDE-ORIENTATION RECORD AUDIT")
print("=" * 92)

print(
    "NORMAL records:",
    side_counts_cell3_nb64["NORMAL"],
)

print(
    "SWAPPED records:",
    side_counts_cell3_nb64["SWAPPED"],
)

print()
print("[OK] Side-orientation campaign is perfectly balanced.")


# =============================================================================
# 3. AGENT × SIDE PERFORMANCE MATRIX
# =============================================================================

AGENT_SIDE_MATRIX_NB64 = []


for agent_id in AGENT_IDS_NB64:

    for side_mode in SIDE_MODES_NB64:

        rows = [
            row
            for row in NB63_RECORDS
            if (
                row["agent_variant"] == agent_id
                and
                row["side_mode"] == side_mode
            )
        ]

        assert len(rows) == 40

        player_wins = sum(
            1
            for row in rows
            if row["normalized_outcome"] == "PLAYER_WIN"
        )

        opponent_wins = sum(
            1
            for row in rows
            if row["normalized_outcome"] == "OPPONENT_WIN"
        )

        timeouts = sum(
            1
            for row in rows
            if row["normalized_outcome"] == "TIMEOUT_DRAW"
        )

        runtime_failures = sum(
            1
            for row in rows
            if row["runtime_exception"]
        )

        total_moves = sum(
            int(row["total_moves"] or 0)
            for row in rows
        )

        total_damage = sum(
            float(row["total_damage"] or 0.0)
            for row in rows
        )

        unknown_moves = sum(
            int(row["unknown_move_count"] or 0)
            for row in rows
        )

        positive_damage_moves = sum(
            int(row["positive_damage_move_count"] or 0)
            for row in rows
        )

        mean_turns = mean_or_none_cell3_nb64(
            [
                row["turn_count"]
                for row in rows
            ]
        )

        mean_player_final_hp = mean_or_none_cell3_nb64(
            [
                row["player_final_hp"]
                for row in rows
            ]
        )

        mean_opponent_final_hp = mean_or_none_cell3_nb64(
            [
                row["opponent_final_hp"]
                for row in rows
            ]
        )

        AGENT_SIDE_MATRIX_NB64.append({

            "agent_variant":
                agent_id,

            "side_mode":
                side_mode,

            "battle_count":
                len(rows),

            "player_wins":
                player_wins,

            "opponent_wins":
                opponent_wins,

            "timeouts":
                timeouts,

            "runtime_failures":
                runtime_failures,

            "player_win_rate":
                rate_cell3_nb64(
                    player_wins,
                    len(rows),
                ),

            "opponent_win_rate":
                rate_cell3_nb64(
                    opponent_wins,
                    len(rows),
                ),

            "timeout_rate":
                rate_cell3_nb64(
                    timeouts,
                    len(rows),
                ),

            "mean_turn_count":
                mean_turns,

            "total_moves":
                total_moves,

            "mean_moves_per_battle":
                rate_cell3_nb64(
                    total_moves,
                    len(rows),
                ),

            "total_damage":
                total_damage,

            "mean_damage_per_battle":
                rate_cell3_nb64(
                    total_damage,
                    len(rows),
                ),

            "mean_damage_per_move":
                rate_cell3_nb64(
                    total_damage,
                    total_moves,
                ),

            "unknown_move_count":
                unknown_moves,

            "unknown_move_rate":
                rate_cell3_nb64(
                    unknown_moves,
                    total_moves,
                ),

            "positive_damage_move_count":
                positive_damage_moves,

            "positive_damage_move_rate":
                rate_cell3_nb64(
                    positive_damage_moves,
                    total_moves,
                ),

            "mean_player_final_hp":
                mean_player_final_hp,

            "mean_opponent_final_hp":
                mean_opponent_final_hp,

            "mean_final_hp_margin":
                (
                    mean_player_final_hp
                    -
                    mean_opponent_final_hp
                ),
        })


assert len(AGENT_SIDE_MATRIX_NB64) == 6


print()
print("=" * 92)
print("2. AGENT × SIDE PERFORMANCE MATRIX")
print("=" * 92)

print(
    json.dumps(
        AGENT_SIDE_MATRIX_NB64,
        indent=2,
    )
)


# =============================================================================
# 4. NORMAL VS SWAPPED DELTAS
# =============================================================================

SIDE_DELTA_MATRIX_NB64 = []


for agent_id in AGENT_IDS_NB64:

    normal_row = next(
        row
        for row in AGENT_SIDE_MATRIX_NB64
        if (
            row["agent_variant"] == agent_id
            and
            row["side_mode"] == "NORMAL"
        )
    )

    swapped_row = next(
        row
        for row in AGENT_SIDE_MATRIX_NB64
        if (
            row["agent_variant"] == agent_id
            and
            row["side_mode"] == "SWAPPED"
        )
    )

    SIDE_DELTA_MATRIX_NB64.append({

        "agent_variant":
            agent_id,

        "normal_win_rate":
            normal_row["player_win_rate"],

        "swapped_win_rate":
            swapped_row["player_win_rate"],

        "win_rate_delta_normal_minus_swapped":
            (
                normal_row["player_win_rate"]
                -
                swapped_row["player_win_rate"]
            ),

        "normal_mean_turns":
            normal_row["mean_turn_count"],

        "swapped_mean_turns":
            swapped_row["mean_turn_count"],

        "turn_delta_normal_minus_swapped":
            (
                normal_row["mean_turn_count"]
                -
                swapped_row["mean_turn_count"]
            ),

        "normal_damage_per_move":
            normal_row["mean_damage_per_move"],

        "swapped_damage_per_move":
            swapped_row["mean_damage_per_move"],

        "damage_per_move_delta_normal_minus_swapped":
            (
                normal_row["mean_damage_per_move"]
                -
                swapped_row["mean_damage_per_move"]
            ),

        "normal_hp_margin":
            normal_row["mean_final_hp_margin"],

        "swapped_hp_margin":
            swapped_row["mean_final_hp_margin"],

        "hp_margin_delta_normal_minus_swapped":
            (
                normal_row["mean_final_hp_margin"]
                -
                swapped_row["mean_final_hp_margin"]
            ),
    })


print()
print("=" * 92)
print("3. NORMAL VS SWAPPED DELTAS")
print("=" * 92)

print(
    json.dumps(
        SIDE_DELTA_MATRIX_NB64,
        indent=2,
    )
)


# =============================================================================
# 5. DEPLOYMENT SIDE-NEUTRALITY PROFILE
# =============================================================================

deployment_side_delta_nb64 = next(
    row
    for row in SIDE_DELTA_MATRIX_NB64
    if row["agent_variant"] == "DEPLOYMENT_POLICY"
)


DEPLOYMENT_SIDE_NEUTRALITY_NB64 = {

    "normal_win_rate":
        deployment_side_delta_nb64[
            "normal_win_rate"
        ],

    "swapped_win_rate":
        deployment_side_delta_nb64[
            "swapped_win_rate"
        ],

    "absolute_win_rate_delta":
        abs(
            deployment_side_delta_nb64[
                "win_rate_delta_normal_minus_swapped"
            ]
        ),

    "normal_mean_turns":
        deployment_side_delta_nb64[
            "normal_mean_turns"
        ],

    "swapped_mean_turns":
        deployment_side_delta_nb64[
            "swapped_mean_turns"
        ],

    "absolute_turn_delta":
        abs(
            deployment_side_delta_nb64[
                "turn_delta_normal_minus_swapped"
            ]
        ),

    "normal_damage_per_move":
        deployment_side_delta_nb64[
            "normal_damage_per_move"
        ],

    "swapped_damage_per_move":
        deployment_side_delta_nb64[
            "swapped_damage_per_move"
        ],

    "absolute_damage_per_move_delta":
        abs(
            deployment_side_delta_nb64[
                "damage_per_move_delta_normal_minus_swapped"
            ]
        ),

    "normal_hp_margin":
        deployment_side_delta_nb64[
            "normal_hp_margin"
        ],

    "swapped_hp_margin":
        deployment_side_delta_nb64[
            "swapped_hp_margin"
        ],

    "absolute_hp_margin_delta":
        abs(
            deployment_side_delta_nb64[
                "hp_margin_delta_normal_minus_swapped"
            ]
        ),
}


print()
print("=" * 92)
print("4. DEPLOYMENT SIDE-NEUTRALITY PROFILE")
print("=" * 92)

print(
    json.dumps(
        DEPLOYMENT_SIDE_NEUTRALITY_NB64,
        indent=2,
    )
)


# =============================================================================
# 6. SIDE-NEUTRALITY FLAGS
# =============================================================================

DEPLOYMENT_SIDE_NEUTRALITY_FLAGS_NB64 = {

    "win_rate_neutral":
        (
            DEPLOYMENT_SIDE_NEUTRALITY_NB64[
                "absolute_win_rate_delta"
            ]
            == 0.0
        ),

    "turn_efficiency_neutral":
        (
            DEPLOYMENT_SIDE_NEUTRALITY_NB64[
                "absolute_turn_delta"
            ]
            == 0.0
        ),

    "damage_efficiency_neutral":
        (
            DEPLOYMENT_SIDE_NEUTRALITY_NB64[
                "absolute_damage_per_move_delta"
            ]
            == 0.0
        ),

    "final_hp_margin_neutral":
        (
            DEPLOYMENT_SIDE_NEUTRALITY_NB64[
                "absolute_hp_margin_delta"
            ]
            == 0.0
        ),
}


DEPLOYMENT_SIDE_NEUTRALITY_FLAGS_NB64[
    "overall_side_neutral"
] = all(
    DEPLOYMENT_SIDE_NEUTRALITY_FLAGS_NB64.values()
)


print()
print("=" * 92)
print("5. DEPLOYMENT SIDE-NEUTRALITY FLAGS")
print("=" * 92)

print(
    json.dumps(
        DEPLOYMENT_SIDE_NEUTRALITY_FLAGS_NB64,
        indent=2,
    )
)


# =============================================================================
# 7. SIDE NEUTRALITY BY SCENARIO
# =============================================================================

SCENARIO_SIDE_MATRIX_NB64 = []


for scenario_id in SCENARIO_IDS_NB64:

    for agent_id in AGENT_IDS_NB64:

        for side_mode in SIDE_MODES_NB64:

            rows = [
                row
                for row in NB63_RECORDS
                if (
                    row["scenario_id"] == scenario_id
                    and
                    row["agent_variant"] == agent_id
                    and
                    row["side_mode"] == side_mode
                )
            ]

            assert len(rows) == 10

            wins = sum(
                1
                for row in rows
                if (
                    row["normalized_outcome"]
                    ==
                    "PLAYER_WIN"
                )
            )

            total_moves = sum(
                int(row["total_moves"] or 0)
                for row in rows
            )

            total_damage = sum(
                float(row["total_damage"] or 0.0)
                for row in rows
            )

            SCENARIO_SIDE_MATRIX_NB64.append({

                "scenario_id":
                    scenario_id,

                "agent_variant":
                    agent_id,

                "side_mode":
                    side_mode,

                "battle_count":
                    len(rows),

                "player_wins":
                    wins,

                "player_win_rate":
                    wins / len(rows),

                "mean_turn_count":
                    mean_or_none_cell3_nb64(
                        [
                            row["turn_count"]
                            for row in rows
                        ]
                    ),

                "mean_damage_per_move":
                    (
                        total_damage
                        /
                        total_moves
                    ),

                "mean_player_final_hp":
                    mean_or_none_cell3_nb64(
                        [
                            row["player_final_hp"]
                            for row in rows
                        ]
                    ),

                "mean_opponent_final_hp":
                    mean_or_none_cell3_nb64(
                        [
                            row["opponent_final_hp"]
                            for row in rows
                        ]
                    ),
            })


assert len(SCENARIO_SIDE_MATRIX_NB64) == 24


# =============================================================================
# 8. DEPLOYMENT SCENARIO-SIDE DELTAS
# =============================================================================

DEPLOYMENT_SCENARIO_SIDE_DELTAS_NB64 = []


for scenario_id in SCENARIO_IDS_NB64:

    normal_row = next(
        row
        for row in SCENARIO_SIDE_MATRIX_NB64
        if (
            row["scenario_id"] == scenario_id
            and
            row["agent_variant"] == "DEPLOYMENT_POLICY"
            and
            row["side_mode"] == "NORMAL"
        )
    )

    swapped_row = next(
        row
        for row in SCENARIO_SIDE_MATRIX_NB64
        if (
            row["scenario_id"] == scenario_id
            and
            row["agent_variant"] == "DEPLOYMENT_POLICY"
            and
            row["side_mode"] == "SWAPPED"
        )
    )

    DEPLOYMENT_SCENARIO_SIDE_DELTAS_NB64.append({

        "scenario_id":
            scenario_id,

        "normal_win_rate":
            normal_row["player_win_rate"],

        "swapped_win_rate":
            swapped_row["player_win_rate"],

        "win_rate_delta":
            (
                normal_row["player_win_rate"]
                -
                swapped_row["player_win_rate"]
            ),

        "normal_mean_turns":
            normal_row["mean_turn_count"],

        "swapped_mean_turns":
            swapped_row["mean_turn_count"],

        "turn_delta":
            (
                normal_row["mean_turn_count"]
                -
                swapped_row["mean_turn_count"]
            ),

        "normal_damage_per_move":
            normal_row["mean_damage_per_move"],

        "swapped_damage_per_move":
            swapped_row["mean_damage_per_move"],

        "damage_per_move_delta":
            (
                normal_row["mean_damage_per_move"]
                -
                swapped_row["mean_damage_per_move"]
            ),
    })


print()
print("=" * 92)
print("6. DEPLOYMENT SCENARIO-SIDE DELTAS")
print("=" * 92)

print(
    json.dumps(
        DEPLOYMENT_SCENARIO_SIDE_DELTAS_NB64,
        indent=2,
    )
)


# =============================================================================
# 9. SIDE-NEUTRALITY CLASSIFICATION
# =============================================================================

all_scenario_win_deltas_zero_cell3_nb64 = all(
    abs(row["win_rate_delta"]) == 0.0
    for row in DEPLOYMENT_SCENARIO_SIDE_DELTAS_NB64
)


if (
    DEPLOYMENT_SIDE_NEUTRALITY_FLAGS_NB64[
        "overall_side_neutral"
    ]
    and
    all_scenario_win_deltas_zero_cell3_nb64
):

    SIDE_NEUTRALITY_CLASSIFICATION_NB64 = (
        "CONTROLLED_EXACT_SIDE_NEUTRALITY_OBSERVED"
    )

else:

    SIDE_NEUTRALITY_CLASSIFICATION_NB64 = (
        "CONTROLLED_SIDE_DIFFERENCE_OBSERVED"
    )


SIDE_NEUTRALITY_INTERPRETATION_NB64 = {

    "classification":
        SIDE_NEUTRALITY_CLASSIFICATION_NB64,

    "overall_side_neutral":
        DEPLOYMENT_SIDE_NEUTRALITY_FLAGS_NB64[
            "overall_side_neutral"
        ],

    "all_scenario_win_deltas_zero":
        all_scenario_win_deltas_zero_cell3_nb64,

    "scope":
        "NB63_CONTROLLED_ATTACK_ONLY_SYNTHETIC_CAMPAIGN",

    "supported_claim":
        (
            "Within the paired deterministic NB63 campaign, "
            "the deployment policy showed no measurable dependence "
            "on NORMAL versus SWAPPED orientation."
        ),

    "unsupported_claim":
        (
            "This does not establish universal side neutrality "
            "under complete official Pokémon TCG gameplay."
        ),
}


print()
print("=" * 92)
print("7. SIDE-NEUTRALITY CLASSIFICATION")
print("=" * 92)

print(
    json.dumps(
        SIDE_NEUTRALITY_INTERPRETATION_NB64,
        indent=2,
    )
)


# =============================================================================
# 10. BASELINE SIDE COMPARISON
# =============================================================================

BASELINE_SIDE_COMPARISON_NB64 = {}


for row in SIDE_DELTA_MATRIX_NB64:

    BASELINE_SIDE_COMPARISON_NB64[
        row["agent_variant"]
    ] = {

        "win_rate_delta":
            row[
                "win_rate_delta_normal_minus_swapped"
            ],

        "turn_delta":
            row[
                "turn_delta_normal_minus_swapped"
            ],

        "damage_per_move_delta":
            row[
                "damage_per_move_delta_normal_minus_swapped"
            ],

        "hp_margin_delta":
            row[
                "hp_margin_delta_normal_minus_swapped"
            ],

        "exact_win_rate_neutrality":
            (
                row[
                    "win_rate_delta_normal_minus_swapped"
                ]
                == 0.0
            ),
    }


print()
print("=" * 92)
print("8. BASELINE SIDE COMPARISON")
print("=" * 92)

print(
    json.dumps(
        BASELINE_SIDE_COMPARISON_NB64,
        indent=2,
    )
)


# =============================================================================
# 11. PERSIST CELL 3 EVIDENCE
# =============================================================================

cell3_payload_nb64 = {

    "notebook":
        64,

    "cell":
        3,

    "purpose":
        "SIDE_ORIENTATION_AND_SIDE_NEUTRALITY_INTERPRETATION",

    "campaign_configuration_sha256":
        NB63_CORE_HANDOFF[
            "campaign_configuration_sha256"
        ],

    "battle_records_analyzed":
        len(NB63_RECORDS),

    "agent_side_matrix":
        AGENT_SIDE_MATRIX_NB64,

    "side_delta_matrix":
        SIDE_DELTA_MATRIX_NB64,

    "deployment_side_neutrality_profile":
        DEPLOYMENT_SIDE_NEUTRALITY_NB64,

    "deployment_side_neutrality_flags":
        DEPLOYMENT_SIDE_NEUTRALITY_FLAGS_NB64,

    "scenario_side_matrix":
        SCENARIO_SIDE_MATRIX_NB64,

    "deployment_scenario_side_deltas":
        DEPLOYMENT_SCENARIO_SIDE_DELTAS_NB64,

    "side_neutrality_interpretation":
        SIDE_NEUTRALITY_INTERPRETATION_NB64,

    "baseline_side_comparison":
        BASELINE_SIDE_COMPARISON_NB64,

    "analysis_scope":
        "CONTROLLED_ATTACK_ONLY_SYNTHETIC_BATTLE_STATES",

    "simulation_executed":
        False,

    "training_executed":
        False,

    "submission_modified":
        False,

    "model_modified":
        False,
}


cell3_artifact_nb64 = (
    NB64_ARTIFACT_DIR
    /
    "cell3_side_neutrality_analysis.json"
)

cell3_output_nb64 = (
    NB64_OUTPUT_DIR
    /
    "cell3_side_neutrality_analysis.json"
)

cell3_report_nb64 = (
    NB64_REPORT_DIR
    /
    "notebook64_side_neutrality_analysis.json"
)


for path in [
    cell3_artifact_nb64,
    cell3_output_nb64,
    cell3_report_nb64,
]:

    path.write_text(
        json.dumps(
            cell3_payload_nb64,
            indent=2,
        ),
        encoding="utf-8",
    )


# =============================================================================
# 12. CSV EXPORT
# =============================================================================

cell3_csv_nb64 = (
    NB64_OUTPUT_DIR
    /
    "cell3_agent_side_matrix.csv"
)


csv_fields_cell3_nb64 = [
    "agent_variant",
    "side_mode",
    "battle_count",
    "player_wins",
    "opponent_wins",
    "timeouts",
    "runtime_failures",
    "player_win_rate",
    "opponent_win_rate",
    "timeout_rate",
    "mean_turn_count",
    "total_moves",
    "mean_moves_per_battle",
    "total_damage",
    "mean_damage_per_battle",
    "mean_damage_per_move",
    "unknown_move_count",
    "unknown_move_rate",
    "positive_damage_move_count",
    "positive_damage_move_rate",
    "mean_player_final_hp",
    "mean_opponent_final_hp",
    "mean_final_hp_margin",
]


with cell3_csv_nb64.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=csv_fields_cell3_nb64,
    )

    writer.writeheader()

    writer.writerows(
        AGENT_SIDE_MATRIX_NB64
    )


print()
print("=" * 92)
print("9. EVIDENCE PERSISTENCE")
print("=" * 92)

print("[OK]", cell3_artifact_nb64)
print("[OK]", cell3_output_nb64)
print("[OK]", cell3_report_nb64)
print("[OK]", cell3_csv_nb64)


# =============================================================================
# 13. CERTIFIED-ASSET INTEGRITY
# =============================================================================

assert (
    sha256_file_nb64(FINAL_MAIN)
    ==
    EXPECTED_MAIN_SHA256
)

assert (
    sha256_file_nb64(CERTIFIED_MODEL)
    ==
    EXPECTED_MODEL_SHA256
)

assert (
    sha256_file_nb64(DEPLOYMENT_PPO_CHECKPOINT)
    ==
    EXPECTED_PPO_SHA256
)


print()
print("=" * 92)
print("10. CERTIFIED-ASSET INTEGRITY")
print("=" * 92)

print("[OK] main.py unchanged.")
print("[OK] Certified RF model unchanged.")
print("[OK] Deployment PPO checkpoint unchanged.")
print("[OK] No simulation executed.")
print("[OK] No training/retraining executed.")


# =============================================================================
# 14. FINAL STATUS
# =============================================================================

CELL3_NB64_STATUS = {

    "cell2_pass":
        True,

    "side_orientation_analysis_complete":
        True,

    "battle_records_analyzed":
        len(NB63_RECORDS),

    "normal_records":
        side_counts_cell3_nb64["NORMAL"],

    "swapped_records":
        side_counts_cell3_nb64["SWAPPED"],

    "deployment_normal_win_rate":
        DEPLOYMENT_SIDE_NEUTRALITY_NB64[
            "normal_win_rate"
        ],

    "deployment_swapped_win_rate":
        DEPLOYMENT_SIDE_NEUTRALITY_NB64[
            "swapped_win_rate"
        ],

    "deployment_absolute_win_rate_delta":
        DEPLOYMENT_SIDE_NEUTRALITY_NB64[
            "absolute_win_rate_delta"
        ],

    "deployment_side_neutral":
        DEPLOYMENT_SIDE_NEUTRALITY_FLAGS_NB64[
            "overall_side_neutral"
        ],

    "side_neutrality_classification":
        SIDE_NEUTRALITY_CLASSIFICATION_NB64,

    "analysis_only":
        True,

    "simulation_executed":
        False,

    "training_executed":
        False,

    "submission_modified":
        False,

    "model_modified":
        False,

    "main_integrity_preserved":
        True,

    "certified_model_integrity_preserved":
        True,

    "deployment_checkpoint_integrity_preserved":
        True,

    "validation_status":
        "PASS",
}


print()
print("=" * 92)
print("NOTEBOOK 64 — CELL 3 STATUS: PASS")
print("=" * 92)

print(
    json.dumps(
        CELL3_NB64_STATUS,
        indent=2,
    )
)

print()
print(
    "Side-orientation and side-neutrality interpretation complete."
)

print(
    "NEXT STEP: Cell 4 — decision-efficiency and baseline-gap interpretation."
)


# In[4]:


# =============================================================================
# NOTEBOOK 64 — CELL 4
# DECISION EFFICIENCY + BASELINE-GAP INTERPRETATION
# =============================================================================

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict


print("=" * 92)
print("NOTEBOOK 64 — CELL 4")
print("DECISION EFFICIENCY + BASELINE-GAP INTERPRETATION")
print("=" * 92)


# =============================================================================
# 0. PREREQUISITES
# =============================================================================

assert CELL3_NB64_STATUS["validation_status"] == "PASS"
assert CELL3_NB64_STATUS["battle_records_analyzed"] == 240
assert CELL3_NB64_STATUS["analysis_only"] is True
assert CELL3_NB64_STATUS["simulation_executed"] is False
assert CELL3_NB64_STATUS["training_executed"] is False

assert len(NB63_RECORDS) == 240

print()
print("[OK] Cell 3 PASS confirmed.")
print("[OK] 240 NB63 battle records available.")
print("[OK] Analysis-only restrictions preserved.")
print("[OK] No simulation or training will execute.")


# =============================================================================
# 1. HELPERS
# =============================================================================

def mean_cell4(values):
    values = [
        float(v)
        for v in values
        if v is not None
    ]
    return float(statistics.mean(values)) if values else None


def safe_div_cell4(a, b):
    if b in (0, 0.0, None):
        return None
    return float(a / b)


def pct_reduction_cell4(reference, candidate):
    if reference in (0, 0.0, None):
        return None
    return float((reference - candidate) / reference)


# =============================================================================
# 2. OVERALL AGENT PERFORMANCE PROFILE
# =============================================================================

OVERALL_AGENT_PROFILE_NB64 = []

for agent_id in AGENT_IDS_NB64:

    rows = [
        row
        for row in NB63_RECORDS
        if row["agent_variant"] == agent_id
    ]

    assert len(rows) == 80

    wins = sum(
        row["normalized_outcome"] == "PLAYER_WIN"
        for row in rows
    )

    losses = sum(
        row["normalized_outcome"] == "OPPONENT_WIN"
        for row in rows
    )

    timeouts = sum(
        row["normalized_outcome"] == "TIMEOUT_DRAW"
        for row in rows
    )

    runtime_failures = sum(
        bool(row["runtime_exception"])
        for row in rows
    )

    total_moves = sum(
        int(row["total_moves"] or 0)
        for row in rows
    )

    total_damage = sum(
        float(row["total_damage"] or 0.0)
        for row in rows
    )

    positive_damage_moves = sum(
        int(row["positive_damage_move_count"] or 0)
        for row in rows
    )

    unknown_moves = sum(
        int(row["unknown_move_count"] or 0)
        for row in rows
    )

    mean_player_hp = mean_cell4(
        row["player_final_hp"]
        for row in rows
    )

    mean_opponent_hp = mean_cell4(
        row["opponent_final_hp"]
        for row in rows
    )

    profile = {
        "agent_variant": agent_id,
        "battle_count": len(rows),
        "player_wins": wins,
        "opponent_wins": losses,
        "timeouts": timeouts,
        "runtime_failures": runtime_failures,
        "player_win_rate": safe_div_cell4(wins, len(rows)),
        "mean_turn_count": mean_cell4(
            row["turn_count"]
            for row in rows
        ),
        "total_moves": total_moves,
        "mean_moves_per_battle": safe_div_cell4(
            total_moves,
            len(rows),
        ),
        "total_damage": total_damage,
        "mean_damage_per_battle": safe_div_cell4(
            total_damage,
            len(rows),
        ),
        "mean_damage_per_move": safe_div_cell4(
            total_damage,
            total_moves,
        ),
        "positive_damage_move_rate": safe_div_cell4(
            positive_damage_moves,
            total_moves,
        ),
        "unknown_move_rate": safe_div_cell4(
            unknown_moves,
            total_moves,
        ),
        "mean_player_final_hp": mean_player_hp,
        "mean_opponent_final_hp": mean_opponent_hp,
        "mean_final_hp_margin": (
            mean_player_hp - mean_opponent_hp
        ),
    }

    OVERALL_AGENT_PROFILE_NB64.append(profile)


assert len(OVERALL_AGENT_PROFILE_NB64) == 3

print()
print("=" * 92)
print("1. OVERALL AGENT PERFORMANCE PROFILE")
print("=" * 92)

print(
    json.dumps(
        OVERALL_AGENT_PROFILE_NB64,
        indent=2,
    )
)


# =============================================================================
# 3. IDENTIFY CORE AGENTS
# =============================================================================

deployment_profile_nb64 = next(
    row
    for row in OVERALL_AGENT_PROFILE_NB64
    if row["agent_variant"] == "DEPLOYMENT_POLICY"
)

first_legal_profile_nb64 = next(
    row
    for row in OVERALL_AGENT_PROFILE_NB64
    if row["agent_variant"] == "FIRST_LEGAL_MOVE"
)

max_damage_profile_nb64 = next(
    row
    for row in OVERALL_AGENT_PROFILE_NB64
    if row["agent_variant"] == "MAX_DAMAGE_LEGAL_MOVE"
)


# =============================================================================
# 4. DEPLOYMENT VS FIRST-LEGAL EQUIVALENCE
# =============================================================================

DEPLOYMENT_FIRST_LEGAL_GAP_NB64 = {
    "win_rate_gap": (
        deployment_profile_nb64["player_win_rate"]
        -
        first_legal_profile_nb64["player_win_rate"]
    ),
    "mean_turn_gap": (
        deployment_profile_nb64["mean_turn_count"]
        -
        first_legal_profile_nb64["mean_turn_count"]
    ),
    "mean_moves_gap": (
        deployment_profile_nb64["mean_moves_per_battle"]
        -
        first_legal_profile_nb64["mean_moves_per_battle"]
    ),
    "damage_per_battle_gap": (
        deployment_profile_nb64["mean_damage_per_battle"]
        -
        first_legal_profile_nb64["mean_damage_per_battle"]
    ),
    "damage_per_move_gap": (
        deployment_profile_nb64["mean_damage_per_move"]
        -
        first_legal_profile_nb64["mean_damage_per_move"]
    ),
    "hp_margin_gap": (
        deployment_profile_nb64["mean_final_hp_margin"]
        -
        first_legal_profile_nb64["mean_final_hp_margin"]
    ),
}


DEPLOYMENT_FIRST_LEGAL_GAP_NB64[
    "aggregate_exact_match"
] = all(
    abs(value) < 1e-12
    for key, value
    in DEPLOYMENT_FIRST_LEGAL_GAP_NB64.items()
    if key != "aggregate_exact_match"
)


print()
print("=" * 92)
print("2. DEPLOYMENT VS FIRST-LEGAL AGGREGATE GAP")
print("=" * 92)

print(
    json.dumps(
        DEPLOYMENT_FIRST_LEGAL_GAP_NB64,
        indent=2,
    )
)


# =============================================================================
# 5. PAIRED BATTLE-LEVEL EQUIVALENCE
# =============================================================================

PAIR_FIELDS_NB64 = [
    "scenario_id",
    "side_mode",
    "seed",
]

paired_records_nb64 = defaultdict(dict)

for row in NB63_RECORDS:

    key = tuple(
        row[field]
        for field in PAIR_FIELDS_NB64
    )

    paired_records_nb64[key][
        row["agent_variant"]
    ] = row


assert len(paired_records_nb64) == 80


deployment_first_legal_pair_matches_nb64 = 0
deployment_max_damage_pair_matches_nb64 = 0

PAIR_COMPARISON_ROWS_NB64 = []


for pair_key, agent_rows in sorted(
    paired_records_nb64.items(),
    key=lambda item: str(item[0]),
):

    assert set(agent_rows) == set(AGENT_IDS_NB64)

    dep = agent_rows["DEPLOYMENT_POLICY"]
    first = agent_rows["FIRST_LEGAL_MOVE"]
    maxd = agent_rows["MAX_DAMAGE_LEGAL_MOVE"]

    dep_first_match = (
        dep["normalized_outcome"]
        ==
        first["normalized_outcome"]
        and
        dep["turn_count"]
        ==
        first["turn_count"]
        and
        dep["total_moves"]
        ==
        first["total_moves"]
        and
        math.isclose(
            float(dep["total_damage"]),
            float(first["total_damage"]),
            abs_tol=1e-12,
        )
        and
        math.isclose(
            float(dep["player_final_hp"]),
            float(first["player_final_hp"]),
            abs_tol=1e-12,
        )
        and
        math.isclose(
            float(dep["opponent_final_hp"]),
            float(first["opponent_final_hp"]),
            abs_tol=1e-12,
        )
    )

    dep_max_match = (
        dep["normalized_outcome"]
        ==
        maxd["normalized_outcome"]
        and
        dep["turn_count"]
        ==
        maxd["turn_count"]
        and
        dep["total_moves"]
        ==
        maxd["total_moves"]
        and
        math.isclose(
            float(dep["total_damage"]),
            float(maxd["total_damage"]),
            abs_tol=1e-12,
        )
    )

    deployment_first_legal_pair_matches_nb64 += int(
        dep_first_match
    )

    deployment_max_damage_pair_matches_nb64 += int(
        dep_max_match
    )

    PAIR_COMPARISON_ROWS_NB64.append({
        "scenario_id": pair_key[0],
        "side_mode": pair_key[1],
        "seed": pair_key[2],
        "deployment_first_legal_match": dep_first_match,
        "deployment_max_damage_match": dep_max_match,
        "deployment_outcome": dep["normalized_outcome"],
        "first_legal_outcome": first["normalized_outcome"],
        "max_damage_outcome": maxd["normalized_outcome"],
        "deployment_turns": dep["turn_count"],
        "first_legal_turns": first["turn_count"],
        "max_damage_turns": maxd["turn_count"],
        "deployment_total_damage": dep["total_damage"],
        "first_legal_total_damage": first["total_damage"],
        "max_damage_total_damage": maxd["total_damage"],
    })


PAIRED_EQUIVALENCE_NB64 = {
    "paired_battles": len(paired_records_nb64),
    "deployment_first_legal_exact_matches":
        deployment_first_legal_pair_matches_nb64,
    "deployment_first_legal_exact_match_rate":
        safe_div_cell4(
            deployment_first_legal_pair_matches_nb64,
            len(paired_records_nb64),
        ),
    "deployment_max_damage_exact_matches":
        deployment_max_damage_pair_matches_nb64,
    "deployment_max_damage_exact_match_rate":
        safe_div_cell4(
            deployment_max_damage_pair_matches_nb64,
            len(paired_records_nb64),
        ),
}


print()
print("=" * 92)
print("3. PAIRED BATTLE-LEVEL EQUIVALENCE")
print("=" * 92)

print(
    json.dumps(
        PAIRED_EQUIVALENCE_NB64,
        indent=2,
    )
)


# =============================================================================
# 6. DEPLOYMENT VS MAX-DAMAGE EFFICIENCY GAP
# =============================================================================

DEPLOYMENT_MAX_DAMAGE_GAP_NB64 = {
    "deployment_win_rate":
        deployment_profile_nb64["player_win_rate"],

    "max_damage_win_rate":
        max_damage_profile_nb64["player_win_rate"],

    "win_rate_gap_deployment_minus_max_damage":
        (
            deployment_profile_nb64["player_win_rate"]
            -
            max_damage_profile_nb64["player_win_rate"]
        ),

    "deployment_mean_turns":
        deployment_profile_nb64["mean_turn_count"],

    "max_damage_mean_turns":
        max_damage_profile_nb64["mean_turn_count"],

    "turn_gap_deployment_minus_max_damage":
        (
            deployment_profile_nb64["mean_turn_count"]
            -
            max_damage_profile_nb64["mean_turn_count"]
        ),

    "max_damage_turn_reduction_vs_deployment":
        pct_reduction_cell4(
            deployment_profile_nb64["mean_turn_count"],
            max_damage_profile_nb64["mean_turn_count"],
        ),

    "deployment_damage_per_move":
        deployment_profile_nb64["mean_damage_per_move"],

    "max_damage_damage_per_move":
        max_damage_profile_nb64["mean_damage_per_move"],

    "damage_per_move_gap_max_minus_deployment":
        (
            max_damage_profile_nb64["mean_damage_per_move"]
            -
            deployment_profile_nb64["mean_damage_per_move"]
        ),

    "damage_per_move_ratio_max_over_deployment":
        safe_div_cell4(
            max_damage_profile_nb64["mean_damage_per_move"],
            deployment_profile_nb64["mean_damage_per_move"],
        ),

    "deployment_hp_margin":
        deployment_profile_nb64["mean_final_hp_margin"],

    "max_damage_hp_margin":
        max_damage_profile_nb64["mean_final_hp_margin"],

    "hp_margin_gap_max_minus_deployment":
        (
            max_damage_profile_nb64["mean_final_hp_margin"]
            -
            deployment_profile_nb64["mean_final_hp_margin"]
        ),
}


print()
print("=" * 92)
print("4. DEPLOYMENT VS MAX-DAMAGE EFFICIENCY GAP")
print("=" * 92)

print(
    json.dumps(
        DEPLOYMENT_MAX_DAMAGE_GAP_NB64,
        indent=2,
    )
)


# =============================================================================
# 7. SCENARIO-LEVEL EFFICIENCY COMPARISON
# =============================================================================

SCENARIO_EFFICIENCY_NB64 = []

for scenario_id in SCENARIO_IDS_NB64:

    scenario_profiles = {}

    for agent_id in AGENT_IDS_NB64:

        rows = [
            row
            for row in NB63_RECORDS
            if (
                row["scenario_id"] == scenario_id
                and
                row["agent_variant"] == agent_id
            )
        ]

        assert len(rows) == 20

        total_moves = sum(
            int(row["total_moves"] or 0)
            for row in rows
        )

        total_damage = sum(
            float(row["total_damage"] or 0.0)
            for row in rows
        )

        wins = sum(
            row["normalized_outcome"] == "PLAYER_WIN"
            for row in rows
        )

        scenario_profiles[agent_id] = {
            "win_rate": wins / len(rows),
            "mean_turns": mean_cell4(
                row["turn_count"]
                for row in rows
            ),
            "mean_damage_per_move":
                safe_div_cell4(
                    total_damage,
                    total_moves,
                ),
            "mean_player_final_hp":
                mean_cell4(
                    row["player_final_hp"]
                    for row in rows
                ),
            "mean_opponent_final_hp":
                mean_cell4(
                    row["opponent_final_hp"]
                    for row in rows
                ),
        }

    dep = scenario_profiles["DEPLOYMENT_POLICY"]
    first = scenario_profiles["FIRST_LEGAL_MOVE"]
    maxd = scenario_profiles["MAX_DAMAGE_LEGAL_MOVE"]

    SCENARIO_EFFICIENCY_NB64.append({
        "scenario_id": scenario_id,

        "deployment_win_rate":
            dep["win_rate"],

        "first_legal_win_rate":
            first["win_rate"],

        "max_damage_win_rate":
            maxd["win_rate"],

        "deployment_first_legal_win_gap":
            dep["win_rate"] - first["win_rate"],

        "deployment_max_damage_win_gap":
            dep["win_rate"] - maxd["win_rate"],

        "deployment_mean_turns":
            dep["mean_turns"],

        "first_legal_mean_turns":
            first["mean_turns"],

        "max_damage_mean_turns":
            maxd["mean_turns"],

        "deployment_first_legal_turn_gap":
            dep["mean_turns"] - first["mean_turns"],

        "deployment_max_damage_turn_gap":
            dep["mean_turns"] - maxd["mean_turns"],

        "deployment_damage_per_move":
            dep["mean_damage_per_move"],

        "first_legal_damage_per_move":
            first["mean_damage_per_move"],

        "max_damage_damage_per_move":
            maxd["mean_damage_per_move"],

        "deployment_first_legal_damage_gap":
            (
                dep["mean_damage_per_move"]
                -
                first["mean_damage_per_move"]
            ),

        "max_damage_advantage_damage_per_move":
            (
                maxd["mean_damage_per_move"]
                -
                dep["mean_damage_per_move"]
            ),
    })


print()
print("=" * 92)
print("5. SCENARIO-LEVEL EFFICIENCY COMPARISON")
print("=" * 92)

print(
    json.dumps(
        SCENARIO_EFFICIENCY_NB64,
        indent=2,
    )
)


# =============================================================================
# 8. STRATEGIC CLASSIFICATION
# =============================================================================

deployment_matches_first_legal_nb64 = (
    PAIRED_EQUIVALENCE_NB64[
        "deployment_first_legal_exact_match_rate"
    ]
    == 1.0
)

same_win_rate_as_max_nb64 = math.isclose(
    deployment_profile_nb64["player_win_rate"],
    max_damage_profile_nb64["player_win_rate"],
    abs_tol=1e-12,
)

max_damage_more_turn_efficient_nb64 = (
    max_damage_profile_nb64["mean_turn_count"]
    <
    deployment_profile_nb64["mean_turn_count"]
)

max_damage_more_damage_efficient_nb64 = (
    max_damage_profile_nb64["mean_damage_per_move"]
    >
    deployment_profile_nb64["mean_damage_per_move"]
)


if (
    deployment_matches_first_legal_nb64
    and
    same_win_rate_as_max_nb64
    and
    max_damage_more_turn_efficient_nb64
    and
    max_damage_more_damage_efficient_nb64
):
    STRATEGIC_CLASSIFICATION_NB64 = (
        "DEPLOYMENT_MATCHES_FIRST_LEGAL_WITH_MAX_DAMAGE_EFFICIENCY_GAP"
    )
else:
    STRATEGIC_CLASSIFICATION_NB64 = (
        "MIXED_CONTROLLED_DECISION_PATTERN"
    )


STRATEGIC_INTERPRETATION_NB64 = {
    "classification":
        STRATEGIC_CLASSIFICATION_NB64,

    "deployment_first_legal_exact_pair_match":
        deployment_matches_first_legal_nb64,

    "deployment_and_max_damage_same_aggregate_win_rate":
        same_win_rate_as_max_nb64,

    "max_damage_more_turn_efficient":
        max_damage_more_turn_efficient_nb64,

    "max_damage_more_damage_efficient":
        max_damage_more_damage_efficient_nb64,

    "supported_findings": [
        (
            "The deployment policy behaviorally matched "
            "FIRST_LEGAL_MOVE across all paired controlled battles."
        ),
        (
            "MAX_DAMAGE_LEGAL_MOVE achieved the same aggregate "
            "controlled win rate while requiring fewer turns."
        ),
        (
            "MAX_DAMAGE_LEGAL_MOVE produced substantially more "
            "damage per move in the controlled attack-only states."
        ),
        (
            "The deployment policy executed legally and reliably, "
            "with zero unknown moves and no runtime failures."
        ),
    ],

    "important_limitation": (
        "These results come from deterministic, attack-only synthetic "
        "battle states. They diagnose controlled decision behavior but "
        "do not establish comparative strength under complete official "
        "Pokémon TCG gameplay."
    ),

    "strategy_implication": (
        "The controlled evidence identifies a decision-quality gap: "
        "the deployment policy does not exploit the immediately "
        "higher-damage legal attack when that choice is available."
    ),
}


print()
print("=" * 92)
print("6. STRATEGIC INTERPRETATION")
print("=" * 92)

print(
    json.dumps(
        STRATEGIC_INTERPRETATION_NB64,
        indent=2,
    )
)


# =============================================================================
# 9. CELL 3 SIDE-NEUTRALITY RECONCILIATION
# =============================================================================

SIDE_NEUTRALITY_RECONCILIATION_NB64 = {
    "aggregate_result":
        (
            "NORMAL and SWAPPED aggregate deployment metrics "
            "are identical."
        ),

    "scenario_conditioned_result":
        (
            "S3 and S4 reverse their outcome pattern when the "
            "damage-advantaged side is swapped."
        ),

    "interpretation":
        (
            "This is consistent with aggregate orientation neutrality "
            "combined with scenario-conditioned state advantage. "
            "It should not be described as universal scenario-level "
            "side invariance."
        ),

    "recommended_wording":
        (
            "The deployment policy showed aggregate side neutrality "
            "in the paired controlled campaign, while outcomes in "
            "asymmetric scenarios followed the side holding the "
            "synthetic damage advantage."
        ),
}


print()
print("=" * 92)
print("7. SIDE-NEUTRALITY RECONCILIATION")
print("=" * 92)

print(
    json.dumps(
        SIDE_NEUTRALITY_RECONCILIATION_NB64,
        indent=2,
    )
)


# =============================================================================
# 10. PERSIST EVIDENCE
# =============================================================================

CELL4_PAYLOAD_NB64 = {
    "notebook": 64,
    "cell": 4,
    "purpose":
        "DECISION_EFFICIENCY_AND_BASELINE_GAP_INTERPRETATION",

    "campaign_configuration_sha256":
        NB63_CORE_HANDOFF[
            "campaign_configuration_sha256"
        ],

    "battle_records_analyzed":
        len(NB63_RECORDS),

    "overall_agent_profile":
        OVERALL_AGENT_PROFILE_NB64,

    "deployment_first_legal_gap":
        DEPLOYMENT_FIRST_LEGAL_GAP_NB64,

    "paired_equivalence":
        PAIRED_EQUIVALENCE_NB64,

    "deployment_max_damage_gap":
        DEPLOYMENT_MAX_DAMAGE_GAP_NB64,

    "scenario_efficiency":
        SCENARIO_EFFICIENCY_NB64,

    "strategic_interpretation":
        STRATEGIC_INTERPRETATION_NB64,

    "side_neutrality_reconciliation":
        SIDE_NEUTRALITY_RECONCILIATION_NB64,

    "analysis_scope":
        "CONTROLLED_ATTACK_ONLY_SYNTHETIC_BATTLE_STATES",

    "simulation_executed": False,
    "training_executed": False,
    "submission_modified": False,
    "model_modified": False,
}


cell4_artifact_nb64 = (
    NB64_ARTIFACT_DIR
    /
    "cell4_decision_efficiency_analysis.json"
)

cell4_output_nb64 = (
    NB64_OUTPUT_DIR
    /
    "cell4_decision_efficiency_analysis.json"
)

cell4_report_nb64 = (
    NB64_REPORT_DIR
    /
    "notebook64_decision_efficiency_analysis.json"
)


for path in [
    cell4_artifact_nb64,
    cell4_output_nb64,
    cell4_report_nb64,
]:
    path.write_text(
        json.dumps(
            CELL4_PAYLOAD_NB64,
            indent=2,
        ),
        encoding="utf-8",
    )


# =============================================================================
# 11. CSV EXPORTS
# =============================================================================

overall_csv_nb64 = (
    NB64_OUTPUT_DIR
    /
    "cell4_overall_agent_profile.csv"
)

with overall_csv_nb64.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=list(
            OVERALL_AGENT_PROFILE_NB64[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(OVERALL_AGENT_PROFILE_NB64)


pair_csv_nb64 = (
    NB64_OUTPUT_DIR
    /
    "cell4_paired_agent_comparison.csv"
)

with pair_csv_nb64.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=list(
            PAIR_COMPARISON_ROWS_NB64[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(PAIR_COMPARISON_ROWS_NB64)


print()
print("=" * 92)
print("8. EVIDENCE PERSISTENCE")
print("=" * 92)

print("[OK]", cell4_artifact_nb64)
print("[OK]", cell4_output_nb64)
print("[OK]", cell4_report_nb64)
print("[OK]", overall_csv_nb64)
print("[OK]", pair_csv_nb64)


# =============================================================================
# 12. CERTIFIED-ASSET INTEGRITY
# =============================================================================

assert sha256_file_nb64(FINAL_MAIN) == EXPECTED_MAIN_SHA256
assert sha256_file_nb64(CERTIFIED_MODEL) == EXPECTED_MODEL_SHA256
assert (
    sha256_file_nb64(DEPLOYMENT_PPO_CHECKPOINT)
    ==
    EXPECTED_PPO_SHA256
)

print()
print("=" * 92)
print("9. CERTIFIED-ASSET INTEGRITY")
print("=" * 92)

print("[OK] main.py unchanged.")
print("[OK] Certified RF model unchanged.")
print("[OK] Deployment PPO checkpoint unchanged.")
print("[OK] No simulation executed.")
print("[OK] No training/retraining executed.")


# =============================================================================
# 13. REQUIRED ASSERTIONS
# =============================================================================

assert (
    PAIRED_EQUIVALENCE_NB64[
        "deployment_first_legal_exact_match_rate"
    ]
    == 1.0
), (
    "Deployment policy no longer matches FIRST_LEGAL_MOVE "
    "across all paired NB63 battles."
)

assert (
    deployment_profile_nb64["unknown_move_rate"]
    == 0.0
)

assert (
    deployment_profile_nb64["positive_damage_move_rate"]
    == 1.0
)

assert (
    deployment_profile_nb64["runtime_failures"]
    == 0
)

assert (
    max_damage_profile_nb64["mean_turn_count"]
    <
    deployment_profile_nb64["mean_turn_count"]
)

assert (
    max_damage_profile_nb64["mean_damage_per_move"]
    >
    deployment_profile_nb64["mean_damage_per_move"]
)


# =============================================================================
# 14. FINAL STATUS
# =============================================================================

CELL4_NB64_STATUS = {
    "cell3_pass": True,

    "decision_efficiency_analysis_complete": True,

    "battle_records_analyzed":
        len(NB63_RECORDS),

    "deployment_first_legal_exact_match_rate":
        PAIRED_EQUIVALENCE_NB64[
            "deployment_first_legal_exact_match_rate"
        ],

    "deployment_max_damage_exact_match_rate":
        PAIRED_EQUIVALENCE_NB64[
            "deployment_max_damage_exact_match_rate"
        ],

    "deployment_win_rate":
        deployment_profile_nb64["player_win_rate"],

    "max_damage_win_rate":
        max_damage_profile_nb64["player_win_rate"],

    "deployment_mean_turns":
        deployment_profile_nb64["mean_turn_count"],

    "max_damage_mean_turns":
        max_damage_profile_nb64["mean_turn_count"],

    "deployment_damage_per_move":
        deployment_profile_nb64["mean_damage_per_move"],

    "max_damage_damage_per_move":
        max_damage_profile_nb64["mean_damage_per_move"],

    "strategic_classification":
        STRATEGIC_CLASSIFICATION_NB64,

    "analysis_only": True,
    "simulation_executed": False,
    "training_executed": False,
    "submission_modified": False,
    "model_modified": False,

    "main_integrity_preserved": True,
    "certified_model_integrity_preserved": True,
    "deployment_checkpoint_integrity_preserved": True,

    "validation_status": "PASS",
}


print()
print("=" * 92)
print("NOTEBOOK 64 — CELL 4 STATUS: PASS")
print("=" * 92)

print(
    json.dumps(
        CELL4_NB64_STATUS,
        indent=2,
    )
)

print()
print(
    "Decision-efficiency and baseline-gap interpretation complete."
)

print(
    "NEXT STEP: Cell 5 — integrate NB57–NB63 evidence "
    "into final Track 2 strategic conclusions."
)


# In[5]:


# =============================================================================
# NOTEBOOK 64 — CELL 5
# NB57–NB63 EVIDENCE INTEGRATION + FINAL TRACK 2 STRATEGIC SYNTHESIS
# =============================================================================

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


print("=" * 92)
print("NOTEBOOK 64 — CELL 5")
print("NB57–NB63 EVIDENCE INTEGRATION + FINAL TRACK 2 STRATEGIC SYNTHESIS")
print("=" * 92)


# =============================================================================
# 0. PREREQUISITES
# =============================================================================

assert (
    CELL4_NB64_STATUS[
        "validation_status"
    ]
    ==
    "PASS"
)

assert (
    CELL4_NB64_STATUS[
        "decision_efficiency_analysis_complete"
    ]
    is True
)

assert (
    CELL4_NB64_STATUS[
        "battle_records_analyzed"
    ]
    ==
    240
)

assert (
    CELL4_NB64_STATUS[
        "analysis_only"
    ]
    is True
)

assert (
    CELL4_NB64_STATUS[
        "simulation_executed"
    ]
    is False
)

assert (
    CELL4_NB64_STATUS[
        "training_executed"
    ]
    is False
)


print()
print("[OK] Cell 4 PASS confirmed.")
print("[OK] Decision-efficiency evidence available.")
print("[OK] Track 2 final strategic synthesis authorized.")
print("[OK] Analysis-only restrictions preserved.")


# =============================================================================
# 1. UPSTREAM NOTEBOOK EVIDENCE ROOTS
# =============================================================================

UPSTREAM_NOTEBOOKS_NB64 = [
    57,
    58,
    59,
    60,
    61,
    62,
    63,
]


UPSTREAM_ROOTS_NB64 = {}


for notebook_number in UPSTREAM_NOTEBOOKS_NB64:

    UPSTREAM_ROOTS_NB64[
        notebook_number
    ] = {

        "artifacts":
            (
                PROJECT_ROOT
                /
                "artifacts"
                /
                f"notebook{notebook_number}"
            ),

        "outputs":
            (
                PROJECT_ROOT
                /
                "outputs"
                /
                f"notebook{notebook_number}"
            ),

        "reports":
            (
                PROJECT_ROOT
                /
                "reports"
                /
                f"notebook{notebook_number}"
            ),
    }


print()
print("=" * 92)
print("1. UPSTREAM EVIDENCE ROOTS")
print("=" * 92)


for notebook_number, roots in (
    UPSTREAM_ROOTS_NB64.items()
):

    print()
    print(
        f"Notebook {notebook_number}"
    )

    for root_type, path in (
        roots.items()
    ):

        print(
            f"  {root_type:<10}: "
            f"{path} "
            f"| exists={path.is_dir()}"
        )


# =============================================================================
# 2. SAFE EVIDENCE INVENTORY
# =============================================================================
#
# Inventory only.
# No model files are loaded.
# No notebooks are executed.
# No submission files are modified.
# =============================================================================

ALLOWED_EVIDENCE_SUFFIXES_NB64 = {
    ".json",
    ".csv",
    ".md",
    ".txt",
}


MAX_TEXT_EVIDENCE_BYTES_NB64 = (
    5
    *
    1024
    *
    1024
)


UPSTREAM_EVIDENCE_INVENTORY_NB64 = []


for notebook_number, roots in (
    UPSTREAM_ROOTS_NB64.items()
):

    for root_type, root_path in (
        roots.items()
    ):

        if not root_path.is_dir():
            continue

        for path in sorted(
            root_path.rglob("*")
        ):

            if not path.is_file():
                continue

            if (
                path.suffix.lower()
                not in
                ALLOWED_EVIDENCE_SUFFIXES_NB64
            ):
                continue

            file_size = (
                path.stat().st_size
            )

            if (
                file_size
                >
                MAX_TEXT_EVIDENCE_BYTES_NB64
            ):
                continue

            UPSTREAM_EVIDENCE_INVENTORY_NB64.append(
                {
                    "notebook":
                        notebook_number,

                    "root_type":
                        root_type,

                    "relative_path":
                        str(
                            path.relative_to(
                                PROJECT_ROOT
                            )
                        ),

                    "bytes":
                        file_size,

                    "sha256":
                        sha256_file_nb64(
                            path
                        ),
                }
            )


print()
print("=" * 92)
print("2. UPSTREAM TEXT-EVIDENCE INVENTORY")
print("=" * 92)

print(
    "Evidence files discovered:",
    len(
        UPSTREAM_EVIDENCE_INVENTORY_NB64
    ),
)


inventory_count_by_notebook_nb64 = {}


for notebook_number in (
    UPSTREAM_NOTEBOOKS_NB64
):

    count = sum(
        1
        for record
        in UPSTREAM_EVIDENCE_INVENTORY_NB64
        if (
            record[
                "notebook"
            ]
            ==
            notebook_number
        )
    )

    inventory_count_by_notebook_nb64[
        notebook_number
    ] = count

    print(
        f"Notebook {notebook_number}: "
        f"{count} evidence files"
    )


# =============================================================================
# 3. REQUIRED CERTIFIED LATE-STAGE EVIDENCE
# =============================================================================

NB60_CERTIFIED_CONTRACT_NB64 = (
    PROJECT_ROOT
    /
    "artifacts"
    /
    "notebook60"
    /
    "notebook60_certified_model_contract.json"
)


NB61_FINAL_EVALUATION_NB64 = (
    PROJECT_ROOT
    /
    "artifacts"
    /
    "notebook61"
    /
    "notebook61_final_competition_evaluation.json"
)


NB62_PRESERVATION_MANIFEST_NB64 = (
    PROJECT_ROOT
    /
    "artifacts"
    /
    "notebook62"
    /
    "notebook62_final_preservation_manifest.csv"
)


NB63_FINAL_EVIDENCE_NB64 = (
    PROJECT_ROOT
    /
    "artifacts"
    /
    "notebook63"
    /
    "notebook63_final_evidence_package.json"
)


required_late_stage_evidence_nb64 = [
    NB60_CERTIFIED_CONTRACT_NB64,
    NB61_FINAL_EVALUATION_NB64,
    NB62_PRESERVATION_MANIFEST_NB64,
    NB63_FINAL_EVIDENCE_NB64,
]


print()
print("=" * 92)
print("3. REQUIRED LATE-STAGE EVIDENCE")
print("=" * 92)


for path in required_late_stage_evidence_nb64:

    assert path.is_file(), (
        f"Required evidence missing: {path}"
    )

    print(
        "[OK]",
        path.relative_to(
            PROJECT_ROOT
        ),
    )


# =============================================================================
# 4. LOAD CERTIFIED LATE-STAGE EVIDENCE
# =============================================================================

NB60_CERTIFIED_CONTRACT = json.loads(
    NB60_CERTIFIED_CONTRACT_NB64.read_text(
        encoding="utf-8"
    )
)


NB61_FINAL_EVALUATION = json.loads(
    NB61_FINAL_EVALUATION_NB64.read_text(
        encoding="utf-8"
    )
)


NB63_FINAL_EVIDENCE = json.loads(
    NB63_FINAL_EVIDENCE_NB64.read_text(
        encoding="utf-8"
    )
)


with NB62_PRESERVATION_MANIFEST_NB64.open(
    "r",
    encoding="utf-8",
    newline="",
) as handle:

    NB62_PRESERVATION_MANIFEST = list(
        csv.DictReader(
            handle
        )
    )


assert (
    len(
        NB62_PRESERVATION_MANIFEST
    )
    >
    0
)


print()
print(
    "[OK] NB60 certified contract loaded."
)

print(
    "[OK] NB61 final evaluation loaded."
)

print(
    "[OK] NB62 preservation manifest loaded."
)

print(
    "[OK] NB63 final simulation evidence loaded."
)


# =============================================================================
# 5. EVIDENCE-LAYER ROLES
# =============================================================================

EVIDENCE_LAYER_ROLES_NB64 = {

    "NB57":
        (
            "Explainability foundation and global feature-attribution evidence."
        ),

    "NB58":
        (
            "Local explainability and decision-level interpretation evidence."
        ),

    "NB59":
        (
            "Robustness, calibration, and stress-analysis evidence."
        ),

    "NB60":
        (
            "Tournament-scale evaluation and certified model contract."
        ),

    "NB61":
        (
            "Final benchmark / competition-evaluation evidence."
        ),

    "NB62":
        (
            "Final deployment packaging, interface validation, "
            "submission preservation, and certified-asset freeze."
        ),

    "NB63":
        (
            "Controlled simulation runtime, semantic validity, "
            "paired behavior, and side-orientation evidence."
        ),

    "NB64":
        (
            "Strategic interpretation across matchup, orientation, "
            "decision efficiency, limitations, and evidence synthesis."
        ),
}


print()
print("=" * 92)
print("4. EVIDENCE-LAYER ROLES")
print("=" * 92)

print(
    json.dumps(
        EVIDENCE_LAYER_ROLES_NB64,
        indent=2,
    )
)


# =============================================================================
# 6. TRACK 2 CORE FINDINGS
# =============================================================================

TRACK2_CORE_FINDINGS_NB64 = {

    "runtime_reliability": {

        "battle_records_analyzed":
            240,

        "runtime_failures":
            0,

        "unknown_move_rate":
            deployment_profile_nb64[
                "unknown_move_rate"
            ],

        "positive_damage_move_rate":
            deployment_profile_nb64[
                "positive_damage_move_rate"
            ],

        "interpretation":
            (
                "Deployment runtime was stable and semantically valid "
                "throughout the corrected controlled campaign."
            ),
    },


    "matchup_profile": {

        "aggregate_win_rate":
            deployment_profile_nb64[
                "player_win_rate"
            ],

        "controlled_player_favored_scenarios": [
            record[
                "scenario_id"
            ]
            for record
            in MATCHUP_MATRIX_COMPACT_NB64
            if (
                record[
                    "deployment_win_rate"
                ]
                >
                0.5
            )
        ],

        "controlled_balanced_scenarios": [
            record[
                "scenario_id"
            ]
            for record
            in MATCHUP_MATRIX_COMPACT_NB64
            if (
                record[
                    "deployment_win_rate"
                ]
                ==
                0.5
            )
        ],

        "interpretation":
            (
                "Observed matchup outcomes are scenario-conditioned "
                "and must remain scoped to the controlled attack-only environment."
            ),
    },


    "side_orientation": {

        "normal_win_rate":
            CELL3_NB64_STATUS[
                "deployment_normal_win_rate"
            ],

        "swapped_win_rate":
            CELL3_NB64_STATUS[
                "deployment_swapped_win_rate"
            ],

        "aggregate_win_rate_delta":
            CELL3_NB64_STATUS[
                "deployment_absolute_win_rate_delta"
            ],

        "classification":
            CELL3_NB64_STATUS[
                "side_neutrality_classification"
            ],

        "interpretation":
            (
                "Aggregate orientation metrics were neutral, "
                "while asymmetric S3/S4 outcomes followed the side "
                "holding the synthetic damage advantage."
            ),
    },


    "decision_behavior": {

        "deployment_first_legal_exact_match_rate":
            CELL4_NB64_STATUS[
                "deployment_first_legal_exact_match_rate"
            ],

        "deployment_max_damage_exact_match_rate":
            CELL4_NB64_STATUS[
                "deployment_max_damage_exact_match_rate"
            ],

        "interpretation":
            (
                "The deployment policy was behaviorally equivalent "
                "to FIRST_LEGAL_MOVE throughout the paired controlled campaign."
            ),
    },


    "decision_efficiency_gap": {

        "deployment_mean_turns":
            CELL4_NB64_STATUS[
                "deployment_mean_turns"
            ],

        "max_damage_mean_turns":
            CELL4_NB64_STATUS[
                "max_damage_mean_turns"
            ],

        "deployment_damage_per_move":
            CELL4_NB64_STATUS[
                "deployment_damage_per_move"
            ],

        "max_damage_damage_per_move":
            CELL4_NB64_STATUS[
                "max_damage_damage_per_move"
            ],

        "max_damage_turn_reduction_fraction":
            DEPLOYMENT_MAX_DAMAGE_GAP_NB64[
                "max_damage_turn_reduction_vs_deployment"
            ],

        "max_damage_damage_ratio":
            DEPLOYMENT_MAX_DAMAGE_GAP_NB64[
                "damage_per_move_ratio_max_over_deployment"
            ],

        "interpretation":
            (
                "A simple max-damage legal-action rule achieved "
                "the same aggregate controlled win rate with substantially "
                "greater immediate damage and substantially fewer turns."
            ),
    },
}


print()
print("=" * 92)
print("5. TRACK 2 CORE FINDINGS")
print("=" * 92)

print(
    json.dumps(
        TRACK2_CORE_FINDINGS_NB64,
        indent=2,
    )
)


# =============================================================================
# 7. CLAIM / NON-CLAIM BOUNDARIES
# =============================================================================

TRACK2_SUPPORTED_CLAIMS_NB64 = [

    (
        "The deployment runtime completed the corrected controlled "
        "campaign without runtime failures."
    ),

    (
        "The corrected deployment path produced legal, named, "
        "positive-damage actions in the controlled campaign."
    ),

    (
        "Aggregate NORMAL and SWAPPED performance was identical "
        "across the full paired campaign."
    ),

    (
        "Asymmetric scenario outcomes followed the side holding "
        "the synthetic damage advantage."
    ),

    (
        "The deployment policy behaviorally matched FIRST_LEGAL_MOVE "
        "in all 80 paired controlled comparisons."
    ),

    (
        "MAX_DAMAGE_LEGAL_MOVE achieved the same aggregate controlled "
        "win rate while requiring fewer turns and producing more damage per move."
    ),

    (
        "The certified submission assets remained unchanged throughout "
        "NB63 and NB64 analysis."
    ),
]


TRACK2_PROHIBITED_CLAIMS_NB64 = [

    (
        "Do not claim that NB63/NB64 proves the deployment policy "
        "is superior to deterministic baselines."
    ),

    (
        "Do not claim that the controlled synthetic campaign represents "
        "complete official Pokémon TCG gameplay."
    ),

    (
        "Do not claim universal side neutrality from the aggregate "
        "NORMAL/SWAPPED equality."
    ),

    (
        "Do not claim that attack-only damage efficiency alone "
        "determines full-game strategic quality."
    ),

    (
        "Do not replace or invalidate the certified upstream benchmark "
        "evidence from NB57-NB62 with the simplified NB63 campaign."
    ),
]


print()
print("=" * 92)
print("6. CLAIM BOUNDARIES")
print("=" * 92)

print()
print("SUPPORTED CLAIMS")

for index, claim in enumerate(
    TRACK2_SUPPORTED_CLAIMS_NB64,
    start=1,
):

    print(
        f"{index}. {claim}"
    )


print()
print("PROHIBITED / UNSUPPORTED CLAIMS")

for index, claim in enumerate(
    TRACK2_PROHIBITED_CLAIMS_NB64,
    start=1,
):

    print(
        f"{index}. {claim}"
    )


# =============================================================================
# 8. FINAL STRATEGIC CONCLUSIONS
# =============================================================================

TRACK2_FINAL_STRATEGIC_CONCLUSIONS_NB64 = {

    "conclusion_1_runtime":
        (
            "The final deployment stack is executable and stable "
            "under the controlled simulator interface."
        ),

    "conclusion_2_semantics":
        (
            "The corrected attack schema eliminated the prior "
            "Unknown Move artifact and produced valid legal attacks."
        ),

    "conclusion_3_matchup":
        (
            "Controlled outcome patterns were driven primarily by "
            "the synthetic state configuration and damage advantage."
        ),

    "conclusion_4_orientation":
        (
            "The full campaign showed aggregate orientation neutrality, "
            "but asymmetric scenario outcomes appropriately reversed "
            "when the advantaged side was swapped."
        ),

    "conclusion_5_policy_behavior":
        (
            "Within this attack-only environment, the deployment policy "
            "did not demonstrate behavior distinct from FIRST_LEGAL_MOVE."
        ),

    "conclusion_6_efficiency":
        (
            "The max-damage baseline exposed an immediate-action efficiency "
            "gap: comparable aggregate outcomes were reached with fewer turns "
            "and substantially more damage per move."
        ),

    "conclusion_7_scope":
        (
            "NB63/NB64 should therefore be presented as controlled "
            "runtime/semantic/behavioral evidence rather than proof "
            "of deployment-policy superiority."
        ),

    "conclusion_8_upstream_integration":
        (
            "The strategic interpretation should be combined with, "
            "not substituted for, the stronger certified benchmark, "
            "robustness, explainability, and packaging evidence from NB57-NB62."
        ),
}


print()
print("=" * 92)
print("7. FINAL STRATEGIC CONCLUSIONS")
print("=" * 92)

print(
    json.dumps(
        TRACK2_FINAL_STRATEGIC_CONCLUSIONS_NB64,
        indent=2,
    )
)


# =============================================================================
# 9. COMPETITION-WRITEUP EVIDENCE BLOCK
# =============================================================================

COMPETITION_WRITEUP_EVIDENCE_NB64 = {

    "safe_summary":
        (
            "We evaluated the frozen deployment stack in a controlled "
            "attack-only simulation campaign spanning four scenario classes, "
            "two side orientations, ten seeds, and three decision policies. "
            "All 240 runs completed without runtime failures. "
            "After correcting the evaluation card schema, the deployment path "
            "produced valid named attacks with no Unknown Move artifacts. "
            "The deployment policy matched the first-legal-action baseline "
            "across all paired controlled battles, while a max-damage baseline "
            "resolved the same scenarios more efficiently. "
            "These simulations are treated as deployment and behavioral "
            "diagnostics rather than evidence of full-game policy superiority."
        ),

    "headline_metrics": {

        "controlled_battles":
            240,

        "runtime_failures":
            0,

        "deployment_win_rate":
            deployment_profile_nb64[
                "player_win_rate"
            ],

        "deployment_first_legal_pair_equivalence":
            PAIRED_EQUIVALENCE_NB64[
                "deployment_first_legal_exact_match_rate"
            ],

        "deployment_mean_turns":
            deployment_profile_nb64[
                "mean_turn_count"
            ],

        "max_damage_mean_turns":
            max_damage_profile_nb64[
                "mean_turn_count"
            ],

        "deployment_damage_per_move":
            deployment_profile_nb64[
                "mean_damage_per_move"
            ],

        "max_damage_damage_per_move":
            max_damage_profile_nb64[
                "mean_damage_per_move"
            ],
    },

    "required_disclaimer":
        (
            "The campaign uses controlled synthetic attack-only states "
            "and is not a complete official Pokémon TCG rules simulation."
        ),
}


print()
print("=" * 92)
print("8. COMPETITION-WRITEUP EVIDENCE BLOCK")
print("=" * 92)

print(
    json.dumps(
        COMPETITION_WRITEUP_EVIDENCE_NB64,
        indent=2,
    )
)


# =============================================================================
# 10. TRACK 2 COMPLETION READINESS
# =============================================================================

TRACK2_COMPLETION_READINESS_NB64 = {

    "matchup_interpretation_complete":
        True,

    "side_orientation_interpretation_complete":
        True,

    "decision_efficiency_interpretation_complete":
        True,

    "baseline_gap_interpretation_complete":
        True,

    "upstream_evidence_roles_mapped":
        True,

    "claim_boundaries_defined":
        True,

    "competition_writeup_block_created":
        True,

    "final_track2_package_ready":
        True,
}


assert all(
    TRACK2_COMPLETION_READINESS_NB64.values()
)


print()
print("=" * 92)
print("9. TRACK 2 COMPLETION READINESS")
print("=" * 92)

print(
    json.dumps(
        TRACK2_COMPLETION_READINESS_NB64,
        indent=2,
    )
)


# =============================================================================
# 11. PERSIST CELL 5 EVIDENCE
# =============================================================================

CELL5_PAYLOAD_NB64 = {

    "notebook":
        64,

    "cell":
        5,

    "purpose":
        "NB57_NB63_EVIDENCE_INTEGRATION_AND_TRACK2_STRATEGIC_SYNTHESIS",

    "evidence_layer_roles":
        EVIDENCE_LAYER_ROLES_NB64,

    "upstream_evidence_inventory":
        UPSTREAM_EVIDENCE_INVENTORY_NB64,

    "inventory_count_by_notebook":
        inventory_count_by_notebook_nb64,

    "track2_core_findings":
        TRACK2_CORE_FINDINGS_NB64,

    "supported_claims":
        TRACK2_SUPPORTED_CLAIMS_NB64,

    "prohibited_claims":
        TRACK2_PROHIBITED_CLAIMS_NB64,

    "final_strategic_conclusions":
        TRACK2_FINAL_STRATEGIC_CONCLUSIONS_NB64,

    "competition_writeup_evidence":
        COMPETITION_WRITEUP_EVIDENCE_NB64,

    "track2_completion_readiness":
        TRACK2_COMPLETION_READINESS_NB64,

    "analysis_only":
        True,

    "simulation_executed":
        False,

    "training_executed":
        False,

    "submission_modified":
        False,

    "model_modified":
        False,
}


cell5_artifact_nb64 = (
    NB64_ARTIFACT_DIR
    /
    "cell5_track2_strategic_synthesis.json"
)


cell5_output_nb64 = (
    NB64_OUTPUT_DIR
    /
    "cell5_track2_strategic_synthesis.json"
)


cell5_report_nb64 = (
    NB64_REPORT_DIR
    /
    "notebook64_track2_strategic_synthesis.json"
)


for path in [
    cell5_artifact_nb64,
    cell5_output_nb64,
    cell5_report_nb64,
]:

    path.write_text(
        json.dumps(
            CELL5_PAYLOAD_NB64,
            indent=2,
        ),
        encoding="utf-8",
    )


print()
print("=" * 92)
print("10. CELL 5 EVIDENCE PERSISTENCE")
print("=" * 92)

print(
    "[OK]",
    cell5_artifact_nb64,
)

print(
    "[OK]",
    cell5_output_nb64,
)

print(
    "[OK]",
    cell5_report_nb64,
)


# =============================================================================
# 12. CERTIFIED-ASSET INTEGRITY
# =============================================================================

assert (
    sha256_file_nb64(
        FINAL_MAIN
    )
    ==
    EXPECTED_MAIN_SHA256
)


assert (
    sha256_file_nb64(
        CERTIFIED_MODEL
    )
    ==
    EXPECTED_MODEL_SHA256
)


assert (
    sha256_file_nb64(
        DEPLOYMENT_PPO_CHECKPOINT
    )
    ==
    EXPECTED_PPO_SHA256
)


print()
print("=" * 92)
print("11. CERTIFIED-ASSET INTEGRITY")
print("=" * 92)

print(
    "[OK] main.py unchanged."
)

print(
    "[OK] Certified RF model unchanged."
)

print(
    "[OK] Deployment PPO checkpoint unchanged."
)

print(
    "[OK] No simulation executed."
)

print(
    "[OK] No training/retraining executed."
)


# =============================================================================
# 13. FINAL STATUS
# =============================================================================

CELL5_NB64_STATUS = {

    "cell4_pass":
        True,

    "upstream_evidence_integration_complete":
        True,

    "track2_strategic_synthesis_complete":
        True,

    "battle_records_analyzed":
        240,

    "deployment_first_legal_exact_match_rate":
        PAIRED_EQUIVALENCE_NB64[
            "deployment_first_legal_exact_match_rate"
        ],

    "deployment_win_rate":
        deployment_profile_nb64[
            "player_win_rate"
        ],

    "max_damage_win_rate":
        max_damage_profile_nb64[
            "player_win_rate"
        ],

    "claim_boundaries_defined":
        True,

    "competition_writeup_evidence_ready":
        True,

    "final_track2_package_ready":
        True,

    "analysis_only":
        True,

    "simulation_executed":
        False,

    "training_executed":
        False,

    "submission_modified":
        False,

    "model_modified":
        False,

    "main_integrity_preserved":
        True,

    "certified_model_integrity_preserved":
        True,

    "deployment_checkpoint_integrity_preserved":
        True,

    "validation_status":
        "PASS",
}


print()
print("=" * 92)
print("NOTEBOOK 64 — CELL 5 STATUS: PASS")
print("=" * 92)

print(
    json.dumps(
        CELL5_NB64_STATUS,
        indent=2,
    )
)


print()
print(
    "NB57–NB63 evidence integration complete."
)

print(
    "Track 2 strategic synthesis complete."
)

print(
    "NEXT STEP: Cell 6 — final Notebook 64 / Track 2 "
    "evidence package and project handoff."
)


# In[6]:


# =============================================================================
# NOTEBOOK 64 — CELL 6
# FINAL TRACK 2 EVIDENCE PACKAGE + PROJECT HANDOFF
# =============================================================================

from __future__ import annotations

import hashlib
import json
from pathlib import Path


print("=" * 92)
print("NOTEBOOK 64 — CELL 6")
print("FINAL TRACK 2 EVIDENCE PACKAGE + PROJECT HANDOFF")
print("=" * 92)


# =============================================================================
# 0. PREREQUISITES
# =============================================================================

assert (
    CELL5_NB64_STATUS[
        "validation_status"
    ]
    ==
    "PASS"
)

assert (
    CELL5_NB64_STATUS[
        "track2_strategic_synthesis_complete"
    ]
    is True
)

assert (
    CELL5_NB64_STATUS[
        "competition_writeup_evidence_ready"
    ]
    is True
)

assert (
    CELL5_NB64_STATUS[
        "final_track2_package_ready"
    ]
    is True
)

assert (
    CELL5_NB64_STATUS[
        "analysis_only"
    ]
    is True
)


print()
print("[OK] Cell 5 PASS confirmed.")
print("[OK] Track 2 strategic synthesis complete.")
print("[OK] Competition-writeup evidence ready.")
print("[OK] Final Track 2 package authorized.")


# =============================================================================
# 1. FINAL NOTEBOOK 64 FINDINGS
# =============================================================================

NB64_FINAL_FINDINGS = {

    "track":
        "MATCHUP_DECISION_STRATEGY_INTERPRETATION",

    "track_status":
        "COMPLETE",

    "battle_records_analyzed":
        240,

    "deployment_win_rate":
        CELL5_NB64_STATUS[
            "deployment_win_rate"
        ],

    "max_damage_win_rate":
        CELL5_NB64_STATUS[
            "max_damage_win_rate"
        ],

    "deployment_first_legal_exact_match_rate":
        CELL5_NB64_STATUS[
            "deployment_first_legal_exact_match_rate"
        ],

    "deployment_mean_turns":
        CELL4_NB64_STATUS[
            "deployment_mean_turns"
        ],

    "max_damage_mean_turns":
        CELL4_NB64_STATUS[
            "max_damage_mean_turns"
        ],

    "deployment_damage_per_move":
        CELL4_NB64_STATUS[
            "deployment_damage_per_move"
        ],

    "max_damage_damage_per_move":
        CELL4_NB64_STATUS[
            "max_damage_damage_per_move"
        ],

    "aggregate_normal_win_rate":
        CELL3_NB64_STATUS[
            "deployment_normal_win_rate"
        ],

    "aggregate_swapped_win_rate":
        CELL3_NB64_STATUS[
            "deployment_swapped_win_rate"
        ],

    "aggregate_side_win_rate_delta":
        CELL3_NB64_STATUS[
            "deployment_absolute_win_rate_delta"
        ],

    "side_interpretation":
        (
            "Aggregate orientation neutrality was observed, "
            "while asymmetric scenario outcomes tracked the side "
            "holding the synthetic damage advantage."
        ),

    "decision_interpretation":
        (
            "Deployment behavior matched FIRST_LEGAL_MOVE in all "
            "paired controlled battles."
        ),

    "efficiency_interpretation":
        (
            "MAX_DAMAGE_LEGAL_MOVE reached the same aggregate "
            "controlled win rate with fewer turns and greater "
            "damage per move."
        ),

    "scope":
        (
            "CONTROLLED_ATTACK_ONLY_SYNTHETIC_BATTLE_STATES"
        ),
}


print()
print("=" * 92)
print("1. FINAL NOTEBOOK 64 FINDINGS")
print("=" * 92)

print(
    json.dumps(
        NB64_FINAL_FINDINGS,
        indent=2,
    )
)


# =============================================================================
# 2. FINAL CLAIM BOUNDARIES
# =============================================================================

NB64_FINAL_CLAIM_BOUNDARIES = {

    "supported_claims":
        TRACK2_SUPPORTED_CLAIMS_NB64,

    "unsupported_claims":
        TRACK2_PROHIBITED_CLAIMS_NB64,

    "required_scope_statement":
        (
            "NB63/NB64 controlled simulations are deployment, "
            "semantic, and behavioral diagnostics. They are not "
            "a complete official Pokémon TCG gameplay benchmark."
        ),

    "upstream_evidence_relationship":
        (
            "NB64 complements rather than replaces NB57-NB62 "
            "certified explainability, robustness, benchmark, "
            "evaluation, and packaging evidence."
        ),
}


print()
print("=" * 92)
print("2. FINAL CLAIM BOUNDARIES")
print("=" * 92)

print(
    json.dumps(
        NB64_FINAL_CLAIM_BOUNDARIES,
        indent=2,
    )
)


# =============================================================================
# 3. NOTEBOOK 64 EVIDENCE INVENTORY
# =============================================================================

NB64_EXPECTED_EVIDENCE = [

    NB64_ARTIFACT_DIR
    / "cell1_nb63_handoff_initialization.json",

    NB64_ARTIFACT_DIR
    / "cell2_scenario_matchup_matrix.json",

    NB64_ARTIFACT_DIR
    / "cell3_side_neutrality_analysis.json",

    NB64_ARTIFACT_DIR
    / "cell4_decision_efficiency_analysis.json",

    NB64_ARTIFACT_DIR
    / "cell5_track2_strategic_synthesis.json",
]


NB64_EVIDENCE_INVENTORY = []


for path in NB64_EXPECTED_EVIDENCE:

    exists = path.is_file()

    NB64_EVIDENCE_INVENTORY.append(
        {
            "path":
                str(path),

            "exists":
                exists,

            "bytes":
                (
                    path.stat().st_size
                    if exists
                    else None
                ),

            "sha256":
                (
                    sha256_file_nb64(
                        path
                    )
                    if exists
                    else None
                ),
        }
    )


missing_nb64_evidence = [
    record
    for record
    in NB64_EVIDENCE_INVENTORY
    if not record["exists"]
]


print()
print("=" * 92)
print("3. NOTEBOOK 64 EVIDENCE INVENTORY")
print("=" * 92)


for record in NB64_EVIDENCE_INVENTORY:

    print(
        "[OK]"
        if record["exists"]
        else
        "[MISSING]",
        record["path"],
    )


assert (
    not missing_nb64_evidence
), (
    "Notebook 64 evidence package is incomplete."
)


print()
print(
    "[OK] All required Notebook 64 evidence files are present."
)


# =============================================================================
# 4. AUTHORITATIVE IDENTIFIERS
# =============================================================================

NB64_FINAL_IDENTIFIERS = {

    "campaign_configuration_sha256":
        NB63_CORE_HANDOFF[
            "campaign_configuration_sha256"
        ],

    "main_sha256":
        EXPECTED_MAIN_SHA256,

    "certified_rf_sha256":
        EXPECTED_MODEL_SHA256,

    "deployment_ppo_sha256":
        EXPECTED_PPO_SHA256,

    "nb63_final_package_sha256":
        NB63_FINAL_PACKAGE.get(
            "final_package_sha256"
        ),

    "nb64_source_notebook":
        "64_matchup_decision_strategy_interpretation.ipynb",
}


print()
print("=" * 92)
print("4. AUTHORITATIVE IDENTIFIERS")
print("=" * 92)

print(
    json.dumps(
        NB64_FINAL_IDENTIFIERS,
        indent=2,
    )
)


# =============================================================================
# 5. TWO-TRACK COMPLETION SUMMARY
# =============================================================================

TWO_TRACK_COMPLETION_SUMMARY = {

    "track_1": {

        "notebook":
            63,

        "name":
            "SIMULATION_AND_EVIDENCE",

        "status":
            "COMPLETE",

        "battle_count":
            240,

        "runtime_failures":
            0,

        "key_result":
            (
                "Controlled deployment runtime and semantic "
                "validation completed successfully."
            ),
    },


    "track_2": {

        "notebook":
            64,

        "name":
            "MATCHUP_DECISION_STRATEGY_INTERPRETATION",

        "status":
            "COMPLETE",

        "key_result":
            (
                "Matchup, orientation, decision-efficiency, "
                "baseline-gap, limitation, and writeup interpretation "
                "completed."
            ),
    },


    "two_track_plan_complete":
        True,
}


print()
print("=" * 92)
print("5. TWO-TRACK COMPLETION SUMMARY")
print("=" * 92)

print(
    json.dumps(
        TWO_TRACK_COMPLETION_SUMMARY,
        indent=2,
    )
)


# =============================================================================
# 6. COMPETITION HANDOFF
# =============================================================================

FINAL_COMPETITION_HANDOFF_NB64 = {

    "status":
        "READY_FOR_FINAL_COMPETITION_REVIEW",

    "certified_submission":
        str(
            SUBMISSION_DIR
        ),

    "certified_main":
        str(
            FINAL_MAIN
        ),

    "certified_rf_model":
        str(
            CERTIFIED_MODEL
        ),

    "deployment_ppo_checkpoint":
        str(
            DEPLOYMENT_PPO_CHECKPOINT
        ),

    "nb63_final_evidence":
        str(
            NB63_FINAL_PACKAGE_PATH
        ),

    "nb64_final_strategy_report":
        str(
            NB64_REPORT_DIR
            /
            "notebook64_final_track2_package.json"
        ),

    "competition_writeup_block":
        COMPETITION_WRITEUP_EVIDENCE_NB64,

    "final_review_focus": [

        "submission contents",

        "submission ZIP integrity",

        "competition writeup",

        "claims versus evidence",

        "required disclaimer",

        "final file preservation",

        "submission portal readiness",
    ],

    "additional_notebook_required":
        False,
}


print()
print("=" * 92)
print("6. FINAL COMPETITION HANDOFF")
print("=" * 92)

print(
    json.dumps(
        FINAL_COMPETITION_HANDOFF_NB64,
        indent=2,
    )
)


# =============================================================================
# 7. BUILD FINAL TRACK 2 PACKAGE
# =============================================================================

NB64_FINAL_PACKAGE = {

    "notebook":
        64,

    "title":
        "MATCHUP_DECISION_STRATEGY_INTERPRETATION",

    "status":
        "COMPLETE",

    "final_findings":
        NB64_FINAL_FINDINGS,

    "claim_boundaries":
        NB64_FINAL_CLAIM_BOUNDARIES,

    "strategic_conclusions":
        TRACK2_FINAL_STRATEGIC_CONCLUSIONS_NB64,

    "competition_writeup_evidence":
        COMPETITION_WRITEUP_EVIDENCE_NB64,

    "track2_core_findings":
        TRACK2_CORE_FINDINGS_NB64,

    "two_track_completion":
        TWO_TRACK_COMPLETION_SUMMARY,

    "identifiers":
        NB64_FINAL_IDENTIFIERS,

    "evidence_inventory":
        NB64_EVIDENCE_INVENTORY,

    "competition_handoff":
        FINAL_COMPETITION_HANDOFF_NB64,
}


canonical_nb64_json = json.dumps(
    NB64_FINAL_PACKAGE,
    sort_keys=True,
    separators=(
        ",",
        ":",
    ),
)


NB64_FINAL_PACKAGE_SHA256 = (
    hashlib.sha256(
        canonical_nb64_json.encode(
            "utf-8"
        )
    ).hexdigest()
)


NB64_FINAL_PACKAGE[
    "final_package_sha256"
] = NB64_FINAL_PACKAGE_SHA256


print()
print("=" * 92)
print("7. FINAL TRACK 2 PACKAGE SHA256")
print("=" * 92)

print(
    NB64_FINAL_PACKAGE_SHA256
)


# =============================================================================
# 8. PERSIST FINAL PACKAGE
# =============================================================================

NB64_FINAL_ARTIFACT = (
    NB64_ARTIFACT_DIR
    /
    "notebook64_final_track2_package.json"
)


NB64_FINAL_OUTPUT = (
    NB64_OUTPUT_DIR
    /
    "notebook64_final_track2_package.json"
)


NB64_FINAL_REPORT = (
    NB64_REPORT_DIR
    /
    "notebook64_final_track2_package.json"
)


NB64_COMPETITION_HANDOFF = (
    NB64_ARTIFACT_DIR
    /
    "notebook64_final_competition_handoff.json"
)


for path in [
    NB64_FINAL_ARTIFACT,
    NB64_FINAL_OUTPUT,
    NB64_FINAL_REPORT,
]:

    path.write_text(
        json.dumps(
            NB64_FINAL_PACKAGE,
            indent=2,
        ),
        encoding="utf-8",
    )


NB64_COMPETITION_HANDOFF.write_text(
    json.dumps(
        FINAL_COMPETITION_HANDOFF_NB64,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 92)
print("8. FINAL PACKAGE PERSISTENCE")
print("=" * 92)

print(
    "[OK]",
    NB64_FINAL_ARTIFACT,
)

print(
    "[OK]",
    NB64_FINAL_OUTPUT,
)

print(
    "[OK]",
    NB64_FINAL_REPORT,
)

print(
    "[OK]",
    NB64_COMPETITION_HANDOFF,
)


# =============================================================================
# 9. FINAL CERTIFIED-ASSET INTEGRITY
# =============================================================================

final_main_hash_nb64 = (
    sha256_file_nb64(
        FINAL_MAIN
    )
)

final_rf_hash_nb64 = (
    sha256_file_nb64(
        CERTIFIED_MODEL
    )
)

final_ppo_hash_nb64 = (
    sha256_file_nb64(
        DEPLOYMENT_PPO_CHECKPOINT
    )
)


assert (
    final_main_hash_nb64
    ==
    EXPECTED_MAIN_SHA256
)

assert (
    final_rf_hash_nb64
    ==
    EXPECTED_MODEL_SHA256
)

assert (
    final_ppo_hash_nb64
    ==
    EXPECTED_PPO_SHA256
)


print()
print("=" * 92)
print("9. FINAL CERTIFIED-ASSET INTEGRITY")
print("=" * 92)

print(
    "main.py:",
    final_main_hash_nb64,
)

print(
    "Certified RF:",
    final_rf_hash_nb64,
)

print(
    "Deployment PPO:",
    final_ppo_hash_nb64,
)

print()
print("[OK] Certified main.py unchanged.")
print("[OK] Certified RF model unchanged.")
print("[OK] Deployment PPO checkpoint unchanged.")
print("[OK] No training/retraining occurred.")
print("[OK] No submission mutation occurred.")


# =============================================================================
# 10. FINAL ASSERTIONS
# =============================================================================

assert (
    TWO_TRACK_COMPLETION_SUMMARY[
        "track_1"
    ][
        "status"
    ]
    ==
    "COMPLETE"
)

assert (
    TWO_TRACK_COMPLETION_SUMMARY[
        "track_2"
    ][
        "status"
    ]
    ==
    "COMPLETE"
)

assert (
    TWO_TRACK_COMPLETION_SUMMARY[
        "two_track_plan_complete"
    ]
    is True
)

assert (
    FINAL_COMPETITION_HANDOFF_NB64[
        "additional_notebook_required"
    ]
    is False
)


# =============================================================================
# 11. FINAL STATUS
# =============================================================================

CELL6_NB64_STATUS = {

    "cell5_pass":
        True,

    "notebook64_complete":
        True,

    "track2_complete":
        True,

    "two_track_plan_complete":
        True,

    "track1_status":
        "COMPLETE",

    "track2_status":
        "COMPLETE",

    "battle_records_analyzed":
        240,

    "final_track2_package_created":
        True,

    "competition_handoff_created":
        True,

    "competition_writeup_evidence_ready":
        True,

    "additional_notebook_required":
        False,

    "final_package_sha256":
        NB64_FINAL_PACKAGE_SHA256,

    "analysis_only":
        True,

    "simulation_executed":
        False,

    "training_executed":
        False,

    "submission_modified":
        False,

    "model_modified":
        False,

    "main_integrity_preserved":
        True,

    "certified_model_integrity_preserved":
        True,

    "deployment_checkpoint_integrity_preserved":
        True,

    "validation_status":
        "PASS",
}


print()
print("=" * 92)
print("NOTEBOOK 64 — CELL 6 STATUS: PASS")
print("=" * 92)

print(
    json.dumps(
        CELL6_NB64_STATUS,
        indent=2,
    )
)


print()
print("=" * 92)
print("NOTEBOOK 64 STATUS: COMPLETE")
print("TRACK 2 — STRATEGY INTERPRETATION: COMPLETE")
print("TWO-TRACK POST-NB62 PLAN: COMPLETE")
print("=" * 92)


print()
print(
    "Notebook 64 final Track 2 package created."
)

print(
    "No additional analysis notebook is required by the "
    "confirmed two-track completion plan."
)

print(
    "NEXT STEP: save and back up Notebook 64, then perform "
    "the final competition/submission/writeup review."
)


# In[ ]:




