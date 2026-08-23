#!/usr/bin/env python
# coding: utf-8

# # Notebook 16 — Kaggle Submission Compatibility
# 
# ## The Pokémon Company — PTCG AI Battle Challenge Simulation
# 
# ### Team Jesus
# 
# This notebook prepares, validates, and packages a competition-compatible
# Pokémon TCG agent submission.
# 
# ## Main objectives
# 
# 1. Confirm the required Kaggle submission structure.
# 2. Create a minimal valid `main.py`.
# 3. validate the competition deck file.
# 4. Build `submission.tar.gz`.
# 5. Inspect the archive contents.
# 6. Prepare the first Simulation submission.
# 7. Later connect the production Team Jesus engine.
# 
# ## Required archive structure
# 
# ```text
# submission.tar.gz
# ├── main.py
# └── deck.csv

# In[2]:


from __future__ import annotations

import csv
import importlib.util
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Any, Iterable

print("Python version:", sys.version)
print("Current working directory:", Path.cwd())


# # Cell 3 — Find the project root
# 
# ## This cell safely detects the project directory whether the notebook is opened from the root or from notebooks.

# In[3]:


def find_project_root(start: Path | None = None) -> Path:
    """
    Locate the PTCG project root.

    The project root is identified using common project folders or files.
    """

    current = (start or Path.cwd()).resolve()

    markers = [
        "src",
        "notebooks",
        "README.md",
        "requirements.txt",
        "pyproject.toml",
    ]

    for candidate in [current, *current.parents]:
        marker_count = sum((candidate / marker).exists() for marker in markers)

        if marker_count >= 2:
            return candidate

    # Safe fallback when the notebook is inside the notebooks directory.
    if current.name.lower() == "notebooks":
        return current.parent

    return current


PROJECT_ROOT = find_project_root()

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"

SUBMISSION_WORK_DIR = PROJECT_ROOT / "submission_work"
SUBMISSION_DIR = SUBMISSION_WORK_DIR / "submission"
SUBMISSION_ARCHIVE = SUBMISSION_WORK_DIR / "submission.tar.gz"

print("Project root:          ", PROJECT_ROOT)
print("Notebooks directory:   ", NOTEBOOKS_DIR)
print("Source directory:      ", SRC_DIR)
print("Data directory:        ", DATA_DIR)
print("Submission directory:  ", SUBMISSION_DIR)
print("Submission archive:    ", SUBMISSION_ARCHIVE)


# # Cell 4 — Inspect the current project structure

# In[4]:


def show_directory(
    directory: Path,
    *,
    max_depth: int = 2,
    max_items: int = 100,
) -> None:
    """
    Display a compact directory tree without printing thousands of files.
    """

    directory = directory.resolve()

    if not directory.exists():
        print(f"[MISSING] {directory}")
        return

    print(directory.name + "/")

    displayed = 0

    for path in sorted(directory.rglob("*")):
        try:
            relative = path.relative_to(directory)
        except ValueError:
            continue

        depth = len(relative.parts)

        if depth > max_depth:
            continue

        indent = "    " * depth
        suffix = "/" if path.is_dir() else ""

        print(f"{indent}{path.name}{suffix}")

        displayed += 1

        if displayed >= max_items:
            print(f"\n... display stopped after {max_items} items")
            break


show_directory(PROJECT_ROOT, max_depth=2, max_items=120)


# # Cell 5 — Define official Kaggle paths

# In[6]:


from pathlib import Path

COMPETITION_INPUT = Path(
    "/kaggle/input/competitions/pokemon-tcg-ai-battle"
)

SAMPLE_SUBMISSION_ROOT = (
    COMPETITION_INPUT
    / "sample_submission"
    / "sample_submission"
)

KAGGLE_WORKING = Path("/kaggle/working")
BASELINE_BUILD_DIR = KAGGLE_WORKING / "team_jesus_baseline"

OFFICIAL_MAIN = SAMPLE_SUBMISSION_ROOT / "main.py"
OFFICIAL_DECK = SAMPLE_SUBMISSION_ROOT / "deck.csv"
OFFICIAL_CG = SAMPLE_SUBMISSION_ROOT / "cg"

BASELINE_MAIN = BASELINE_BUILD_DIR / "main.py"
BASELINE_DECK = BASELINE_BUILD_DIR / "deck.csv"
BASELINE_CG = BASELINE_BUILD_DIR / "cg"

BASELINE_ARCHIVE = KAGGLE_WORKING / "submission.tar.gz"

print("Competition input:", COMPETITION_INPUT)
print("Sample submission:", SAMPLE_SUBMISSION_ROOT)
print("Build directory:", BASELINE_BUILD_DIR)
print("Output archive:", BASELINE_ARCHIVE)


# # Cell 6 — Verify the official files

# In[11]:


from pathlib import Path

SAMPLE_SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "kaggle_sample_submission"
)

DOWNLOADED_MAIN = SAMPLE_SOURCE_DIR / "main.py"
DOWNLOADED_DECK = SAMPLE_SOURCE_DIR / "deck.csv"
DOWNLOADED_ARCHIVE = SAMPLE_SOURCE_DIR / "submission.tar.gz"

SUBMISSION_WORK_DIR = PROJECT_ROOT / "submission_work"
EXTRACTED_SAMPLE_DIR = SUBMISSION_WORK_DIR / "official_sample_extracted"
BASELINE_BUILD_DIR = SUBMISSION_WORK_DIR / "team_jesus_baseline"
BASELINE_ARCHIVE = SUBMISSION_WORK_DIR / "team_jesus_baseline.tar.gz"

print("Sample source directory:", SAMPLE_SOURCE_DIR)
print("Downloaded main.py:     ", DOWNLOADED_MAIN)
print("Downloaded deck.csv:    ", DOWNLOADED_DECK)
print("Downloaded archive:     ", DOWNLOADED_ARCHIVE)
print("Extraction directory:   ", EXTRACTED_SAMPLE_DIR)
print("Baseline build folder:  ", BASELINE_BUILD_DIR)
print("Baseline archive:       ", BASELINE_ARCHIVE)


# # Cell 7 — Safely inspect the archive

# In[14]:


import tarfile


def inspect_tar_archive(archive_path: Path) -> list[str]:
    """
    Return every file stored inside a .tar.gz archive.
    """

    if not archive_path.exists():
        raise FileNotFoundError(archive_path)

    with tarfile.open(archive_path, "r:gz") as tar:
        return sorted(member.name for member in tar.getmembers())


archive_members = inspect_tar_archive(DOWNLOADED_ARCHIVE)

print(f"Archive contains {len(archive_members)} entries\n")

for member in archive_members:
    print(member)


# # Cell 8 — Validate the archive structure

# In[15]:


REQUIRED_ARCHIVE_FILES = {
    "main.py",
    "deck.csv",
    "cg/__init__.py",
    "cg/api.py",
    "cg/game.py",
    "cg/sim.py",
    "cg/utils.py",
}

archive_member_set = {
    name.rstrip("/")
    for name in archive_members
}

missing_archive_files = (
    REQUIRED_ARCHIVE_FILES - archive_member_set
)

if missing_archive_files:
    print("Missing required files:")

    for name in sorted(missing_archive_files):
        print("-", name)

    raise ValueError(
        "The downloaded submission archive is incomplete."
    )

native_libraries = sorted(
    name
    for name in archive_member_set
    if name.endswith((".dll", ".so"))
)

print("Required archive files found.")
print("Native libraries found:")

for library in native_libraries:
    print("-", library)

if not native_libraries:
    raise ValueError(
        "No native cg library was found."
    )

print("\nArchive structure validation passed.")


# # Cell 9 — Safely extract the archive

# In[16]:


import shutil
import tarfile


def safe_extract_tar(
    archive_path: Path,
    destination: Path,
) -> None:
    """
    Safely extract a .tar.gz archive while preventing path traversal.
    """

    destination = destination.resolve()

    if destination.exists():
        shutil.rmtree(destination)

    destination.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as tar:

        for member in tar.getmembers():

            target = (destination / member.name).resolve()

            try:
                target.relative_to(destination)
            except ValueError:
                raise ValueError(
                    f"Unsafe archive member: {member.name}"
                )

        tar.extractall(destination)


safe_extract_tar(
    DOWNLOADED_ARCHIVE,
    EXTRACTED_SAMPLE_DIR,
)

print("Archive extracted successfully.")
print(EXTRACTED_SAMPLE_DIR)


# # Cell 10 — Display the extracted folder

# In[17]:


show_directory(
    EXTRACTED_SAMPLE_DIR,
    max_depth=3,
    max_items=100,
)


# # Cell 11 — Read the official main.py

# In[18]:


OFFICIAL_MAIN = EXTRACTED_SAMPLE_DIR / "main.py"

with open(OFFICIAL_MAIN, "r", encoding="utf-8") as f:
    main_source = f.read()

print(f"Characters : {len(main_source):,}")
print(f"Lines      : {len(main_source.splitlines())}")


# # Cell 12 — Preview the first 100 lines

# In[19]:


preview_lines = main_source.splitlines()

for i, line in enumerate(preview_lines[:100], start=1):
    print(f"{i:4}: {line}")


# # Cell 13 — Locate important sections automatically

# In[20]:


KEYWORDS = [
    "def agent",
    "obs.select",
    "to_observation_class",
    "all_card_data",
    "with open",
    "return my_deck",
    "return",
]

lines = main_source.splitlines()

for keyword in KEYWORDS:
    print("=" * 70)
    print(keyword)
    print("=" * 70)

    found = False

    for i, line in enumerate(lines):
        if keyword in line:
            found = True

            start = max(0, i - 3)
            end = min(len(lines), i + 8)

            for j in range(start, end):
                print(f"{j+1:4}: {lines[j]}")

            print()

    if not found:
        print("Not found.\n")


# # Cell 14 — Notebook Notes

# ## Official Agent Observations
# 
# ### Entry Point
# 
# The Kaggle simulator calls:
# 
# ```python
# agent(obs_dict)

# # Cell 14 — Display the entire agent() function

# In[21]:


lines = main_source.splitlines()

start = None
end = None

for i, line in enumerate(lines):

    if line.startswith("def agent("):
        start = i

        continue

    if start is not None:

        if (
            line.startswith("def ")
            and i > start
        ):
            end = i
            break

if start is None:
    raise ValueError("agent() not found")

if end is None:
    end = len(lines)

print(f"agent() starts at line {start+1}")
print(f"agent() ends   at line {end}")

print()

for i in range(start, end):
    print(f"{i+1:4}: {lines[i]}")


# #  Cell 15 — Build Our Integration Blueprint

# # Team Jesus Agent Integration Blueprint
# 
# ## Official Kaggle Flow
# 
# ```text
# Kaggle Simulator
#         │
#         ▼
# agent(obs_dict)
#         │
#         ▼
# Observation Object
#         │
#         ▼
# Legal Options (select.option)
#         │
#         ▼
# Choose Option Indices
#         │
#         ▼
# Return list[int]
# ```
# 
# ## Team Jesus Flow
# 
# ```text
# Kaggle Simulator
#         │
#         ▼
# Observation Adapter
#         │
#         ▼
# Internal Battle State
#         │
#         ▼
# Evaluation Function
#         │
#         ▼
# Alpha-Beta Search
#         │
#         ▼
# Move Ranking
#         │
#         ▼
# Return Best Option Indices
# ```
# 
# ## Integration Strategy
# 
# Notebook 17
# - Learn the official card database.
# 
# Notebook 18
# - Convert Kaggle observations into our internal engine representation.
# 
# Notebook 19+
# - Replace heuristic scoring with Team Jesus search and evaluation.

# # Cell 16 — Create the Team Jesus baseline build folder

# In[22]:


import shutil


def create_baseline_build(
    source_dir: Path,
    build_dir: Path,
) -> Path:
    """
    Create a clean Team Jesus baseline submission folder.
    """

    required_items = [
        source_dir / "main.py",
        source_dir / "deck.csv",
        source_dir / "cg",
    ]

    missing_items = [
        path for path in required_items
        if not path.exists()
    ]

    if missing_items:
        missing_text = "\n".join(
            f"- {path}" for path in missing_items
        )
        raise FileNotFoundError(
            "Cannot create the baseline build. "
            f"Missing:\n{missing_text}"
        )

    if build_dir.exists():
        shutil.rmtree(build_dir)

    build_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        source_dir / "main.py",
        build_dir / "main.py",
    )

    shutil.copy2(
        source_dir / "deck.csv",
        build_dir / "deck.csv",
    )

    shutil.copytree(
        source_dir / "cg",
        build_dir / "cg",
    )

    return build_dir


created_build_dir = create_baseline_build(
    EXTRACTED_SAMPLE_DIR,
    BASELINE_BUILD_DIR,
)

print("Team Jesus baseline build created:")
print(created_build_dir)


# # Cell 17 — Display the baseline folder

# In[24]:


show_directory(
    BASELINE_BUILD_DIR,
    max_depth=3,
    max_items=100,
)


# # Cell 18 — Validate the 60-card deck

# from collections import Counter
# 
# 
# def load_deck(deck_path: Path) -> list[int]:
#     """
#     Load one numeric Card ID from each nonblank line.
#     """
# 
#     if not deck_path.is_file():
#         raise FileNotFoundError(
#             f"Deck file not found: {deck_path}"
#         )
# 
#     raw_lines = deck_path.read_text(
#         encoding="utf-8"
#     ).splitlines()
# 
#     values = [
#         line.strip()
#         for line in raw_lines
#         if line.strip()
#     ]
# 
#     card_ids: list[int] = []
# 
#     for line_number, value in enumerate(
#         values,
#         start=1,
#     ):
#         try:
#             card_id = int(value)
#         except ValueError as exc:
#             raise ValueError(
#                 f"Invalid Card ID on line "
#                 f"{line_number}: {value!r}"
#             ) from exc
# 
#         if card_id < 0:
#             raise ValueError(
#                 f"Negative Card ID on line "
#                 f"{line_number}: {card_id}"
#             )
# 
#         card_ids.append(card_id)
# 
#     return card_ids
# 
# 
# team_jesus_deck = load_deck(
#     BASELINE_BUILD_DIR / "deck.csv"
# )
# 
# deck_counts = Counter(team_jesus_deck)
# 
# print("Deck size:", len(team_jesus_deck))
# print("Unique Card IDs:", len(deck_counts))
# print("First 10 cards:", team_jesus_deck[:10])
# print()
# 
# print("Card counts:")
# 
# for card_id, count in sorted(deck_counts.items()):
#     print(f"Card ID {card_id:4}: {count}")

# # Cell 19 — Enforce baseline deck requirements

# In[26]:


deck_errors: list[str] = []

if len(team_jesus_deck) != 60:
    deck_errors.append(
        f"Deck contains {len(team_jesus_deck)} "
        "cards instead of 60."
    )

if not all(
    isinstance(card_id, int)
    for card_id in team_jesus_deck
):
    deck_errors.append(
        "Every deck entry must be an integer."
    )

if any(
    card_id < 0
    for card_id in team_jesus_deck
):
    deck_errors.append(
        "Deck contains a negative Card ID."
    )

if deck_errors:
    print("Deck validation failed:")

    for error in deck_errors:
        print("-", error)

    raise ValueError(
        "Baseline deck validation failed."
    )

print("Deck validation passed.")
print("The baseline deck contains exactly 60 Card IDs.")


# # Cell 20 — Validate main.py compatibility

# In[27]:


baseline_main_path = (
    BASELINE_BUILD_DIR / "main.py"
)

baseline_main_source = baseline_main_path.read_text(
    encoding="utf-8"
)

required_main_fragments = {
    "agent entry point":
        "def agent(obs_dict: dict) -> list[int]:",

    "observation conversion":
        "to_observation_class(obs_dict)",

    "initial deck handling":
        "if obs.select == None:",

    "deck return":
        "return my_deck",

    "legal-option access":
        "select.option",

    "final action return":
        "return desc_indices[:select.maxCount]",
}

main_errors: list[str] = []

for label, fragment in required_main_fragments.items():
    found = fragment in baseline_main_source
    status = "FOUND" if found else "MISSING"

    print(f"{status:8} | {label}")

    if not found:
        main_errors.append(
            f"Missing {label}: {fragment}"
        )

if main_errors:
    raise ValueError(
        "main.py compatibility validation failed:\n"
        + "\n".join(main_errors)
    )

print("\nmain.py compatibility validation passed.")


# # Cell 21 — Build our baseline archive

# In[28]:


import tarfile


def build_submission_archive(
    build_dir: Path,
    archive_path: Path,
) -> Path:
    """
    Create a .tar.gz archive with all submission
    files stored at the archive's top level.
    """

    if not build_dir.is_dir():
        raise NotADirectoryError(
            f"Build directory not found: {build_dir}"
        )

    archive_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if archive_path.exists():
        archive_path.unlink()

    with tarfile.open(
        archive_path,
        mode="w:gz",
    ) as tar:

        for item in sorted(build_dir.iterdir()):
            tar.add(
                item,
                arcname=item.name,
                recursive=True,
            )

    return archive_path


created_archive = build_submission_archive(
    BASELINE_BUILD_DIR,
    BASELINE_ARCHIVE,
)

print("Baseline archive created:")
print(created_archive)
print()
print(
    "Archive size:",
    f"{created_archive.stat().st_size:,}",
    "bytes",
)


# # Cell 22 — Inspect the archive we built

# In[29]:


team_jesus_archive_members = inspect_tar_archive(
    BASELINE_ARCHIVE
)

print(
    "Team Jesus archive contains "
    f"{len(team_jesus_archive_members)} entries\n"
)

for member in team_jesus_archive_members:
    print(member)


# # Cell 23 — Independent final validation

# In[30]:


FINAL_REQUIRED_MEMBERS = {
    "main.py",
    "deck.csv",
    "cg",
    "cg/__init__.py",
    "cg/api.py",
    "cg/cg.dll",
    "cg/game.py",
    "cg/libcg.so",
    "cg/sim.py",
    "cg/utils.py",
}


def validate_final_submission(
    archive_path: Path,
) -> dict[str, object]:
    """
    Perform independent final validation of the
    Team Jesus baseline submission archive.
    """

    errors: list[str] = []

    if not archive_path.is_file():
        return {
            "valid": False,
            "errors": [
                f"Archive does not exist: {archive_path}"
            ],
        }

    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        member_names = {
            member.name.rstrip("/")
            for member in members
        }

    missing_members = (
        FINAL_REQUIRED_MEMBERS - member_names
    )

    unexpected_top_level = {
        name.split("/")[0]
        for name in member_names
    } - {
        "main.py",
        "deck.csv",
        "cg",
    }

    if missing_members:
        errors.append(
            "Missing archive members: "
            + ", ".join(sorted(missing_members))
        )

    if unexpected_top_level:
        errors.append(
            "Unexpected top-level items: "
            + ", ".join(sorted(unexpected_top_level))
        )

    # Validate deck directly from the archive.
    with tarfile.open(archive_path, "r:gz") as tar:
        deck_member = tar.extractfile("deck.csv")

        if deck_member is None:
            errors.append(
                "deck.csv could not be read from archive."
            )
            archived_deck: list[int] = []
        else:
            deck_text = deck_member.read().decode("utf-8")
            archived_deck = [
                int(line.strip())
                for line in deck_text.splitlines()
                if line.strip()
            ]

    if len(archived_deck) != 60:
        errors.append(
            "Archived deck contains "
            f"{len(archived_deck)} cards instead of 60."
        )

    # Validate main.py directly from the archive.
    with tarfile.open(archive_path, "r:gz") as tar:
        main_member = tar.extractfile("main.py")

        if main_member is None:
            errors.append(
                "main.py could not be read from archive."
            )
            archived_main = ""
        else:
            archived_main = (
                main_member.read().decode("utf-8")
            )

    required_fragments = [
        "def agent(obs_dict: dict) -> list[int]:",
        "to_observation_class(obs_dict)",
        "return my_deck",
        "select.option",
    ]

    for fragment in required_fragments:
        if fragment not in archived_main:
            errors.append(
                f"main.py is missing: {fragment}"
            )

    return {
        "valid": not errors,
        "errors": errors,
        "member_count": len(member_names),
        "deck_size": len(archived_deck),
        "archive_size_bytes":
            archive_path.stat().st_size,
        "archive_path": archive_path,
    }


final_validation = validate_final_submission(
    BASELINE_ARCHIVE
)

print("Final archive validation")
print("=" * 50)

for key, value in final_validation.items():
    if key != "errors":
        print(f"{key}: {value}")

if final_validation["errors"]:
    print("\nErrors:")

    for error in final_validation["errors"]:
        print("-", error)

    raise ValueError(
        "Final submission validation failed."
    )

print("\nFINAL VALIDATION PASSED")
print("The baseline archive is ready for Kaggle upload.")


# # Cell 24 — Completion summary

# # Notebook 16 Complete
# 
# ## Kaggle Submission Compatibility
# 
# The project successfully:
# 
# - Located the downloaded Kaggle sample submission.
# - Verified `main.py`, `deck.csv`, and `submission.tar.gz`.
# - Inspected and safely extracted the official archive.
# - Confirmed the complete `cg/` runtime.
# - Identified the official `agent(obs_dict)` entry point.
# - Confirmed that the agent returns legal-option indices.
# - Validated the 60-card deck.
# - Created an independent Team Jesus baseline build.
# - Rebuilt `team_jesus_baseline.tar.gz`.
# - Performed final archive validation.
# 
# ## Final baseline archive
# 
# ```text
# submission_work/team_jesus_baseline.tar.gz

# In[ ]:




