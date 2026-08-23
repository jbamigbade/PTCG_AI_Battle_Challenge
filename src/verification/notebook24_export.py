#!/usr/bin/env python
# coding: utf-8

# # Notebook 24 — Final Submission Validation and Runtime Smoke Test
# 
# ## The Pokémon Company — PTCG AI Battle Challenge
# 
# ### Team Jesus
# 
# Notebook 23 created the packaged submission archive.
# 
# Notebook 24 validates the archive as a standalone artifact before any Kaggle upload.
# 
# ## Objectives
# 
# 1. Locate the final submission ZIP.
# 2. Extract it into a clean temporary directory.
# 3. Verify the manifest and checksums.
# 4. Confirm the 60-card deck.
# 5. Compile every packaged Python file.
# 6. Inspect imports and runtime dependencies.
# 7. Load the packaged policy and agent exports.
# 8. Validate the packaged production entry point.
# 9. Test deck-request behavior.
# 10. Test action-selection behavior.
# 11. Run repeated smoke tests.
# 12. Detect missing, duplicate, or unexpected files.
# 13. Produce a final readiness report.
# 14. Save Notebook 24 through PowerShell.

# # Cell 2 — Imports

# In[1]:


from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import types
import uuid
import zipfile

from pathlib import Path
from typing import Any

print("Python:", sys.version)
print("Working directory:", Path.cwd())


# # Cell 3 — Locate the project and archive

# In[4]:


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()

    markers = {
        "notebooks",
        "scripts",
        "src",
        "submission_work",
    }

    for candidate in [current, *current.parents]:
        found = {
            marker
            for marker in markers
            if (candidate / marker).exists()
        }

        if len(found) >= 3:
            return candidate

    return current


PROJECT_ROOT = find_project_root()

ZIP_FILE = (
    PROJECT_ROOT
    / "submission_work"
    / "team_jesus_submission.zip"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "notebook24"
)

EXTRACT_DIR = (
    PROJECT_ROOT
    / "submission_work"
    / "notebook24_extracted"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

if EXTRACT_DIR.exists():
    shutil.rmtree(EXTRACT_DIR)

EXTRACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("Project root:", PROJECT_ROOT)
print("Submission ZIP:", ZIP_FILE)
print("Extraction directory:", EXTRACT_DIR)
print("Reports:", REPORT_DIR)


# # Cell 4 — Validate the ZIP exists

# In[6]:


if not ZIP_FILE.is_file():
    raise FileNotFoundError(
        "Final submission ZIP was not found:\n"
        f"{ZIP_FILE}"
    )

print("Submission ZIP located.")
print("Size:", ZIP_FILE.stat().st_size, "bytes")


# # Cell 5 — Extract the submission archive

# In[7]:


with zipfile.ZipFile(ZIP_FILE) as archive:
    archive.extractall(EXTRACT_DIR)
    archive_names = archive.namelist()

print("Archive entries:", len(archive_names))

for name in archive_names:
    print("-", name)

print("\nArchive extracted successfully.")


# # Cell 6 — Define required packaged files

# In[8]:


REQUIRED_PACKAGE_FILES = {
    "manifest": EXTRACT_DIR / "manifest.json",
    "deck": EXTRACT_DIR / "data" / "deck.csv",
    "agent": (
        EXTRACT_DIR
        / "agent"
        / "notebook21_export.py"
    ),
    "evaluation": (
        EXTRACT_DIR
        / "evaluation"
        / "notebook22_export.py"
    ),
    "policy": (
        EXTRACT_DIR
        / "policy"
        / "notebook20_export.py"
    ),
}

missing_files = []

for name, path in REQUIRED_PACKAGE_FILES.items():
    exists = path.is_file()

    print(
        f"{'[FOUND]' if exists else '[MISSING]'} "
        f"{name}: {path}"
    )

    if not exists:
        missing_files.append(str(path))

if missing_files:
    raise FileNotFoundError(
        "Packaged submission is incomplete:\n"
        + "\n".join(missing_files)
    )

print("\nAll required packaged files located.")


# # Cell 7 — Validate the Manifest

# In[9]:


import json

MANIFEST_PATH = REQUIRED_PACKAGE_FILES["manifest"]

with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest = json.load(f)

print("Project:", manifest["project"])
print("Team:", manifest["team"])
print("Repository cards:", manifest["repository_cards"])
print("Official cards:", manifest["official_cards"])
print("Deck size:", manifest["deck_size"])

assert manifest["project"] == "PTCG AI Battle Challenge"
assert manifest["team"] == "Team Jesus"
assert manifest["repository_cards"] == 1267
assert manifest["official_cards"] == 1267
assert manifest["deck_size"] == 60

print("\nManifest validated.")


# # Cell 8 — Verify SHA256 checksums

# In[11]:


# Cell 8 — Verify SHA256 checksums

def sha256_file(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


checksum_results = {}

for logical_name, details in manifest["files"].items():
    relative_path = details["relative_path"]

    packaged_path = (
        EXTRACT_DIR
        / Path(relative_path)
    )

    expected_checksum = details["sha256"]
    actual_checksum = sha256_file(packaged_path)

    passed = actual_checksum == expected_checksum

    checksum_results[logical_name] = passed

    print(
        f"{'[OK]' if passed else '[FAIL]'} "
        f"{logical_name}"
    )
    print(" expected:", expected_checksum)
    print(" actual:  ", actual_checksum)
    print()

assert all(checksum_results.values())

print("All packaged checksums validated.")


# # Cell 9 — Compile every packaged Python file

# In[12]:


python_files = sorted(
    EXTRACT_DIR.rglob("*.py")
)

print("Python files found:", len(python_files))

compile_results = {}

for python_file in python_files:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            str(python_file),
        ],
        capture_output=True,
        text=True,
    )

    passed = completed.returncode == 0
    compile_results[str(python_file)] = passed

    print(
        f"{'[OK]' if passed else '[FAIL]'} "
        f"{python_file.relative_to(EXTRACT_DIR)}"
    )

    if not passed:
        print(completed.stderr)

assert python_files
assert all(compile_results.values())

print("\nAll packaged Python files compiled successfully.")


# # Cell 10 — Validate deck contents

# In[13]:


deck_path = REQUIRED_PACKAGE_FILES["deck"]

deck_ids = tuple(
    int(line.strip())
    for line in deck_path.read_text(
        encoding="utf-8-sig"
    ).splitlines()
    if line.strip()
)

print("Deck size:", len(deck_ids))
print("Unique card IDs:", len(set(deck_ids)))
print("First 10 cards:", deck_ids[:10])

assert len(deck_ids) == 60
assert len(set(deck_ids)) > 1
assert all(card_id > 0 for card_id in deck_ids)

print("\nPackaged deck validated.")


# # Cell 11 — Check for unexpected files

# In[15]:


expected_files = {
    "manifest.json",
    "data/deck.csv",
    "agent/notebook21_export.py",
    "evaluation/notebook22_export.py",
    "policy/notebook20_export.py",
}

actual_files = {
    str(path.relative_to(EXTRACT_DIR)).replace("\\", "/")
    for path in EXTRACT_DIR.rglob("*")
    if path.is_file()
}

# Ignore Python cache files
actual_files = {
    f
    for f in actual_files
    if "__pycache__" not in f
    and not f.endswith(".pyc")
}

unexpected_files = actual_files - expected_files
missing_expected = expected_files - actual_files

print("Actual files:")

for name in sorted(actual_files):
    print("-", name)

print()
print("Unexpected files:", sorted(unexpected_files))
print("Missing expected files:", sorted(missing_expected))

assert not unexpected_files
assert not missing_expected

print("\nPackage file inventory validated.")


# # Cell 12 - Load the packaged agent report

# In[16]:


AGENT_EXPORT = REQUIRED_PACKAGE_FILES["agent"]

source_21 = AGENT_EXPORT.read_text(
    encoding="utf-8-sig"
)

cleaned_lines = [
    line
    for line in source_21.splitlines()
    if line.strip() != "from __future__ import annotations"
]

cleaned_source = (
    "from __future__ import annotations\n"
    + "\n".join(cleaned_lines)
)

module_name = (
    "packaged_notebook21_"
    + uuid.uuid4().hex
)

packaged_agent_module = types.ModuleType(
    module_name
)

packaged_agent_module.__file__ = str(
    AGENT_EXPORT
)

packaged_agent_module.__package__ = ""

sys.modules[module_name] = packaged_agent_module

compiled_agent = compile(
    cleaned_source,
    str(AGENT_EXPORT),
    "exec",
)

exec(
    compiled_agent,
    packaged_agent_module.__dict__,
)

print("Packaged agent module loaded.")


# # Cell 13 — Verify packaged agent objects

# In[17]:


required_agent_objects = [
    "KaggleBattleAgent",
    "production_agent",
    "kaggle_agent",
    "deck_ids",
    "sample_snapshot",
]

missing_agent_objects = [
    name
    for name in required_agent_objects
    if not hasattr(packaged_agent_module, name)
]

for name in required_agent_objects:
    print(
        f"{'[OK]' if hasattr(packaged_agent_module, name) else '[MISSING]'} "
        f"{name}"
    )

if missing_agent_objects:
    raise AttributeError(
        "Packaged agent module is missing:\n"
        + "\n".join(missing_agent_objects)
    )

print("\nPackaged agent interface verified.")


# # Cell 14 — Smoke-test deck mode

# In[18]:


class DeckRequest:
    select = None


packaged_kaggle_agent = (
    packaged_agent_module.kaggle_agent
)

packaged_deck_response = (
    packaged_kaggle_agent.choose(
        DeckRequest()
    )
)

print(
    "Returned deck size:",
    len(packaged_deck_response),
)

assert packaged_deck_response == list(
    packaged_agent_module.deck_ids
)

assert len(packaged_deck_response) == 60

print("\nPackaged deck mode passed.")


# # Cell 15 — Smoke-test action mode

# In[20]:


packaged_snapshot = (
    packaged_agent_module.sample_snapshot
)

packaged_action = (
    packaged_kaggle_agent.choose(
        packaged_snapshot
    )
)

print("Returned action:", packaged_action)

assert packaged_action == [0]

print("\nPackaged action mode passed.")


# # Cell 16 — Final Validation Summary

# In[25]:


# Cell 16 — Final validation summary

checksum_count = sum(
    1
    for details in manifest["files"].values()
    if details.get("sha256")
)

python_files = sorted(
    path
    for path in EXTRACT_DIR.rglob("*.py")
    if path.is_file()
)

print("=" * 72)
print("Notebook 24 — Submission Verification")
print("=" * 72)
print()

print(
    "Repository cards:",
    manifest["repository_cards"],
)
print(
    "Official cards:",
    manifest["official_cards"],
)
print(
    "Deck size:",
    manifest["deck_size"],
)

print()

print(
    "Package files:",
    len(manifest["files"]),
)
print(
    "Checksums:",
    checksum_count,
)
print(
    "Python modules:",
    len(python_files),
)

print()

print("Packaged deck: PASS")
print("Packaged action: PASS")
print("Archive integrity: PASS")
print("Manifest validation: PASS")
print("Checksum validation: PASS")

print()

print("Submission ZIP:", ZIP_FILE.name)

assert manifest["repository_cards"] == 1267
assert manifest["official_cards"] == 1267
assert manifest["deck_size"] == 60
assert len(manifest["files"]) == 4
assert checksum_count == 4
assert len(python_files) == 3
assert len(packaged_deck_response) == 60
assert packaged_action == [0]

print()
print("NOTEBOOK 24 COMPLETED SUCCESSFULLY")
print("Submission package fully verified.")


# In[ ]:




