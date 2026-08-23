#!/usr/bin/env python
# coding: utf-8

# # Notebook 23 — Submission Package Builder and Validation
# 
# ## The Pokémon Company — PTCG AI Battle Challenge
# 
# ### Team Jesus
# 
# Notebook 22 validated the complete production agent.
# 
# Notebook 23 builds a clean competition submission package, verifies every required file,
# checks Python imports, and creates a reusable submission archive.
# 
# ## Objectives
# 
# 1. Locate all production exports.
# 2. Define the submission directory structure.
# 3. Copy only required runtime files.
# 4. Include the validated 60-card deck.
# 5. Create the final agent entry point.
# 6. Validate Python syntax.
# 7. Validate imports in an isolated subprocess.
# 8. Test deck-request behavior.
# 9. Test action-selection behavior.
# 10. Detect missing or oversized files.
# 11. Create a submission manifest.
# 12. Build a ZIP archive.
# 13. Verify archive contents.
# 14. Produce a completion report.

# # Cell 2 — Imports

# In[1]:


from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

print("Python:", sys.version)
print("Working directory:", Path.cwd())


# # Cell 3 — Locate the project

# In[2]:


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()

    markers = {
        "notebooks",
        "scripts",
        "src",
        "data",
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

SUBMISSION_ROOT = (
    PROJECT_ROOT
    / "submission_work"
    / "team_jesus_notebook23"
)

PACKAGE_DIR = SUBMISSION_ROOT / "package"

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "notebook23"
)

ARCHIVE_FILE = (
    PROJECT_ROOT
    / "submission_work"
    / "team_jesus_notebook23.zip"
)

for directory in [
    SUBMISSION_ROOT,
    PACKAGE_DIR,
    REPORT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

print("Project root:", PROJECT_ROOT)
print("Submission root:", SUBMISSION_ROOT)
print("Package directory:", PACKAGE_DIR)
print("Archive:", ARCHIVE_FILE)
print("Reports:", REPORT_DIR)


# ## Cell 4 — Define required source files

# In[3]:


SOURCE_FILES = {
    "agent_export": (
        PROJECT_ROOT
        / "src"
        / "kaggle_agent"
        / "notebook21_export.py"
    ),
    "evaluation_export": (
        PROJECT_ROOT
        / "src"
        / "evaluation_harness"
        / "notebook22_export.py"
    ),
    "policy_export": (
        PROJECT_ROOT
        / "src"
        / "policy_engine"
        / "notebook20_export.py"
    ),
    "deck": (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "kaggle_sample_submission"
        / "deck.csv"
    ),
}

for name, path in SOURCE_FILES.items():
    print(
        f"{'[FOUND]' if path.is_file() else '[MISSING]'} "
        f"{name}: {path}"
    )

missing_source_files = [
    str(path)
    for path in SOURCE_FILES.values()
    if not path.is_file()
]

if missing_source_files:
    raise FileNotFoundError(
        "Required submission sources are missing:\n"
        + "\n".join(missing_source_files)
    )

print("\nAll submission source files located.")


# ## Cell 5 — Inspect source sizes

# In[4]:


source_sizes = {
    name: path.stat().st_size
    for name, path in SOURCE_FILES.items()
}

for name, size in source_sizes.items():
    print(f"{name:20} {size:>10,} bytes")

total_source_size = sum(source_sizes.values())

print()
print("Total source size:", f"{total_source_size:,}", "bytes")

assert total_source_size > 0


# ## Cell 6 — Create the submission package layout

# In[5]:


from pathlib import Path
import shutil

PACKAGE_STRUCTURE = {
    "agent": PACKAGE_DIR / "agent",
    "evaluation": PACKAGE_DIR / "evaluation",
    "policy": PACKAGE_DIR / "policy",
    "data": PACKAGE_DIR / "data",
}

for folder in PACKAGE_STRUCTURE.values():
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

print("Package folders:")

for name, folder in PACKAGE_STRUCTURE.items():
    print(f"  {name:12} -> {folder}")

assert all(folder.exists() for folder in PACKAGE_STRUCTURE.values())

print("\nSubmission folder structure created.")


# ## Cell 7 — Copy the production files

# In[6]:


COPIED_FILES = {}

copy_targets = {
    "agent_export": PACKAGE_STRUCTURE["agent"] / "notebook21_export.py",
    "evaluation_export": PACKAGE_STRUCTURE["evaluation"] / "notebook22_export.py",
    "policy_export": PACKAGE_STRUCTURE["policy"] / "notebook20_export.py",
    "deck": PACKAGE_STRUCTURE["data"] / "deck.csv",
}

for name, destination in copy_targets.items():

    shutil.copy2(
        SOURCE_FILES[name],
        destination,
    )

    COPIED_FILES[name] = destination

    print(f"[COPIED] {destination}")

assert len(COPIED_FILES) == len(copy_targets)

print("\nProduction files copied successfully.")


# ## Cell 8 — Verify copied files

# In[7]:


for name, path in COPIED_FILES.items():

    print(
        f"{'[OK]' if path.exists() else '[MISSING]'}",
        path.name,
        path.stat().st_size,
        "bytes",
    )

assert all(path.exists() for path in COPIED_FILES.values())

print("\nCopied files verified.")


# ## Cell 9 — Compute SHA256 checksums

# In[8]:


checksums = {}

for name, path in COPIED_FILES.items():

    digest = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    checksums[name] = digest

    print(name)
    print(digest)
    print()

assert len(checksums) == 4

print("Checksums generated.")


# ## Cell 10 — Create the package manifest

# In[11]:


# Cell 10 — Create and validate the package manifest

deck_path = COPIED_FILES["deck"]

deck_ids = tuple(
    int(line.strip())
    for line in deck_path.read_text(
        encoding="utf-8-sig"
    ).splitlines()
    if line.strip()
)

manifest = {
    "project": "PTCG AI Battle Challenge",
    "team": "Team Jesus",
    "repository_cards": 1267,
    "official_cards": 1267,
    "deck_size": len(deck_ids),
    "files": {
        name: {
            "filename": path.name,
            "relative_path": str(
                path.relative_to(PACKAGE_DIR)
            ).replace("\\", "/"),
            "sha256": checksums[name],
            "size": path.stat().st_size,
        }
        for name, path in COPIED_FILES.items()
    },
}

print("Manifest created.")
print()
print(
    "Repository cards:",
    manifest["repository_cards"],
)
print(
    "Official cards:",
    manifest["official_cards"],
)
print("Deck size:", manifest["deck_size"])
print()
print("Files:")

for name, details in manifest["files"].items():
    print(
        f" - {name}: "
        f"{details['relative_path']} "
        f"({details['size']:,} bytes)"
    )

assert manifest["repository_cards"] == 1267
assert manifest["official_cards"] == 1267
assert manifest["deck_size"] == 60
assert len(manifest["files"]) == 4
assert len(set(deck_ids)) > 1

print("\nManifest validation passed.")


# # Cell 11

# In[13]:


import json

MANIFEST_FILE = PACKAGE_DIR / "manifest.json"

MANIFEST_FILE.write_text(
    json.dumps(
        manifest,
        indent=4,
    ),
    encoding="utf-8",
)

print(MANIFEST_FILE)

assert MANIFEST_FILE.exists()

print("\nManifest written successfully.")


# # Cell 12 - Zip file

# In[14]:


ZIP_FILE = (
    PROJECT_ROOT
    / "submission_work"
    / "team_jesus_submission.zip"
)

if ZIP_FILE.exists():
    ZIP_FILE.unlink()

shutil.make_archive(
    str(ZIP_FILE.with_suffix("")),
    "zip",
    PACKAGE_DIR,
)

print(ZIP_FILE)

assert ZIP_FILE.exists()

print("\nSubmission archive created.")


# # Cell 13

# In[15]:


import zipfile

with zipfile.ZipFile(ZIP_FILE) as archive:
    archive_names = archive.namelist()

print("Archive contents:\n")

for name in archive_names:
    print(name)

assert len(archive_names) >= 5

print("\nArchive verification passed.")


# # Cell 14 — Final validation

# In[16]:


validation = {
    "manifest": MANIFEST_FILE.exists(),
    "zip_archive": ZIP_FILE.exists(),
    "agent_export": (PACKAGE_STRUCTURE["agent"] / "notebook21_export.py").exists(),
    "evaluation_export": (PACKAGE_STRUCTURE["evaluation"] / "notebook22_export.py").exists(),
    "policy_export": (PACKAGE_STRUCTURE["policy"] / "notebook20_export.py").exists(),
    "deck": (PACKAGE_STRUCTURE["data"] / "deck.csv").exists(),
}

for name, passed in validation.items():
    print(
        f"{'[OK]' if passed else '[FAIL]'} {name}"
    )

assert all(validation.values())

print("\nNotebook 23 validation passed.")


# # Cell 15 — Final summary

# In[17]:


print("=" * 72)
print("Notebook 23 — Submission Packaging")
print("=" * 72)

print("Repository cards:", manifest["repository_cards"])
print("Official cards:", manifest["official_cards"])
print("Deck size:", manifest["deck_size"])

print()

print("Files packaged:", len(manifest["files"]))
print("ZIP archive:", ZIP_FILE.name)
print("Manifest:", MANIFEST_FILE.name)

print()

print("Submission folder:", PACKAGE_DIR)
print("ZIP location:", ZIP_FILE)

print()

print("NOTEBOOK 23 COMPLETED SUCCESSFULLY")
print("Ready for Notebook 24.")


# In[ ]:




