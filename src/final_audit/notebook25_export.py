#!/usr/bin/env python
# coding: utf-8

# # Notebook 25 — Final Project Audit and Submission Readiness
# 
# ## The Pokémon Company — PTCG AI Battle Challenge
# 
# ### Team Jesus
# 
# Notebook 24 verified the packaged submission artifact.
# 
# Notebook 25 performs the final project-wide audit and produces a formal submission-readiness report.
# 
# ## Objectives
# 
# 1. Verify all production notebooks and exports.
# 2. Verify all reusable source modules.
# 3. Verify completion reports.
# 4. Validate the final submission ZIP.
# 5. Confirm repository, card lookup, and deck counts.
# 6. Check for missing production files.
# 7. Check Python syntax across final exports.
# 8. Confirm the final agent entry point.
# 9. Confirm zero errors and zero fallbacks.
# 10. Generate a final readiness checklist.
# 11. Save a machine-readable audit report.
# 12. Produce the final “READY TO SUBMIT” status.

# # Cell 2 - Imports

# In[1]:


from __future__ import annotations

import json
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

    required = {
        "notebooks",
        "scripts",
        "src",
        "reports",
        "submission_work",
    }

    for candidate in [current, *current.parents]:
        found = {
            name
            for name in required
            if (candidate / name).exists()
        }

        if len(found) >= 4:
            return candidate

    return current


PROJECT_ROOT = find_project_root()

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "notebook25"
)

FINAL_ZIP = (
    PROJECT_ROOT
    / "submission_work"
    / "team_jesus_submission.zip"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("Project root:", PROJECT_ROOT)
print("Final ZIP:", FINAL_ZIP)
print("Report directory:", REPORT_DIR)


# # Cell 4 — Define final required artifacts

# In[3]:


REQUIRED_ARTIFACTS = {
    "notebook20": (
        PROJECT_ROOT
        / "notebooks"
        / "20_policy_engine.ipynb"
    ),
    "notebook21": (
        PROJECT_ROOT
        / "notebooks"
        / "21_kaggle_battle_agent.ipynb"
    ),
    "notebook22": (
        PROJECT_ROOT
        / "notebooks"
        / "22_end_to_end_agent_integration.ipynb"
    ),
    "notebook23": (
        PROJECT_ROOT
        / "notebooks"
        / "23_submission_package_builder.ipynb"
    ),
    "notebook24": (
        PROJECT_ROOT
        / "notebooks"
        / "24_final_submission_validation.ipynb"
    ),
    "script20": (
        PROJECT_ROOT
        / "scripts"
        / "20_policy_engine.py"
    ),
    "script21": (
        PROJECT_ROOT
        / "scripts"
        / "21_kaggle_battle_agent.py"
    ),
    "script22": (
        PROJECT_ROOT
        / "scripts"
        / "22_end_to_end_agent_integration.py"
    ),
    "script23": (
        PROJECT_ROOT
        / "scripts"
        / "23_submission_package_builder.py"
    ),
    "script24": (
        PROJECT_ROOT
        / "scripts"
        / "24_final_submission_validation.py"
    ),
    "submission_zip": FINAL_ZIP,
}


# # Cell 5 — Check required artifacts

# In[4]:


missing_artifacts = []

for name, path in REQUIRED_ARTIFACTS.items():
    exists = path.is_file()

    print(
        f"{'[FOUND]' if exists else '[MISSING]'} "
        f"{name}: {path}"
    )

    if not exists:
        missing_artifacts.append(str(path))

if missing_artifacts:
    raise FileNotFoundError(
        "Final audit found missing artifacts:\n"
        + "\n".join(missing_artifacts)
    )

print("\nAll final required artifacts located.")


# # Cell 6 — Verify reusable source exports

# In[5]:


REQUIRED_SOURCE_EXPORTS = {
    "policy_engine": (
        PROJECT_ROOT
        / "src"
        / "policy_engine"
        / "notebook20_export.py"
    ),
    "kaggle_agent": (
        PROJECT_ROOT
        / "src"
        / "kaggle_agent"
        / "notebook21_export.py"
    ),
    "evaluation_harness": (
        PROJECT_ROOT
        / "src"
        / "evaluation_harness"
        / "notebook22_export.py"
    ),
    "submission_builder": (
        PROJECT_ROOT
        / "src"
        / "submission_builder"
        / "notebook23_export.py"
    ),
    "verification": (
        PROJECT_ROOT
        / "src"
        / "verification"
        / "notebook24_export.py"
    ),
}

missing_source_exports = []

for name, path in REQUIRED_SOURCE_EXPORTS.items():
    exists = path.is_file()

    print(
        f"{'[FOUND]' if exists else '[MISSING]'} "
        f"{name}: {path}"
    )

    if not exists:
        missing_source_exports.append(str(path))

if missing_source_exports:
    raise FileNotFoundError(
        "Missing reusable source exports:\n"
        + "\n".join(missing_source_exports)
    )

print("\nAll reusable source exports located.")


# # Cell 7 — Verify completion reports

# In[6]:


REQUIRED_COMPLETION_REPORTS = {
    notebook_number: (
        PROJECT_ROOT
        / "reports"
        / f"notebook{notebook_number}"
        / "completion_status.txt"
    )
    for notebook_number in range(20, 25)
}

missing_reports = []

for number, path in REQUIRED_COMPLETION_REPORTS.items():
    exists = path.is_file()

    print(
        f"{'[FOUND]' if exists else '[MISSING]'} "
        f"Notebook {number}: {path}"
    )

    if not exists:
        missing_reports.append(str(path))

if missing_reports:
    raise FileNotFoundError(
        "Missing completion reports:\n"
        + "\n".join(missing_reports)
    )

print("\nAll completion reports located.")


# # Cell 8 — Compile final production scripts

# In[7]:


compile_targets = [
    *[
        path
        for name, path in REQUIRED_ARTIFACTS.items()
        if name.startswith("script")
    ],
    *REQUIRED_SOURCE_EXPORTS.values(),
]

compile_results = {}

for path in compile_targets:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            str(path),
        ],
        capture_output=True,
        text=True,
    )

    passed = completed.returncode == 0
    compile_results[str(path)] = passed

    print(
        f"{'[OK]' if passed else '[FAIL]'} "
        f"{path.relative_to(PROJECT_ROOT)}"
    )

    if not passed:
        print(completed.stderr)

assert all(compile_results.values())

print("\nAll final production scripts compiled successfully.")


# # Cell 9 — Validate the final submission ZIP

# In[8]:


with zipfile.ZipFile(FINAL_ZIP) as archive:
    zip_names = {
        name
        for name in archive.namelist()
        if not name.endswith("/")
    }

expected_zip_files = {
    "manifest.json",
    "data/deck.csv",
    "agent/notebook21_export.py",
    "evaluation/notebook22_export.py",
    "policy/notebook20_export.py",
}

missing_zip_files = expected_zip_files - zip_names
unexpected_zip_files = zip_names - expected_zip_files

print("ZIP files:")

for name in sorted(zip_names):
    print("-", name)

print()
print("Missing:", sorted(missing_zip_files))
print("Unexpected:", sorted(unexpected_zip_files))

assert not missing_zip_files
assert not unexpected_zip_files

print("\nFinal submission ZIP inventory passed.")


# # Cell 10 — Read and validate the manifest

# In[9]:


with zipfile.ZipFile(FINAL_ZIP) as archive:
    manifest_data = json.loads(
        archive.read("manifest.json").decode("utf-8")
    )

print("Project:", manifest_data["project"])
print("Team:", manifest_data["team"])
print("Repository cards:", manifest_data["repository_cards"])
print("Official cards:", manifest_data["official_cards"])
print("Deck size:", manifest_data["deck_size"])

assert manifest_data["project"] == "PTCG AI Battle Challenge"
assert manifest_data["team"] == "Team Jesus"
assert manifest_data["repository_cards"] == 1267
assert manifest_data["official_cards"] == 1267
assert manifest_data["deck_size"] == 60
assert len(manifest_data["files"]) == 4

print("\nFinal manifest validation passed.")


# # Cell 11 — Create the final audit checklist

# In[10]:


final_checks = {
    "required_artifacts": not missing_artifacts,
    "source_exports": not missing_source_exports,
    "completion_reports": not missing_reports,
    "production_compilation": all(compile_results.values()),
    "submission_zip_exists": FINAL_ZIP.is_file(),
    "zip_inventory": not missing_zip_files and not unexpected_zip_files,
    "manifest_project": (
        manifest_data["project"]
        == "PTCG AI Battle Challenge"
    ),
    "manifest_team": (
        manifest_data["team"]
        == "Team Jesus"
    ),
    "repository_cards": (
        manifest_data["repository_cards"]
        == 1267
    ),
    "official_cards": (
        manifest_data["official_cards"]
        == 1267
    ),
    "deck_size": (
        manifest_data["deck_size"]
        == 60
    ),
}

for name, passed in final_checks.items():
    print(
        f"{'[OK]' if passed else '[FAIL]'} "
        f"{name}"
    )

assert all(final_checks.values())

print("\nFinal project checklist passed.")


# # Cell 12 — Save the machine-readable audit report

# In[11]:


audit_report = {
    "project": "PTCG AI Battle Challenge",
    "team": "Team Jesus",
    "status": "READY TO SUBMIT",
    "checks": final_checks,
    "submission_zip": str(FINAL_ZIP),
    "repository_cards": manifest_data["repository_cards"],
    "official_cards": manifest_data["official_cards"],
    "deck_size": manifest_data["deck_size"],
    "compiled_files": len(compile_results),
}

AUDIT_JSON = REPORT_DIR / "final_audit.json"

AUDIT_JSON.write_text(
    json.dumps(
        audit_report,
        indent=4,
    ),
    encoding="utf-8",
)

print("Audit report:", AUDIT_JSON)

assert AUDIT_JSON.is_file()

print("\nMachine-readable audit report saved.")


# # Cell 13 — Final summary

# In[12]:


print("=" * 72)
print("Notebook 25 — Final Project Audit")
print("=" * 72)
print()

print("Project:", audit_report["project"])
print("Team:", audit_report["team"])
print("Repository cards:", audit_report["repository_cards"])
print("Official cards:", audit_report["official_cards"])
print("Deck size:", audit_report["deck_size"])
print("Compiled files:", audit_report["compiled_files"])

print()
print("Submission ZIP:", FINAL_ZIP.name)
print("Audit report:", AUDIT_JSON.name)

print()
print("FINAL PROJECT AUDIT PASSED")
print("STATUS: READY TO SUBMIT")
print()
print("NOTEBOOK 25 COMPLETED SUCCESSFULLY")


# In[ ]:




