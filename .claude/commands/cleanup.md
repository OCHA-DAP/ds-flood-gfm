# Repo Cleanup Review

Analyze the current repository state and identify technical debt. Focus on:

## 1. Categorize Untracked Files
Review `git status` and group files into:
- **One-off debugging/testing scripts** (test_*.py, debug_*.py, check_*.py, quick_*.py, minimal_*.py, simple_*.py)
- **Experimental code** (in experiments/ or scattered elsewhere)
- **Potentially useful utilities** (scripts that might belong in scripts/ or src/)
- **Documentation** (.md files, especially duplicates or drafts)
- **Generated outputs** (.png, .pdf, .html files)
- **Config examples** (.env.example, etc.)

## 2. Suggest Actions
For each category, recommend:
- **ARCHIVE** - Move to to_delete/ folder for review (temporary/debugging files)
- **MOVE** - Relocate useful scripts to proper directories (scripts/, src/, etc.)
- **REVIEW** - Files that might be worth keeping but need user decision - makes suggestions on files to commit
- **IGNORE** - Files that should be added to .gitignore

## 3. Check for Duplication
- Identify files with similar names or purposes (e.g., multiple "check" or "test" scripts)
- Flag potential consolidation opportunities

## 4. Directory Structure Health
- Count how many untracked files are at repo root vs proper subdirectories
- Warn if experiments/ is getting bloated
- Suggest organization improvements

## Output Format
Present as a clean, actionable checklist organized by category. Be concise. Only include high-confidence recommendations. Let the user make final decisions on borderline cases.

**Note:** The to_delete/ folder should be in .gitignore as a staging area for files to be deleted later.
