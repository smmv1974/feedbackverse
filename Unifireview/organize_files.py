"""Organize project files for university submission."""
import shutil
from pathlib import Path

# Get project root
root = Path(__file__).parent

# Create directory structure
(root / "docs" / "markdown").mkdir(parents=True, exist_ok=True)

# Files to move to docs/markdown
md_files_to_move = [
    "CHANGELOG.md",
    "COMPLETE_PROJECT_JOURNEY.md",
    "FINAL_ANSWERS_ALL_QUESTIONS.md",
    "QUICKSTART.md",
    "REORGANIZATION_COMPLETE.md",
    "SUBMISSION_GUIDE.md",
    "SUBMISSION_VERIFICATION.md",
    "WORD_DOC_UPDATE_GUIDE.md",
]

print("Moving markdown files to docs/markdown...")
for filename in md_files_to_move:
    src = root / filename
    if src.exists():
        dst = root / "docs" / "markdown" / filename
        shutil.move(str(src), str(dst))
        print(f"  + Moved {filename}")
    else:
        print(f"  - Skipped {filename} (not found)")

print("\n✅ File organization complete!")
print("\nDirectory structure:")
print(f"  docs/markdown/        - All markdown documentation")
print(f"  Doc_submission_final/ - University submission package")
print(f"  archive/             - Old session notes")
