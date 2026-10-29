"""
Project Cleanup Script

Organizes messy root directory into proper folders.
Moves test files, logs, scripts, and legacy files.

Usage:
    python scripts/cleanup_project.py --dry-run  # See what would happen
    python scripts/cleanup_project.py            # Actually move files
"""

import argparse
import io
import shutil
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (AttributeError, io.UnsupportedOperation):
        pass

# Base directory
BASE_DIR = Path(__file__).parent.parent

# File organization rules
RULES = {
    # Test files → tests/
    'tests/': [
        'test_*.py',
        'check_*.py',
        'diagnostic_*.py',
        'smoke_checks.py',
    ],

    # Log files → logs/
    'logs/': [
        '*.log',
        'FINAL SUMMARY.txt',
        'nul',
    ],

    # Scripts → scripts/
    'scripts/': [
        'scrape_url.py',
        'quick_test_analysis.py',
        'validate_results.py',
        'validate.py',
        'setup_and_run.ps1',
        'main.py',
    ],

    # Legacy/archive files
    'archive/': [
        'run_pipeline_cpu.py',
        'pipeline*.log',
        'Objective 2 - Multi-Platform Review .*',
        'streamlit_app.py',
        'test_models.py',
        'test_multi_model.py',
        'test_ollama.py',
        'test_analysis.py',
        'test_sentiment.py',
    ],

    # Keep in root (do not move)
    'ROOT': [
        'requirements.txt',
        'run_pipeline.py',
        'run_pipeline_with_audit.py',
        'START_HERE.md',
        'QUICK_START.md',
        'README.md',
        '.gitignore',
        '.git',
        '.venv',
        '.claude',
    ],
}


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)


def find_files_to_move():
    """Find all files that need to be moved."""
    moves = {}  # destination → [files]

    for dest_dir, patterns in RULES.items():
        if dest_dir == 'ROOT':
            continue  # Skip files to keep

        moves[dest_dir] = []

        for pattern in patterns:
            # Find matching files in root
            matching = list(BASE_DIR.glob(pattern))
            for file in matching:
                if file.is_file():
                    # Check if it's not in keep list
                    should_keep = False
                    for keep_pattern in RULES['ROOT']:
                        if file.match(keep_pattern):
                            should_keep = True
                            break

                    if not should_keep:
                        moves[dest_dir].append(file)

    return moves


def move_files(moves, dry_run=False):
    """Move files to their destinations."""
    total_moved = 0

    for dest_dir, files in moves.items():
        if not files:
            continue

        dest_path = BASE_DIR / dest_dir

        print(f"\n📁 {dest_dir}")
        print("-" * 80)

        for file in files:
            target = dest_path / file.name

            if dry_run:
                print(f"   [DRY RUN] Would move: {file.name} → {dest_dir}")
            else:
                try:
                    # Create destination directory
                    dest_path.mkdir(parents=True, exist_ok=True)

                    # Move file
                    shutil.move(str(file), str(target))
                    print(f"   ✅ Moved: {file.name} → {dest_dir}")
                    total_moved += 1
                except Exception as e:
                    print(f"   ❌ Failed to move {file.name}: {e}")

    return total_moved


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Clean up project directory")
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be moved without actually moving'
    )

    args = parser.parse_args()

    print_header("PROJECT CLEANUP SCRIPT")

    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No files will be moved")

    # Find files to move
    print("\n🔍 Scanning root directory for files to organize...")
    moves = find_files_to_move()

    # Count total files
    total_files = sum(len(files) for files in moves.values())

    if total_files == 0:
        print("\n✅ Root directory is already clean!")
        print("   No files need to be moved.")
        return

    print(f"\n📊 Found {total_files} files to organize:")
    for dest_dir, files in moves.items():
        if files:
            print(f"   - {len(files)} files → {dest_dir}")

    # Move files
    if args.dry_run:
        print("\n" + "=" * 80)
        print(" DRY RUN - Showing what would happen")
        print("=" * 80)
        move_files(moves, dry_run=True)
        print("\n" + "=" * 80)
        print(" To actually move files, run without --dry-run:")
        print("   python scripts/cleanup_project.py")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print(" MOVING FILES")
        print("=" * 80)

        moved = move_files(moves, dry_run=False)

        print("\n" + "=" * 80)
        print(" CLEANUP COMPLETE")
        print("=" * 80)
        print(f"\n✅ Moved {moved} files")
        print("\n📁 Final structure:")
        print("   - tests/      : Test files")
        print("   - logs/       : Log files")
        print("   - scripts/    : Utility scripts")
        print("   - archive/    : Legacy files")
        print("   - ROOT        : Essential files only")

        print("\n📋 Files kept in root:")
        for pattern in RULES['ROOT']:
            matching = list(BASE_DIR.glob(pattern))
            for file in matching:
                if file.is_file():
                    print(f"   - {file.name}")


if __name__ == "__main__":
    main()
