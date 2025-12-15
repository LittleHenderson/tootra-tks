#!/usr/bin/env python3
"""
Generate release notes from CHANGELOG.md for a specific version.

Usage:
    python scripts/generate_release_notes.py --version 0.2.2
    python scripts/generate_release_notes.py --version v0.2.2
    python scripts/generate_release_notes.py --latest

Canon Guardrails (for reference):
- Operators: +, -, +T, -T, ->, <-, *T, /T, o (ALLOWED_OPS=9)
- Worlds: A/B/C/D
- Noetics: 1-10 (involution pairs: 2<->3, 5<->6, 8<->9; self-duals: 1,4,7,10)
- Foundations: 1-7
- Sub-foundations: 7x4=28 combinations
"""

import argparse
import re
import sys
from pathlib import Path


def extract_version_notes(changelog_path: Path, version: str) -> str:
    """
    Extract release notes for a specific version from CHANGELOG.md.

    Args:
        changelog_path: Path to CHANGELOG.md
        version: Version string (with or without 'v' prefix)

    Returns:
        Extracted release notes as string
    """
    # Normalize version (remove 'v' prefix if present)
    version_num = version.lstrip('v')

    if not changelog_path.exists():
        return f"## Release v{version_num}\n\nNo CHANGELOG.md found."

    content = changelog_path.read_text(encoding='utf-8')
    lines = content.split('\n')

    # Find the start of the version section
    start_idx = None
    end_idx = None
    version_pattern = re.compile(rf'##\s*\[v?{re.escape(version_num)}\]')
    next_version_pattern = re.compile(r'##\s*\[v?\d+\.\d+')

    for i, line in enumerate(lines):
        if start_idx is None and version_pattern.match(line):
            start_idx = i
        elif start_idx is not None and i > start_idx and next_version_pattern.match(line):
            end_idx = i
            break

    if start_idx is None:
        return f"## Release v{version_num}\n\nVersion not found in CHANGELOG.md."

    # Extract the section
    if end_idx is None:
        section = lines[start_idx:]
    else:
        section = lines[start_idx:end_idx]

    return '\n'.join(section).strip()


def get_latest_version(changelog_path: Path) -> str:
    """
    Get the latest version from CHANGELOG.md.

    Args:
        changelog_path: Path to CHANGELOG.md

    Returns:
        Latest version string
    """
    if not changelog_path.exists():
        return "0.0.0"

    content = changelog_path.read_text(encoding='utf-8')

    # Match version headers like ## [0.2.2] or ## [v0.2.2]
    version_pattern = re.compile(r'##\s*\[v?(\d+\.\d+\.\d+)\]')
    versions = version_pattern.findall(content)

    if not versions:
        return "0.0.0"

    # Return the first (most recent) version found
    return versions[0]


def main():
    parser = argparse.ArgumentParser(
        description='Generate release notes from CHANGELOG.md'
    )
    parser.add_argument(
        '--version', '-v',
        help='Version to extract notes for (e.g., 0.2.2 or v0.2.2)'
    )
    parser.add_argument(
        '--latest', '-l',
        action='store_true',
        help='Extract notes for the latest version'
    )
    parser.add_argument(
        '--changelog', '-c',
        default='CHANGELOG.md',
        help='Path to CHANGELOG.md (default: CHANGELOG.md)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output file path (default: stdout)'
    )

    args = parser.parse_args()

    # Resolve changelog path
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    changelog_path = project_root / args.changelog

    if not changelog_path.exists():
        changelog_path = Path(args.changelog)

    # Determine version
    if args.latest:
        version = get_latest_version(changelog_path)
        print(f"Latest version: {version}", file=sys.stderr)
    elif args.version:
        version = args.version
    else:
        print("Error: Specify --version or --latest", file=sys.stderr)
        sys.exit(1)

    # Extract notes
    notes = extract_version_notes(changelog_path, version)

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(notes, encoding='utf-8')
        print(f"Release notes written to: {output_path}", file=sys.stderr)
    else:
        # Handle potential encoding issues on Windows console
        try:
            print(notes)
        except UnicodeEncodeError:
            # Fall back to ASCII with replacements for non-ASCII chars
            print(notes.encode('ascii', 'replace').decode('ascii'))


if __name__ == '__main__':
    main()
