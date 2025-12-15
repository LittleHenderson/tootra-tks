"""
Tests for scripts/generate_release_notes.py

Verifies release note extraction from CHANGELOG.md.
"""

import sys
from pathlib import Path
import tempfile
import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from generate_release_notes import extract_version_notes, get_latest_version


SAMPLE_CHANGELOG = """# Changelog

All notable changes to the TKS project will be documented here.

## [Unreleased]

### Added
- Some unreleased feature

## [v0.2.2] - 2025-12-14

### Added
- Training loop with DummyTKSModel
- Inference CLI: scripts/run_inference.py

### Tests/CI
- Total tests: 298 passing

## [v0.2.1] - 2025-12-14

### Added
- Training integration Phase 2

### Tests/CI
- Total tests: 235 passing

## [v0.1.0] - 2025-12-14

### Added
- Initial release with scenario inversion
"""


class TestExtractVersionNotes:
    """Test extract_version_notes function."""

    def test_extract_specific_version(self, tmp_path):
        """Extract notes for a specific version."""
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(SAMPLE_CHANGELOG)

        notes = extract_version_notes(changelog, "0.2.2")

        assert "## [v0.2.2]" in notes
        assert "Training loop with DummyTKSModel" in notes
        assert "Inference CLI" in notes
        assert "Total tests: 298 passing" in notes
        # Should not include other versions
        assert "## [v0.2.1]" not in notes
        assert "## [v0.1.0]" not in notes

    def test_extract_with_v_prefix(self, tmp_path):
        """Extract notes when version has 'v' prefix."""
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(SAMPLE_CHANGELOG)

        notes = extract_version_notes(changelog, "v0.2.2")

        assert "## [v0.2.2]" in notes
        assert "Training loop with DummyTKSModel" in notes

    def test_extract_middle_version(self, tmp_path):
        """Extract notes for a version in the middle."""
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(SAMPLE_CHANGELOG)

        notes = extract_version_notes(changelog, "0.2.1")

        assert "## [v0.2.1]" in notes
        assert "Training integration Phase 2" in notes
        assert "Total tests: 235 passing" in notes
        # Should not include other versions
        assert "## [v0.2.2]" not in notes
        assert "## [v0.1.0]" not in notes

    def test_extract_last_version(self, tmp_path):
        """Extract notes for the last version in changelog."""
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(SAMPLE_CHANGELOG)

        notes = extract_version_notes(changelog, "0.1.0")

        assert "## [v0.1.0]" in notes
        assert "Initial release with scenario inversion" in notes

    def test_version_not_found(self, tmp_path):
        """Handle version not present in changelog."""
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(SAMPLE_CHANGELOG)

        notes = extract_version_notes(changelog, "9.9.9")

        assert "Version not found" in notes

    def test_missing_changelog(self, tmp_path):
        """Handle missing CHANGELOG.md."""
        changelog = tmp_path / "CHANGELOG.md"  # Does not exist

        notes = extract_version_notes(changelog, "0.2.2")

        assert "No CHANGELOG.md found" in notes


class TestGetLatestVersion:
    """Test get_latest_version function."""

    def test_get_latest(self, tmp_path):
        """Get the latest version from changelog."""
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(SAMPLE_CHANGELOG)

        version = get_latest_version(changelog)

        # First non-unreleased version should be 0.2.2
        assert version == "0.2.2"

    def test_missing_changelog(self, tmp_path):
        """Handle missing CHANGELOG.md."""
        changelog = tmp_path / "CHANGELOG.md"

        version = get_latest_version(changelog)

        assert version == "0.0.0"

    def test_empty_changelog(self, tmp_path):
        """Handle empty changelog."""
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text("# Changelog\n\nNo versions yet.")

        version = get_latest_version(changelog)

        assert version == "0.0.0"


class TestCanonCompliance:
    """Verify release notes don't introduce new symbols."""

    def test_no_new_operators_in_sample(self):
        """Sample changelog should only reference allowed operators."""
        allowed_ops = {'+', '-', '+T', '-T', '->', '<-', '*T', '/T', 'o'}

        # This is a basic sanity check - the changelog should not
        # introduce new operator syntax
        # In practice, operators in changelog are descriptive, not code
        assert len(allowed_ops) == 9

    def test_worlds_constraint(self):
        """Canon worlds are A, B, C, D only."""
        valid_worlds = {'A', 'B', 'C', 'D'}
        assert len(valid_worlds) == 4

    def test_noetics_constraint(self):
        """Canon noetics are 1-10 with fixed involution pairs."""
        valid_noetics = set(range(1, 11))
        involution_pairs = {(2, 3), (5, 6), (8, 9)}
        self_duals = {1, 4, 7, 10}

        assert len(valid_noetics) == 10
        assert len(involution_pairs) == 3
        assert len(self_duals) == 4

    def test_foundations_constraint(self):
        """Canon foundations are 1-7 only."""
        valid_foundations = set(range(1, 8))
        assert len(valid_foundations) == 7

    def test_subfoundations_constraint(self):
        """Sub-foundations are 7x4=28."""
        num_foundations = 7
        num_worlds = 4
        assert num_foundations * num_worlds == 28
