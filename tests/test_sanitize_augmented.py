"""
Tests for TKS Data Sanitizer

Tests data quality validation, duplicate detection, and cleaning functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.sanitize_augmented import (
    validate_entry,
    validate_required_fields,
    validate_operators,
    validate_elements,
    validate_structure,
    compute_content_hash,
    scan_jsonl,
    clean_entries,
    SanitizationReport,
    SanitizationIssue,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def valid_entry():
    """A valid augmented entry."""
    return {
        "id": "entry_001",
        "story": "A spiritual teacher causes enlightenment in a seeking student",
        "expr": "A5 -> D2",
        "expr_elements": ["A5", "D2"],
        "expr_ops": ["->"],
        "aug_type": "original",
        "source_id": "entry_001",
        "validator_pass": True
    }


@pytest.fixture
def duplicate_entry():
    """A duplicate entry with same ID."""
    return {
        "id": "entry_001",  # Duplicate ID
        "story": "Different story with same ID",
        "expr": "B2 -> C3",
        "expr_elements": ["B2", "C3"],
        "expr_ops": ["->"],
        "aug_type": "original",
        "source_id": "entry_001",
        "validator_pass": True
    }


@pytest.fixture
def invalid_operator_entry():
    """Entry with invalid operator."""
    return {
        "id": "entry_002",
        "story": "Invalid operator test",
        "expr": "A1 ** B2",  # ** is not a valid operator
        "expr_elements": ["A1", "B2"],
        "expr_ops": ["**"],  # Invalid
        "aug_type": "original",
        "source_id": "entry_002",
        "validator_pass": False
    }


@pytest.fixture
def invalid_world_entry():
    """Entry with invalid world (not A/B/C/D)."""
    return {
        "id": "entry_003",
        "story": "Invalid world test",
        "expr": "E5 -> A2",  # E is not valid (only A/B/C/D)
        "expr_elements": ["E5", "A2"],
        "expr_ops": ["->"],
        "aug_type": "original",
        "source_id": "entry_003",
        "validator_pass": False
    }


@pytest.fixture
def invalid_noetic_entry():
    """Entry with invalid noetic (not 1-10)."""
    return {
        "id": "entry_004",
        "story": "Invalid noetic test",
        "expr": "A11 -> B2",  # 11 is not valid (only 1-10)
        "expr_elements": ["A11", "B2"],
        "expr_ops": ["->"],
        "aug_type": "original",
        "source_id": "entry_004",
        "validator_pass": False
    }


@pytest.fixture
def missing_fields_entry():
    """Entry with missing required fields."""
    return {
        "id": "entry_005",
        # Missing: story, expr, aug_type, validator_pass
        "expr_elements": ["A1"],
        "expr_ops": []
    }


@pytest.fixture
def structural_error_entry():
    """Entry with structural inconsistency."""
    return {
        "id": "entry_006",
        "story": "Structural error test",
        "expr": "A1 -> B2 -> C3",
        "expr_elements": ["A1", "B2", "C3"],  # 3 elements
        "expr_ops": ["->"],  # Only 1 operator (should be 2)
        "aug_type": "original",
        "source_id": "entry_006",
        "validator_pass": False
    }


@pytest.fixture
def sample_jsonl_content():
    """Sample JSONL content with mixed valid and invalid entries."""
    return [
        {
            "id": "clean_001",
            "story": "A spiritual teacher causes enlightenment",
            "expr": "A5 -> D2",
            "expr_elements": ["A5", "D2"],
            "expr_ops": ["->"],
            "aug_type": "original",
            "validator_pass": True
        },
        {
            "id": "clean_002",
            "story": "Mental clarity produces physical action",
            "expr": "B2 -> D5",
            "expr_elements": ["B2", "D5"],
            "expr_ops": ["->"],
            "aug_type": "original",
            "validator_pass": True
        },
        {
            "id": "dup_001",
            "story": "First occurrence",
            "expr": "A1 +T B2",
            "expr_elements": ["A1", "B2"],
            "expr_ops": ["+T"],
            "aug_type": "original",
            "validator_pass": True
        },
        {
            "id": "dup_001",  # Duplicate ID
            "story": "Second occurrence with same ID",
            "expr": "C3 -T D4",
            "expr_elements": ["C3", "D4"],
            "expr_ops": ["-T"],
            "aug_type": "inversion",
            "validator_pass": True
        },
        {
            "id": "invalid_op",
            "story": "Invalid operator",
            "expr": "A1 ** B2",
            "expr_elements": ["A1", "B2"],
            "expr_ops": ["**"],  # Invalid operator
            "aug_type": "original",
            "validator_pass": False
        },
        {
            "id": "invalid_world",
            "story": "Invalid world E",
            "expr": "E5 -> A2",
            "expr_elements": ["E5", "A2"],  # E is invalid
            "expr_ops": ["->"],
            "aug_type": "original",
            "validator_pass": False
        },
        {
            "id": "invalid_noetic",
            "story": "Invalid noetic 11",
            "expr": "A11 -> B2",
            "expr_elements": ["A11", "B2"],  # 11 is invalid
            "expr_ops": ["->"],
            "aug_type": "original",
            "validator_pass": False
        },
        {
            "id": "missing_story",
            # Missing story field
            "expr": "A1",
            "expr_elements": ["A1"],
            "expr_ops": [],
            "aug_type": "original",
            "validator_pass": False
        }
    ]


# =============================================================================
# VALIDATION TESTS
# =============================================================================

def test_validate_valid_entry(valid_entry):
    """Test that a valid entry passes validation."""
    is_valid, issues = validate_entry(valid_entry)
    assert is_valid
    assert len(issues) == 0


def test_validate_invalid_operator(invalid_operator_entry):
    """Test detection of invalid operators."""
    is_valid, issues = validate_entry(invalid_operator_entry)
    assert not is_valid
    assert any(issue.issue_type == "invalid_operator" for issue in issues)
    assert any("**" in issue.description for issue in issues)


def test_validate_invalid_world(invalid_world_entry):
    """Test detection of invalid worlds."""
    is_valid, issues = validate_entry(invalid_world_entry)
    assert not is_valid
    assert any(issue.issue_type == "invalid_world" for issue in issues)
    assert any("E" in issue.description for issue in issues)


def test_validate_invalid_noetic(invalid_noetic_entry):
    """Test detection of invalid noetics."""
    is_valid, issues = validate_entry(invalid_noetic_entry)
    assert not is_valid
    assert any(issue.issue_type == "invalid_noetic" for issue in issues)
    assert any("11" in issue.description for issue in issues)


def test_validate_missing_fields(missing_fields_entry):
    """Test detection of missing required fields."""
    is_valid, issues = validate_entry(missing_fields_entry)
    assert not is_valid

    missing_field_issues = [i for i in issues if i.issue_type == "missing_field"]
    assert len(missing_field_issues) > 0

    # Check that specific fields are flagged
    missing_fields = [i.field for i in missing_field_issues]
    assert "story" in missing_fields
    assert "expr" in missing_fields
    assert "aug_type" in missing_fields


def test_validate_structural_error(structural_error_entry):
    """Test detection of structural inconsistencies."""
    is_valid, issues = validate_entry(structural_error_entry)
    assert not is_valid
    assert any(issue.issue_type == "structural_error" for issue in issues)
    assert any("inconsistency" in issue.description.lower() for issue in issues)


def test_validate_operators_function(invalid_operator_entry):
    """Test the validate_operators function directly."""
    issues = validate_operators(invalid_operator_entry)
    assert len(issues) > 0
    assert issues[0].issue_type == "invalid_operator"
    assert "**" in issues[0].description


def test_validate_elements_function(invalid_world_entry):
    """Test the validate_elements function directly."""
    issues = validate_elements(invalid_world_entry)
    assert len(issues) > 0
    assert any(issue.issue_type == "invalid_world" for issue in issues)


def test_validate_structure_function(structural_error_entry):
    """Test the validate_structure function directly."""
    issues = validate_structure(structural_error_entry)
    assert len(issues) > 0
    assert issues[0].issue_type == "structural_error"


def test_validate_required_fields_function(missing_fields_entry):
    """Test the validate_required_fields function directly."""
    issues = validate_required_fields(missing_fields_entry)
    assert len(issues) > 0
    assert all(issue.issue_type == "missing_field" for issue in issues)


# =============================================================================
# HASH AND DUPLICATE TESTS
# =============================================================================

def test_compute_content_hash():
    """Test content hash computation."""
    entry1 = {
        "id": "a",
        "story": "Test story",
        "expr": "A1 -> B2",
        "aug_type": "original"
    }
    entry2 = {
        "id": "b",  # Different ID
        "story": "Test story",  # Same content
        "expr": "A1 -> B2",
        "aug_type": "original"
    }
    entry3 = {
        "id": "c",
        "story": "Different story",
        "expr": "A1 -> B2",
        "aug_type": "original"
    }

    hash1 = compute_content_hash(entry1)
    hash2 = compute_content_hash(entry2)
    hash3 = compute_content_hash(entry3)

    # Same content should produce same hash (regardless of ID)
    assert hash1 == hash2
    # Different content should produce different hash
    assert hash1 != hash3


# =============================================================================
# SCANNING TESTS
# =============================================================================

def test_scan_jsonl_with_valid_entries(valid_entry, tmp_path):
    """Test scanning a file with only valid entries."""
    # Create temp JSONL file
    test_file = tmp_path / "test_valid.jsonl"
    with open(test_file, 'w', encoding='utf-8') as f:
        for i in range(3):
            entry = valid_entry.copy()
            entry["id"] = f"entry_{i:03d}"
            f.write(json.dumps(entry) + '\n')

    # Scan file
    entries, report = scan_jsonl(test_file)

    assert len(entries) == 3
    assert report.total_entries == 3
    assert report.clean_entries == 3
    assert len(report.issues) == 0


def test_scan_jsonl_with_mixed_entries(sample_jsonl_content, tmp_path):
    """Test scanning a file with mixed valid and invalid entries."""
    # Create temp JSONL file
    test_file = tmp_path / "test_mixed.jsonl"
    with open(test_file, 'w', encoding='utf-8') as f:
        for entry in sample_jsonl_content:
            f.write(json.dumps(entry) + '\n')

    # Scan file
    entries, report = scan_jsonl(test_file)

    assert len(entries) == len(sample_jsonl_content)
    assert report.total_entries == len(sample_jsonl_content)

    # Should detect issues
    assert report.duplicate_entries > 0  # dup_001 appears twice
    assert report.invalid_operators > 0  # invalid_op entry
    assert report.invalid_worlds > 0     # invalid_world entry
    assert report.invalid_noetics > 0    # invalid_noetic entry
    assert report.missing_fields > 0     # missing_story entry

    # Check clean entries
    # clean_001, clean_002, and dup_001 (first occurrence) are all clean
    # Only the second occurrence of dup_001 is flagged as duplicate
    assert report.clean_entries == 3


def test_scan_jsonl_duplicate_detection(tmp_path):
    """Test that duplicate IDs are detected."""
    test_file = tmp_path / "test_duplicates.jsonl"

    entries = [
        {
            "id": "dup_id",
            "story": "First",
            "expr": "A1",
            "expr_elements": ["A1"],
            "expr_ops": [],
            "aug_type": "original",
            "validator_pass": True
        },
        {
            "id": "dup_id",  # Duplicate
            "story": "Second",
            "expr": "B2",
            "expr_elements": ["B2"],
            "expr_ops": [],
            "aug_type": "original",
            "validator_pass": True
        }
    ]

    with open(test_file, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

    _, report = scan_jsonl(test_file)

    assert report.duplicate_entries > 0
    assert "dup_id" in report.duplicates_by_id


def test_scan_jsonl_content_hash_duplicates(tmp_path):
    """Test that content duplicates are detected by hash."""
    test_file = tmp_path / "test_content_dup.jsonl"

    entries = [
        {
            "id": "id1",
            "story": "Same story",
            "expr": "A1 -> B2",
            "expr_elements": ["A1", "B2"],
            "expr_ops": ["->"],
            "aug_type": "original",
            "validator_pass": True
        },
        {
            "id": "id2",  # Different ID
            "story": "Same story",  # Same content
            "expr": "A1 -> B2",
            "expr_elements": ["A1", "B2"],
            "expr_ops": ["->"],
            "aug_type": "original",
            "validator_pass": True
        }
    ]

    with open(test_file, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

    _, report = scan_jsonl(test_file)

    # Check that content hash detected duplicates
    content_dups = {k: v for k, v in report.duplicates_by_hash.items() if len(v) > 1}
    assert len(content_dups) > 0


# =============================================================================
# CLEANING TESTS
# =============================================================================

def test_clean_entries_keep_all(sample_jsonl_content, tmp_path):
    """Test that cleaning with drop_invalid=False keeps all entries."""
    test_file = tmp_path / "test_clean_keep.jsonl"

    with open(test_file, 'w', encoding='utf-8') as f:
        for entry in sample_jsonl_content:
            f.write(json.dumps(entry) + '\n')

    entries, report = scan_jsonl(test_file)
    cleaned = clean_entries(entries, report, drop_invalid=False)

    assert len(cleaned) == len(entries)


def test_clean_entries_drop_invalid(sample_jsonl_content, tmp_path):
    """Test that cleaning with drop_invalid=True removes invalid entries."""
    test_file = tmp_path / "test_clean_drop.jsonl"

    with open(test_file, 'w', encoding='utf-8') as f:
        for entry in sample_jsonl_content:
            f.write(json.dumps(entry) + '\n')

    entries, report = scan_jsonl(test_file)
    cleaned = clean_entries(entries, report, drop_invalid=True)

    # Should remove all entries with any issue (including duplicates)
    # Note: clean_entries removes ALL entries whose ID appears in any issue,
    # so both occurrences of dup_001 are removed (not just the duplicate)
    assert len(cleaned) < len(entries)
    # Cleaned should have clean_001, clean_002 (2 entries)
    # dup_001 is removed because its ID appears in duplicate issue
    assert len(cleaned) == 2


# =============================================================================
# REPORT TESTS
# =============================================================================

def test_sanitization_report_creation():
    """Test SanitizationReport creation and methods."""
    report = SanitizationReport()

    assert report.total_entries == 0
    assert report.clean_entries == 0
    assert len(report.issues) == 0


def test_sanitization_report_add_issue():
    """Test adding issues to report."""
    report = SanitizationReport()

    issue = SanitizationIssue(
        entry_id="test_001",
        issue_type="invalid_operator",
        description="Test issue",
        severity="error"
    )

    report.add_issue(issue)

    assert len(report.issues) == 1
    assert report.invalid_operators == 1


def test_sanitization_report_to_dict():
    """Test converting report to dictionary."""
    report = SanitizationReport()
    report.total_entries = 10
    report.clean_entries = 7

    report_dict = report.to_dict()

    assert "summary" in report_dict
    assert report_dict["summary"]["total_entries"] == 10
    assert report_dict["summary"]["clean_entries"] == 7
    assert "pass_rate" in report_dict["summary"]


def test_sanitization_issue_creation():
    """Test SanitizationIssue creation."""
    issue = SanitizationIssue(
        entry_id="test_001",
        issue_type="invalid_world",
        description="Invalid world E",
        severity="error",
        field="expr_elements"
    )

    assert issue.entry_id == "test_001"
    assert issue.issue_type == "invalid_world"
    assert issue.field == "expr_elements"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_full_pipeline_valid_data(tmp_path):
    """Test full pipeline with valid data."""
    input_file = tmp_path / "input.jsonl"
    output_file = tmp_path / "output.jsonl"

    valid_entries = [
        {
            "id": f"entry_{i:03d}",
            "story": f"Story {i}",
            "expr": "A1 -> B2",
            "expr_elements": ["A1", "B2"],
            "expr_ops": ["->"],
            "aug_type": "original",
            "validator_pass": True
        }
        for i in range(5)
    ]

    with open(input_file, 'w', encoding='utf-8') as f:
        for entry in valid_entries:
            f.write(json.dumps(entry) + '\n')

    # Scan and clean
    entries, report = scan_jsonl(input_file)
    cleaned = clean_entries(entries, report, drop_invalid=True)

    assert len(cleaned) == 5
    assert report.clean_entries == 5
    assert len(report.issues) == 0


def test_full_pipeline_mixed_data(sample_jsonl_content, tmp_path):
    """Test full pipeline with mixed valid/invalid data."""
    input_file = tmp_path / "input_mixed.jsonl"

    with open(input_file, 'w', encoding='utf-8') as f:
        for entry in sample_jsonl_content:
            f.write(json.dumps(entry) + '\n')

    # Scan
    entries, report = scan_jsonl(input_file)

    # Verify detection
    assert report.total_entries == len(sample_jsonl_content)
    # clean_001, clean_002, and dup_001 (first occurrence) are all clean
    assert report.clean_entries == 3
    assert len(report.issues) > 0

    # Clean - removes all entries whose ID appears in any issue
    cleaned = clean_entries(entries, report, drop_invalid=True)
    # Only clean_001 and clean_002 remain (dup_001 removed due to duplicate ID issue)
    assert len(cleaned) == 2


def test_multiple_issues_per_entry(tmp_path):
    """Test that multiple issues can be detected in a single entry."""
    test_file = tmp_path / "test_multiple.jsonl"

    # Entry with multiple issues (include required fields so all validations run)
    bad_entry = {
        "id": "bad_entry",
        "story": "Test story with multiple validation errors",
        "expr": "E11 ** F12",  # Invalid world E, invalid noetic 11, invalid operator **
        "expr_elements": ["E11", "F12"],  # E and F invalid, 11 and 12 invalid
        "expr_ops": ["**"],  # Invalid operator
        "aug_type": "original",
        "validator_pass": False
    }

    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(bad_entry) + '\n')

    _, report = scan_jsonl(test_file)

    # Should detect multiple issues (operators, worlds, noetics)
    assert report.invalid_operators > 0
    assert report.invalid_worlds > 0
    assert report.invalid_noetics > 0

    # All issues should be for the same entry
    entry_ids = [issue.entry_id for issue in report.issues]
    assert all(eid == "bad_entry" for eid in entry_ids)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
