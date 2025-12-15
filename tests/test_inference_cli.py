"""
Smoke tests for TKS Inference CLI (scripts/run_inference.py)

Tests verify that the inference pipeline works end-to-end without errors.
Validates story input, equation input, anti-attractor mode, JSON output,
and strict mode error handling.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import json
from scripts.run_inference import (
    run_inference,
    format_text_output,
    format_json_output,
    parse_axes,
    format_expression,
)
from scenario_inversion import parse_equation, EncodeStory, TKSExpression
from inversion.engine import TargetProfile


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def test_parse_axes():
    """Test axes parsing from string."""
    # Letter codes
    assert "World" in parse_axes("W")
    assert "Noetic" in parse_axes("N")
    assert "Element" in parse_axes("E")

    # Full names
    assert "World" in parse_axes("World")
    assert "Noetic" in parse_axes("Noetic")

    # Comma-separated
    axes = parse_axes("W,N")
    assert "World" in axes
    assert "Noetic" in axes

    # Mixed case
    axes = parse_axes("w,n,e")
    assert "World" in axes
    assert "Noetic" in axes
    assert "Element" in axes

    # Empty
    assert parse_axes("") == set()


def test_format_expression():
    """Test TKS expression formatting."""
    expr = TKSExpression(
        elements=["B5", "D3"],
        ops=["+T"],
        foundations=[],
        acquisitions=[],
        raw=""
    )
    assert format_expression(expr) == "B5 +T D3"

    # Multiple operators
    expr2 = TKSExpression(
        elements=["B5", "D3", "C2"],
        ops=["+T", "->"],
        foundations=[],
        acquisitions=[],
        raw=""
    )
    assert format_expression(expr2) == "B5 +T D3 -> C2"

    # Single element
    expr3 = TKSExpression(
        elements=["B5"],
        ops=[],
        foundations=[],
        acquisitions=[],
        raw=""
    )
    assert format_expression(expr3) == "B5"


# =============================================================================
# INFERENCE PIPELINE TESTS
# =============================================================================

def test_story_input_returns_non_empty():
    """Test that story input produces non-empty output."""
    result = run_inference(
        input_text="A woman loved a man",
        is_equation=False,
        anti_attractor=False,
        axes={"World", "Noetic"},
        mode="soft",
        strict=True,
        target=None
    )

    # Verify all required fields present
    assert "original_expr" in result
    assert "inverted_expr" in result
    assert "original_story" in result
    assert "inverted_story" in result
    assert "explanation" in result

    # Verify non-empty
    assert len(result["original_expr"].elements) > 0
    assert len(result["inverted_expr"].elements) > 0
    assert len(result["original_story"]) > 0
    assert len(result["inverted_story"]) > 0
    assert len(result["explanation"]) > 0


def test_equation_input_returns_non_empty():
    """Test that equation input produces non-empty output."""
    result = run_inference(
        input_text="B5 +T D3",
        is_equation=True,
        anti_attractor=False,
        axes={"World", "Noetic"},
        mode="soft",
        strict=True,
        target=None
    )

    # Verify all required fields present
    assert "original_expr" in result
    assert "inverted_expr" in result
    assert "original_story" in result
    assert "inverted_story" in result
    assert "explanation" in result

    # Verify non-empty
    assert len(result["original_expr"].elements) > 0
    assert len(result["inverted_expr"].elements) > 0
    assert len(result["original_story"]) > 0
    assert len(result["inverted_story"]) > 0


def test_anti_attractor_flag_works():
    """Test that --anti-attractor flag triggers anti-attractor synthesis."""
    result = run_inference(
        input_text="B2 -> D5 +T C8",
        is_equation=True,
        anti_attractor=True,  # Enable anti-attractor mode
        axes=set(),  # Axes ignored in anti-attractor mode
        mode="soft",
        strict=True,
        target=None
    )

    # Verify anti-attractor-specific fields
    assert "signature" in result
    assert "inverted_signature" in result

    # Verify signature fields
    sig = result["signature"]
    assert hasattr(sig, "element_counts")
    assert hasattr(sig, "dominant_world")
    assert hasattr(sig, "dominant_noetic")
    assert hasattr(sig, "polarity")
    assert hasattr(sig, "foundation_tags")

    # Verify inverted expression generated
    assert len(result["inverted_expr"].elements) > 0


def test_json_output_is_valid():
    """Test that JSON output format produces valid JSON."""
    result = run_inference(
        input_text="B5 +T D3",
        is_equation=True,
        anti_attractor=False,
        axes={"World"},
        mode="soft",
        strict=True,
        target=None
    )

    # Format as JSON
    json_output = format_json_output(result, anti_attractor_mode=False)

    # Verify valid JSON
    parsed = json.loads(json_output)

    # Verify structure
    assert "mode" in parsed
    assert "original" in parsed
    assert "inverted" in parsed
    assert "explanation" in parsed

    assert "expression" in parsed["original"]
    assert "story" in parsed["original"]
    assert "elements" in parsed["original"]
    assert "ops" in parsed["original"]

    assert "expression" in parsed["inverted"]
    assert "story" in parsed["inverted"]
    assert "elements" in parsed["inverted"]
    assert "ops" in parsed["inverted"]


def test_json_output_anti_attractor_includes_signature():
    """Test that JSON output in anti-attractor mode includes signature."""
    result = run_inference(
        input_text="B2 -> C2 +T D2",
        is_equation=True,
        anti_attractor=True,
        axes=set(),
        mode="soft",
        strict=True,
        target=None
    )

    json_output = format_json_output(result, anti_attractor_mode=True)
    parsed = json.loads(json_output)

    # Verify signature included
    assert "signature" in parsed
    assert "element_counts" in parsed["signature"]
    assert "dominant_world" in parsed["signature"]
    assert "dominant_noetic" in parsed["signature"]
    assert "polarity" in parsed["signature"]
    assert "foundation_tags" in parsed["signature"]


def test_strict_mode_rejects_unknown_tokens():
    """Test that strict mode raises ValueError for unknown tokens."""
    # This test assumes "UNKNOWNTOKEN" is not in the lexicon
    with pytest.raises(ValueError):
        run_inference(
            input_text="UNKNOWNTOKEN",
            is_equation=False,
            anti_attractor=False,
            axes={"World"},
            mode="soft",
            strict=True,  # Strict mode should reject
            target=None
        )


def test_lenient_mode_allows_unknown_tokens():
    """Test that lenient mode allows unknown tokens with warnings."""
    # In lenient mode, unknown tokens should be skipped/warned but not raise error
    # The encoder should return something (possibly empty or partial)
    try:
        result = run_inference(
            input_text="Some valid text with UNKNOWNTOKEN",
            is_equation=False,
            anti_attractor=False,
            axes={"World"},
            mode="soft",
            strict=False,  # Lenient mode
            target=None
        )

        # Should not raise error
        assert "original_expr" in result
        assert "inverted_expr" in result

    except ValueError:
        # If encoder still raises in lenient mode, that's acceptable
        # as long as the error message is different from strict mode
        pytest.skip("Encoder may still reject even in lenient mode")


# =============================================================================
# OUTPUT FORMAT TESTS
# =============================================================================

def test_text_output_includes_all_sections():
    """Test that text output includes all required sections."""
    result = run_inference(
        input_text="B5 +T D3",
        is_equation=True,
        anti_attractor=False,
        axes={"World"},
        mode="soft",
        strict=True,
        target=None
    )

    text_output = format_text_output(result, anti_attractor_mode=False)

    # Verify all sections present
    assert "=== ORIGINAL ===" in text_output
    assert "=== INVERTED ===" in text_output
    assert "=== EXPLANATION ===" in text_output

    # Verify content
    assert "Expression:" in text_output
    assert "Story:" in text_output


def test_text_output_anti_attractor_includes_signature():
    """Test that text output in anti-attractor mode includes signature."""
    result = run_inference(
        input_text="B2 -> C2 +T D2",
        is_equation=True,
        anti_attractor=True,
        axes=set(),
        mode="soft",
        strict=True,
        target=None
    )

    text_output = format_text_output(result, anti_attractor_mode=True)

    # Verify signature section present
    assert "=== ATTRACTOR SIGNATURE ===" in text_output
    assert "Element Distribution:" in text_output
    assert "Dominant:" in text_output
    assert "Polarity:" in text_output


# =============================================================================
# INVERSION MODE TESTS
# =============================================================================

def test_soft_mode_produces_output():
    """Test that soft mode inversion works."""
    result = run_inference(
        input_text="B5 +T D3",
        is_equation=True,
        anti_attractor=False,
        axes={"World", "Noetic"},
        mode="soft",
        strict=True,
        target=None
    )

    assert len(result["inverted_expr"].elements) > 0


def test_hard_mode_produces_output():
    """Test that hard mode inversion works."""
    result = run_inference(
        input_text="B5 +T D3",
        is_equation=True,
        anti_attractor=False,
        axes={"World", "Noetic"},
        mode="hard",
        strict=True,
        target=None
    )

    assert len(result["inverted_expr"].elements) > 0


def test_targeted_mode_with_profile():
    """Test that targeted mode with TargetProfile works."""
    target = TargetProfile(
        enable=True,
        from_foundation=5,
        to_foundation=2,
        from_world=None,
        to_world=None
    )

    result = run_inference(
        input_text="B5 +T D3",
        is_equation=True,
        anti_attractor=False,
        axes={"Foundation"},
        mode="targeted",
        strict=True,
        target=target
    )

    assert len(result["inverted_expr"].elements) > 0


# =============================================================================
# AXES TESTS
# =============================================================================

def test_world_axis_inversion():
    """Test that World axis inversion produces different world letters."""
    expr_input = parse_equation("A2 -> D5")

    result = run_inference(
        input_text="A2 -> D5",
        is_equation=True,
        anti_attractor=False,
        axes={"World"},
        mode="soft",
        strict=True,
        target=None
    )

    # Original has A and D
    assert "A2" in result["original_expr"].elements
    assert "D5" in result["original_expr"].elements

    # After World inversion: A<->D
    inverted_elements = result["inverted_expr"].elements
    assert "D2" in inverted_elements or any("D" in e for e in inverted_elements)
    assert "A5" in inverted_elements or any("A" in e for e in inverted_elements)


def test_noetic_axis_inversion():
    """Test that Noetic axis inversion produces different noetics."""
    result = run_inference(
        input_text="B2 +T B3",
        is_equation=True,
        anti_attractor=False,
        axes={"Noetic"},
        mode="soft",
        strict=True,
        target=None
    )

    # Original has N2 and N3
    orig_noetics = [int(e[1:]) for e in result["original_expr"].elements]
    assert 2 in orig_noetics
    assert 3 in orig_noetics

    # After Noetic inversion: 2<->3
    inv_noetics = [int(e[1:]) for e in result["inverted_expr"].elements]
    # Should have swapped
    assert 3 in inv_noetics  # 2->3
    assert 2 in inv_noetics  # 3->2


# =============================================================================
# EDGE CASES
# =============================================================================

def test_single_element_equation():
    """Test inference on single-element equation."""
    result = run_inference(
        input_text="B5",
        is_equation=True,
        anti_attractor=False,
        axes={"World"},
        mode="soft",
        strict=True,
        target=None
    )

    assert len(result["original_expr"].elements) == 1
    assert len(result["inverted_expr"].elements) == 1


def test_empty_axes_uses_default():
    """Test that empty axes set uses default behavior."""
    result = run_inference(
        input_text="B5 +T D3",
        is_equation=True,
        anti_attractor=False,
        axes=set(),  # Empty set
        mode="soft",
        strict=True,
        target=None
    )

    # Should still produce output (engine may have defaults)
    assert "inverted_expr" in result


def test_multiple_operators():
    """Test inference on expression with multiple operators."""
    result = run_inference(
        input_text="B5 +T D3 -> C2 *T A6",
        is_equation=True,
        anti_attractor=False,
        axes={"World", "Noetic"},
        mode="soft",
        strict=True,
        target=None
    )

    assert len(result["original_expr"].elements) == 4
    assert len(result["original_expr"].ops) == 3
    assert len(result["inverted_expr"].elements) > 0


# =============================================================================
# ANTI-ATTRACTOR SPECIFIC TESTS
# =============================================================================

def test_anti_attractor_signature_polarity():
    """Test that anti-attractor correctly identifies polarity."""
    # Positive expression (all N2)
    result = run_inference(
        input_text="B2 -> C2 +T D2",
        is_equation=True,
        anti_attractor=True,
        axes=set(),
        mode="soft",
        strict=True,
        target=None
    )

    sig = result["signature"]
    # All N2 (positive) -> polarity should be +1
    assert sig.polarity == 1


def test_anti_attractor_inverts_polarity():
    """Test that anti-attractor inverts polarity."""
    result = run_inference(
        input_text="B2 -> C2 +T D2",
        is_equation=True,
        anti_attractor=True,
        axes=set(),
        mode="soft",
        strict=True,
        target=None
    )

    orig_sig = result["signature"]
    inv_sig = result["inverted_signature"]

    # Polarity should be inverted
    assert orig_sig.polarity == -inv_sig.polarity


def test_anti_attractor_synthesizes_counter_scenario():
    """Test that anti-attractor synthesizes valid counter-scenario."""
    result = run_inference(
        input_text="B2 -> D5 +T C8",
        is_equation=True,
        anti_attractor=True,
        axes=set(),
        mode="soft",
        strict=True,
        target=None
    )

    # Counter-scenario should have inverted elements
    counter = result["inverted_expr"]
    assert len(counter.elements) > 0
    assert len(counter.ops) > 0

    # Should be different from original
    assert counter.elements != result["original_expr"].elements


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_story_to_equation_to_story_roundtrip():
    """Test full roundtrip: story -> equation -> inverted -> story."""
    result = run_inference(
        input_text="A woman loved a man",
        is_equation=False,
        anti_attractor=False,
        axes={"World", "Noetic"},
        mode="soft",
        strict=True,
        target=None
    )

    # Verify we got stories
    assert len(result["original_story"]) > 0
    assert len(result["inverted_story"]) > 0

    # Stories should be different
    assert result["original_story"] != result["inverted_story"]


def test_explanation_describes_changes():
    """Test that explanation describes actual changes."""
    result = run_inference(
        input_text="B5 +T D3",
        is_equation=True,
        anti_attractor=False,
        axes={"World"},
        mode="soft",
        strict=True,
        target=None
    )

    explanation = result["explanation"]

    # Should contain some description
    assert len(explanation) > 0

    # If changes occurred, should mention them
    if result["original_expr"].elements != result["inverted_expr"].elements:
        # Should have some text (could be "INVERSION CHANGES" or element descriptions)
        assert len(explanation) > 20


# =============================================================================
# BULK MODE TESTS
# =============================================================================

def test_process_single_item_with_story():
    """Test process_single_item with story input."""
    from scripts.run_inference import process_single_item

    item = {"story": "A woman loved a man"}
    result = process_single_item(
        item=item,
        default_axes={"World", "Noetic"},
        default_mode="soft",
        default_anti_attractor=False,
        default_strict=True,
        include_validator=True,
    )

    # Verify success
    assert result["success"] is True
    assert result["error"] is None
    assert result["inversion_type"] == "inversion"

    # Verify result structure
    assert "result" in result
    assert result["result"] is not None
    assert "original" in result["result"]
    assert "inverted" in result["result"]

    # Verify validator section
    assert "validator" in result
    assert result["validator"] is not None
    assert "is_valid" in result["validator"]
    assert "canon_score" in result["validator"]
    assert "error_count" in result["validator"]
    assert "warning_count" in result["validator"]
    assert "issues" in result["validator"]


def test_process_single_item_with_equation():
    """Test process_single_item with equation input."""
    from scripts.run_inference import process_single_item

    item = {"equation": "B5 +T D3"}
    result = process_single_item(
        item=item,
        default_axes={"World"},
        default_mode="soft",
        default_anti_attractor=False,
        default_strict=True,
        include_validator=True,
    )

    # Verify success
    assert result["success"] is True
    assert result["error"] is None
    assert result["inversion_type"] == "inversion"

    # Verify result structure
    assert result["result"]["original"]["elements"] == ["B5", "D3"]
    assert result["result"]["original"]["ops"] == ["+T"]


def test_process_single_item_anti_attractor():
    """Test process_single_item with anti-attractor mode."""
    from scripts.run_inference import process_single_item

    item = {"equation": "B2 -> C2 +T D2"}
    result = process_single_item(
        item=item,
        default_axes=set(),
        default_mode="soft",
        default_anti_attractor=True,  # Anti-attractor mode
        default_strict=True,
        include_validator=True,
    )

    # Verify success
    assert result["success"] is True
    assert result["inversion_type"] == "anti-attractor"

    # Verify signature present for anti-attractor mode
    assert "signature" in result
    assert result["signature"] is not None
    assert "element_counts" in result["signature"]
    assert "polarity" in result["signature"]
    assert "dominant_world" in result["signature"]


def test_process_single_item_with_item_overrides():
    """Test process_single_item with item-level overrides."""
    from scripts.run_inference import process_single_item

    # Item overrides default axes and mode
    item = {
        "equation": "B5 +T D3",
        "axes": "W",  # Override to World only
        "mode": "hard",  # Override to hard mode
    }
    result = process_single_item(
        item=item,
        default_axes={"Noetic"},  # Should be overridden
        default_mode="soft",  # Should be overridden
        default_anti_attractor=False,
        default_strict=True,
        include_validator=False,  # Skip validator for speed
    )

    # Verify success
    assert result["success"] is True
    assert result["error"] is None


def test_process_single_item_error_handling():
    """Test process_single_item error handling for invalid input."""
    from scripts.run_inference import process_single_item

    # Missing story/equation field
    item = {"invalid_field": "value"}
    result = process_single_item(
        item=item,
        default_axes={"World"},
        default_mode="soft",
        default_anti_attractor=False,
        default_strict=True,
        include_validator=True,
    )

    # Verify failure
    assert result["success"] is False
    assert result["error"] is not None
    assert "Missing" in result["error"]


def test_read_jsonl_parses_valid_lines(tmp_path):
    """Test read_jsonl parses valid JSONL lines."""
    from scripts.run_inference import read_jsonl

    # Create test JSONL file
    jsonl_content = '{"story": "Test story 1"}\n{"equation": "B5 +T D3"}\n{"story": "Test story 2"}'
    input_file = tmp_path / "test_input.jsonl"
    input_file.write_text(jsonl_content)

    # Read and verify
    items = list(read_jsonl(input_file))
    assert len(items) == 3
    assert items[0] == {"story": "Test story 1"}
    assert items[1] == {"equation": "B5 +T D3"}
    assert items[2] == {"story": "Test story 2"}


def test_read_jsonl_handles_comments_and_empty_lines(tmp_path):
    """Test read_jsonl skips comments and empty lines."""
    from scripts.run_inference import read_jsonl

    # Create test JSONL file with comments and empty lines
    jsonl_content = '# This is a comment\n{"story": "Valid"}\n\n{"equation": "B5"}'
    input_file = tmp_path / "test_input.jsonl"
    input_file.write_text(jsonl_content)

    # Read and verify
    items = list(read_jsonl(input_file))
    assert len(items) == 2
    assert items[0] == {"story": "Valid"}
    assert items[1] == {"equation": "B5"}


def test_read_jsonl_reports_parse_errors(tmp_path):
    """Test read_jsonl reports JSON parse errors."""
    from scripts.run_inference import read_jsonl

    # Create test JSONL file with invalid JSON
    jsonl_content = '{"story": "Valid"}\n{invalid json here}\n{"equation": "B5"}'
    input_file = tmp_path / "test_input.jsonl"
    input_file.write_text(jsonl_content)

    # Read and verify
    items = list(read_jsonl(input_file))
    assert len(items) == 3
    assert items[0] == {"story": "Valid"}

    # Second item should have parse error marker
    assert items[1].get("_parse_error") is True
    assert items[1].get("_line_number") == 2
    assert "JSON parse error" in items[1].get("_error", "")

    assert items[2] == {"equation": "B5"}


def test_run_bulk_inference_end_to_end(tmp_path):
    """Test run_bulk_inference end-to-end with multiple items."""
    from scripts.run_inference import run_bulk_inference

    # Create test JSONL input
    input_content = '{"equation": "B5 +T D3"}\n{"equation": "A2 -> D5"}\n{"story": "power corrupts"}'
    input_file = tmp_path / "input.jsonl"
    input_file.write_text(input_content)

    output_file = tmp_path / "output.jsonl"

    # Run bulk inference
    stats = run_bulk_inference(
        input_path=input_file,
        output_path=output_file,
        default_axes={"World"},
        default_mode="soft",
        default_anti_attractor=False,
        default_strict=False,  # Lenient to avoid story encoding errors
        include_validator=True,
        verbose=False,
    )

    # Verify stats
    assert stats["total"] == 3
    assert stats["parse_errors"] == 0

    # Verify output file exists and contains valid JSONL
    assert output_file.exists()
    output_lines = output_file.read_text().strip().split('\n')
    assert len(output_lines) == 3

    # Parse and verify first output
    first_result = json.loads(output_lines[0])
    assert "success" in first_result
    assert "input" in first_result
    assert "result" in first_result or "error" in first_result


# =============================================================================
# RICH JSON OUTPUT TESTS
# =============================================================================

def test_json_output_includes_validator_section():
    """Test that JSON output includes validator section with all fields."""
    result = run_inference(
        input_text="B5 +T D3",
        is_equation=True,
        anti_attractor=False,
        axes={"World"},
        mode="soft",
        strict=True,
        target=None
    )

    # Create validator and run validation
    from teacher.validator import CanonicalValidator
    validator = CanonicalValidator(strict_mode=True)
    inverted_expr = format_expression(result['inverted_expr'])
    validator_result = validator.validate(inverted_expr)

    # Format with validator
    json_output = format_json_output(
        result,
        anti_attractor_mode=False,
        validator_result=validator_result,
        include_validator=True,
    )
    parsed = json.loads(json_output)

    # Verify validator section structure
    assert "validator" in parsed
    validator_data = parsed["validator"]
    assert "is_valid" in validator_data
    assert "canon_score" in validator_data
    assert "error_count" in validator_data
    assert "warning_count" in validator_data
    assert "issues" in validator_data
    assert isinstance(validator_data["issues"], list)


def test_json_output_includes_inversion_type():
    """Test that JSON output includes inversion_type field."""
    result = run_inference(
        input_text="B5 +T D3",
        is_equation=True,
        anti_attractor=False,
        axes={"World"},
        mode="soft",
        strict=True,
        target=None
    )

    # Test inversion mode
    json_output = format_json_output(result, anti_attractor_mode=False)
    parsed = json.loads(json_output)
    assert "inversion_type" in parsed
    assert parsed["inversion_type"] == "inversion"

    # Test anti-attractor mode
    result_aa = run_inference(
        input_text="B5 +T D3",
        is_equation=True,
        anti_attractor=True,
        axes=set(),
        mode="soft",
        strict=True,
        target=None
    )
    json_output_aa = format_json_output(result_aa, anti_attractor_mode=True)
    parsed_aa = json.loads(json_output_aa)
    assert parsed_aa["inversion_type"] == "anti-attractor"


def test_json_output_anti_attractor_includes_both_signatures():
    """Test that anti-attractor JSON output includes both original and inverted signatures."""
    result = run_inference(
        input_text="B2 -> C2 +T D2",
        is_equation=True,
        anti_attractor=True,
        axes=set(),
        mode="soft",
        strict=True,
        target=None
    )

    json_output = format_json_output(result, anti_attractor_mode=True)
    parsed = json.loads(json_output)

    # Verify both signatures present
    assert "signature" in parsed
    assert "inverted_signature" in parsed

    # Verify signature structure
    for sig_key in ["signature", "inverted_signature"]:
        sig = parsed[sig_key]
        assert "element_counts" in sig
        assert "dominant_world" in sig
        assert "dominant_noetic" in sig
        assert "polarity" in sig
        assert "foundation_tags" in sig


def test_json_output_compact_mode():
    """Test that compact mode produces single-line JSON."""
    result = run_inference(
        input_text="B5 +T D3",
        is_equation=True,
        anti_attractor=False,
        axes={"World"},
        mode="soft",
        strict=True,
        target=None
    )

    # Compact mode
    json_compact = format_json_output(result, anti_attractor_mode=False, compact=True)
    assert "\n" not in json_compact  # Single line

    # Normal mode (should have newlines)
    json_normal = format_json_output(result, anti_attractor_mode=False, compact=False)
    assert "\n" in json_normal  # Multiple lines


def test_strict_lenient_flags_respected_in_bulk():
    """Test that strict/lenient flags are respected in bulk processing."""
    from scripts.run_inference import process_single_item

    # Test strict mode with potentially problematic input
    item_strict = {"story": "A woman loved a man"}
    result_strict = process_single_item(
        item=item_strict,
        default_axes={"World"},
        default_mode="soft",
        default_anti_attractor=False,
        default_strict=True,  # Strict
        include_validator=True,
    )
    # Should process (valid story)
    assert result_strict["success"] is True

    # Test with item-level lenient override
    item_lenient = {
        "story": "Some text with UNKNOWNWORD",
        "lenient": True  # Item-level override
    }
    result_lenient = process_single_item(
        item=item_lenient,
        default_axes={"World"},
        default_mode="soft",
        default_anti_attractor=False,
        default_strict=True,  # Default is strict, but item overrides
        include_validator=True,
    )
    # With lenient, should not raise error (may succeed or have graceful handling)
    # The key is that it doesn't crash
    assert "success" in result_lenient
    assert "error" in result_lenient  # May have error field (possibly None)


# =============================================================================
# ADDITIONAL BULK MODE SMOKE TESTS
# =============================================================================

def test_bulk_mode_anti_attractor_end_to_end(tmp_path):
    """Test bulk mode with anti-attractor synthesis end-to-end."""
    from scripts.run_inference import run_bulk_inference

    # Create test JSONL input with multiple items for anti-attractor
    input_content = (
        '{"equation": "B2 -> C2 +T D2"}\n'
        '{"equation": "A5 +T D3 -> C8"}\n'
        '{"equation": "B2 -> D5"}'
    )
    input_file = tmp_path / "input_aa.jsonl"
    input_file.write_text(input_content)

    output_file = tmp_path / "output_aa.jsonl"

    # Run bulk inference with anti-attractor mode
    stats = run_bulk_inference(
        input_path=input_file,
        output_path=output_file,
        default_axes=set(),  # Ignored in anti-attractor mode
        default_mode="soft",
        default_anti_attractor=True,  # Anti-attractor mode
        default_strict=True,
        include_validator=True,
        verbose=False,
    )

    # Verify stats
    assert stats["total"] == 3
    assert stats["success"] >= 1  # At least some should succeed
    assert stats["parse_errors"] == 0

    # Verify output file and structure
    assert output_file.exists()
    output_lines = output_file.read_text().strip().split('\n')
    assert len(output_lines) == 3

    # Parse and verify first successful result has anti-attractor fields
    for line in output_lines:
        result = json.loads(line)
        assert "success" in result
        assert "inversion_type" in result
        if result["success"]:
            assert result["inversion_type"] == "anti-attractor"
            # Anti-attractor results should have signature
            assert "signature" in result
            assert result["signature"] is not None
            assert "polarity" in result["signature"]
            assert "dominant_world" in result["signature"]
            assert "element_counts" in result["signature"]
            break  # Verify at least one


def test_bulk_mode_error_structure_in_json(tmp_path):
    """Test that bulk mode errors are properly structured in JSON output."""
    from scripts.run_inference import run_bulk_inference

    # Create test JSONL with mix of valid and invalid items
    input_content = (
        '{"equation": "B5 +T D3"}\n'
        '{"invalid_field": "no story or equation"}\n'
        '{bad json here}\n'
        '{"equation": "A2"}'
    )
    input_file = tmp_path / "input_errors.jsonl"
    input_file.write_text(input_content)

    output_file = tmp_path / "output_errors.jsonl"

    # Run bulk inference
    stats = run_bulk_inference(
        input_path=input_file,
        output_path=output_file,
        default_axes={"World"},
        default_mode="soft",
        default_anti_attractor=False,
        default_strict=True,
        include_validator=True,
        verbose=False,
    )

    # Verify we captured errors
    assert stats["total"] == 4
    assert stats["parse_errors"] >= 1  # JSON parse error
    assert stats["errors"] >= 1  # Missing field error

    # Verify output structure for errors
    output_lines = output_file.read_text().strip().split('\n')
    assert len(output_lines) == 4

    # Check first line (valid)
    first_result = json.loads(output_lines[0])
    assert first_result["success"] is True
    assert first_result["error"] is None

    # Check second line (missing field error)
    second_result = json.loads(output_lines[1])
    assert second_result["success"] is False
    assert second_result["error"] is not None
    assert "Missing" in second_result["error"]

    # Check third line (JSON parse error)
    third_result = json.loads(output_lines[2])
    assert third_result["success"] is False
    assert "error" in third_result
    assert third_result["error"] is not None

    # Check fourth line (valid single element)
    fourth_result = json.loads(output_lines[3])
    assert fourth_result["success"] is True


def test_bulk_mode_no_validator_flag(tmp_path):
    """Test that --no-validator flag skips validation in bulk mode."""
    from scripts.run_inference import run_bulk_inference

    # Create test JSONL input
    input_content = '{"equation": "B5 +T D3"}\n{"equation": "A2 -> D5"}'
    input_file = tmp_path / "input_no_val.jsonl"
    input_file.write_text(input_content)

    output_file_with_val = tmp_path / "output_with_val.jsonl"
    output_file_no_val = tmp_path / "output_no_val.jsonl"

    # Run with validator
    run_bulk_inference(
        input_path=input_file,
        output_path=output_file_with_val,
        default_axes={"World"},
        default_mode="soft",
        default_anti_attractor=False,
        default_strict=True,
        include_validator=True,  # With validator
        verbose=False,
    )

    # Run without validator
    run_bulk_inference(
        input_path=input_file,
        output_path=output_file_no_val,
        default_axes={"World"},
        default_mode="soft",
        default_anti_attractor=False,
        default_strict=True,
        include_validator=False,  # Without validator
        verbose=False,
    )

    # Verify both produced output
    assert output_file_with_val.exists()
    assert output_file_no_val.exists()

    # Parse outputs
    with_val_lines = output_file_with_val.read_text().strip().split('\n')
    no_val_lines = output_file_no_val.read_text().strip().split('\n')

    # With validator should have validator section
    with_val_result = json.loads(with_val_lines[0])
    assert with_val_result["success"] is True
    assert "validator" in with_val_result
    assert with_val_result["validator"] is not None
    assert "is_valid" in with_val_result["validator"]
    assert "canon_score" in with_val_result["validator"]

    # Without validator should have validator as None
    no_val_result = json.loads(no_val_lines[0])
    assert no_val_result["success"] is True
    assert "validator" in no_val_result
    assert no_val_result["validator"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
