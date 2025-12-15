"""
Smoke tests for TKS Inference HTTP API Server

Tests the serve_inference.py HTTP endpoints to ensure proper functionality.

Usage:
    pytest tests/test_serve_inference.py -v
    pytest tests/test_serve_inference.py::test_health_endpoint -v
"""

import pytest
import json
import threading
import time
from http.server import HTTPServer
from urllib.request import urlopen, Request
from urllib.error import HTTPError
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.serve_inference import TKSInferenceHandler, CANON_CONFIG, SERVER_MODE


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def server():
    """Start test server in background thread."""
    host = "localhost"
    port = 8765  # Use different port to avoid conflicts

    # Create server
    server_address = (host, port)
    httpd = HTTPServer(server_address, TKSInferenceHandler)

    # Start server in background thread
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    # Wait for server to start
    time.sleep(0.5)

    # Yield server info
    yield {
        "host": host,
        "port": port,
        "url": f"http://{host}:{port}",
        "httpd": httpd,
    }

    # Cleanup: shutdown server
    httpd.shutdown()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def make_get_request(url: str) -> dict:
    """
    Make GET request and return JSON response.

    Args:
        url: Full URL to request

    Returns:
        Parsed JSON response

    Raises:
        HTTPError: If request fails
    """
    with urlopen(url) as response:
        data = response.read()
        return json.loads(data.decode('utf-8'))


def make_post_request(url: str, data: dict) -> dict:
    """
    Make POST request with JSON body and return JSON response.

    Args:
        url: Full URL to request
        data: JSON data to send

    Returns:
        Parsed JSON response

    Raises:
        HTTPError: If request fails
    """
    json_data = json.dumps(data).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    req = Request(url, data=json_data, headers=headers, method='POST')

    with urlopen(req) as response:
        response_data = response.read()
        return json.loads(response_data.decode('utf-8'))


# =============================================================================
# HEALTH ENDPOINT TESTS
# =============================================================================

def test_health_endpoint(server):
    """Test GET /health endpoint returns canonical config."""
    url = f"{server['url']}/health"
    response = make_get_request(url)

    # Check response structure
    assert "status" in response
    assert "canon" in response
    assert "mode" in response
    assert "timestamp" in response

    # Check status
    assert response["status"] == "ok"

    # Check canonical config
    canon = response["canon"]
    assert canon["ops"] == 9
    assert canon["worlds"] == 4
    assert canon["noetics"] == 10
    assert canon["foundations"] == 7
    assert canon["subfoundations"] == 28

    # Check mode is set
    assert response["mode"] in ["strict", "lenient"]


def test_health_endpoint_cors(server):
    """Test /health endpoint includes CORS headers."""
    url = f"{server['url']}/health"
    req = Request(url, method='GET')

    with urlopen(req) as response:
        headers = dict(response.headers)
        # Check CORS headers are present
        assert "Access-Control-Allow-Origin" in headers
        assert headers["Access-Control-Allow-Origin"] == "*"


# =============================================================================
# ENCODE ENDPOINT TESTS
# =============================================================================

def test_encode_endpoint_basic(server):
    """Test POST /encode with basic story input."""
    url = f"{server['url']}/encode"
    request_data = {
        "story": "A woman loved a man"
    }

    response = make_post_request(url, request_data)

    # Check response structure
    assert response["success"] is True
    assert "expression" in response
    assert "elements" in response
    assert "ops" in response
    assert "story" in response
    assert "timestamp" in response

    # Check expression is valid
    assert isinstance(response["expression"], str)
    assert len(response["expression"]) > 0

    # Check elements and ops are lists
    assert isinstance(response["elements"], list)
    assert isinstance(response["ops"], list)
    assert len(response["elements"]) > 0

    # Check story matches input
    assert response["story"] == "A woman loved a man"


def test_encode_endpoint_missing_story(server):
    """Test POST /encode with missing 'story' field returns error."""
    url = f"{server['url']}/encode"
    request_data = {}

    try:
        response = make_post_request(url, request_data)
        # Should not reach here
        assert False, "Expected HTTPError"
    except HTTPError as e:
        # Should return 400 Bad Request
        assert e.code == 400
        error_data = json.loads(e.read().decode('utf-8'))
        assert error_data["success"] is False
        assert "error" in error_data


def test_encode_endpoint_invalid_json(server):
    """Test POST /encode with invalid JSON returns error."""
    url = f"{server['url']}/encode"

    # Send invalid JSON
    req = Request(url, data=b"not valid json", headers={'Content-Type': 'application/json'}, method='POST')

    try:
        with urlopen(req) as response:
            pass
        # Should not reach here
        assert False, "Expected HTTPError"
    except HTTPError as e:
        # Should return 400 Bad Request
        assert e.code == 400
        error_data = json.loads(e.read().decode('utf-8'))
        assert error_data["success"] is False
        assert "error" in error_data


# =============================================================================
# INVERT ENDPOINT TESTS
# =============================================================================

def test_invert_endpoint_basic(server):
    """Test POST /invert with basic story input."""
    url = f"{server['url']}/invert"
    request_data = {
        "story": "A woman loved a man",
        "axes": ["W", "N"],
        "mode": "soft"
    }

    response = make_post_request(url, request_data)

    # Check response structure
    assert response["success"] is True
    assert "result" in response
    assert "validator" in response
    assert "timestamp" in response

    # Check result structure
    result = response["result"]
    assert "mode" in result
    assert result["mode"] == "inversion"
    assert "original" in result
    assert "inverted" in result
    assert "explanation" in result

    # Check original
    original = result["original"]
    assert "expression" in original
    assert "elements" in original
    assert "ops" in original
    assert "story" in original

    # Check inverted
    inverted = result["inverted"]
    assert "expression" in inverted
    assert "elements" in inverted
    assert "ops" in inverted
    assert "story" in inverted

    # Check validator
    validator = response["validator"]
    assert "is_valid" in validator
    assert "canon_score" in validator
    assert "error_count" in validator
    assert "warning_count" in validator
    assert "issues" in validator


def test_invert_endpoint_default_axes(server):
    """Test POST /invert with default axes (W, N)."""
    url = f"{server['url']}/invert"
    request_data = {
        "story": "A woman loved a man",
        "strict": False  # Use lenient mode for this test
    }

    response = make_post_request(url, request_data)

    # Should succeed with default axes
    assert response["success"] is True
    assert "result" in response


def test_invert_endpoint_equation_input(server):
    """Test POST /invert with equation input instead of story."""
    url = f"{server['url']}/invert"
    request_data = {
        "equation": "B5 +T D3",
        "axes": ["W"],
        "mode": "soft"
    }

    response = make_post_request(url, request_data)

    # Should succeed
    assert response["success"] is True
    assert "result" in response

    # Check that expression was parsed
    result = response["result"]
    assert "original" in result
    assert "B5" in result["original"]["elements"]


def test_invert_endpoint_invalid_mode(server):
    """Test POST /invert with invalid mode returns error."""
    url = f"{server['url']}/invert"
    request_data = {
        "story": "A woman loved a man",
        "mode": "invalid_mode"
    }

    try:
        response = make_post_request(url, request_data)
        # Should not reach here
        assert False, "Expected HTTPError"
    except HTTPError as e:
        # Should return 400 Bad Request
        assert e.code == 400
        error_data = json.loads(e.read().decode('utf-8'))
        assert error_data["success"] is False
        assert "error" in error_data
        assert "invalid mode" in error_data["error"].lower()


def test_invert_endpoint_missing_input(server):
    """Test POST /invert with missing story/equation returns error."""
    url = f"{server['url']}/invert"
    request_data = {
        "axes": ["W", "N"]
    }

    try:
        response = make_post_request(url, request_data)
        # Should not reach here
        assert False, "Expected HTTPError"
    except HTTPError as e:
        # Should return 400 Bad Request
        assert e.code == 400
        error_data = json.loads(e.read().decode('utf-8'))
        assert error_data["success"] is False
        assert "error" in error_data


# =============================================================================
# ANTI-ATTRACTOR ENDPOINT TESTS
# =============================================================================

def test_anti_attractor_endpoint_basic(server):
    """Test POST /anti-attractor with basic story input."""
    url = f"{server['url']}/anti-attractor"
    request_data = {
        "story": "A woman loved a man",
        "strict": False  # Use lenient mode for this test
    }

    response = make_post_request(url, request_data)

    # Check response structure
    assert response["success"] is True
    assert "result" in response
    assert "validator" in response
    assert "timestamp" in response

    # Check result structure
    result = response["result"]
    assert "mode" in result
    assert result["mode"] == "anti_attractor"
    assert "original" in result
    assert "inverted" in result
    assert "explanation" in result

    # Check signature fields (specific to anti-attractor)
    assert "signature" in result
    assert "inverted_signature" in result

    # Check signature structure
    signature = result["signature"]
    assert "element_counts" in signature
    assert "dominant_world" in signature
    assert "dominant_noetic" in signature
    assert "polarity" in signature
    assert "foundation_tags" in signature

    # Check inverted signature structure
    inv_signature = result["inverted_signature"]
    assert "element_counts" in inv_signature
    assert "dominant_world" in inv_signature
    assert "dominant_noetic" in inv_signature
    assert "polarity" in inv_signature
    assert "foundation_tags" in inv_signature


def test_anti_attractor_endpoint_equation_input(server):
    """Test POST /anti-attractor with equation input."""
    url = f"{server['url']}/anti-attractor"
    request_data = {
        "equation": "B5 +T D3 -> C2"
    }

    response = make_post_request(url, request_data)

    # Should succeed
    assert response["success"] is True
    assert "result" in response

    # Check anti-attractor mode
    result = response["result"]
    assert result["mode"] == "anti_attractor"


def test_anti_attractor_endpoint_missing_input(server):
    """Test POST /anti-attractor with missing story/equation returns error."""
    url = f"{server['url']}/anti-attractor"
    request_data = {}

    try:
        response = make_post_request(url, request_data)
        # Should not reach here
        assert False, "Expected HTTPError"
    except HTTPError as e:
        # Should return 400 Bad Request
        assert e.code == 400
        error_data = json.loads(e.read().decode('utf-8'))
        assert error_data["success"] is False
        assert "error" in error_data


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

def test_invalid_endpoint(server):
    """Test request to invalid endpoint returns 404."""
    url = f"{server['url']}/invalid"

    try:
        response = make_get_request(url)
        # Should not reach here
        assert False, "Expected HTTPError"
    except HTTPError as e:
        # Should return 404 Not Found
        assert e.code == 404
        error_data = json.loads(e.read().decode('utf-8'))
        assert error_data["success"] is False
        assert "error" in error_data


def test_post_to_health_endpoint(server):
    """Test POST to /health endpoint returns error."""
    url = f"{server['url']}/health"
    request_data = {}

    try:
        response = make_post_request(url, request_data)
        # Should not reach here
        assert False, "Expected HTTPError"
    except HTTPError as e:
        # Should return error (either 404 or 405)
        assert e.code in [404, 405]


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_full_pipeline_story_to_inversion(server):
    """Test full pipeline: encode -> invert -> validate."""
    # Step 1: Encode story
    encode_url = f"{server['url']}/encode"
    encode_data = {"story": "A woman loved a man"}
    encode_response = make_post_request(encode_url, encode_data)

    assert encode_response["success"] is True
    expression = encode_response["expression"]

    # Step 2: Invert using expression
    invert_url = f"{server['url']}/invert"
    invert_data = {
        "equation": expression,
        "axes": ["W", "N"],
        "mode": "soft"
    }
    invert_response = make_post_request(invert_url, invert_data)

    assert invert_response["success"] is True
    result = invert_response["result"]

    # Check that original expression matches
    assert result["original"]["expression"] == expression

    # Check that inverted expression is different
    assert result["inverted"]["expression"] != expression

    # Check validator ran
    validator = invert_response["validator"]
    assert "is_valid" in validator


def test_full_pipeline_story_to_anti_attractor(server):
    """Test full pipeline: encode -> anti-attractor -> validate."""
    # Step 1: Encode story
    encode_url = f"{server['url']}/encode"
    encode_data = {"story": "A woman loved a man", "strict": False}
    encode_response = make_post_request(encode_url, encode_data)

    assert encode_response["success"] is True
    expression = encode_response["expression"]

    # Step 2: Anti-attractor synthesis
    aa_url = f"{server['url']}/anti-attractor"
    aa_data = {"equation": expression, "strict": False}
    aa_response = make_post_request(aa_url, aa_data)

    assert aa_response["success"] is True
    result = aa_response["result"]

    # Check that original expression matches
    assert result["original"]["expression"] == expression

    # Check signature fields exist
    assert "signature" in result
    assert "inverted_signature" in result

    # Check validator ran
    validator = aa_response["validator"]
    assert "is_valid" in validator


# =============================================================================
# EDGE CASES
# =============================================================================

def test_empty_story(server):
    """Test encoding empty story returns error."""
    url = f"{server['url']}/encode"
    request_data = {"story": ""}

    try:
        response = make_post_request(url, request_data)
        # May succeed with empty expression or return error
        # Just check it doesn't crash
        assert "success" in response
    except HTTPError:
        # Error is acceptable for empty story
        pass


def test_very_long_story(server):
    """Test encoding very long story doesn't crash server."""
    url = f"{server['url']}/encode"
    long_story = "A woman loved a man. " * 100  # Repeat 100 times
    request_data = {"story": long_story}

    try:
        response = make_post_request(url, request_data)
        # Should succeed or fail gracefully
        assert "success" in response
    except HTTPError as e:
        # Error is acceptable for very long story
        assert e.code in [400, 500]


def test_special_characters_in_story(server):
    """Test encoding story with special characters."""
    url = f"{server['url']}/encode"
    request_data = {"story": "Love & hate; unity + division"}

    try:
        response = make_post_request(url, request_data)
        # Should succeed or fail gracefully
        assert "success" in response
    except HTTPError:
        # Error is acceptable for special characters
        pass


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

def test_concurrent_requests(server):
    """Test server handles multiple concurrent requests."""
    url = f"{server['url']}/health"

    # Make 10 concurrent requests
    import concurrent.futures

    def make_request():
        return make_get_request(url)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(10)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # All requests should succeed
    assert len(results) == 10
    for result in results:
        assert result["status"] == "ok"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
