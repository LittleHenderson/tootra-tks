#!/usr/bin/env python3
"""
TKS Inference HTTP API Server

Serves TKS inference endpoints via HTTP REST API using Python's built-in http.server.

ENDPOINTS
=========

POST /invert
    Accept JSON {"story": "...", "axes": ["W"], "mode": "soft"}
    Return inverted story + expression with validator flags

    Request body:
    {
        "story": "A woman loved a man",       // OR "equation": "B5 +T D3"
        "axes": ["W", "N"],                   // Optional, default: ["W", "N"]
        "mode": "soft",                        // Optional, default: "soft" (soft/hard/targeted)
        "strict": true                         // Optional, default: true
    }

    Response:
    {
        "success": true,
        "result": {
            "mode": "inversion",
            "original": {"expression": "...", "story": "..."},
            "inverted": {"expression": "...", "story": "..."},
            "explanation": "..."
        },
        "validator": {
            "is_valid": true,
            "canon_score": 1.0,
            "error_count": 0,
            "warning_count": 0,
            "issues": []
        }
    }

POST /anti-attractor
    Accept JSON {"story": "..."}
    Return anti-attractor output

    Request body:
    {
        "story": "Power corrupts",            // OR "equation": "B5 +T D3"
        "strict": true                         // Optional, default: true
    }

    Response:
    {
        "success": true,
        "result": {
            "mode": "anti_attractor",
            "original": {"expression": "...", "story": "..."},
            "inverted": {"expression": "...", "story": "..."},
            "explanation": "...",
            "signature": {...},
            "inverted_signature": {...}
        },
        "validator": {...}
    }

POST /encode
    Accept JSON {"story": "..."}
    Return TKS expression

    Request body:
    {
        "story": "A woman loved a man",
        "strict": true                         // Optional, default: true
    }

    Response:
    {
        "success": true,
        "expression": "B5 +T D3",
        "elements": ["B5", "D3"],
        "ops": ["+T"],
        "story": "A woman loved a man"
    }

GET /health
    Return server health status and canonical configuration

    Response:
    {
        "status": "ok",
        "canon": {
            "ops": 9,
            "worlds": 4,
            "noetics": 10,
            "foundations": 7,
            "subfoundations": 28
        },
        "mode": "strict"  // or "lenient"
    }

USAGE
=====

Start server:
    python scripts/serve_inference.py --port 8000
    python scripts/serve_inference.py --port 8080 --lenient
    python scripts/serve_inference.py --port 8000 --strict --log-file server.log

Test with curl:
    # Health check
    curl http://localhost:8000/health

    # Encode story
    curl -X POST http://localhost:8000/encode \\
         -H "Content-Type: application/json" \\
         -d '{"story": "A woman loved a man"}'

    # Invert story
    curl -X POST http://localhost:8000/invert \\
         -H "Content-Type: application/json" \\
         -d '{"story": "A woman loved a man", "axes": ["W", "N"], "mode": "soft"}'

    # Anti-attractor synthesis
    curl -X POST http://localhost:8000/anti-attractor \\
         -H "Content-Type: application/json" \\
         -d '{"story": "Power corrupts"}'

CANONICAL GUARANTEES
====================

This server enforces the following canonical constraints:
- Canon: worlds A/B/C/D; noetics fixed (pairs 2<->3, 5<->6, 8<->9; self-duals 1,4,7,10)
- Foundations 1..7; sub-foundations 7x4 only
- ALLOWED_OPS fixed: +, -, +T, -T, ->, <-, *T, /T, o (9 operators)
- No new symbols/metaphysics. Keep code ASCII, deterministic, type-safe
"""

import argparse
import json
import sys
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse, parse_qs
from datetime import datetime
from collections import defaultdict
from threading import Lock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scenario_inversion import (
    EncodeStory,
    DecodeStory,
    ScenarioInvert,
    ExplainInversion,
    parse_equation,
    TKSExpression,
    AXES_MAP,
)
from anti_attractor import (
    compute_anti_attractor,
    compute_attractor_signature,
    explain_signature,
)
from inversion.engine import TargetProfile
from teacher.validator import CanonicalValidator, ValidationResult

# =============================================================================
# CONFIGURATION
# =============================================================================

# Server defaults
DEFAULT_PORT = 8000
DEFAULT_HOST = "localhost"

# Canonical configuration (fixed per spec)
CANON_CONFIG = {
    "ops": 9,           # +, -, +T, -T, ->, <-, *T, /T, o
    "worlds": 4,        # A, B, C, D
    "noetics": 10,      # 1..10 (pairs: 2<->3, 5<->6, 8<->9; self-duals: 1,4,7,10)
    "foundations": 7,   # 1..7
    "subfoundations": 28,  # 7 foundations x 4 worlds
}

# Global server mode (can be set via CLI)
SERVER_MODE = "strict"  # "strict" or "lenient"

# Security configuration (can be set via CLI)
AUTH_TOKEN = None  # Optional shared secret for authentication
RATE_LIMIT_ENABLED = False  # Enable rate limiting
RATE_LIMIT_REQUESTS = 60  # Max requests per minute per IP
RATE_LIMIT_WINDOW = 60  # Time window in seconds

# Logger configuration
logger = logging.getLogger("tks_inference_server")

# Rate limiting state
rate_limit_store = defaultdict(list)
rate_limit_lock = Lock()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_rate_limit(client_ip: str) -> bool:
    """
    Check if client has exceeded rate limit.

    Args:
        client_ip: Client IP address

    Returns:
        True if request is allowed, False if rate limit exceeded
    """
    if not RATE_LIMIT_ENABLED:
        return True

    current_time = datetime.utcnow().timestamp()

    with rate_limit_lock:
        # Get request timestamps for this IP
        timestamps = rate_limit_store[client_ip]

        # Remove timestamps outside the time window
        timestamps[:] = [ts for ts in timestamps if current_time - ts < RATE_LIMIT_WINDOW]

        # Check if limit exceeded
        if len(timestamps) >= RATE_LIMIT_REQUESTS:
            logger.warning(f"Rate limit exceeded for {client_ip}: {len(timestamps)} requests in {RATE_LIMIT_WINDOW}s")
            return False

        # Add current timestamp
        timestamps.append(current_time)
        return True


def verify_auth_token(headers: dict) -> bool:
    """
    Verify authentication token from request headers.

    Args:
        headers: Request headers

    Returns:
        True if authentication is successful or not required, False otherwise
    """
    if AUTH_TOKEN is None:
        return True  # No authentication required

    # Check for Authorization header
    auth_header = headers.get('Authorization', '').strip()

    if not auth_header:
        logger.warning("Missing Authorization header")
        return False

    # Support both "Bearer <token>" and plain token
    if auth_header.startswith('Bearer '):
        token = auth_header[7:].strip()
    else:
        token = auth_header

    if token != AUTH_TOKEN:
        logger.warning("Invalid authentication token")
        return False

    return True


def parse_axes(axes_list: List[str]) -> set:
    """
    Parse list of axes strings into set of axis names.

    Args:
        axes_list: List of axis codes (e.g., ["W", "N"] or ["World", "Noetic"])

    Returns:
        Set of normalized axis names
    """
    if not axes_list:
        return set()

    axes = set()
    for a in axes_list:
        a = a.strip()
        a_upper = a.upper()

        # Try letter code first (e.g., "W" -> "World")
        if a_upper in AXES_MAP:
            axes.add(AXES_MAP[a_upper])
        # Try full name with proper capitalization
        elif a.capitalize() in AXES_MAP.values():
            axes.add(a.capitalize())
        elif a in AXES_MAP.values():
            axes.add(a)

    return axes


def format_expression(expr: TKSExpression) -> str:
    """
    Format TKSExpression for display.

    Args:
        expr: TKS expression to format

    Returns:
        Formatted string (e.g., "B5 +T D3 -> C2")
    """
    parts = []
    for i, elem in enumerate(expr.elements):
        parts.append(elem)
        if i < len(expr.ops):
            parts.append(expr.ops[i])
    return " ".join(parts)


def run_inference(
    input_text: str,
    is_equation: bool,
    anti_attractor: bool,
    axes: set,
    mode: str,
    strict: bool,
    target: Optional[TargetProfile] = None
) -> Dict[str, Any]:
    """
    Execute the complete inference pipeline.

    Pipeline stages:
    1. Encode input to TKS expression
    2. Apply inversion or anti-attractor synthesis
    3. Decode to natural language

    Args:
        input_text: Natural language story or TKS equation
        is_equation: True if input is equation, False if story
        anti_attractor: Enable anti-attractor synthesis
        axes: Set of axes for inversion
        mode: Inversion mode (soft/hard/targeted)
        strict: Strict mode for encoding
        target: Optional target profile for targeted mode

    Returns:
        Dict with original_expr, inverted_expr, original_story, inverted_story,
        explanation, and optionally signature

    Raises:
        ValueError: If strict=True and unknown tokens detected
    """
    # Stage 1: Encode input to TKS expression
    if is_equation:
        expr = parse_equation(input_text)
    else:
        expr = EncodeStory(input_text, strict=strict)

    # Stage 2: Apply inversion or anti-attractor
    if anti_attractor:
        # Anti-attractor synthesis mode
        orig_sig, inv_sig, inverted_expr = compute_anti_attractor(expr)

        # Generate explanation
        explanation = explain_signature(orig_sig, "Original") + "\n\n" + \
                     explain_signature(inv_sig, "Anti-Attractor")

        result = {
            'original_expr': expr,
            'inverted_expr': inverted_expr,
            'signature': orig_sig,
            'inverted_signature': inv_sig,
            'explanation': explanation,
        }
    else:
        # Standard multi-axis inversion mode
        inverted_expr = ScenarioInvert(expr, axes, mode, target)
        explanation = ExplainInversion(expr, inverted_expr)

        result = {
            'original_expr': expr,
            'inverted_expr': inverted_expr,
            'explanation': explanation,
        }

    # Stage 3: Decode to natural language
    original_story = DecodeStory(expr)
    inverted_story = DecodeStory(inverted_expr)

    result['original_story'] = original_story
    result['inverted_story'] = inverted_story

    return result


def format_json_response(
    result: Dict[str, Any],
    anti_attractor_mode: bool,
    validator_result: Optional[ValidationResult] = None,
) -> Dict[str, Any]:
    """
    Format inference results as JSON API response.

    Args:
        result: Inference pipeline result
        anti_attractor_mode: Whether anti-attractor synthesis was used
        validator_result: Optional validation result

    Returns:
        JSON-serializable dict
    """
    output = {
        "mode": "anti_attractor" if anti_attractor_mode else "inversion",
        "original": {
            "expression": format_expression(result['original_expr']),
            "elements": result['original_expr'].elements,
            "ops": result['original_expr'].ops,
            "story": result['original_story'],
        },
        "inverted": {
            "expression": format_expression(result['inverted_expr']),
            "elements": result['inverted_expr'].elements,
            "ops": result['inverted_expr'].ops,
            "story": result['inverted_story'],
        },
        "explanation": result['explanation'],
    }

    # Add signature info if anti-attractor mode
    if anti_attractor_mode and 'signature' in result:
        sig = result['signature']
        output["signature"] = {
            "element_counts": {f"{w}{n}": c for (w, n), c in sig.element_counts.items()},
            "dominant_world": sig.dominant_world,
            "dominant_noetic": sig.dominant_noetic,
            "polarity": sig.polarity,
            "foundation_tags": sorted(sig.foundation_tags),
        }

        # Add inverted signature if available
        if 'inverted_signature' in result:
            inv_sig = result['inverted_signature']
            output["inverted_signature"] = {
                "element_counts": {f"{w}{n}": c for (w, n), c in inv_sig.element_counts.items()},
                "dominant_world": inv_sig.dominant_world,
                "dominant_noetic": inv_sig.dominant_noetic,
                "polarity": inv_sig.polarity,
                "foundation_tags": sorted(inv_sig.foundation_tags),
            }

    # Add validator section if provided
    if validator_result is not None:
        output["validator"] = {
            "is_valid": validator_result.is_valid,
            "canon_score": validator_result.canon_score,
            "error_count": validator_result.error_count,
            "warning_count": validator_result.warning_count,
            "issues": [
                {
                    "rule": issue.rule,
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "location": issue.location,
                    "suggestion": issue.suggestion,
                }
                for issue in validator_result.issues
            ] if validator_result.issues else [],
        }

    return output


# =============================================================================
# HTTP REQUEST HANDLER
# =============================================================================

class TKSInferenceHandler(BaseHTTPRequestHandler):
    """HTTP request handler for TKS inference API."""

    def log_message(self, format, *args):
        """Override to use logger instead of stderr."""
        logger.info("%s - - [%s] %s" % (
            self.address_string(),
            self.log_date_time_string(),
            format % args
        ))

    def _check_security(self) -> bool:
        """
        Check rate limiting and authentication.

        Returns:
            True if request passes security checks, False otherwise
        """
        # Check rate limit
        client_ip = self.client_address[0]
        if not check_rate_limit(client_ip):
            self._send_error_json("Rate limit exceeded. Please try again later.", 429)
            return False

        # Check authentication
        if not verify_auth_token(self.headers):
            self._send_error_json("Unauthorized: Invalid or missing authentication token", 401)
            return False

        return True

    def _set_headers(self, status_code: int = 200, content_type: str = "application/json"):
        """Set HTTP response headers."""
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")  # CORS
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _send_json(self, data: Dict[str, Any], status_code: int = 200):
        """Send JSON response."""
        self._set_headers(status_code)
        response = json.dumps(data, indent=2, ensure_ascii=False)
        self.wfile.write(response.encode('utf-8'))

        # Log response
        logger.info(f"Response ({status_code}): {json.dumps(data, ensure_ascii=False)}")

    def _send_error_json(self, message: str, status_code: int = 400):
        """Send JSON error response."""
        self._send_json({
            "success": False,
            "error": message,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }, status_code)

    def _read_json_body(self) -> Optional[Dict[str, Any]]:
        """Read and parse JSON request body."""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                return None

            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            # Log request
            logger.info(f"Request body: {json.dumps(data, ensure_ascii=False)}")

            return data
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error reading request body: {e}")
            return None

    def do_OPTIONS(self):
        """Handle OPTIONS request for CORS preflight."""
        self._set_headers(204)

    def do_GET(self):
        """Handle GET requests."""
        # Security checks
        if not self._check_security():
            return

        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path == "/health":
            self._handle_health()
        else:
            self._send_error_json(f"Endpoint not found: {path}", 404)

    def do_POST(self):
        """Handle POST requests."""
        # Security checks
        if not self._check_security():
            return

        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path == "/invert":
            self._handle_invert()
        elif path == "/anti-attractor":
            self._handle_anti_attractor()
        elif path == "/encode":
            self._handle_encode()
        else:
            self._send_error_json(f"Endpoint not found: {path}", 404)

    def _handle_health(self):
        """Handle GET /health endpoint."""
        response = {
            "status": "ok",
            "canon": CANON_CONFIG,
            "mode": SERVER_MODE,
            "security": {
                "auth_required": AUTH_TOKEN is not None,
                "rate_limit_enabled": RATE_LIMIT_ENABLED,
                "rate_limit_requests": RATE_LIMIT_REQUESTS if RATE_LIMIT_ENABLED else None,
                "rate_limit_window": RATE_LIMIT_WINDOW if RATE_LIMIT_ENABLED else None,
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        self._send_json(response)

    def _handle_encode(self):
        """Handle POST /encode endpoint."""
        data = self._read_json_body()

        if data is None:
            self._send_error_json("Invalid JSON body", 400)
            return

        # Extract parameters
        story = data.get("story")
        if not story:
            self._send_error_json("Missing 'story' field in request body", 400)
            return

        strict = data.get("strict", SERVER_MODE == "strict")

        try:
            # Encode story to TKS expression
            expr = EncodeStory(story, strict=strict)

            response = {
                "success": True,
                "expression": format_expression(expr),
                "elements": expr.elements,
                "ops": expr.ops,
                "story": story,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

            self._send_json(response)

        except ValueError as e:
            self._send_error_json(f"Encoding error: {str(e)}", 400)
        except Exception as e:
            logger.error(f"Unexpected error in /encode: {e}", exc_info=True)
            self._send_error_json(f"Internal server error: {str(e)}", 500)

    def _handle_invert(self):
        """Handle POST /invert endpoint."""
        data = self._read_json_body()

        if data is None:
            self._send_error_json("Invalid JSON body", 400)
            return

        # Extract parameters
        story = data.get("story")
        equation = data.get("equation")

        if not story and not equation:
            self._send_error_json("Missing 'story' or 'equation' field in request body", 400)
            return

        is_equation = equation is not None
        input_text = equation if is_equation else story

        # Parse axes
        axes_list = data.get("axes", ["W", "N"])
        axes = parse_axes(axes_list)
        if not axes:
            axes = {"World", "Noetic"}  # Default

        mode = data.get("mode", "soft")
        if mode not in ["soft", "hard", "targeted"]:
            self._send_error_json(f"Invalid mode: {mode}. Must be 'soft', 'hard', or 'targeted'", 400)
            return

        strict = data.get("strict", SERVER_MODE == "strict")

        # Build target profile if specified
        target = None
        if mode == "targeted":
            target = TargetProfile(
                enable=bool(data.get("from_foundation") or data.get("from_world")),
                from_foundation=data.get("from_foundation"),
                to_foundation=data.get("to_foundation"),
                from_world=data.get("from_world"),
                to_world=data.get("to_world"),
            )

        try:
            # Run inference
            result = run_inference(
                input_text=input_text,
                is_equation=is_equation,
                anti_attractor=False,
                axes=axes,
                mode=mode,
                strict=strict,
                target=target,
            )

            # Run validator
            validator = CanonicalValidator(strict_mode=strict)
            inverted_story = result.get('inverted_story', '')
            inverted_expr = format_expression(result['inverted_expr'])
            validation_text = f"{inverted_expr}\n{inverted_story}"
            validator_result = validator.validate(validation_text)

            # Format response
            response_data = format_json_response(
                result,
                anti_attractor_mode=False,
                validator_result=validator_result,
            )

            response = {
                "success": True,
                "result": response_data,
                "validator": response_data.pop("validator", None),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

            self._send_json(response)

        except ValueError as e:
            self._send_error_json(f"Validation error: {str(e)}", 400)
        except Exception as e:
            logger.error(f"Unexpected error in /invert: {e}", exc_info=True)
            self._send_error_json(f"Internal server error: {str(e)}", 500)

    def _handle_anti_attractor(self):
        """Handle POST /anti-attractor endpoint."""
        data = self._read_json_body()

        if data is None:
            self._send_error_json("Invalid JSON body", 400)
            return

        # Extract parameters
        story = data.get("story")
        equation = data.get("equation")

        if not story and not equation:
            self._send_error_json("Missing 'story' or 'equation' field in request body", 400)
            return

        is_equation = equation is not None
        input_text = equation if is_equation else story

        strict = data.get("strict", SERVER_MODE == "strict")

        try:
            # Run inference with anti-attractor mode
            result = run_inference(
                input_text=input_text,
                is_equation=is_equation,
                anti_attractor=True,
                axes=set(),  # Not used in anti-attractor mode
                mode="soft",  # Not used in anti-attractor mode
                strict=strict,
                target=None,
            )

            # Run validator
            validator = CanonicalValidator(strict_mode=strict)
            inverted_story = result.get('inverted_story', '')
            inverted_expr = format_expression(result['inverted_expr'])
            validation_text = f"{inverted_expr}\n{inverted_story}"
            validator_result = validator.validate(validation_text)

            # Format response
            response_data = format_json_response(
                result,
                anti_attractor_mode=True,
                validator_result=validator_result,
            )

            response = {
                "success": True,
                "result": response_data,
                "validator": response_data.pop("validator", None),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

            self._send_json(response)

        except ValueError as e:
            self._send_error_json(f"Validation error: {str(e)}", 400)
        except Exception as e:
            logger.error(f"Unexpected error in /anti-attractor: {e}", exc_info=True)
            self._send_error_json(f"Internal server error: {str(e)}", 500)


# =============================================================================
# SERVER STARTUP
# =============================================================================

def run_server(host: str, port: int, log_file: Optional[str] = None, verbose: bool = False):
    """
    Start the TKS inference HTTP server.

    Args:
        host: Host address to bind to
        port: Port number to listen on
        log_file: Optional log file path
        verbose: Enable verbose logging
    """
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    if log_file:
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[logging.StreamHandler(sys.stdout)]
        )

    # Create server
    server_address = (host, port)
    httpd = HTTPServer(server_address, TKSInferenceHandler)

    logger.info("=" * 70)
    logger.info("TKS INFERENCE HTTP API SERVER")
    logger.info("=" * 70)
    logger.info(f"Server mode: {SERVER_MODE}")
    logger.info(f"Canonical config: {CANON_CONFIG}")
    logger.info(f"Listening on http://{host}:{port}")
    logger.info("")
    logger.info("Security Settings:")
    logger.info(f"  Authentication: {'ENABLED' if AUTH_TOKEN else 'DISABLED'}")
    logger.info(f"  Rate limiting: {'ENABLED' if RATE_LIMIT_ENABLED else 'DISABLED'}")
    if RATE_LIMIT_ENABLED:
        logger.info(f"  Rate limit: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW}s per IP")
    logger.info(f"  Verbose logging: {'ENABLED' if verbose else 'DISABLED'}")
    logger.info("")
    logger.info("Available endpoints:")
    logger.info("  GET  /health           - Server health check")
    logger.info("  POST /encode           - Encode story to TKS expression")
    logger.info("  POST /invert           - Invert story/expression")
    logger.info("  POST /anti-attractor   - Anti-attractor synthesis")
    logger.info("")
    logger.info("Press Ctrl+C to stop server")
    logger.info("=" * 70)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("\nShutting down server...")
        httpd.shutdown()
        logger.info("Server stopped.")


# =============================================================================
# CLI ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="TKS Inference HTTP API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server on default port (8000)
  python scripts/serve_inference.py

  # Start server on custom port
  python scripts/serve_inference.py --port 8080

  # Start server in lenient mode
  python scripts/serve_inference.py --lenient

  # Start server with logging to file
  python scripts/serve_inference.py --log-file server.log

  # Test with curl
  curl http://localhost:8000/health
  curl -X POST http://localhost:8000/encode \\
       -H "Content-Type: application/json" \\
       -d '{"story": "A woman loved a man"}'
        """
    )

    p.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"Host address to bind to (default: {DEFAULT_HOST})"
    )
    p.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port number to listen on (default: {DEFAULT_PORT})"
    )

    # Mode selector (mutually exclusive)
    mode_group = p.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--strict",
        action="store_true",
        help="Use strict mode for unknown tokens (default)"
    )
    mode_group.add_argument(
        "--lenient",
        action="store_true",
        help="Use lenient mode for unknown tokens"
    )

    p.add_argument(
        "--log-file",
        type=str,
        dest="log_file",
        help="Log file path (logs to stdout by default)"
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose/debug logging"
    )

    # Security options
    security_group = p.add_argument_group('security options')
    security_group.add_argument(
        "--auth-token",
        type=str,
        dest="auth_token",
        help="Shared secret token for authentication (optional). Clients must send this in Authorization header."
    )
    security_group.add_argument(
        "--rate-limit",
        type=int,
        dest="rate_limit",
        metavar="N",
        help="Enable rate limiting: max N requests per minute per IP (e.g., --rate-limit 60)"
    )
    security_group.add_argument(
        "--rate-limit-window",
        type=int,
        dest="rate_limit_window",
        default=60,
        metavar="SECONDS",
        help="Rate limit time window in seconds (default: 60)"
    )

    return p.parse_args()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main CLI entry point."""
    global SERVER_MODE, AUTH_TOKEN, RATE_LIMIT_ENABLED, RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW

    args = parse_args()

    # Set server mode
    if args.lenient:
        SERVER_MODE = "lenient"
    else:
        SERVER_MODE = "strict"

    # Set authentication token
    if args.auth_token:
        AUTH_TOKEN = args.auth_token
        logger.info("Authentication enabled")

    # Set rate limiting
    if args.rate_limit:
        RATE_LIMIT_ENABLED = True
        RATE_LIMIT_REQUESTS = args.rate_limit
        RATE_LIMIT_WINDOW = args.rate_limit_window
        logger.info(f"Rate limiting enabled: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW}s")

    # Start server
    try:
        run_server(args.host, args.port, args.log_file, args.verbose)
    except Exception as e:
        print(f"ERROR: Failed to start server: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
