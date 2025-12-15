# TKS Inference HTTP API Documentation

## Overview

The TKS Inference HTTP API provides RESTful endpoints for encoding natural language stories to TKS expressions, performing scenario inversions, and anti-attractor synthesis. The API is served via `scripts/serve_inference.py` using Python's built-in HTTP server.

## Quick Start

### Starting the Server

```bash
# Start server on default port (8000)
python scripts/serve_inference.py

# Start server on custom port
python scripts/serve_inference.py --port 8080

# Start server in lenient mode (allows unknown tokens)
python scripts/serve_inference.py --lenient

# Start server with logging to file
python scripts/serve_inference.py --log-file server.log

# Start server with verbose logging
python scripts/serve_inference.py --verbose

# Start server with authentication (shared secret token)
python scripts/serve_inference.py --auth-token "my-secret-token-123"

# Start server with rate limiting (60 requests per minute per IP)
python scripts/serve_inference.py --rate-limit 60

# Start server with rate limiting (custom window: 100 requests per 120 seconds)
python scripts/serve_inference.py --rate-limit 100 --rate-limit-window 120

# Start server with all security features enabled
python scripts/serve_inference.py --auth-token "my-secret-token-123" --rate-limit 60 --verbose
```

### Testing the Server

```bash
# Health check
curl http://localhost:8000/health

# Encode a story
curl -X POST http://localhost:8000/encode \
     -H "Content-Type: application/json" \
     -d '{"story": "A woman loved a man"}'

# Invert a story
curl -X POST http://localhost:8000/invert \
     -H "Content-Type: application/json" \
     -d '{"story": "A woman loved a man", "axes": ["W", "N"], "mode": "soft"}'

# Anti-attractor synthesis
curl -X POST http://localhost:8000/anti-attractor \
     -H "Content-Type: application/json" \
     -d '{"story": "Power corrupts"}'

# With authentication (if enabled with --auth-token)
curl -X POST http://localhost:8000/encode \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer my-secret-token-123" \
     -d '{"story": "A woman loved a man"}'

# Alternative: plain token (without "Bearer ")
curl -X POST http://localhost:8000/encode \
     -H "Content-Type: application/json" \
     -H "Authorization: my-secret-token-123" \
     -d '{"story": "A woman loved a man"}'
```

## Endpoints

### GET /health

Returns server health status and canonical configuration.

**Request:**
```
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "canon": {
    "ops": 9,
    "worlds": 4,
    "noetics": 10,
    "foundations": 7,
    "subfoundations": 28
  },
  "mode": "strict",
  "security": {
    "auth_required": true,
    "rate_limit_enabled": true,
    "rate_limit_requests": 60,
    "rate_limit_window": 60
  },
  "timestamp": "2025-12-14T10:30:00.000000Z"
}
```

**Canonical Configuration:**
- **ops**: 9 operators (`+`, `-`, `+T`, `-T`, `->`, `<-`, `*T`, `/T`, `o`)
- **worlds**: 4 worlds (`A`, `B`, `C`, `D`)
- **noetics**: 10 noetics (1..10, pairs 2<->3, 5<->6, 8<->9; self-duals 1,4,7,10)
- **foundations**: 7 foundations (1..7)
- **subfoundations**: 28 subfoundations (7 foundations x 4 worlds)

### POST /encode

Encodes a natural language story to a TKS expression.

**Request:**
```json
{
  "story": "A woman loved a man",
  "strict": true  // Optional, default: true
}
```

**Response (Success):**
```json
{
  "success": true,
  "expression": "B5 +T D3",
  "elements": ["B5", "D3"],
  "ops": ["+T"],
  "story": "A woman loved a man",
  "timestamp": "2025-12-14T10:30:00.000000Z"
}
```

**Response (Error):**
```json
{
  "success": false,
  "error": "Encoding error: Unknown token 'xyz'",
  "timestamp": "2025-12-14T10:30:00.000000Z"
}
```

**Parameters:**
- `story` (required): Natural language story to encode
- `strict` (optional): If `true` (default), reject unknown tokens. If `false`, skip unknown tokens with warnings.

**Status Codes:**
- `200 OK`: Encoding successful
- `400 Bad Request`: Invalid input or encoding error
- `500 Internal Server Error`: Unexpected server error

### POST /invert

Performs scenario inversion on a story or TKS equation.

**Request:**
```json
{
  "story": "A woman loved a man",  // OR "equation": "B5 +T D3"
  "axes": ["W", "N"],               // Optional, default: ["W", "N"]
  "mode": "soft",                   // Optional, default: "soft"
  "strict": true                    // Optional, default: true
}
```

**Response (Success):**
```json
{
  "success": true,
  "result": {
    "mode": "inversion",
    "original": {
      "expression": "B5 +T D3",
      "elements": ["B5", "D3"],
      "ops": ["+T"],
      "story": "A woman loved a man"
    },
    "inverted": {
      "expression": "C6 +T A2",
      "elements": ["C6", "A2"],
      "ops": ["+T"],
      "story": "A man hated a woman"
    },
    "explanation": "Inverted World axis: B->C, D->A. Inverted Noetic axis: 5->6, 3->2."
  },
  "validator": {
    "is_valid": true,
    "canon_score": 1.0,
    "error_count": 0,
    "warning_count": 0,
    "issues": []
  },
  "timestamp": "2025-12-14T10:30:00.000000Z"
}
```

**Parameters:**
- `story` or `equation` (required): Input to invert (provide one or the other)
- `axes` (optional): List of axes to invert. Default: `["W", "N"]`
  - `"N"` = Noetic (involution pairs: 2<->3, 5<->6, 8<->9)
  - `"E"` = Element (full element inversion: world + noetic)
  - `"W"` = World (world mirror: A<->D, B<->C)
  - `"F"` = Foundation (1<->7, 2<->6, 3<->5, 4 self-dual)
  - `"S"` = SubFoundation (foundation + world compound)
  - `"A"` = Acquisition (negation toggle)
  - `"P"` = Polarity (valence flip)
- `mode` (optional): Inversion mode. Default: `"soft"`
  - `"soft"`: Invert only where canonical dual/opposite exists
  - `"hard"`: Apply on all selected axes unconditionally
  - `"targeted"`: Apply TargetProfile remaps; others unchanged
- `strict` (optional): Strict encoding mode. Default: `true`

**Targeted Mode Parameters** (only used when `mode` = `"targeted"`):
- `from_foundation` (optional): Source foundation for targeted remap (1-7)
- `to_foundation` (optional): Target foundation for targeted remap (1-7)
- `from_world` (optional): Source world for targeted remap (A/B/C/D)
- `to_world` (optional): Target world for targeted remap (A/B/C/D)

**Status Codes:**
- `200 OK`: Inversion successful
- `400 Bad Request`: Invalid input or validation error
- `500 Internal Server Error`: Unexpected server error

### POST /anti-attractor

Performs anti-attractor synthesis on a story or TKS equation.

**Request:**
```json
{
  "story": "Power corrupts",  // OR "equation": "B5 +T D3"
  "strict": true               // Optional, default: true
}
```

**Response (Success):**
```json
{
  "success": true,
  "result": {
    "mode": "anti_attractor",
    "original": {
      "expression": "B5 +T D3",
      "elements": ["B5", "D3"],
      "ops": ["+T"],
      "story": "Power corrupts"
    },
    "inverted": {
      "expression": "C6 -> A2",
      "elements": ["C6", "A2"],
      "ops": ["->"],
      "story": "Wisdom liberates"
    },
    "explanation": "Original Signature:\n- Dominant: B5 (Mental-Female)\n- Polarity: +1 (Positive)\n\nAnti-Attractor Signature:\n- Dominant: C6 (Emotional-Male)\n- Polarity: -1 (Negative)",
    "signature": {
      "element_counts": {"B5": 1, "D3": 1},
      "dominant_world": "B",
      "dominant_noetic": 5,
      "polarity": 1,
      "foundation_tags": [5]
    },
    "inverted_signature": {
      "element_counts": {"C6": 1, "A2": 1},
      "dominant_world": "C",
      "dominant_noetic": 6,
      "polarity": -1,
      "foundation_tags": [2]
    }
  },
  "validator": {
    "is_valid": true,
    "canon_score": 1.0,
    "error_count": 0,
    "warning_count": 0,
    "issues": []
  },
  "timestamp": "2025-12-14T10:30:00.000000Z"
}
```

**Parameters:**
- `story` or `equation` (required): Input for anti-attractor synthesis
- `strict` (optional): Strict encoding mode. Default: `true`

**Status Codes:**
- `200 OK`: Anti-attractor synthesis successful
- `400 Bad Request`: Invalid input or validation error
- `500 Internal Server Error`: Unexpected server error

## Error Handling

All endpoints return consistent error responses:

```json
{
  "success": false,
  "error": "Error message describing what went wrong",
  "timestamp": "2025-12-14T10:30:00.000000Z"
}
```

**Common Error Scenarios:**

1. **Missing required field:**
   ```json
   {
     "success": false,
     "error": "Missing 'story' or 'equation' field in request body",
     "timestamp": "2025-12-14T10:30:00.000000Z"
   }
   ```

2. **Invalid JSON:**
   ```json
   {
     "success": false,
     "error": "Invalid JSON body",
     "timestamp": "2025-12-14T10:30:00.000000Z"
   }
   ```

3. **Unknown tokens in strict mode:**
   ```json
   {
     "success": false,
     "error": "Encoding error: Unknown token 'xyz'",
     "timestamp": "2025-12-14T10:30:00.000000Z"
   }
   ```

4. **Invalid mode:**
   ```json
   {
     "success": false,
     "error": "Invalid mode: xyz. Must be 'soft', 'hard', or 'targeted'",
     "timestamp": "2025-12-14T10:30:00.000000Z"
   }
   ```

5. **Endpoint not found:**
   ```json
   {
     "success": false,
     "error": "Endpoint not found: /invalid",
     "timestamp": "2025-12-14T10:30:00.000000Z"
   }
   ```

## Validation

All inversion and anti-attractor responses include a `validator` section with canonical validation results:

```json
{
  "validator": {
    "is_valid": true,
    "canon_score": 1.0,
    "error_count": 0,
    "warning_count": 0,
    "issues": [
      {
        "rule": "RULE_NAME",
        "severity": "error",  // or "warning"
        "message": "Description of the issue",
        "location": "Location in the expression",
        "suggestion": "How to fix it"
      }
    ]
  }
}
```

**Validator Fields:**
- `is_valid`: Boolean indicating if the output passes all canonical rules
- `canon_score`: Float score (0.0 to 1.0) representing canonical compliance
- `error_count`: Number of validation errors
- `warning_count`: Number of validation warnings
- `issues`: List of validation issues (errors and warnings)

## Server Modes

### Strict Mode (Default)

```bash
python scripts/serve_inference.py --strict
```

In strict mode:
- Unknown tokens in stories will cause encoding to fail with `400 Bad Request`
- All canonical rules are enforced strictly
- Recommended for production use

### Lenient Mode

```bash
python scripts/serve_inference.py --lenient
```

In lenient mode:
- Unknown tokens are skipped with warnings (encoding continues)
- Some canonical rules are relaxed
- Useful for development and testing

You can override the server mode on a per-request basis using the `strict` parameter:

```json
{
  "story": "Some text with unknown words",
  "strict": false  // Override server mode for this request
}
```

## Security Features

### Authentication

The server supports optional token-based authentication using a shared secret. When enabled, all requests must include the authentication token in the `Authorization` header.

**Enable authentication:**
```bash
python scripts/serve_inference.py --auth-token "your-secret-token-here"
```

**Making authenticated requests:**
```bash
# Using Bearer token format (recommended)
curl -X POST http://localhost:8000/encode \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your-secret-token-here" \
     -d '{"story": "A woman loved a man"}'

# Using plain token format
curl -X POST http://localhost:8000/encode \
     -H "Content-Type: application/json" \
     -H "Authorization: your-secret-token-here" \
     -d '{"story": "A woman loved a man"}'
```

**Error responses:**

Unauthorized (missing or invalid token):
```json
{
  "success": false,
  "error": "Unauthorized: Invalid or missing authentication token",
  "timestamp": "2025-12-14T10:30:00.000000Z"
}
```

Status code: `401 Unauthorized`

### Rate Limiting

The server supports per-IP rate limiting to prevent abuse. When enabled, each IP address is limited to a maximum number of requests within a time window.

**Enable rate limiting:**
```bash
# Limit to 60 requests per minute (60 seconds) per IP
python scripts/serve_inference.py --rate-limit 60

# Custom time window: 100 requests per 120 seconds
python scripts/serve_inference.py --rate-limit 100 --rate-limit-window 120
```

**Rate limit behavior:**
- Requests are counted per IP address
- Old requests outside the time window are automatically removed
- When limit is exceeded, requests return `429 Too Many Requests`
- The rate limit resets as the time window slides

**Error responses:**

Rate limit exceeded:
```json
{
  "success": false,
  "error": "Rate limit exceeded. Please try again later.",
  "timestamp": "2025-12-14T10:30:00.000000Z"
}
```

Status code: `429 Too Many Requests`

**Check rate limit status:**

The `/health` endpoint includes rate limit configuration:
```json
{
  "security": {
    "auth_required": false,
    "rate_limit_enabled": true,
    "rate_limit_requests": 60,
    "rate_limit_window": 60
  }
}
```

### Verbose Logging

Enable verbose logging to track all requests, security events, and errors:

```bash
python scripts/serve_inference.py --verbose
```

Verbose mode logs:
- All incoming requests with client IP
- Request bodies (JSON payloads)
- Response bodies
- Rate limit violations
- Authentication failures
- Detailed error traces

**Example log output:**
```
2025-12-14 10:30:00,123 - tks_inference_server - INFO - 127.0.0.1 - - [14/Dec/2025 10:30:00] "POST /encode HTTP/1.1" 200 -
2025-12-14 10:30:00,124 - tks_inference_server - INFO - Request body: {"story": "A woman loved a man"}
2025-12-14 10:30:00,234 - tks_inference_server - INFO - Response (200): {"success": true, "expression": "B5 +T D3", ...}
2025-12-14 10:30:01,456 - tks_inference_server - WARNING - Rate limit exceeded for 192.168.1.100: 61 requests in 60s
2025-12-14 10:30:02,789 - tks_inference_server - WARNING - Invalid authentication token
```

### Combined Security Setup

For production deployments, combine all security features:

```bash
python scripts/serve_inference.py \
    --auth-token "$(openssl rand -hex 32)" \
    --rate-limit 60 \
    --rate-limit-window 60 \
    --verbose \
    --log-file /var/log/tks/server.log \
    --strict
```

This configuration:
- Requires authentication for all requests
- Limits to 60 requests per minute per IP
- Enables verbose logging to file and stdout
- Uses strict mode for canonical validation

## CORS Support

The API includes CORS headers to allow cross-origin requests:

```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, OPTIONS
Access-Control-Allow-Headers: Content-Type
```

This allows the API to be called from web applications running on different domains.

## Logging

### Console Logging (Default)

By default, all requests and responses are logged to stdout:

```
2025-12-14 10:30:00,000 - tks_inference_server - INFO - Request body: {"story": "A woman loved a man"}
2025-12-14 10:30:00,100 - tks_inference_server - INFO - Response (200): {"success": true, "expression": "B5 +T D3", ...}
```

### File Logging

Enable file logging with the `--log-file` option:

```bash
python scripts/serve_inference.py --log-file server.log
```

Logs will be written to both the file and stdout.

## Testing

Run the smoke tests to verify the API is working correctly:

```bash
# Run all tests
pytest tests/test_serve_inference.py -v

# Run specific test
pytest tests/test_serve_inference.py::test_health_endpoint -v

# Run with coverage
pytest tests/test_serve_inference.py --cov=scripts.serve_inference --cov-report=term-missing
```

## Examples

### Example 1: Simple Story Encoding

```bash
curl -X POST http://localhost:8000/encode \
     -H "Content-Type: application/json" \
     -d '{"story": "Unity brings peace"}'
```

Response:
```json
{
  "success": true,
  "expression": "A1 -> D4",
  "elements": ["A1", "D4"],
  "ops": ["->"],
  "story": "Unity brings peace"
}
```

### Example 2: World Axis Inversion

```bash
curl -X POST http://localhost:8000/invert \
     -H "Content-Type: application/json" \
     -d '{
       "story": "A woman loved a man",
       "axes": ["W"],
       "mode": "soft"
     }'
```

Response shows World axis inversion (B<->C, A<->D).

### Example 3: Multi-Axis Inversion

```bash
curl -X POST http://localhost:8000/invert \
     -H "Content-Type: application/json" \
     -d '{
       "equation": "B5 +T D3",
       "axes": ["W", "N"],
       "mode": "soft"
     }'
```

Response shows both World and Noetic axis inversions.

### Example 4: Anti-Attractor Synthesis

```bash
curl -X POST http://localhost:8000/anti-attractor \
     -H "Content-Type: application/json" \
     -d '{"story": "Power corrupts absolutely"}'
```

Response includes attractor signature analysis and synthesized anti-attractor.

### Example 5: Targeted Foundation Remap

```bash
curl -X POST http://localhost:8000/invert \
     -H "Content-Type: application/json" \
     -d '{
       "story": "Power corrupts",
       "axes": ["F"],
       "mode": "targeted",
       "from_foundation": 5,
       "to_foundation": 2
     }'
```

Response shows Foundation 5 (Power) remapped to Foundation 2 (Wisdom).

### Example 6: Lenient Mode for Unknown Tokens

```bash
curl -X POST http://localhost:8000/encode \
     -H "Content-Type: application/json" \
     -d '{
       "story": "Some unknown words here",
       "strict": false
     }'
```

Response will encode known tokens and skip unknown ones.

## Integration Examples

### Python Client

```python
import requests
import json

# Server URL
BASE_URL = "http://localhost:8000"

# Encode a story
def encode_story(story):
    response = requests.post(
        f"{BASE_URL}/encode",
        json={"story": story}
    )
    return response.json()

# Invert a story
def invert_story(story, axes=["W", "N"], mode="soft"):
    response = requests.post(
        f"{BASE_URL}/invert",
        json={
            "story": story,
            "axes": axes,
            "mode": mode
        }
    )
    return response.json()

# Anti-attractor synthesis
def anti_attractor(story):
    response = requests.post(
        f"{BASE_URL}/anti-attractor",
        json={"story": story}
    )
    return response.json()

# Example usage
if __name__ == "__main__":
    # Encode
    result = encode_story("A woman loved a man")
    print("Encoded:", result["expression"])

    # Invert
    result = invert_story("Power corrupts", axes=["F"], mode="soft")
    print("Inverted:", result["result"]["inverted"]["story"])

    # Anti-attractor
    result = anti_attractor("Unity brings peace")
    print("Anti-attractor:", result["result"]["inverted"]["story"])
```

### JavaScript Client (Browser)

```javascript
const BASE_URL = "http://localhost:8000";

// Encode a story
async function encodeStory(story) {
    const response = await fetch(`${BASE_URL}/encode`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({story})
    });
    return await response.json();
}

// Invert a story
async function invertStory(story, axes = ["W", "N"], mode = "soft") {
    const response = await fetch(`${BASE_URL}/invert`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({story, axes, mode})
    });
    return await response.json();
}

// Anti-attractor synthesis
async function antiAttractor(story) {
    const response = await fetch(`${BASE_URL}/anti-attractor`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({story})
    });
    return await response.json();
}

// Example usage
(async () => {
    // Encode
    let result = await encodeStory("A woman loved a man");
    console.log("Encoded:", result.expression);

    // Invert
    result = await invertStory("Power corrupts", ["F"], "soft");
    console.log("Inverted:", result.result.inverted.story);

    // Anti-attractor
    result = await antiAttractor("Unity brings peace");
    console.log("Anti-attractor:", result.result.inverted.story);
})();
```

## Canonical Guarantees

The TKS Inference API enforces the following canonical constraints:

1. **Worlds**: Fixed set of 4 worlds (A, B, C, D)
   - A = Spiritual/Abstract
   - B = Mental/Intellectual
   - C = Emotional/Relational
   - D = Physical/Material

2. **Noetics**: Fixed set of 10 noetic principles (1..10)
   - Pairs: 2<->3 (Positive/Negative), 5<->6 (Female/Male), 8<->9 (Cause/Effect)
   - Self-duals: 1 (Mind), 4 (Vibration), 7 (Rhythm), 10 (Idea)

3. **Foundations**: Fixed set of 7 foundations (1..7)
   - 1 = Unity, 2 = Wisdom, 3 = Life, 4 = Companionship, 5 = Power, 6 = Material, 7 = Lust
   - Opposites: 1<->7, 2<->6, 3<->5, 4 (self-dual)

4. **Sub-foundations**: 28 combinations (7 foundations x 4 worlds)

5. **Operators**: Fixed set of 9 operators
   - Transcendent: `+T`, `-T`, `*T`, `/T`
   - Causal: `->`, `<-`
   - Compositional: `o`
   - Basic: `+`, `-`

6. **ASCII-only**: No Unicode symbols or special characters

7. **Deterministic**: Same input always produces same output

8. **Type-safe**: Strong typing throughout the pipeline

## Performance Considerations

- The server uses Python's built-in `http.server`, which is single-threaded
- For production use, consider deploying behind a reverse proxy (nginx, Apache)
- Each request is processed synchronously
- Typical response times:
  - `/health`: < 10ms
  - `/encode`: 10-100ms (depending on story length)
  - `/invert`: 50-200ms (depending on complexity)
  - `/anti-attractor`: 100-500ms (requires signature analysis)

## Troubleshooting

### Server won't start

```bash
# Check if port is already in use
netstat -an | grep 8000  # Linux/Mac
netstat -an | findstr 8000  # Windows

# Use a different port
python scripts/serve_inference.py --port 8080
```

### Encoding errors in strict mode

If you get encoding errors for unknown tokens, try lenient mode:

```bash
# Start server in lenient mode
python scripts/serve_inference.py --lenient

# Or pass strict=false in the request
curl -X POST http://localhost:8000/encode \
     -H "Content-Type: application/json" \
     -d '{"story": "Some text", "strict": false}'
```

### Invalid JSON errors

Ensure your JSON is properly formatted:

```bash
# Use a JSON validator
echo '{"story": "test"}' | python -m json.tool

# Or use jq
echo '{"story": "test"}' | jq .
```

## See Also

- [Scenario Inversion Guide](SCENARIO_INVERSION.md)
- [Anti-Attractor Guide](anti_attractor_guide.md)
- [Canonical Validation](Agent3_Validation_Implementation.md)
- [CLI Inference](../scripts/run_inference.py)
