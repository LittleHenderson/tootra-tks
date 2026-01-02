#!/usr/bin/env python3
"""
TKS Inference HTTP API Client Example

Demonstrates how to use the TKS Inference HTTP API from Python.

Requirements:
    pip install requests  # Or use urllib (built-in)

Usage:
    # Start the server first:
    python scripts/serve_inference.py

    # Then run this example:
    python examples/api_client_example.py
"""

import requests
import json

# Server configuration
BASE_URL = "http://localhost:8000"


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def health_check():
    """Check server health and canonical configuration."""
    print_section("HEALTH CHECK")

    response = requests.get(f"{BASE_URL}/health")
    data = response.json()

    print(f"Status: {data['status']}")
    print(f"Mode: {data['mode']}")
    print(f"\nCanonical Configuration:")
    for key, value in data['canon'].items():
        print(f"  {key}: {value}")

    return data


def encode_story(story, strict=True):
    """Encode a natural language story to TKS expression."""
    print_section(f"ENCODE STORY: {story}")

    response = requests.post(
        f"{BASE_URL}/encode",
        json={"story": story, "strict": strict}
    )

    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data['success']}")
        print(f"Expression: {data['expression']}")
        print(f"Elements: {data['elements']}")
        print(f"Operators: {data['ops']}")
        return data
    else:
        error = response.json()
        print(f"Error: {error['error']}")
        return None


def invert_story(story, axes=None, mode="soft", strict=True):
    """Perform scenario inversion on a story."""
    if axes is None:
        axes = ["W", "N"]

    print_section(f"INVERT STORY: {story}")
    print(f"Axes: {axes}, Mode: {mode}")

    response = requests.post(
        f"{BASE_URL}/invert",
        json={
            "story": story,
            "axes": axes,
            "mode": mode,
            "strict": strict
        }
    )

    if response.status_code == 200:
        data = response.json()
        result = data['result']
        validator = data['validator']

        print(f"\nOriginal:")
        print(f"  Expression: {result['original']['expression']}")
        print(f"  Story: {result['original']['story']}")

        print(f"\nInverted:")
        print(f"  Expression: {result['inverted']['expression']}")
        print(f"  Story: {result['inverted']['story']}")

        print(f"\nExplanation:")
        print(f"  {result['explanation'][:200]}...")

        print(f"\nValidation:")
        print(f"  Valid: {validator['is_valid']}")
        print(f"  Canon Score: {validator['canon_score']}")
        print(f"  Errors: {validator['error_count']}")
        print(f"  Warnings: {validator['warning_count']}")

        return data
    else:
        error = response.json()
        print(f"Error: {error['error']}")
        return None


def anti_attractor(story, strict=True):
    """Perform anti-attractor synthesis."""
    print_section(f"ANTI-ATTRACTOR: {story}")

    response = requests.post(
        f"{BASE_URL}/anti-attractor",
        json={"story": story, "strict": strict}
    )

    if response.status_code == 200:
        data = response.json()
        result = data['result']
        validator = data['validator']

        print(f"\nOriginal:")
        print(f"  Expression: {result['original']['expression']}")
        print(f"  Story: {result['original']['story']}")

        if 'signature' in result:
            sig = result['signature']
            print(f"\nOriginal Signature:")
            print(f"  Dominant: {sig['dominant_world']}{sig['dominant_noetic']}")
            print(f"  Polarity: {sig['polarity']}")
            print(f"  Foundations: {sig['foundation_tags']}")

        print(f"\nAnti-Attractor:")
        print(f"  Expression: {result['inverted']['expression']}")
        print(f"  Story: {result['inverted']['story']}")

        if 'inverted_signature' in result:
            inv_sig = result['inverted_signature']
            print(f"\nInverted Signature:")
            print(f"  Dominant: {inv_sig['dominant_world']}{inv_sig['dominant_noetic']}")
            print(f"  Polarity: {inv_sig['polarity']}")
            print(f"  Foundations: {inv_sig['foundation_tags']}")

        print(f"\nValidation:")
        print(f"  Valid: {validator['is_valid']}")
        print(f"  Canon Score: {validator['canon_score']}")

        return data
    else:
        error = response.json()
        print(f"Error: {error['error']}")
        return None


def main():
    """Main example function."""
    print("\n" + "=" * 70)
    print("  TKS INFERENCE HTTP API CLIENT EXAMPLE")
    print("=" * 70)
    print(f"\nConnecting to: {BASE_URL}")

    try:
        # 1. Health check
        health_check()

        # 2. Encode a story
        encode_story("A woman loved a man", strict=False)

        # 3. Invert across World and Noetic axes
        invert_story(
            "A woman loved a man",
            axes=["W", "N"],
            mode="soft",
            strict=False
        )

        # 4. Invert across Foundation axis
        invert_story(
            "A woman loved a man",
            axes=["F"],
            mode="soft",
            strict=False
        )

        # 5. Anti-attractor synthesis
        anti_attractor("A woman loved a man", strict=False)

        # 6. Equation input example
        print_section("EQUATION INPUT EXAMPLE")
        response = requests.post(
            f"{BASE_URL}/invert",
            json={
                "equation": "B5 +T D3",
                "axes": ["W"],
                "mode": "soft"
            }
        )
        if response.status_code == 200:
            data = response.json()
            result = data['result']
            print(f"Original: {result['original']['expression']}")
            print(f"Inverted: {result['inverted']['expression']}")

        print("\n" + "=" * 70)
        print("  EXAMPLES COMPLETE")
        print("=" * 70)

    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to server.")
        print("Please start the server first:")
        print("  python scripts/serve_inference.py")
    except Exception as e:
        print(f"\nERROR: {e}")


if __name__ == "__main__":
    main()
