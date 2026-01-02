#!/usr/bin/env python3
"""
Demo: NL→Equation Retrieval

Demonstrates how to use the NL→equation dataset for retrieval tasks.
Given a natural language query, find the most similar equations.
"""

import torch
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer


# TKS element mapping
TKS_ELEMENTS = (
    ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"] +
    ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10"] +
    ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"] +
    ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10"]
)

OPERATOR_NAMES = ["+", "-", "×", "÷"]


def format_equation(triplet: torch.Tensor) -> str:
    """Format a triplet as an equation string."""
    left_idx, right_idx, op_idx = triplet.tolist()
    left = TKS_ELEMENTS[left_idx]
    right = TKS_ELEMENTS[right_idx]
    op = OPERATOR_NAMES[op_idx]
    return f"{left} {op} {right}"


def retrieve_similar(
    query: str,
    nl_embeddings: torch.Tensor,
    triplets: torch.Tensor,
    model: SentenceTransformer,
    top_k: int = 5
) -> list:
    """
    Retrieve top-k most similar equations for a query.

    Args:
        query: Natural language query
        nl_embeddings: [N, 384] tensor of interpretation embeddings
        triplets: [N, 3] tensor of equation triplets
        model: Sentence transformer model
        top_k: Number of results to return

    Returns:
        List of (equation_str, triplet, similarity_score) tuples
    """
    # Encode query
    query_embedding = model.encode(
        [query.lower().strip()],
        convert_to_tensor=True,
        normalize_embeddings=True
    )  # [1, 384]

    # Compute similarities
    similarities = torch.mm(query_embedding, nl_embeddings.T).squeeze(0)  # [N]

    # Get top-k
    top_k = min(top_k, len(similarities))
    top_scores, top_indices = torch.topk(similarities, k=top_k)

    # Format results
    results = []
    for idx, score in zip(top_indices, top_scores):
        triplet = triplets[idx]
        eq_str = format_equation(triplet)
        results.append((eq_str, triplet, score.item()))

    return results


def main():
    """Main demo."""
    # Paths
    repo_root = Path(__file__).parent.parent
    data_dir = repo_root / "data" / "nl_equation_pairs"
    corpus_path = data_dir / "nl_corpus.pt"
    jsonl_path = data_dir / "nl_corpus.jsonl"

    print("="*70)
    print("NL→Equation Retrieval Demo")
    print("="*70)

    # Load dataset
    print(f"\nLoading dataset from {corpus_path}")
    corpus = torch.load(corpus_path, weights_only=False)
    nl_embeddings = corpus["nl_embeddings"]
    triplets = corpus["triplets"]

    print(f"Loaded {len(nl_embeddings)} equation interpretations")

    # Load model
    model_name = "all-MiniLM-L6-v2"
    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Load interpretations for display
    print("Loading interpretations...")
    with open(jsonl_path, 'r') as f:
        interpretations = [json.loads(line)["interpretation"] for line in f]

    # Demo queries
    queries = [
        "spiritual growth through positive energy",
        "mental conflict and emotional turmoil",
        "physical healing with spiritual guidance",
        "balancing male and female energies",
        "overcoming negative thoughts with positive action"
    ]

    print("\n" + "="*70)
    print("Running Demo Queries")
    print("="*70)

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}: \"{query}\"")
        print("="*70)

        results = retrieve_similar(query, nl_embeddings, triplets, model, top_k=3)

        for j, (eq_str, triplet, score) in enumerate(results, 1):
            idx = ((triplets == triplet).all(dim=1).nonzero(as_tuple=True)[0][0]).item()
            interpretation = interpretations[idx]

            print(f"\n  Result {j}: {eq_str}")
            print(f"  Similarity: {score:.4f}")
            print(f"  Triplet: {tuple(triplet.tolist())}")
            print(f"  Interpretation: {interpretation[:200]}...")

    # Interactive mode
    print("\n" + "="*70)
    print("Interactive Mode")
    print("="*70)
    print("\nEnter your own queries (or 'quit' to exit):")

    while True:
        try:
            query = input("\nQuery: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break

            if not query:
                continue

            results = retrieve_similar(query, nl_embeddings, triplets, model, top_k=3)

            print(f"\nTop 3 results for: \"{query}\"")
            print("-" * 70)

            for j, (eq_str, triplet, score) in enumerate(results, 1):
                idx = ((triplets == triplet).all(dim=1).nonzero(as_tuple=True)[0][0]).item()
                interpretation = interpretations[idx]

                print(f"\n{j}. {eq_str} (similarity: {score:.4f})")
                print(f"   {interpretation[:150]}...")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "="*70)
    print("Demo Complete")
    print("="*70)


if __name__ == "__main__":
    main()
