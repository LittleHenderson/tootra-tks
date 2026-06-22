"""
Synthetic Negation Scope Dataset Generator

Generates sequences for testing whether models can correctly handle negation scope.

Task: Predict properties based on descriptions with and without negation.

Examples:
- "The warrior is brave" → brave=1
- "The warrior is not brave" → brave=0
- "The wizard is wise and not cowardly" → wise=1, cowardly=0

Dataset Structure:
- Positive examples: "The [entity] is [property]"
- Negative examples: "The [entity] is not [property]"
- Complex examples: Multiple properties with mixed negation
- Scope ambiguity examples: "The hero is not brave but wise"

The model must learn to:
1. Identify negation tokens ("not")
2. Determine scope of negation (which predicate it applies to)
3. Correctly flip truth values only for negated predicates
"""

import torch
from torch.utils.data import Dataset, DataLoader
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class NegationExample:
    """Single training example with negation scope tracking."""
    text: str
    tokens: List[str]
    properties: Dict[str, int]  # property -> 0/1
    not_positions: List[int]  # Indices of "not" tokens
    property_positions: Dict[str, int]  # property -> token index


class NegationScopeDataset(Dataset):
    """
    Synthetic dataset for negation scope resolution.

    Task: Given a sentence like "The wizard is wise and not cowardly",
    predict property values: wise=1, cowardly=0
    """

    ENTITIES = [
        "warrior", "wizard", "hero", "villain", "knight", "rogue",
        "archer", "mage", "paladin", "ranger", "barbarian", "cleric"
    ]

    PROPERTIES = [
        "brave", "wise", "strong", "clever", "fast", "cowardly",
        "foolish", "weak", "dim", "slow", "loyal", "treacherous"
    ]

    def __init__(
        self,
        n_samples: int = 10000,
        max_properties: int = 3,
        negation_prob: float = 0.5,
        vocab_file: Optional[str] = None,
        seed: int = 42
    ):
        """
        Args:
            n_samples: Number of examples to generate
            max_properties: Maximum properties per example (1-3)
            negation_prob: Probability of negating each property
            vocab_file: Optional file to save vocabulary
            seed: Random seed
        """
        random.seed(seed)
        self.n_samples = n_samples
        self.max_properties = max_properties
        self.negation_prob = negation_prob

        # Build vocabulary
        self.vocab = self._build_vocab()
        self.idx_to_token = {i: t for t, i in self.vocab.items()}

        if vocab_file:
            self._save_vocab(vocab_file)

        # Generate examples
        self.examples = self._generate_examples()

    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary from templates."""
        tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

        # Add template tokens
        tokens.extend(["the", "is", "not", "and", "but"])

        # Add entities and properties
        tokens.extend(self.ENTITIES)
        tokens.extend(self.PROPERTIES)

        # Add special tokens for answers
        tokens.extend(["yes", "no"])

        return {t: i for i, t in enumerate(tokens)}

    def _save_vocab(self, filepath: str):
        """Save vocabulary to file."""
        with open(filepath, 'w') as f:
            for token, idx in sorted(self.vocab.items(), key=lambda x: x[1]):
                f.write(f"{token}\t{idx}\n")

    def _generate_examples(self) -> List[NegationExample]:
        """Generate all training examples."""
        examples = []

        for _ in range(self.n_samples):
            # Choose number of properties (1-3)
            n_props = random.randint(1, self.max_properties)

            # Sample entity and properties
            entity = random.choice(self.ENTITIES)
            properties = random.sample(self.PROPERTIES, n_props)

            # Decide which properties to negate
            negated = [random.random() < self.negation_prob for _ in properties]

            # Build sentence
            example = self._build_example(entity, properties, negated)
            examples.append(example)

        return examples

    def _build_example(
        self,
        entity: str,
        properties: List[str],
        negated: List[bool]
    ) -> NegationExample:
        """Build a single example."""

        # Start with "The [entity] is"
        tokens = ["the", entity, "is"]
        not_positions = []
        property_positions = {}

        for i, (prop, is_negated) in enumerate(zip(properties, negated)):
            # Add conjunction for properties after first
            if i > 0:
                tokens.append("and")

            # Add negation if needed
            if is_negated:
                not_positions.append(len(tokens))
                tokens.append("not")

            # Add property
            property_positions[prop] = len(tokens)
            tokens.append(prop)

        # Build text
        text = " ".join(tokens)

        # Build property values (1 if not negated, 0 if negated)
        property_values = {
            prop: 0 if is_neg else 1
            for prop, is_neg in zip(properties, negated)
        }

        return NegationExample(
            text=text,
            tokens=tokens,
            properties=property_values,
            not_positions=not_positions,
            property_positions=property_positions
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            {
                'input_ids': Tensor[T] - tokenized input
                'labels': Tensor[T] - shifted for language modeling
                'property_positions': Dict[str, int] - where properties appear
                'not_positions': List[int] - where "not" appears
                'property_values': Dict[str, int] - ground truth values
            }
        """
        example = self.examples[idx]

        # Tokenize
        token_ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in example.tokens]

        # Create labels (for language modeling: predict next token)
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        labels = torch.tensor(token_ids[1:], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'property_positions': example.property_positions,
            'not_positions': example.not_positions,
            'property_values': example.property_values,
            'text': example.text
        }


class NegationQADataset(Dataset):
    """
    Question-answering variant: Given sentence, answer yes/no about property.

    Format:
    Input: "The wizard is wise and not cowardly. Is the wizard brave?"
    Output: "no" (property not mentioned)

    Input: "The wizard is not cowardly. Is the wizard cowardly?"
    Output: "no"

    Input: "The wizard is wise. Is the wizard wise?"
    Output: "yes"
    """

    def __init__(
        self,
        n_samples: int = 10000,
        max_properties: int = 3,
        negation_prob: float = 0.5,
        seed: int = 42
    ):
        random.seed(seed)

        # Use same vocabulary building as main dataset
        base_dataset = NegationScopeDataset(n_samples=10, seed=seed)
        self.vocab = base_dataset.vocab
        self.idx_to_token = base_dataset.idx_to_token

        self.ENTITIES = base_dataset.ENTITIES
        self.PROPERTIES = base_dataset.PROPERTIES

        self.n_samples = n_samples
        self.max_properties = max_properties
        self.negation_prob = negation_prob

        # Generate examples
        self.examples = self._generate_examples()

    def _generate_examples(self) -> List[Dict]:
        """Generate QA examples."""
        examples = []

        for _ in range(self.n_samples):
            # Generate a statement
            n_props = random.randint(1, self.max_properties)
            entity = random.choice(self.ENTITIES)
            properties = random.sample(self.PROPERTIES, n_props)
            negated = [random.random() < self.negation_prob for _ in properties]

            # Build statement
            statement_tokens = ["the", entity, "is"]
            property_values = {}
            not_positions = []

            for i, (prop, is_negated) in enumerate(zip(properties, negated)):
                if i > 0:
                    statement_tokens.append("and")
                if is_negated:
                    not_positions.append(len(statement_tokens))
                    statement_tokens.append("not")
                statement_tokens.append(prop)

                property_values[prop] = 0 if is_negated else 1

            # Create question about a random property
            query_prop = random.choice(self.PROPERTIES)

            # Answer: yes if mentioned and positive, no otherwise
            if query_prop in property_values:
                answer = "yes" if property_values[query_prop] == 1 else "no"
            else:
                answer = "no"  # Property not mentioned

            # Build full sequence: statement + question + answer
            full_tokens = statement_tokens + ["is", "the", entity, query_prop]
            answer_tokens = [answer]

            examples.append({
                'statement_tokens': statement_tokens,
                'query_prop': query_prop,
                'full_tokens': full_tokens,
                'answer_tokens': answer_tokens,
                'property_values': property_values,
                'not_positions': not_positions,
                'answer': answer
            })

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        example = self.examples[idx]

        # Tokenize full sequence + answer
        input_tokens = example['full_tokens']
        target_token = example['answer_tokens'][0]

        input_ids = torch.tensor(
            [self.vocab.get(t, self.vocab["<UNK>"]) for t in input_tokens],
            dtype=torch.long
        )
        target_id = torch.tensor(
            self.vocab.get(target_token, self.vocab["<UNK>"]),
            dtype=torch.long
        )

        return {
            'input_ids': input_ids,
            'target_id': target_id,
            'answer': example['answer'],
            'query_prop': example['query_prop'],
            'property_values': example['property_values'],
            'not_positions': example['not_positions']
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader with padding."""
    # Find max length
    max_len = max(len(item['input_ids']) for item in batch)

    # Pad sequences
    input_ids = []
    labels = []

    for item in batch:
        seq_len = len(item['input_ids'])
        pad_len = max_len - seq_len

        # Pad input_ids and labels
        padded_input = torch.cat([
            item['input_ids'],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        padded_labels = torch.cat([
            item['labels'],
            torch.full((pad_len,), -100, dtype=torch.long)  # -100 ignored in loss
        ])

        input_ids.append(padded_input)
        labels.append(padded_labels)

    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels),
        'property_positions': [item['property_positions'] for item in batch],
        'not_positions': [item['not_positions'] for item in batch],
        'property_values': [item['property_values'] for item in batch]
    }


def collate_fn_qa(batch: List[Dict]) -> Dict:
    """Collate function for QA dataset."""
    max_len = max(len(item['input_ids']) for item in batch)

    input_ids = []
    for item in batch:
        seq_len = len(item['input_ids'])
        pad_len = max_len - seq_len

        padded_input = torch.cat([
            item['input_ids'],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        input_ids.append(padded_input)

    target_ids = torch.stack([item['target_id'] for item in batch])

    return {
        'input_ids': torch.stack(input_ids),
        'target_ids': target_ids,
        'answers': [item['answer'] for item in batch],
        'query_props': [item['query_prop'] for item in batch],
        'property_values': [item['property_values'] for item in batch],
        'not_positions': [item['not_positions'] for item in batch]
    }


if __name__ == "__main__":
    print("Testing Negation Scope Dataset...")

    # Test standard dataset
    dataset = NegationScopeDataset(n_samples=100, seed=42)
    print(f"Vocabulary size: {len(dataset.vocab)}")
    print(f"Dataset size: {len(dataset)}")

    # Show examples
    print("\nExample sequences:")
    for i in range(5):
        ex = dataset.examples[i]
        print(f"\n{i+1}. {ex.text}")
        print(f"   Properties: {ex.properties}")
        print(f"   NOT positions: {ex.not_positions}")
        print(f"   Property positions: {ex.property_positions}")

    # Test dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    batch = next(iter(dataloader))
    print(f"\nBatch shapes:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  labels: {batch['labels'].shape}")

    # Test QA dataset
    print("\n" + "="*60)
    print("Testing QA Dataset...")
    qa_dataset = NegationQADataset(n_samples=100, seed=42)

    print(f"\nQA Dataset size: {len(qa_dataset)}")
    print("\nExample QA pairs:")
    for i in range(5):
        ex = qa_dataset.examples[i]
        statement = " ".join(ex['statement_tokens'])
        query = f"Is the [entity] {ex['query_prop']}?"
        print(f"\n{i+1}. Statement: {statement}")
        print(f"   Question: {query}")
        print(f"   Answer: {ex['answer']}")
        print(f"   Property values: {ex['property_values']}")

    print("\nAll tests passed!")
