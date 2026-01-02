"""
Direct generation of story-equation pairs - standalone script.

This generates the dataset without external dependencies.
"""
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Allow running as a script: `python scripts/generate_pairs_direct.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

from tks_rules.noetics import NOETICS, SENSES
from tks_rules.operators import OPERATORS
from tks_rules.rpm import RPM_MAPPINGS
from tks_rules.worlds import WORLDS
# Use the 9-operator subset used across this repo's training data (exclude * and /)
CANONICAL_OPERATORS = [op for op in OPERATORS.keys() if op not in {"*", "/"}]
VALID_SENSES = SENSES  # Alias for compatibility
VALID_FOUNDATIONS = ['_d1', '_d2', '_d3', '_d4', '_d5', '_d6', '_d7']
FOUNDATION_NAMES = {'_d1': 'Unity', '_d2': 'Wisdom', '_d3': 'Life', '_d4': 'Companionship', '_d5': 'Power', '_d6': 'Material', '_d7': 'Lust'}
WORLD_DESCRIPTIONS = {'A': ('Spiritual', 'the spiritual realm', 'spiritual essence'), 'B': ('Mental', 'the mental plane', 'mental energy'), 'C': ('Emotional', 'the emotional sphere', 'emotional force'), 'D': ('Physical', 'the physical world', 'physical manifestation')}
NOETIC_DESCRIPTIONS = {1: ('Mind', 'mentalism', 'consciousness'), 2: ('Positive', 'polarity positive', 'attraction'), 3: ('Negative', 'polarity negative', 'repulsion'), 4: ('Vibration', 'vibrational frequency', 'resonance'), 5: ('Female', 'feminine principle', 'receptivity'), 6: ('Male', 'masculine principle', 'projection'), 7: ('Rhythm', 'rhythmic flow', 'cyclical motion'), 8: ('Cause', 'causation', 'the above'), 9: ('Effect', 'correspondence', 'the below'), 10: ('Idea', 'divine idea', 'the All')}
OPERATOR_DESCRIPTIONS = {'+': ('combines with', 'joins', 'merges with'), '-': ('separates from', 'removes', 'subtracts'), '+T': ('temporally combines with', 'adds over time to', 'accumulates with'), '-T': ('temporally separates from', 'removes over time from', 'diminishes from'), '->': ('flows to', 'transforms into', 'leads to'), '<-': ('receives from', 'draws from', 'is sourced by'), '*T': ('amplifies with', 'multiplies temporally with', 'resonates with'), '/T': ('attenuates through', 'divides temporally by', 'dampens with'), 'o': ('composes with', 'circulates through', 'cycles with')}

def make_element(world: str, noetic: int, sense: Optional[int]=None, foundation: Optional[str]=None) -> str:
    """Create element string."""
    s = f'{world}{noetic}'
    if sense is not None:
        s += f'^{sense}'
    if foundation is not None:
        s += foundation
    return s

def random_element(include_sense: bool=False, include_foundation: bool=False, world: Optional[str]=None) -> Tuple[str, str, int, Optional[int], Optional[str]]:
    """Generate random element and return (string, world, noetic, sense, foundation)."""
    if world is None:
        world = random.choice(list(WORLDS.keys()))
    noetic = random.randint(1, 10)
    sense = random.choice(VALID_SENSES) if include_sense and random.random() < 0.3 else None
    foundation = random.choice(VALID_FOUNDATIONS) if include_foundation and random.random() < 0.2 else None
    return (make_element(world, noetic, sense, foundation), world, noetic, sense, foundation)

def element_phrase(world: str, noetic: int, sense: Optional[int], foundation: Optional[str], context: str='subject') -> str:
    """Generate phrase for element."""
    world_name, world_realm, world_essence = WORLD_DESCRIPTIONS[world]
    noetic_name, noetic_type, noetic_quality = NOETIC_DESCRIPTIONS[noetic]
    templates = {'subject': [f'the {noetic_name} principle in {world_realm}', f'{world_name} {noetic_name}', f'the {noetic_quality} of {world_realm}', f'noetic {noetic_name} within World {world}'], 'object': [f'the {noetic_name} force of {world_realm}', f'{world_name}-{noetic_name} energy', f'the {noetic_quality} from World {world}']}
    phrase = random.choice(templates.get(context, templates['subject']))
    if sense is not None:
        phrases = [f' with sense {sense}', f' at intensity {sense}', f' (sense level {sense})']
        phrase += random.choice(phrases)
    if foundation is not None:
        fname = FOUNDATION_NAMES.get(foundation, foundation)
        phrases = [f' grounded in {fname}', f' with foundation of {fname}', f' anchored in {fname}']
        phrase += random.choice(phrases)
    return phrase

def operator_phrase(op: str) -> str:
    """Get operator phrase."""
    return random.choice(OPERATOR_DESCRIPTIONS[op])

def generate_pair(pair_id: str, chain_length: int=None, pattern: str=None) -> Tuple[Dict, Dict]:
    """Generate a story-equation pair in both directions."""
    if chain_length is None:
        chain_length = random.choice([2, 2, 3, 3, 3, 4, 4, 5, 5, 6])
    elements_data = []
    if pattern == 'single_world':
        world = random.choice(list(WORLDS.keys()))
        for _ in range(chain_length):
            elem, w, n, s, f = random_element(True, True, world)
            elements_data.append((elem, w, n, s, f))
    elif pattern == 'involution':
        pairs = [(2, 3), (5, 6), (8, 9)]
        inv_pair = random.choice(pairs)
        world = random.choice(list(WORLDS.keys()))
        elements_data.append((make_element(world, inv_pair[0]), world, inv_pair[0], None, None))
        elements_data.append((make_element(world, inv_pair[1]), world, inv_pair[1], None, None))
        for _ in range(chain_length - 2):
            elem, w, n, s, f = random_element(True, True)
            elements_data.append((elem, w, n, s, f))
    elif pattern == 'ascending':
        world_order = ['A', 'B', 'C', 'D']
        for i in range(chain_length):
            world = world_order[i % 4]
            elem, w, n, s, f = random_element(True, True, world)
            elements_data.append((elem, w, n, s, f))
    elif pattern == 'descending':
        world_order = ['D', 'C', 'B', 'A']
        for i in range(chain_length):
            world = world_order[i % 4]
            elem, w, n, s, f = random_element(True, True, world)
            elements_data.append((elem, w, n, s, f))
    elif pattern == 'rpm_desire':
        for _ in range(chain_length):
            noetic = random.choice([2, 3])
            world = random.choice(list(WORLDS.keys()))
            elem = make_element(world, noetic)
            elements_data.append((elem, world, noetic, None, None))
    elif pattern == 'rpm_wisdom':
        for _ in range(chain_length):
            noetic = random.choice([1, 4, 5, 6, 7])
            world = random.choice(list(WORLDS.keys()))
            elem = make_element(world, noetic)
            elements_data.append((elem, world, noetic, None, None))
    elif pattern == 'rpm_power':
        for _ in range(chain_length):
            noetic = random.choice([8, 9])
            world = random.choice(list(WORLDS.keys()))
            elem = make_element(world, noetic)
            elements_data.append((elem, world, noetic, None, None))
    else:
        for _ in range(chain_length):
            elem, w, n, s, f = random_element(True, True)
            elements_data.append((elem, w, n, s, f))
    operators = [random.choice(CANONICAL_OPERATORS) for _ in range(chain_length - 1)]
    expr_elements = [e[0] for e in elements_data]
    parts = [expr_elements[0]]
    for i, op in enumerate(operators):
        parts.extend([f' {op} ', expr_elements[i + 1]])
    equation = ''.join(parts)
    openings = ['In this TKS working,', 'This equation describes how', 'The following transformation occurs:', 'In the noetic sequence,', 'This working demonstrates that']
    closings = ['.', ', completing the noetic circuit.', ', manifesting the intended result.', ' in the TKS framework.']
    story_parts = [random.choice(openings)]
    story_parts.append(element_phrase(elements_data[0][1], elements_data[0][2], elements_data[0][3], elements_data[0][4], 'subject'))
    for i, op in enumerate(operators):
        story_parts.append(operator_phrase(op))
        story_parts.append(element_phrase(elements_data[i + 1][1], elements_data[i + 1][2], elements_data[i + 1][3], elements_data[i + 1][4], 'object'))
    story = ' '.join(story_parts) + random.choice(closings)
    if random.random() < 0.3:
        noetics = [e[2] for e in elements_data]
        desire_count = sum((1 for n in noetics if n in RPM_MAPPINGS['desire']))
        wisdom_count = sum((1 for n in noetics if n in RPM_MAPPINGS['wisdom']))
        power_count = sum((1 for n in noetics if n in RPM_MAPPINGS['power']))
        total = desire_count + wisdom_count + power_count
        if total > 0:
            rpm_dist = {'desire': desire_count / total, 'wisdom': wisdom_count / total, 'power': power_count / total}
            dominant = max(rpm_dist, key=rpm_dist.get)
            rpm_phrases = {'desire': ' This working emphasizes Desire (polarity forces).', 'wisdom': ' This working emphasizes Wisdom (mental/vibrational forces).', 'power': ' This working emphasizes Power (causal forces).'}
            story += rpm_phrases[dominant]
    story_to_eq = {'story': story, 'equation': equation, 'expr_elements': expr_elements, 'expr_ops': operators, 'direction': 'story_to_eq', 'pair_id': pair_id}
    eq_to_story = {'story': story, 'equation': equation, 'expr_elements': expr_elements, 'expr_ops': operators, 'direction': 'eq_to_story', 'pair_id': pair_id}
    return (story_to_eq, eq_to_story)

def convert_to_training(pair: Dict) -> Dict:
    """Convert pair to training format."""
    if pair['direction'] == 'story_to_eq':
        return {'task_type': 'story_to_equation', 'input': f"Given this TKS narrative:\n\n{pair['story']}\n\nTranslate this into a TKS equation using canonical notation (worlds A,B,C,D; noetics 1-10).", 'target': pair['equation'], 'metadata': {'elements': pair['expr_elements'], 'operators': pair['expr_ops'], 'pair_id': pair['pair_id']}, 'direction': pair['direction'], 'canon_score': 1.0}
    else:
        return {'task_type': 'equation_to_story', 'input': f"Given the TKS equation: {pair['equation']}\n\nElements: {', '.join(pair['expr_elements'])}\n\nGenerate a natural language narrative describing this TKS working.", 'target': pair['story'], 'metadata': {'elements': pair['expr_elements'], 'operators': pair['expr_ops'], 'pair_id': pair['pair_id']}, 'direction': pair['direction'], 'canon_score': 1.0}

def main():
    random.seed(42)
    patterns = [None, 'single_world', 'involution', 'ascending', 'descending', 'rpm_desire', 'rpm_wisdom', 'rpm_power']
    all_pairs = []
    for i in range(400):
        pair_id = f'pair_{i:04d}'
        pattern = random.choice(patterns)
        story_to_eq, eq_to_story = generate_pair(pair_id, pattern=pattern)
        all_pairs.append(story_to_eq)
        all_pairs.append(eq_to_story)
    pair_ids = list(set((p['pair_id'] for p in all_pairs)))
    random.shuffle(pair_ids)
    split_idx = int(len(pair_ids) * 0.85)
    train_ids = set(pair_ids[:split_idx])
    train_pairs = [p for p in all_pairs if p['pair_id'] in train_ids]
    holdout_pairs = [p for p in all_pairs if p['pair_id'] not in train_ids]
    train_converted = [convert_to_training(p) for p in train_pairs]
    holdout_converted = [convert_to_training(p) for p in holdout_pairs]
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    output_dir = base_dir / 'output'

    def save_jsonl(data, filepath):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    save_jsonl(train_pairs, data_dir / 'story_equation_pairs_train.jsonl')
    save_jsonl(holdout_pairs, data_dir / 'story_equation_pairs_holdout.jsonl')
    save_jsonl(train_converted, output_dir / 'teacher_story_eq_train.jsonl')
    save_jsonl(holdout_converted, output_dir / 'teacher_story_eq_holdout.jsonl')
    print(f'Generated {len(all_pairs)} total samples from {len(pair_ids)} unique pairs')
    print(f'Train: {len(train_pairs)} samples ({len(train_ids)} pairs)')
    print(f'Holdout: {len(holdout_pairs)} samples ({len(pair_ids) - len(train_ids)} pairs)')
    print(f'\nFiles saved:')
    print(f'  data/story_equation_pairs_train.jsonl')
    print(f'  data/story_equation_pairs_holdout.jsonl')
    print(f'  output/teacher_story_eq_train.jsonl')
    print(f'  output/teacher_story_eq_holdout.jsonl')
if __name__ == '__main__':
    main()
