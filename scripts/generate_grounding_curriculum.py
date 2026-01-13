"""
Generate Grounding Curriculum for TKS Model.

Goals:
- Teach "Canon ID <-> Definition" (Bidirectional)
- Teach Disambiguation (Hard Negatives)
- Teach Uniquely Solvable Mapping (Hints)

Data Sources:
- TKS_FORMAL_MANUAL_v6.1_CLEAN_DEFINITIONS.md (Canonical Authority)
"""

import json
import random
import os
from typing import List, Dict, Tuple

# ==============================================================================
# CANONICAL DEFINITIONS (from Manual v6.1)
# ==============================================================================

# 1. Worlds
WORLDS = {
    'A': {'name': 'Spiritual', 'alt': 'Atziluth', 'desc': 'Archetypal/Divine, Source, soul, eternal'},
    'B': {'name': 'Mental', 'alt': 'Briah', 'desc': 'Creative/Intellectual, beliefs, concepts'},
    'C': {'name': 'Emotional', 'alt': 'Yetzirah', 'desc': 'Formative/Astral, feelings, intuition'},
    'D': {'name': 'Physical', 'alt': 'Assiah', 'desc': 'Action/Material, body, concrete'},
}

# 2. Noetics (0-9)
# Note: Manual v6.1 says ^0 is Idea.
NOETICS = {
    0: {'name': 'Idea', 'desc': 'Content/Thing, the neutral baseline'},
    1: {'name': 'Mind', 'desc': 'Processor/Awareness, directed attention'},
    2: {'name': 'Positive', 'desc': 'Attraction/Affinity, the "yes" relation'},
    3: {'name': 'Negative', 'desc': 'Aversion/Repulsion, the "no" relation'},
    4: {'name': 'Vibration', 'desc': 'Resonance/Impression, intensity of impact'},
    5: {'name': 'Female', 'desc': 'Receptivity/Composition (of Mind)'},
    6: {'name': 'Male', 'desc': 'Structure/Composition (of Idea)'},
    7: {'name': 'Rhythm', 'desc': 'Timing/Repetition, exposure over time'},
    8: {'name': 'Above', 'alt': 'Cause', 'desc': 'Trigger/Stimulus, the initiating idea'},
    9: {'name': 'Below', 'alt': 'Effect', 'desc': 'Response/Result, the resulting idea'},
}

# 3. Elements (40)
# Mapping Noetic Index -> Element Suffix
# 1->1, 2->2 ... 9->9, 0->10 (Idea is 10 in Element notation)
NOETIC_TO_ELEMENT_SUFFIX = {
    1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 
    6: '6', 7: '7', 8: '8', 9: '9', 0: '10'
}

ELEMENT_DEFINITIONS = []
for w_char, w_info in WORLDS.items():
    for n_idx, n_info in NOETICS.items():
        suffix = NOETIC_TO_ELEMENT_SUFFIX[n_idx]
        elem_id = f"{w_char}{suffix}"
        
        # Name construction: "Spiritual Mind", "Mental Positive", etc.
        # Exception: "Above" and "Below" often used directly
        n_name = n_info.get('alt', n_info['name']) # Use 'Cause' instead of 'Above' if available? Manual uses 'Above/Cause'
        # Manual Table 4.2 uses "Spiritual Above", "Spiritual Below"
        n_name = n_info['name'] 
        
        full_name = f"{w_info['name']} {n_name}"
        
        # Core meaning (simplified from manual)
        core_meaning = ""
        if n_idx == 1: core_meaning = f"{w_info['name']} awareness or consciousness"
        elif n_idx == 2: core_meaning = f"{w_info['name']} attraction or affinity"
        elif n_idx == 3: core_meaning = f"{w_info['name']} aversion or repulsion"
        elif n_idx == 4: core_meaning = f"{w_info['name']} resonance or frequency"
        elif n_idx == 5: core_meaning = f"{w_info['name']} receptivity or conditioning"
        elif n_idx == 6: core_meaning = f"{w_info['name']} structure or authority"
        elif n_idx == 7: core_meaning = f"{w_info['name']} cycles or timing"
        elif n_idx == 8: core_meaning = f"{w_info['name']} triggers or causes"
        elif n_idx == 9: core_meaning = f"{w_info['name']} results or effects"
        elif n_idx == 0: core_meaning = f"{w_info['name']} content or essence"
        
        ELEMENT_DEFINITIONS.append({
            'id': elem_id,
            'name': full_name,
            'desc': core_meaning,
            'world': w_char,
            'noetic_idx': n_idx
        })

# 4. Foundations (28)
FOUNDATIONS = {
    1: 'Association',
    2: 'Wisdom',
    3: 'Life',
    4: 'Companionship',
    5: 'Power',
    6: 'Wealth',
    7: 'Continuation' # aka Reproduction
}

SUB_FOUNDATIONS = []
for f_num, f_name in FOUNDATIONS.items():
    for w_char, w_info in WORLDS.items():
        # Manual uses a,b,c,d (lowercase)
        w_lower = w_char.lower()
        sf_id = f"{f_num}{w_lower}"
        full_name = f"{w_info['name']} {f_name}"
        
        # Special case for 7 (Continuation)
        if f_num == 7:
            if w_char == 'C': full_name = "Emotional Lust/Bonding"
            elif w_char == 'D': full_name = "Physical Sex/Reproduction"
            else: full_name = f"{w_info['name']} Continuation"
            
        SUB_FOUNDATIONS.append({
            'id': sf_id,
            'name': full_name,
            'foundation': f_name,
            'world': w_char
        })

# 5. Acquisitions (22)
ACQUISITIONS = [
    # F1 Association
    {'id': 'D1', 'name': 'Desire for unity with God', 'f': 1, 'type': 'Desire'},
    {'id': 'W1', 'name': 'Wisdom about unity', 'f': 1, 'type': 'Wisdom'},
    {'id': 'P1', 'name': 'Power to achieve unity', 'f': 1, 'type': 'Power'},
    # F2 Wisdom
    {'id': 'D2', 'name': 'Desire for insight/knowledge', 'f': 2, 'type': 'Desire'},
    {'id': 'W2', 'name': 'Wisdom about knowing', 'f': 2, 'type': 'Wisdom'},
    {'id': 'P2', 'name': 'Power to obtain knowledge', 'f': 2, 'type': 'Power'},
    # F3 Life
    {'id': 'D3', 'name': 'Desire for health/vitality', 'f': 3, 'type': 'Desire'},
    {'id': 'W3', 'name': 'Wisdom about health', 'f': 3, 'type': 'Wisdom'},
    {'id': 'P3', 'name': 'Power to create/maintain health', 'f': 3, 'type': 'Power'},
    # F4 Companionship
    {'id': 'D4', 'name': 'Desire for companionship/love', 'f': 4, 'type': 'Desire'},
    {'id': 'W4', 'name': 'Wisdom about relationships', 'f': 4, 'type': 'Wisdom'},
    {'id': 'P4', 'name': 'Power to form/maintain bonds', 'f': 4, 'type': 'Power'},
    # F5 Power
    {'id': 'D5', 'name': 'Desire for worldly power/influence', 'f': 5, 'type': 'Desire'},
    {'id': 'W5', 'name': 'Wisdom about power', 'f': 5, 'type': 'Wisdom'},
    {'id': 'P5', 'name': 'Power to act effectively', 'f': 5, 'type': 'Power'},
    # F6 Wealth
    {'id': 'D6', 'name': 'Desire for material resources', 'f': 6, 'type': 'Desire'},
    {'id': 'W6', 'name': 'Wisdom about wealth', 'f': 6, 'type': 'Wisdom'},
    {'id': 'P6', 'name': 'Power to acquire/manage resources', 'f': 6, 'type': 'Power'},
    # F7 Continuation
    {'id': 'D7', 'name': 'Desire for continuation/reproduction', 'f': 7, 'type': 'Desire'},
    {'id': 'W7', 'name': 'Wisdom about continuation', 'f': 7, 'type': 'Wisdom'},
    {'id': 'P7', 'name': 'Power to continue/reproduce', 'f': 7, 'type': 'Power'},
    # Special
    {'id': 'A22', 'name': 'Meta-Desire', 'f': 0, 'type': 'Meta', 'desc': 'Desire for desire, global activation'}
]

# ==============================================================================
# GENERATORS
# ==============================================================================

def generate_id_to_def_task(item_type: str, item_id: str, definition: str) -> Dict:
    return {
        'task_type': 'grounding_id_to_def',
        'input': f"Define TKS {item_type} {item_id}",
        'output': definition,
        'metadata': {'id': item_id, 'type': item_type}
    }

def generate_def_to_id_task(item_type: str, item_id: str, definition: str) -> Dict:
    return {
        'task_type': 'grounding_def_to_id',
        'input': f"What is the ID for the TKS {item_type} defined as: {definition}?",
        'output': item_id,
        'metadata': {'id': item_id, 'type': item_type}
    }

def generate_disambiguation_task(item: Dict, all_items: List[Dict], item_type: str) -> Dict:
    # Find a hard negative (same world OR same noetic/foundation)
    correct_id = item['id']
    
    candidates = []
    if item_type == 'Element':
        # Same world, different noetic
        candidates.extend([x for x in all_items if x['world'] == item['world'] and x['id'] != correct_id])
        # Same noetic, different world
        candidates.extend([x for x in all_items if x['noetic_idx'] == item['noetic_idx'] and x['id'] != correct_id])
    elif item_type == 'Sub-Foundation':
        # Same world, different foundation
        candidates.extend([x for x in all_items if x['world'] == item['world'] and x['id'] != correct_id])
        # Same foundation, different world
        candidates.extend([x for x in all_items if x['foundation'] == item['foundation'] and x['id'] != correct_id])
    
    if not candidates:
        return None
        
    distractor = random.choice(candidates)
    distractor_id = distractor['id']
    
    # 50/50 order
    if random.random() < 0.5:
        option_a, option_b = correct_id, distractor_id
        answer = "Option A"
    else:
        option_a, option_b = distractor_id, correct_id
        answer = "Option B"
        
    prompt = f"Which TKS {item_type} corresponds to '{item['name']}'?\nOption A: {option_a}\nOption B: {option_b}"
    
    return {
        'task_type': 'grounding_disambiguation',
        'input': prompt,
        'output': f"{answer} ({correct_id})",
        'metadata': {'correct_id': correct_id, 'distractor_id': distractor_id}
    }

def generate_mapping_with_hints_task(item: Dict, item_type: str) -> Dict:
    # Explicit hints
    hint = ""
    if item_type == 'Element':
        w_name = WORLDS[item['world']]['name']
        hint = f"World: {w_name} ({item['world']})"
    elif item_type == 'Sub-Foundation':
        w_name = WORLDS[item['world']]['name']
        hint = f"World: {w_name} ({item['world']})"
        
    prompt = f"Map this concept to a TKS ID.\nConcept: {item['name']}\nHint: {hint}"
    
    return {
        'task_type': 'grounding_mapping',
        'input': prompt,
        'output': item['id'],
        'metadata': {'id': item['id']}
    }

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    dataset = []
    
    # 1. Elements
    print(f"Generating Element tasks ({len(ELEMENT_DEFINITIONS)} elements)...")
    for elem in ELEMENT_DEFINITIONS:
        # ID <-> Def
        dataset.append(generate_id_to_def_task('Element', elem['id'], elem['name']))
        dataset.append(generate_def_to_id_task('Element', elem['id'], elem['name']))
        # Disambiguation
        disambig = generate_disambiguation_task(elem, ELEMENT_DEFINITIONS, 'Element')
        if disambig: dataset.append(disambig)
        # Mapping
        dataset.append(generate_mapping_with_hints_task(elem, 'Element'))
        
        # Add tasks for description too (variation)
        if elem['desc']:
            dataset.append(generate_id_to_def_task('Element', elem['id'], elem['desc']))
            dataset.append(generate_def_to_id_task('Element', elem['id'], elem['desc']))

    # 2. Noetics
    print(f"Generating Noetic tasks ({len(NOETICS)} noetics)...")
    for idx, info in NOETICS.items():
        nid = f"^{idx}"
        dataset.append(generate_id_to_def_task('Noetic', nid, info['name']))
        dataset.append(generate_def_to_id_task('Noetic', nid, info['name']))
        # Description variation
        dataset.append(generate_id_to_def_task('Noetic', nid, info['desc']))
        
    # 3. Sub-Foundations
    print(f"Generating Sub-Foundation tasks ({len(SUB_FOUNDATIONS)} items)...")
    for sf in SUB_FOUNDATIONS:
        dataset.append(generate_id_to_def_task('Sub-Foundation', sf['id'], sf['name']))
        dataset.append(generate_def_to_id_task('Sub-Foundation', sf['id'], sf['name']))
        disambig = generate_disambiguation_task(sf, SUB_FOUNDATIONS, 'Sub-Foundation')
        if disambig: dataset.append(disambig)
        dataset.append(generate_mapping_with_hints_task(sf, 'Sub-Foundation'))

    # 4. Acquisitions
    print(f"Generating Acquisition tasks ({len(ACQUISITIONS)} items)...")
    for acq in ACQUISITIONS:
        dataset.append(generate_id_to_def_task('Acquisition', acq['id'], acq['name']))
        dataset.append(generate_def_to_id_task('Acquisition', acq['id'], acq['name']))
        
        # Mapping from Foundation + Type
        if acq['id'] != 'A22':
            prompt = f"What is the {acq['type']} Acquisition for Foundation {acq['f']}?"
            dataset.append({
                'task_type': 'grounding_lookup',
                'input': prompt,
                'output': acq['id'],
                'metadata': {'id': acq['id']}
            })

    # Shuffle
    random.seed(42)
    random.shuffle(dataset)
    
    # Save
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "teacher_grounding_curriculum.jsonl")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')
            
    print(f"\nGenerated {len(dataset)} grounding tasks.")
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    main()
