#!/usr/bin/env python3
"""
Generate grounding drills for canonical TKS entities and optionally mix them
into a rebalanced dataset with conflict dedupe.
"""

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    items = []
    if not os.path.exists(path):
        return items
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def write_jsonl(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def normalize_text(text):
    return " ".join(str(text).split()).strip()


def normalize_key(text):
    return normalize_text(text).lower()


class Canon:
    def __init__(self, canon_dir):
        self.canon_dir = canon_dir
        self.elements = {}
        self.elements_list = []
        self.elements_by_world = defaultdict(list)
        self.elements_by_noetic = defaultdict(list)
        self.noetics = {}
        self.noetic_ids = []
        self.subfoundations = {}
        self.subfoundations_list = []
        self.subfoundations_by_world = defaultdict(list)
        self.subfoundations_by_foundation = defaultdict(list)
        self.foundations = {}
        self.acquisitions = {}
        self.worlds = {}
        self._load()

    def _load(self):
        elements = load_json(os.path.join(self.canon_dir, "canon_elements_40.json"))
        for elem in elements:
            symbol = elem["symbol"]
            name = elem["name"]
            definition = elem["definition"]
            short_name = name.split(" ", 1)[-1] if " " in name else name
            noetic_idx = int(symbol[1:])
            item = {
                "id": symbol,
                "name": name,
                "short_name": short_name,
                "definition": definition,
                "world": elem["world"],
                "noetic_idx": noetic_idx,
            }
            self.elements[symbol] = item
            self.elements_list.append(item)
            self.elements_by_world[item["world"]].append(item)
            self.elements_by_noetic[item["noetic_idx"]].append(item)

        for noetic in load_json(os.path.join(self.canon_dir, "canon_noetics_10.json")):
            nid = noetic["id"]
            item = {
                "id": nid,
                "name": noetic["name"],
                "meaning": noetic["meaning"],
            }
            self.noetics[nid] = item
            self.noetic_ids.append(nid)

        subfoundations = load_json(os.path.join(self.canon_dir, "canon_subfoundations_28.json"))
        for sf in subfoundations:
            item = {
                "id": sf["id"],
                "name": sf["name"],
                "foundation_id": sf["foundation_id"],
                "world": sf["world"],
            }
            self.subfoundations[item["id"]] = item
            self.subfoundations_list.append(item)
            self.subfoundations_by_world[item["world"]].append(item)
            self.subfoundations_by_foundation[item["foundation_id"]].append(item)

        for found in load_json(os.path.join(self.canon_dir, "canon_foundations_7.json")):
            self.foundations[found["id"]] = {"id": found["id"], "name": found["name"]}

        for world in load_json(os.path.join(self.canon_dir, "canon_worlds_4.json")):
            self.worlds[world["id"]] = {
                "id": world["id"],
                "name": world["name"],
                "description": world.get("description", ""),
            }

        for acq in load_json(os.path.join(self.canon_dir, "canon_acquisitions_22.json")):
            acq_id = f"W{acq['id']}"
            self.acquisitions[acq_id] = {
                "id": acq_id,
                "name": acq["name"],
                "type": acq["type"],
                "foundation_id": acq["foundation_id"],
            }

        self.elements_list.sort(key=lambda x: x["id"])
        self.subfoundations_list.sort(key=lambda x: x["id"])
        self.noetic_ids.sort()

        self.element_lookup = {}
        for elem in self.elements_list:
            for key in (elem["definition"], elem["short_name"], elem["name"]):
                self.element_lookup.setdefault(normalize_key(key), elem["id"])

        self.noetic_lookup = {}
        for noetic in self.noetics.values():
            for key in (noetic["name"], noetic["meaning"], f"{noetic['name']}: {noetic['meaning']}"):
                self.noetic_lookup.setdefault(normalize_key(key), noetic["id"])

        self.subfoundation_lookup = {}
        for sf in self.subfoundations_list:
            self.subfoundation_lookup.setdefault(normalize_key(sf["name"]), sf["id"])

        self.foundation_lookup = {}
        for found in self.foundations.values():
            self.foundation_lookup.setdefault(normalize_key(found["name"]), found["id"])

        self.world_lookup = {}
        for world in self.worlds.values():
            self.world_lookup.setdefault(normalize_key(world["name"]), world["id"])
            if world["description"]:
                self.world_lookup.setdefault(
                    normalize_key(f"{world['name']}: {world['description']}"), world["id"]
                )

        self.acquisition_lookup = {}
        for acq in self.acquisitions.values():
            self.acquisition_lookup.setdefault(normalize_key(acq["name"]), acq["id"])
            self.acquisition_lookup.setdefault(
                normalize_key(f"{acq['name']} ({acq['type']})"), acq["id"]
            )

    def outputs_for_entity(self, entity_type, entity_id):
        outputs = set()
        if entity_type == "Element":
            elem = self.elements.get(entity_id)
            if elem:
                outputs.update(
                    [
                        elem["definition"],
                        elem["name"],
                        elem["short_name"],
                        f"{elem['short_name']}: {elem['definition']}",
                    ]
                )
        elif entity_type == "Noetic":
            noetic = self.noetics.get(entity_id)
            if noetic:
                outputs.update(
                    [
                        noetic["name"],
                        noetic["meaning"],
                        f"{noetic['name']}: {noetic['meaning']}",
                    ]
                )
        elif entity_type == "Sub-Foundation":
            sf = self.subfoundations.get(entity_id)
            if sf:
                outputs.add(sf["name"])
        elif entity_type == "Foundation":
            found = self.foundations.get(entity_id)
            if found:
                outputs.add(found["name"])
        elif entity_type == "Acquisition":
            acq = self.acquisitions.get(entity_id)
            if acq:
                outputs.add(f"{acq['name']} ({acq['type']})")
        elif entity_type == "World":
            world = self.worlds.get(entity_id)
            if world:
                outputs.add(f"{world['name']}: {world['description']}")
                outputs.add(world["name"])
        return {normalize_text(o) for o in outputs if o}

    def id_for_definition(self, entity_type, definition):
        key = normalize_key(definition)
        if entity_type == "Element":
            return self.element_lookup.get(key)
        if entity_type == "Noetic":
            return self.noetic_lookup.get(key)
        if entity_type == "Sub-Foundation":
            return self.subfoundation_lookup.get(key)
        if entity_type == "Foundation":
            return self.foundation_lookup.get(key)
        if entity_type == "Acquisition":
            return self.acquisition_lookup.get(key)
        if entity_type == "World":
            return self.world_lookup.get(key)
        return None


def render_output(kind, item):
    if kind == "definition":
        return item["definition"]
    if kind == "name":
        return item["name"]
    if kind == "short_name":
        return item["short_name"]
    if kind == "name_definition":
        return f"{item['short_name']}: {item['definition']}"
    if kind == "meaning":
        return item["meaning"]
    if kind == "name_meaning":
        return f"{item['name']}: {item['meaning']}"
    raise ValueError(f"Unknown output kind: {kind}")


def make_task(task_type, input_text, output_text):
    return {
        "task_type": task_type,
        "input": input_text,
        "output": output_text,
        "source": "grounding_drills_v5",
    }


def make_mcq(rng, prompt, options, correct_id):
    letters = "ABCD"
    shuffled = list(options)
    rng.shuffle(shuffled)
    correct_index = shuffled.index(correct_id)
    lines = [prompt]
    for idx, option in enumerate(shuffled):
        lines.append(f"Option {letters[idx]}: {option}")
    return make_task(
        "grounding_disambiguation",
        "\n".join(lines),
        f"Option {letters[correct_index]} ({correct_id})",
    )


CONFUSION_PAIRS = [
    ("^0", "^1"),
    ("^1", "^2"),
    ("^2", "^3"),
    ("^6", "^7"),
    ("^8", "^9"),
]

RPM_TEMPLATES = [
    "Return RPM stack (colon-separated) for {target}.",
    "Provide RPM path (format A1:B2:C3) for {target}.",
    "RPM stack for {target}:",
]

RPM_INPUT_PATTERNS = [
    re.compile(r"^Goal:\s*(.+)$", re.IGNORECASE),
    re.compile(r"^RPM Cascade for\s*(.+)$", re.IGNORECASE),
    re.compile(r"^Deep Plan:\s*I want\s*'([^']+)'$", re.IGNORECASE),
    re.compile(r"^Deep Plan:\s*I want\s*(.+)$", re.IGNORECASE),
]


def build_confusion_map():
    mapping = defaultdict(list)
    for left, right in CONFUSION_PAIRS:
        mapping[left].append(right)
        mapping[right].append(left)
    return mapping


ELEMENT_ID_DEF_TEMPLATES = [
    ("Define TKS Element {id}", "definition"),
    ("What does TKS Element {id} represent?", "definition"),
    ("In TKS, {id} refers to what?", "definition"),
    ("Give the canonical name for TKS Element {id}.", "name"),
    ("What is {id} in TKS?", "name_definition"),
]

ELEMENT_DEF_ID_TEMPLATES = [
    ("What is the ID for the TKS Element defined as: {definition}?", "definition"),
    ("Which TKS Element matches '{short_name}'?", "short_name"),
    ("Identify the TKS Element: {short_name}.", "short_name"),
    ("Which Element is defined as: {definition}?", "definition"),
]

NOETIC_ID_DEF_TEMPLATES = [
    ("Define TKS Noetic {id}", "name_meaning"),
    ("What does TKS Noetic {id} mean?", "meaning"),
    ("In TKS, {id} stands for what?", "name"),
    ("Give the canonical meaning for {id}.", "name_meaning"),
    ("Explain Noetic {id}.", "name_meaning"),
    ("What is the function of Noetic {id}?", "meaning"),
    ("Give the canonical name for Noetic {id}.", "name"),
    ("Noetic {id} is defined as what?", "name_meaning"),
    ("Summarize Noetic {id} in TKS.", "meaning"),
    ("State the official definition of Noetic {id}.", "name_meaning"),
]

NOETIC_DEF_ID_TEMPLATES = [
    ("What is the ID for the TKS Noetic defined as: {name}?", "name"),
    ("Which TKS Noetic corresponds to '{meaning}'?", "meaning"),
    ("Which Noetic is '{name}: {meaning}'?", "name_meaning"),
    ("Identify the Noetic for: {name}.", "name"),
    ("Which Noetic matches '{meaning}' in TKS?", "meaning"),
    ("Which TKS Noetic matches '{name}'?", "name"),
    ("Which noetic ID corresponds to: {name}?", "name"),
    ("Which noetic ID matches: {meaning}?", "meaning"),
    ("Identify the noetic ID for: {name}: {meaning}.", "name_meaning"),
    ("Which TKS noetic represents '{meaning}'?", "meaning"),
]

SUB_ID_DEF_TEMPLATES = [
    ("Define TKS Sub-Foundation {id}", "name"),
    ("What does Sub-Foundation {id} refer to?", "name"),
    ("Give the canonical name for Sub-Foundation {id}.", "name"),
    ("What is Sub-Foundation {id} called?", "name"),
    ("Name the TKS Sub-Foundation {id}.", "name"),
]

SUB_DEF_ID_TEMPLATES = [
    ("What is the ID for the TKS Sub-Foundation defined as: {name}?", "name"),
    ("Which Sub-Foundation matches '{name}'?", "name"),
    ("Identify the Sub-Foundation for: {name}.", "name"),
    ("Which Sub-Foundation ID maps to {name}?", "name"),
    ("What is the TKS ID for Sub-Foundation {name}?", "name"),
]

NOETIC_MCQ_TEMPLATES = [
    "Which TKS Noetic corresponds to '{value}'?",
    "Which Noetic matches '{value}'?",
    "Select the TKS Noetic for '{value}'.",
]

SUBFOUNDATION_MCQ_TEMPLATES = [
    "Which TKS Sub-Foundation corresponds to '{name}'?",
    "Which Sub-Foundation matches '{name}'?",
    "Select the Sub-Foundation for '{name}'.",
]


def _repeat_templates(rng, templates, target_len):
    if not templates:
        return []
    if target_len <= len(templates):
        return list(templates[:target_len])
    repeated = []
    pool = list(templates)
    while len(repeated) < target_len:
        rng.shuffle(pool)
        repeated.extend(pool)
    return repeated[:target_len]


def pick_distractors(rng, candidates, excluded, target_count):
    picks = []
    for candidate in candidates:
        if candidate not in excluded and candidate not in picks:
            picks.append(candidate)
            if len(picks) >= target_count:
                return picks
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    for candidate in shuffled:
        if candidate not in excluded and candidate not in picks:
            picks.append(candidate)
            if len(picks) >= target_count:
                break
    return picks


def generate_element_mcq(rng, canon, elem, prompt_value):
    prompt = f"Which TKS Element corresponds to '{prompt_value}'?"
    distractors = []
    same_world = [
        e["id"]
        for e in canon.elements_by_world[elem["world"]]
        if e["id"] != elem["id"]
    ]
    same_noetic = [
        e["id"]
        for e in canon.elements_by_noetic[elem["noetic_idx"]]
        if e["id"] != elem["id"]
    ]
    distractors.extend(pick_distractors(rng, same_world, {elem["id"]}, 1))
    distractors.extend(pick_distractors(rng, same_noetic, {elem["id"]}, 1))
    all_ids = [e["id"] for e in canon.elements_list if e["id"] != elem["id"]]
    distractors.extend(pick_distractors(rng, all_ids, {elem["id"]} | set(distractors), 3))
    options = [elem["id"]] + distractors[:3]
    return make_mcq(rng, prompt, options, elem["id"])


def generate_noetic_mcq(rng, canon, noetic, prompt_value, confusion_map):
    prompt = f"Which TKS Noetic corresponds to '{prompt_value}'?"
    distractors = []
    for candidate in confusion_map.get(noetic["id"], []):
        if candidate not in distractors and candidate != noetic["id"]:
            distractors.append(candidate)
    all_ids = [nid for nid in canon.noetic_ids if nid != noetic["id"]]
    distractors.extend(
        pick_distractors(rng, all_ids, {noetic["id"]} | set(distractors), 3)
    )
    options = [noetic["id"]] + distractors[:3]
    return make_mcq(rng, prompt, options, noetic["id"])


def generate_subfoundation_mcq(rng, canon, sf, prompt_template):
    prompt = prompt_template.format(name=sf["name"])
    distractors = []
    same_foundation = [
        item["id"]
        for item in canon.subfoundations_by_foundation[sf["foundation_id"]]
        if item["id"] != sf["id"]
    ]
    same_world = [
        item["id"]
        for item in canon.subfoundations_by_world[sf["world"]]
        if item["id"] != sf["id"]
    ]
    distractors.extend(pick_distractors(rng, same_foundation, {sf["id"]}, 1))
    distractors.extend(pick_distractors(rng, same_world, {sf["id"]}, 1))
    all_ids = [item["id"] for item in canon.subfoundations_list if item["id"] != sf["id"]]
    distractors.extend(pick_distractors(rng, all_ids, {sf["id"]} | set(distractors), 3))
    options = [sf["id"]] + distractors[:3]
    return make_mcq(rng, prompt, options, sf["id"])


def generate_element_drills(rng, canon):
    items = []
    for elem in canon.elements_list:
        for template, kind in ELEMENT_ID_DEF_TEMPLATES:
            input_text = template.format(id=elem["id"])
            items.append(make_task("grounding_id_to_def", input_text, render_output(kind, elem)))
        for template, kind in ELEMENT_DEF_ID_TEMPLATES:
            input_text = template.format(definition=elem["definition"], short_name=elem["short_name"])
            items.append(make_task("grounding_def_to_id", input_text, elem["id"]))
        items.append(generate_element_mcq(rng, canon, elem, elem["short_name"]))
        items.append(generate_element_mcq(rng, canon, elem, elem["definition"]))
    return items


def generate_noetic_drills(rng, canon, confusion_map, multiplier):
    items = []
    noetics = sorted(canon.noetics.values(), key=lambda x: x["id"])
    multiplier = max(int(multiplier), 1)
    for noetic in noetics:
        for pass_idx in range(multiplier):
            if pass_idx == 0:
                id_templates = NOETIC_ID_DEF_TEMPLATES
                def_templates = NOETIC_DEF_ID_TEMPLATES
                mcq_templates = NOETIC_MCQ_TEMPLATES
            else:
                id_templates = _repeat_templates(
                    rng, NOETIC_ID_DEF_TEMPLATES, len(NOETIC_ID_DEF_TEMPLATES)
                )
                def_templates = _repeat_templates(
                    rng, NOETIC_DEF_ID_TEMPLATES, len(NOETIC_DEF_ID_TEMPLATES)
                )
                mcq_templates = _repeat_templates(
                    rng, NOETIC_MCQ_TEMPLATES, len(NOETIC_MCQ_TEMPLATES)
                )

            for template, kind in id_templates:
                input_text = template.format(id=noetic["id"])
                items.append(make_task("grounding_id_to_def", input_text, render_output(kind, noetic)))

            for template, kind in def_templates:
                input_text = template.format(
                    name=noetic["name"],
                    meaning=noetic["meaning"],
                )
                items.append(make_task("grounding_def_to_id", input_text, noetic["id"]))

            for template in mcq_templates:
                items.append(
                    generate_noetic_mcq(
                        rng,
                        canon,
                        noetic,
                        template.format(value=noetic["name"]),
                        confusion_map,
                    )
                )
                items.append(
                    generate_noetic_mcq(
                        rng,
                        canon,
                        noetic,
                        template.format(value=noetic["meaning"]),
                        confusion_map,
                    )
                )
                items.append(
                    generate_noetic_mcq(
                        rng,
                        canon,
                        noetic,
                        template.format(value=f"{noetic['name']}: {noetic['meaning']}"),
                        confusion_map,
                    )
                )
    return items


def generate_subfoundation_drills(rng, canon, multiplier):
    items = []
    multiplier = max(int(multiplier), 1)
    for sf in canon.subfoundations_list:
        for pass_idx in range(multiplier):
            if pass_idx == 0:
                id_templates = SUB_ID_DEF_TEMPLATES
                def_templates = SUB_DEF_ID_TEMPLATES
                mcq_templates = SUBFOUNDATION_MCQ_TEMPLATES
            else:
                id_templates = _repeat_templates(
                    rng, SUB_ID_DEF_TEMPLATES, len(SUB_ID_DEF_TEMPLATES)
                )
                def_templates = _repeat_templates(
                    rng, SUB_DEF_ID_TEMPLATES, len(SUB_DEF_ID_TEMPLATES)
                )
                mcq_templates = _repeat_templates(
                    rng, SUBFOUNDATION_MCQ_TEMPLATES, len(SUBFOUNDATION_MCQ_TEMPLATES)
                )

            for template, kind in id_templates:
                input_text = template.format(id=sf["id"])
                items.append(make_task("grounding_id_to_def", input_text, render_output(kind, sf)))

            for template, kind in def_templates:
                input_text = template.format(name=sf["name"])
                items.append(make_task("grounding_def_to_id", input_text, sf["id"]))

            for template in mcq_templates:
                items.append(generate_subfoundation_mcq(rng, canon, sf, template))
    return items


def generate_rpm_drills(rng, base_items):
    items = []
    if not base_items:
        return items

    for item in base_items:
        if item.get("task_type") != "rpm":
            continue
        output = str(item.get("output", "")).strip()
        if not output:
            continue
        input_text = str(item.get("input", "")).strip()
        target = None
        for pattern in RPM_INPUT_PATTERNS:
            match = pattern.match(input_text)
            if match:
                target = match.group(1).strip().rstrip(".")
                break
        if not target:
            continue

        templates = list(RPM_TEMPLATES)
        rng.shuffle(templates)
        for template in templates[:2]:
            items.append(make_task("rpm", template.format(target=target), output))

    return items


def generate_drills(rng, canon, base_items=None, *, noetic_multiplier=1, subfoundation_multiplier=1):
    confusion_map = build_confusion_map()
    items = []
    items.extend(generate_element_drills(rng, canon))
    items.extend(generate_noetic_drills(rng, canon, confusion_map, noetic_multiplier))
    items.extend(generate_subfoundation_drills(rng, canon, subfoundation_multiplier))
    items.extend(generate_rpm_drills(rng, base_items or []))
    rng.shuffle(items)
    return items


RE_DEFINE = re.compile(
    r"Define TKS (?P<type>Element|Noetic|Sub-Foundation|Foundation|Acquisition|World) (?P<id>\\S+)"
)
RE_DEF_TO_ID = re.compile(
    r"What is the ID for the TKS (?P<type>Element|Noetic|Sub-Foundation|Foundation|Acquisition|World) defined as: (?P<def>.+)$"
)
RE_WHAT_IS = re.compile(r"What is (?P<id>\\S+) in TKS\\?")


RE_ELEMENT = re.compile(r"^[ABCD]\\d+$")
RE_NOETIC = re.compile(r"^\\^\\d+$")
RE_SUBFOUNDATION = re.compile(r"^\\d+[abcd]$")
RE_ACQUISITION = re.compile(r"^W\\d+$")
RE_WORLD = re.compile(r"^[ABCD]$")
RE_FOUNDATION = re.compile(r"^\\d+$")


def classify_id(value):
    if RE_NOETIC.match(value):
        return "Noetic"
    if RE_SUBFOUNDATION.match(value):
        return "Sub-Foundation"
    if RE_ACQUISITION.match(value):
        return "Acquisition"
    if RE_ELEMENT.match(value):
        return "Element"
    if RE_WORLD.match(value):
        return "World"
    if RE_FOUNDATION.match(value):
        return "Foundation"
    return None


def parse_input_for_entity(input_text):
    match = RE_DEFINE.search(input_text)
    if match:
        return {
            "kind": "id_to_def",
            "entity_type": match.group("type"),
            "entity_id": match.group("id"),
        }

    match = RE_WHAT_IS.search(input_text)
    if match:
        entity_type = classify_id(match.group("id"))
        if entity_type:
            return {
                "kind": "id_to_def",
                "entity_type": entity_type,
                "entity_id": match.group("id"),
            }

    match = RE_DEF_TO_ID.search(input_text)
    if match:
        definition = match.group("def").strip().rstrip("?").strip().strip("'\"")
        return {
            "kind": "def_to_id",
            "entity_type": match.group("type"),
            "definition": definition,
        }

    return None


def resolve_conflict(input_text, items, canon):
    parsed = parse_input_for_entity(input_text)
    if not parsed:
        return None
    outputs = [normalize_text(item.get("output", "")) for item in items]
    if parsed["kind"] == "id_to_def":
        canonical = canon.outputs_for_entity(parsed["entity_type"], parsed["entity_id"])
        for output in outputs:
            if output in canonical:
                return output
        return None
    if parsed["kind"] == "def_to_id":
        entity_id = canon.id_for_definition(parsed["entity_type"], parsed["definition"])
        if entity_id:
            for output in outputs:
                if output == entity_id:
                    return entity_id
        return None
    return None


def pick_item_with_output(items, desired_output):
    desired = normalize_text(desired_output)
    for item in items:
        if item.get("source") == "grounding_drills_v5" and normalize_text(item.get("output", "")) == desired:
            return item
    for item in items:
        if normalize_text(item.get("output", "")) == desired:
            return item
    return None


def pick_most_common_output(items):
    output_map = defaultdict(list)
    for item in items:
        output_map[normalize_text(item.get("output", ""))].append(item)
    if not output_map:
        return None
    best_output = sorted(output_map.items(), key=lambda kv: (-len(kv[1]), kv[0]))[0][0]
    return output_map[best_output][0]


def dedupe_and_mix(rng, base_items, drill_items, canon):
    combined = base_items + drill_items
    by_input = defaultdict(list)
    for item in combined:
        key = normalize_text(item.get("input", ""))
        if key:
            by_input[key].append(item)

    output = []
    stats = Counter()

    for _, items in by_input.items():
        output_map = defaultdict(list)
        for item in items:
            output_map[normalize_text(item.get("output", ""))].append(item)

        if len(output_map) == 1:
            output.extend(items)
            if len(items) > 1:
                stats["duplicates_kept"] += len(items) - 1
            else:
                stats["unique"] += 1
            continue

        task_types = {item.get("task_type") for item in items}
        if task_types == {"rpm"}:
            chosen = pick_most_common_output(items)
            if chosen:
                output.append(chosen)
                stats["conflicts_resolved"] += 1
                stats["conflict_entries_dropped"] += len(items) - 1
                continue

        stats["conflicts"] += 1
        resolved = resolve_conflict(items[0].get("input", ""), items, canon)
        if resolved:
            chosen = pick_item_with_output(items, resolved)
            if chosen:
                output.append(chosen)
                stats["conflicts_resolved"] += 1
                stats["conflict_entries_dropped"] += len(items) - 1
            else:
                stats["conflicts_dropped"] += 1
                stats["conflict_entries_dropped"] += len(items)
        else:
            stats["conflicts_dropped"] += 1
            stats["conflict_entries_dropped"] += len(items)

    rng.shuffle(output)
    return output, stats


def main():
    parser = argparse.ArgumentParser(description="Generate grounding drills and rebalance mix")
    parser.add_argument("--canon-dir", default="canon/normalized", help="Canonical JSON directory")
    parser.add_argument("--output", default="output/grounding_drills_v5.jsonl", help="Drill output file")
    parser.add_argument(
        "--mix-input", default="output/final_reconciled_mix.jsonl", help="Base mix input"
    )
    parser.add_argument(
        "--mix-output", default="output/rebalanced_mix_v5.jsonl", help="Mixed output"
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--no-mix", action="store_true", help="Skip mixing step")
    parser.add_argument(
        "--noetic-multiplier",
        type=int,
        default=1,
        help="Repeat noetic drills to boost weighting",
    )
    parser.add_argument(
        "--subfoundation-multiplier",
        type=int,
        default=1,
        help="Repeat subfoundation drills to boost weighting",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    canon = Canon(args.canon_dir)

    base_items = load_jsonl(args.mix_input)
    if not base_items:
        print(f"Warning: {args.mix_input} not found or empty.")

    drills = generate_drills(
        rng,
        canon,
        base_items,
        noetic_multiplier=args.noetic_multiplier,
        subfoundation_multiplier=args.subfoundation_multiplier,
    )
    write_jsonl(args.output, drills)
    print(f"Generated {len(drills)} drills -> {args.output}")

    if args.no_mix:
        return

    if not base_items:
        print(f"Warning: {args.mix_input} not found or empty. Skipping mix.")
        return

    mixed, stats = dedupe_and_mix(rng, base_items, drills, canon)
    write_jsonl(args.mix_output, mixed)
    print(f"Mixed dataset size: {len(mixed)} -> {args.mix_output}")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
