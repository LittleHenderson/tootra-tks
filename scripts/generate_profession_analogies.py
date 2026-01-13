#!/usr/bin/env python3
"""
TKS Profession Analogies Generator

NO HEBREW. NO KABBALAH. PURE TKS CANON.

Generates TKS concepts expressed through professional analogies from 25 professions.
Also generates 1:1 mappings between regular explanations and profession analogies.

25 Professions:
- Medical: Doctor, Nurse, Surgeon
- Trades: Electrician, Plumber, Mechanic, Carpenter, Welder, HVAC
- Creative: Musician, Chef, Artist, DJ, Writer
- Technical: Programmer, Engineer, Architect, Data Scientist
- Service: Teacher, Coach, Firefighter, Detective
- Business: Banker, Lawyer, Salesperson

Usage:
    python scripts/generate_profession_analogies.py
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

# =============================================================================
# TKS CANON - NO HEBREW
# =============================================================================

WORLDS = {
    "A": "Spiritual",
    "B": "Mental",
    "C": "Emotional",
    "D": "Physical",
}

POSITIONS = {
    "1": "Mind",
    "2": "Positive",
    "3": "Negative",
    "4": "Vibration",
    "5": "Female",
    "6": "Male",
    "7": "Rhythm",
    "8": "Cause",
    "9": "Effect",
    "10": "All",
}

OPERATORS = ["+T", "-T", "*T", "/T", "+", "-", "->", "<-", "o"]

# =============================================================================
# PROFESSION VOCABULARIES - 25 PROFESSIONS
# =============================================================================

PROFESSIONS = {
    # =========================================================================
    # MEDICAL (3)
    # =========================================================================
    "doctor": {
        "name": "Doctor",
        "worlds": {
            "A": ["patient's spirit", "will to live", "inner healing force", "vital essence"],
            "B": ["diagnosis", "clinical reasoning", "medical judgment", "prognosis"],
            "C": ["patient's mood", "emotional state", "bedside manner", "therapeutic rapport"],
            "D": ["physical symptoms", "body", "vital signs", "lab results"],
        },
        "positions": {
            "1": ["clinical intuition", "medical insight", "diagnostic awareness"],
            "2": ["positive prognosis", "healing response", "recovery signs"],
            "3": ["complications", "adverse reactions", "declining vitals"],
            "4": ["heart rate", "pulse", "respiratory rhythm"],
            "5": ["receptive healing", "absorption of treatment", "patient compliance"],
            "6": ["active intervention", "surgical action", "treatment administration"],
            "7": ["circadian rhythm", "medication schedule", "treatment cycles"],
            "8": ["etiology", "root cause", "pathogen source"],
            "9": ["prognosis", "outcome", "treatment result"],
            "10": ["whole patient care", "holistic health", "complete recovery"],
        },
        "operators": {
            "+T": ["builds up over the treatment course", "accumulates therapeutically"],
            "-T": ["diminishes with treatment", "reduces over recovery"],
            "*T": ["amplifies healing response", "potentiates treatment"],
            "/T": ["attenuates symptoms", "reduces severity"],
            "+": ["combines with", "supplements"],
            "-": ["contraindicates", "removes from protocol"],
            "->": ["progresses to", "leads to diagnosis of"],
            "<-": ["stems from", "originates from condition"],
            "o": ["cycles through treatment", "circulates in system"],
        },
        "templates": [
            "Like how {elem1} {op} {elem2} in patient care",
            "Similar to when {elem1} {op} {elem2} during treatment",
            "Just as {elem1} {op} {elem2} in clinical practice",
        ],
    },

    "nurse": {
        "name": "Nurse",
        "worlds": {
            "A": ["patient's fighting spirit", "will to heal", "inner strength"],
            "B": ["care plan", "nursing assessment", "clinical judgment"],
            "C": ["patient comfort", "emotional support", "compassionate care"],
            "D": ["vital signs", "physical assessment", "wound care"],
        },
        "positions": {
            "1": ["nursing intuition", "patient awareness", "care instinct"],
            "2": ["improvement", "positive response", "stable vitals"],
            "3": ["deterioration", "warning signs", "declining status"],
            "4": ["pulse rhythm", "breathing pattern", "vital fluctuations"],
            "5": ["patient receptivity", "care acceptance", "treatment compliance"],
            "6": ["active nursing intervention", "medication administration", "direct care"],
            "7": ["shift rhythm", "medication rounds", "monitoring schedule"],
            "8": ["underlying condition", "chief complaint", "admission reason"],
            "9": ["patient outcome", "discharge status", "recovery progress"],
            "10": ["total patient care", "holistic nursing", "complete assessment"],
        },
        "operators": {
            "+T": ["builds through shifts", "accumulates over care"],
            "-T": ["reduces with intervention", "decreases through care"],
            "*T": ["enhances patient response", "amplifies healing"],
            "/T": ["minimizes discomfort", "reduces pain levels"],
            "+": ["adds to care plan", "includes in protocol"],
            "-": ["removes from orders", "discontinues"],
            "->": ["transitions to", "progresses toward"],
            "<-": ["responds to", "results from"],
            "o": ["cycles through rounds", "rotates in schedule"],
        },
        "templates": [
            "Like when {elem1} {op} {elem2} during patient rounds",
            "Similar to how {elem1} {op} {elem2} in nursing care",
            "Just as {elem1} {op} {elem2} on the ward",
        ],
    },

    "surgeon": {
        "name": "Surgeon",
        "worlds": {
            "A": ["surgical intuition", "operative instinct", "life force"],
            "B": ["surgical plan", "operative strategy", "procedural decision"],
            "C": ["surgical confidence", "operative calm", "focus under pressure"],
            "D": ["incision site", "operative field", "tissue", "anatomy"],
        },
        "positions": {
            "1": ["surgical awareness", "operative consciousness", "precision focus"],
            "2": ["successful outcome", "clean margins", "optimal result"],
            "3": ["complications", "bleeding", "adverse event"],
            "4": ["heart monitor rhythm", "anesthesia depth", "vital oscillation"],
            "5": ["tissue receptivity", "healing capacity", "surgical access"],
            "6": ["scalpel action", "incision", "surgical intervention"],
            "7": ["operative rhythm", "suture timing", "procedural pace"],
            "8": ["pathology source", "disease origin", "surgical indication"],
            "9": ["surgical outcome", "post-op result", "recovery trajectory"],
            "10": ["complete resection", "total repair", "full reconstruction"],
        },
        "operators": {
            "+T": ["builds through procedure", "layers during operation"],
            "-T": ["excises over time", "removes progressively"],
            "*T": ["amplifies surgical effect", "enhances repair"],
            "/T": ["minimizes tissue damage", "reduces invasiveness"],
            "+": ["joins tissue", "anastomoses"],
            "-": ["separates", "dissects from"],
            "->": ["progresses to next step", "advances to"],
            "<-": ["traces back to", "originates from"],
            "o": ["cycles through sutures", "repeats technique"],
        },
        "templates": [
            "Like how {elem1} {op} {elem2} in the OR",
            "Similar to when {elem1} {op} {elem2} during surgery",
            "Just as {elem1} {op} {elem2} on the operating table",
        ],
    },

    # =========================================================================
    # TRADES (6)
    # =========================================================================
    "electrician": {
        "name": "Electrician",
        "worlds": {
            "A": ["electrical potential", "power source essence", "energy field"],
            "B": ["circuit design", "wiring plan", "electrical schematic"],
            "C": ["safety instinct", "gut feeling about wiring", "electrical intuition"],
            "D": ["wires", "conduit", "breaker box", "outlets"],
        },
        "positions": {
            "1": ["electrical awareness", "circuit consciousness", "load understanding"],
            "2": ["positive terminal", "hot wire", "power supply"],
            "3": ["ground", "neutral", "return path"],
            "4": ["AC frequency", "current oscillation", "voltage fluctuation"],
            "5": ["load receptivity", "power draw", "circuit acceptance"],
            "6": ["power output", "current projection", "voltage supply"],
            "7": ["alternating current cycle", "switching rhythm", "timer pattern"],
            "8": ["power source", "main feed", "service entrance"],
            "9": ["powered device", "load result", "circuit endpoint"],
            "10": ["complete circuit", "whole system", "full installation"],
        },
        "operators": {
            "+T": ["builds charge over time", "accumulates current"],
            "-T": ["drains over time", "bleeds off charge"],
            "*T": ["amplifies through transformer", "boosts voltage"],
            "/T": ["steps down through transformer", "reduces voltage"],
            "+": ["wires in series", "connects to"],
            "-": ["disconnects from", "isolates from"],
            "->": ["flows to", "powers"],
            "<-": ["draws from", "pulls current from"],
            "o": ["cycles through circuit", "alternates in"],
        },
        "templates": [
            "Like how {elem1} {op} {elem2} in a circuit",
            "Similar to when {elem1} {op} {elem2} through the wiring",
            "Just as {elem1} {op} {elem2} in electrical work",
        ],
    },

    "plumber": {
        "name": "Plumber",
        "worlds": {
            "A": ["water pressure source", "flow essence", "hydraulic force"],
            "B": ["plumbing plan", "pipe layout", "system design"],
            "C": ["feel for the pipes", "instinct about flow", "plumber's intuition"],
            "D": ["pipes", "fittings", "valves", "fixtures"],
        },
        "positions": {
            "1": ["flow awareness", "pressure consciousness", "system understanding"],
            "2": ["supply line", "inlet", "pressure source"],
            "3": ["drain", "waste line", "pressure release"],
            "4": ["water hammer", "pressure fluctuation", "flow pulsation"],
            "5": ["fixture intake", "pipe acceptance", "flow receptivity"],
            "6": ["pump output", "pressure projection", "flow emission"],
            "7": ["usage cycle", "demand pattern", "flow rhythm"],
            "8": ["main supply", "water source", "system origin"],
            "9": ["drain endpoint", "waste destination", "flow result"],
            "10": ["complete system", "whole plumbing", "full installation"],
        },
        "operators": {
            "+T": ["builds pressure over time", "accumulates flow"],
            "-T": ["drains over time", "reduces pressure"],
            "*T": ["boosts through pump", "increases pressure"],
            "/T": ["reduces through regulator", "decreases flow"],
            "+": ["tees into", "joins with"],
            "-": ["caps off", "separates from"],
            "->": ["flows to", "drains to"],
            "<-": ["feeds from", "supplies from"],
            "o": ["recirculates through", "cycles in system"],
        },
        "templates": [
            "Like how {elem1} {op} {elem2} through the pipes",
            "Similar to when {elem1} {op} {elem2} in the plumbing",
            "Just as {elem1} {op} {elem2} in a water system",
        ],
    },

    "mechanic": {
        "name": "Mechanic",
        "worlds": {
            "A": ["engine's soul", "mechanical spirit", "power essence"],
            "B": ["diagnostic thinking", "troubleshooting logic", "repair strategy"],
            "C": ["feel for the machine", "mechanical intuition", "engine sense"],
            "D": ["engine parts", "components", "hardware", "chassis"],
        },
        "positions": {
            "1": ["mechanical awareness", "system understanding", "diagnostic sense"],
            "2": ["power stroke", "acceleration", "positive torque"],
            "3": ["resistance", "friction", "mechanical drag"],
            "4": ["engine RPM", "vibration frequency", "oscillation"],
            "5": ["fuel intake", "air intake", "system receptivity"],
            "6": ["power output", "exhaust", "force projection"],
            "7": ["firing order", "timing cycle", "mechanical rhythm"],
            "8": ["ignition source", "starter", "power origin"],
            "9": ["wheel motion", "output result", "mechanical effect"],
            "10": ["complete engine", "whole drivetrain", "full system"],
        },
        "operators": {
            "+T": ["builds RPM over time", "accumulates torque"],
            "-T": ["decelerates over time", "loses power"],
            "*T": ["multiplies through gearing", "amplifies torque"],
            "/T": ["reduces through gear ratio", "decreases speed"],
            "+": ["bolts to", "connects with"],
            "-": ["disconnects from", "removes from"],
            "->": ["transfers power to", "drives"],
            "<-": ["receives power from", "is driven by"],
            "o": ["cycles through engine", "rotates in"],
        },
        "templates": [
            "Like how {elem1} {op} {elem2} in an engine",
            "Similar to when {elem1} {op} {elem2} under the hood",
            "Just as {elem1} {op} {elem2} in the drivetrain",
        ],
    },

    "carpenter": {
        "name": "Carpenter",
        "worlds": {
            "A": ["wood's spirit", "grain essence", "natural character"],
            "B": ["build plan", "blueprint", "design vision"],
            "C": ["craftsman's feel", "wood intuition", "builder's sense"],
            "D": ["lumber", "boards", "joints", "structure"],
        },
        "positions": {
            "1": ["builder's awareness", "structural understanding", "craft consciousness"],
            "2": ["strong joint", "solid connection", "structural integrity"],
            "3": ["weak point", "stress fracture", "structural flaw"],
            "4": ["wood resonance", "material vibration", "grain pattern"],
            "5": ["mortise", "receiving joint", "female connection"],
            "6": ["tenon", "inserting piece", "male connection"],
            "7": ["work rhythm", "hammering cadence", "sawing pace"],
            "8": ["foundation", "base structure", "starting point"],
            "9": ["finished piece", "completed structure", "final build"],
            "10": ["complete structure", "whole building", "full framework"],
        },
        "operators": {
            "+T": ["builds up over project", "layers through construction"],
            "-T": ["wears down over time", "erodes through use"],
            "*T": ["strengthens through bracing", "reinforces"],
            "/T": ["thins through planing", "reduces through sanding"],
            "+": ["joins to", "fastens with"],
            "-": ["separates from", "cuts from"],
            "->": ["supports", "bears load to"],
            "<-": ["rests on", "is supported by"],
            "o": ["interlocks with", "dovetails through"],
        },
        "templates": [
            "Like how {elem1} {op} {elem2} in woodworking",
            "Similar to when {elem1} {op} {elem2} in construction",
            "Just as {elem1} {op} {elem2} in the framework",
        ],
    },

    "welder": {
        "name": "Welder",
        "worlds": {
            "A": ["arc essence", "fusion spirit", "bonding force"],
            "B": ["weld plan", "joint design", "fusion strategy"],
            "C": ["welder's instinct", "arc feel", "puddle sense"],
            "D": ["metal", "electrode", "filler", "base material"],
        },
        "positions": {
            "1": ["arc awareness", "puddle consciousness", "weld understanding"],
            "2": ["strong bead", "full penetration", "solid fusion"],
            "3": ["porosity", "cold lap", "weld defect"],
            "4": ["arc oscillation", "puddle vibration", "heat fluctuation"],
            "5": ["base metal receptivity", "heat acceptance", "fusion zone"],
            "6": ["filler deposition", "metal projection", "arc output"],
            "7": ["weaving pattern", "travel rhythm", "bead cadence"],
            "8": ["arc strike", "ignition point", "weld start"],
            "9": ["completed joint", "finished weld", "fused result"],
            "10": ["complete fabrication", "whole assembly", "full weldment"],
        },
        "operators": {
            "+T": ["builds bead over pass", "deposits through travel"],
            "-T": ["burns through over time", "undercuts"],
            "*T": ["intensifies heat", "amplifies penetration"],
            "/T": ["reduces heat input", "cools through pause"],
            "+": ["fuses to", "joins with"],
            "-": ["gouges from", "grinds away"],
            "->": ["flows into joint", "penetrates to"],
            "<-": ["draws heat from", "pulls from base"],
            "o": ["weaves through joint", "oscillates across"],
        },
        "templates": [
            "Like how {elem1} {op} {elem2} in the weld puddle",
            "Similar to when {elem1} {op} {elem2} under the arc",
            "Just as {elem1} {op} {elem2} in metal fabrication",
        ],
    },

    "hvac": {
        "name": "HVAC Technician",
        "worlds": {
            "A": ["thermal essence", "climate spirit", "comfort force"],
            "B": ["system design", "load calculation", "climate plan"],
            "C": ["comfort sense", "temperature feel", "climate intuition"],
            "D": ["ductwork", "compressor", "coils", "equipment"],
        },
        "positions": {
            "1": ["system awareness", "thermal understanding", "climate consciousness"],
            "2": ["heating mode", "warm air", "thermal gain"],
            "3": ["cooling mode", "cold air", "thermal loss"],
            "4": ["temperature swing", "thermal cycling", "comfort oscillation"],
            "5": ["return air intake", "heat absorption", "thermal receptivity"],
            "6": ["supply air output", "heat projection", "thermal emission"],
            "7": ["thermostat cycle", "compressor rhythm", "on-off pattern"],
            "8": ["heat source", "cold source", "thermal origin"],
            "9": ["conditioned space", "comfort result", "climate effect"],
            "10": ["complete system", "whole HVAC", "full climate control"],
        },
        "operators": {
            "+T": ["heats over cycle", "warms through runtime"],
            "-T": ["cools over cycle", "chills through runtime"],
            "*T": ["boosts capacity", "amplifies output"],
            "/T": ["reduces load", "decreases demand"],
            "+": ["connects to plenum", "joins ductwork"],
            "-": ["isolates from system", "dampers off"],
            "->": ["supplies air to", "conditions"],
            "<-": ["returns from", "draws air from"],
            "o": ["circulates through", "cycles in system"],
        },
        "templates": [
            "Like how {elem1} {op} {elem2} in the HVAC system",
            "Similar to when {elem1} {op} {elem2} through the ductwork",
            "Just as {elem1} {op} {elem2} in climate control",
        ],
    },

    # =========================================================================
    # CREATIVE (5)
    # =========================================================================
    "musician": {
        "name": "Musician",
        "worlds": {
            "A": ["musical soul", "artistic spirit", "creative essence"],
            "B": ["composition", "musical theory", "arrangement"],
            "C": ["musical feeling", "emotional expression", "performance mood"],
            "D": ["instrument", "sound waves", "notes", "physical performance"],
        },
        "positions": {
            "1": ["musical awareness", "tonal consciousness", "harmonic understanding"],
            "2": ["major key", "consonance", "resolution"],
            "3": ["minor key", "dissonance", "tension"],
            "4": ["vibrato", "pitch oscillation", "tonal vibration"],
            "5": ["listening", "receiving the music", "audience receptivity"],
            "6": ["playing", "performing", "projecting sound"],
            "7": ["tempo", "beat", "rhythmic pattern"],
            "8": ["first note", "downbeat", "musical origin"],
            "9": ["final chord", "resolution", "musical conclusion"],
            "10": ["complete composition", "whole piece", "full performance"],
        },
        "operators": {
            "+T": ["builds through crescendo", "swells over measures"],
            "-T": ["fades through decrescendo", "diminishes over bars"],
            "*T": ["harmonizes with", "resonates through octaves"],
            "/T": ["softens through dynamics", "attenuates volume"],
            "+": ["adds voice to", "layers with"],
            "-": ["rests from", "pauses"],
            "->": ["resolves to", "modulates to"],
            "<-": ["derives from motif", "echoes from"],
            "o": ["repeats through chorus", "cycles in verse"],
        },
        "templates": [
            "Like how {elem1} {op} {elem2} in music",
            "Similar to when {elem1} {op} {elem2} in a composition",
            "Just as {elem1} {op} {elem2} during performance",
        ],
    },

    "chef": {
        "name": "Chef",
        "worlds": {
            "A": ["culinary spirit", "cooking soul", "flavor essence"],
            "B": ["recipe", "menu planning", "culinary concept"],
            "C": ["palate", "taste intuition", "flavor feeling"],
            "D": ["ingredients", "cookware", "plated dish", "kitchen"],
        },
        "positions": {
            "1": ["culinary awareness", "flavor consciousness", "taste understanding"],
            "2": ["sweetness", "umami", "pleasant flavor"],
            "3": ["bitterness", "sourness", "sharp notes"],
            "4": ["sizzle", "bubbling", "cooking vibration"],
            "5": ["absorption", "marinating", "flavor receptivity"],
            "6": ["seasoning", "plating", "flavor projection"],
            "7": ["cooking timing", "prep rhythm", "kitchen cadence"],
            "8": ["raw ingredient", "mise en place", "starting point"],
            "9": ["finished dish", "plated result", "culinary outcome"],
            "10": ["complete meal", "whole menu", "full service"],
        },
        "operators": {
            "+T": ["reduces over heat", "concentrates through cooking"],
            "-T": ["evaporates over time", "reduces through simmering"],
            "*T": ["intensifies flavor", "amplifies through reduction"],
            "/T": ["dilutes with stock", "lightens sauce"],
            "+": ["folds into", "combines with"],
            "-": ["strains from", "separates out"],
            "->": ["transforms into", "becomes"],
            "<-": ["derives flavor from", "infuses from"],
            "o": ["stirs through", "tosses in"],
        },
        "templates": [
            "Like how {elem1} {op} {elem2} in the kitchen",
            "Similar to when {elem1} {op} {elem2} during cooking",
            "Just as {elem1} {op} {elem2} in a recipe",
        ],
    },

    "artist": {
        "name": "Artist/Painter",
        "worlds": {
            "A": ["artistic vision", "creative spirit", "inspiration"],
            "B": ["composition", "artistic concept", "design theory"],
            "C": ["artistic feeling", "emotional expression", "creative mood"],
            "D": ["canvas", "paint", "brush strokes", "physical artwork"],
        },
        "positions": {
            "1": ["artistic awareness", "visual consciousness", "aesthetic sense"],
            "2": ["warm colors", "highlights", "positive space"],
            "3": ["cool colors", "shadows", "negative space"],
            "4": ["visual rhythm", "pattern repetition", "color vibration"],
            "5": ["canvas receptivity", "paint absorption", "surface acceptance"],
            "6": ["brush projection", "color application", "stroke expression"],
            "7": ["brushwork rhythm", "painting cadence", "creative flow"],
            "8": ["initial sketch", "underpainting", "artistic origin"],
            "9": ["finished piece", "completed work", "artistic result"],
            "10": ["complete artwork", "whole composition", "full vision"],
        },
        "operators": {
            "+T": ["layers over sessions", "builds through glazing"],
            "-T": ["fades over time", "washes out"],
            "*T": ["intensifies color", "saturates"],
            "/T": ["mutes tone", "desaturates"],
            "+": ["blends with", "mixes into"],
            "-": ["scrapes away", "removes from"],
            "->": ["transitions to", "gradates into"],
            "<-": ["reflects from", "echoes"],
            "o": ["repeats through pattern", "cycles in motif"],
        },
        "templates": [
            "Like how {elem1} {op} {elem2} on canvas",
            "Similar to when {elem1} {op} {elem2} in painting",
            "Just as {elem1} {op} {elem2} in visual art",
        ],
    },

    "dj": {
        "name": "DJ/Producer",
        "worlds": {
            "A": ["sonic spirit", "vibe essence", "energy force"],
            "B": ["set planning", "track selection", "mix strategy"],
            "C": ["crowd feel", "energy reading", "vibe sense"],
            "D": ["speakers", "decks", "mixer", "sound system"],
        },
        "positions": {
            "1": ["sonic awareness", "mix consciousness", "beat understanding"],
            "2": ["drop", "peak energy", "crowd hype"],
            "3": ["breakdown", "tension build", "energy dip"],
            "4": ["bass wobble", "frequency oscillation", "waveform"],
            "5": ["crowd reception", "dancefloor absorption", "vibe intake"],
            "6": ["speaker output", "sound projection", "bass drop"],
            "7": ["BPM", "beat pattern", "tempo rhythm"],
            "8": ["intro", "first track", "set opener"],
            "9": ["outro", "final track", "set closer"],
            "10": ["complete set", "whole mix", "full performance"],
        },
        "operators": {
            "+T": ["builds through buildup", "layers over bars"],
            "-T": ["filters down", "fades through breakdown"],
            "*T": ["boosts through EQ", "amplifies bass"],
            "/T": ["cuts frequency", "attenuates highs"],
            "+": ["mixes with", "blends into"],
            "-": ["cuts from mix", "drops out"],
            "->": ["transitions to", "beatmatches into"],
            "<-": ["samples from", "loops from"],
            "o": ["loops through", "cycles beat"],
        },
        "templates": [
            "Like how {elem1} {op} {elem2} in a DJ set",
            "Similar to when {elem1} {op} {elem2} on the decks",
            "Just as {elem1} {op} {elem2} in the mix",
        ],
    },

    "writer": {
        "name": "Writer",
        "worlds": {
            "A": ["creative muse", "inspiration", "story spirit"],
            "B": ["plot structure", "narrative arc", "story plan"],
            "C": ["emotional tone", "mood", "narrative feeling"],
            "D": ["words", "pages", "manuscript", "text"],
        },
        "positions": {
            "1": ["narrative awareness", "story consciousness", "plot sense"],
            "2": ["rising action", "conflict resolution", "positive arc"],
            "3": ["falling action", "conflict", "tension"],
            "4": ["prose rhythm", "sentence cadence", "word flow"],
            "5": ["reader reception", "audience engagement", "story absorption"],
            "6": ["authorial voice", "narrative projection", "story telling"],
            "7": ["chapter rhythm", "scene pacing", "narrative beat"],
            "8": ["opening line", "inciting incident", "story origin"],
            "9": ["conclusion", "denouement", "story ending"],
            "10": ["complete manuscript", "whole story", "full narrative"],
        },
        "operators": {
            "+T": ["develops over chapters", "builds through scenes"],
            "-T": ["diminishes tension", "resolves through pages"],
            "*T": ["intensifies conflict", "heightens drama"],
            "/T": ["softens tone", "reduces tension"],
            "+": ["weaves with subplot", "adds to narrative"],
            "-": ["edits out", "cuts from draft"],
            "->": ["leads to climax", "progresses to"],
            "<-": ["flashbacks to", "references"],
            "o": ["echoes through motif", "repeats theme"],
        },
        "templates": [
            "Like how {elem1} {op} {elem2} in storytelling",
            "Similar to when {elem1} {op} {elem2} in narrative",
            "Just as {elem1} {op} {elem2} in writing",
        ],
    },

    # =========================================================================
    # TECHNICAL (4)
    # =========================================================================
    "programmer": {
        "name": "Programmer",
        "worlds": {
            "A": ["code elegance", "algorithmic beauty", "software spirit"],
            "B": ["architecture", "system design", "code logic"],
            "C": ["developer intuition", "code feel", "debugging sense"],
            "D": ["code", "data", "hardware", "servers"],
        },
        "positions": {
            "1": ["system awareness", "code consciousness", "architectural understanding"],
            "2": ["success state", "valid output", "positive result"],
            "3": ["error state", "exception", "bug"],
            "4": ["clock cycle", "processing frequency", "loop iteration"],
            "5": ["input handling", "data reception", "parameter intake"],
            "6": ["output generation", "return value", "data emission"],
            "7": ["execution loop", "process cycle", "runtime rhythm"],
            "8": ["initialization", "entry point", "main function"],
            "9": ["return value", "output result", "program termination"],
            "10": ["complete system", "full stack", "whole application"],
        },
        "operators": {
            "+T": ["accumulates in loop", "builds through iterations"],
            "-T": ["decrements over cycles", "reduces through processing"],
            "*T": ["scales through parallelism", "multiplies threads"],
            "/T": ["partitions data", "divides workload"],
            "+": ["concatenates with", "appends to"],
            "-": ["removes from array", "filters out"],
            "->": ["passes to function", "pipes to"],
            "<-": ["receives from callback", "pulls from API"],
            "o": ["iterates through", "loops over"],
        },
        "templates": [
            "Like how {elem1} {op} {elem2} in code",
            "Similar to when {elem1} {op} {elem2} in programming",
            "Just as {elem1} {op} {elem2} in software",
        ],
    },

    "engineer": {
        "name": "Engineer",
        "worlds": {
            "A": ["engineering elegance", "design spirit", "innovation essence"],
            "B": ["specifications", "technical design", "engineering plan"],
            "C": ["engineering intuition", "design sense", "problem-solving feel"],
            "D": ["materials", "structures", "mechanisms", "hardware"],
        },
        "positions": {
            "1": ["systems thinking", "engineering awareness", "design consciousness"],
            "2": ["load capacity", "structural strength", "positive tolerance"],
            "3": ["stress point", "failure mode", "structural weakness"],
            "4": ["resonant frequency", "vibration mode", "oscillation"],
            "5": ["load bearing", "force reception", "structural intake"],
            "6": ["force transmission", "load distribution", "structural output"],
            "7": ["operational cycle", "maintenance rhythm", "design lifecycle"],
            "8": ["design origin", "initial concept", "engineering source"],
            "9": ["final product", "built result", "engineering outcome"],
            "10": ["complete system", "whole design", "full engineering solution"],
        },
        "operators": {
            "+T": ["accumulates stress over cycles", "builds fatigue"],
            "-T": ["dissipates over time", "reduces through damping"],
            "*T": ["amplifies through leverage", "multiplies force"],
            "/T": ["distributes load", "reduces through factor"],
            "+": ["integrates with system", "joins to"],
            "-": ["decouples from", "isolates"],
            "->": ["transfers force to", "transmits to"],
            "<-": ["receives load from", "is driven by"],
            "o": ["cycles through operation", "rotates in mechanism"],
        },
        "templates": [
            "Like how {elem1} {op} {elem2} in engineering",
            "Similar to when {elem1} {op} {elem2} in design",
            "Just as {elem1} {op} {elem2} in the system",
        ],
    },

    "architect": {
        "name": "Architect",
        "worlds": {
            "A": ["design vision", "architectural spirit", "spatial essence"],
            "B": ["blueprints", "floor plan", "design concept"],
            "C": ["spatial feeling", "design intuition", "aesthetic sense"],
            "D": ["building", "materials", "structure", "space"],
        },
        "positions": {
            "1": ["spatial awareness", "design consciousness", "architectural vision"],
            "2": ["open space", "light", "positive volume"],
            "3": ["solid mass", "shadow", "negative space"],
            "4": ["spatial rhythm", "module repetition", "design pattern"],
            "5": ["interior space", "enclosed volume", "spatial reception"],
            "6": ["facade", "exterior expression", "spatial projection"],
            "7": ["circulation pattern", "flow rhythm", "spatial sequence"],
            "8": ["site", "foundation", "design origin"],
            "9": ["completed building", "built form", "design result"],
            "10": ["complete design", "whole building", "full vision"],
        },
        "operators": {
            "+T": ["develops through phases", "builds over construction"],
            "-T": ["weathers over time", "ages through use"],
            "*T": ["amplifies through scale", "expands volume"],
            "/T": ["subdivides space", "partitions area"],
            "+": ["connects to", "joins with"],
            "-": ["separates from", "carves out"],
            "->": ["leads to", "opens onto"],
            "<-": ["receives light from", "is accessed from"],
            "o": ["circulates through", "flows around"],
        },
        "templates": [
            "Like how {elem1} {op} {elem2} in architecture",
            "Similar to when {elem1} {op} {elem2} in building design",
            "Just as {elem1} {op} {elem2} in spatial planning",
        ],
    },

    "data_scientist": {
        "name": "Data Scientist",
        "worlds": {
            "A": ["data insight", "analytical spirit", "pattern essence"],
            "B": ["model architecture", "algorithm design", "analysis plan"],
            "C": ["data intuition", "pattern sense", "analytical feel"],
            "D": ["datasets", "features", "metrics", "outputs"],
        },
        "positions": {
            "1": ["analytical awareness", "data consciousness", "pattern recognition"],
            "2": ["positive correlation", "signal", "significant result"],
            "3": ["negative correlation", "noise", "outlier"],
            "4": ["variance", "data oscillation", "feature fluctuation"],
            "5": ["input features", "training data", "model intake"],
            "6": ["predictions", "model output", "inference"],
            "7": ["training cycle", "epoch rhythm", "iteration pattern"],
            "8": ["raw data", "data source", "input origin"],
            "9": ["final prediction", "model result", "analytical outcome"],
            "10": ["complete pipeline", "full model", "whole analysis"],
        },
        "operators": {
            "+T": ["accumulates over epochs", "trains through iterations"],
            "-T": ["decays learning rate", "reduces over time"],
            "*T": ["amplifies weights", "boosts feature importance"],
            "/T": ["normalizes", "scales down"],
            "+": ["concatenates features", "merges datasets"],
            "-": ["drops features", "removes outliers"],
            "->": ["feeds into model", "transforms to"],
            "<-": ["derives from data", "extracts from"],
            "o": ["cross-validates through folds", "iterates over batches"],
        },
        "templates": [
            "Like how {elem1} {op} {elem2} in data science",
            "Similar to when {elem1} {op} {elem2} in machine learning",
            "Just as {elem1} {op} {elem2} in analysis",
        ],
    },

    # =========================================================================
    # SERVICE (4)
    # =========================================================================
    "teacher": {
        "name": "Teacher",
        "worlds": {
            "A": ["teaching spirit", "educational vision", "wisdom essence"],
            "B": ["lesson plan", "curriculum", "pedagogical design"],
            "C": ["classroom mood", "student rapport", "teaching intuition"],
            "D": ["classroom", "materials", "assignments", "grades"],
        },
        "positions": {
            "1": ["educational awareness", "learning consciousness", "teaching sense"],
            "2": ["student success", "positive feedback", "learning gain"],
            "3": ["student struggle", "learning gap", "comprehension block"],
            "4": ["attention span", "engagement fluctuation", "focus rhythm"],
            "5": ["student receptivity", "learning intake", "knowledge absorption"],
            "6": ["instruction delivery", "knowledge projection", "teaching output"],
            "7": ["class schedule", "lesson rhythm", "academic calendar"],
            "8": ["introduction", "lesson opener", "teaching start"],
            "9": ["assessment result", "learning outcome", "student achievement"],
            "10": ["complete education", "whole curriculum", "full learning"],
        },
        "operators": {
            "+T": ["builds over semester", "develops through units"],
            "-T": ["fades over break", "diminishes without practice"],
            "*T": ["reinforces through repetition", "strengthens understanding"],
            "/T": ["simplifies for comprehension", "breaks down concept"],
            "+": ["scaffolds with", "builds upon"],
            "-": ["differentiates from", "distinguishes"],
            "->": ["leads to understanding", "connects to"],
            "<-": ["builds from prior knowledge", "derives from"],
            "o": ["reviews through cycle", "spirals back to"],
        },
        "templates": [
            "Like how {elem1} {op} {elem2} in teaching",
            "Similar to when {elem1} {op} {elem2} in the classroom",
            "Just as {elem1} {op} {elem2} in education",
        ],
    },

    "coach": {
        "name": "Coach/Athlete",
        "worlds": {
            "A": ["competitive spirit", "athletic drive", "winning essence"],
            "B": ["game plan", "strategy", "play design"],
            "C": ["team morale", "competitive fire", "athletic intuition"],
            "D": ["body", "equipment", "field", "court"],
        },
        "positions": {
            "1": ["game awareness", "athletic consciousness", "sport IQ"],
            "2": ["momentum", "winning streak", "positive play"],
            "3": ["deficit", "losing streak", "negative momentum"],
            "4": ["heart rate", "breathing rhythm", "athletic tempo"],
            "5": ["defensive stance", "receiving position", "athletic reception"],
            "6": ["offensive drive", "attacking motion", "athletic projection"],
            "7": ["game rhythm", "play cadence", "training cycle"],
            "8": ["opening play", "first possession", "game start"],
            "9": ["final score", "game result", "athletic outcome"],
            "10": ["complete game", "whole season", "full performance"],
        },
        "operators": {
            "+T": ["builds through quarters", "accumulates through game"],
            "-T": ["fatigues over time", "diminishes through exertion"],
            "*T": ["amplifies through teamwork", "multiplies effort"],
            "/T": ["paces through conservation", "distributes energy"],
            "+": ["combines with teammate", "adds to lineup"],
            "-": ["subs out", "benches"],
            "->": ["passes to", "advances toward"],
            "<-": ["receives from", "gets assist from"],
            "o": ["rotates through positions", "cycles plays"],
        },
        "templates": [
            "Like how {elem1} {op} {elem2} in sports",
            "Similar to when {elem1} {op} {elem2} on the field",
            "Just as {elem1} {op} {elem2} in athletics",
        ],
    },

    "firefighter": {
        "name": "Firefighter",
        "worlds": {
            "A": ["service spirit", "heroic drive", "life-saving essence"],
            "B": ["incident command", "tactical plan", "rescue strategy"],
            "C": ["situational awareness", "danger sense", "firefighter intuition"],
            "D": ["fire", "equipment", "structure", "victims"],
        },
        "positions": {
            "1": ["scene awareness", "fire consciousness", "tactical sense"],
            "2": ["containment", "fire suppression", "positive outcome"],
            "3": ["spread", "flashover risk", "dangerous condition"],
            "4": ["flame oscillation", "heat fluctuation", "fire behavior"],
            "5": ["water intake", "hose reception", "equipment acceptance"],
            "6": ["water projection", "foam discharge", "suppression output"],
            "7": ["shift rhythm", "response cycle", "rotation schedule"],
            "8": ["ignition source", "fire origin", "point of origin"],
            "9": ["fire out", "rescue complete", "incident resolved"],
            "10": ["complete response", "whole operation", "full incident"],
        },
        "operators": {
            "+T": ["intensifies over time", "spreads through structure"],
            "-T": ["suppresses over time", "reduces through attack"],
            "*T": ["accelerates with fuel", "amplifies with oxygen"],
            "/T": ["smothers with foam", "cuts oxygen supply"],
            "+": ["connects hose to", "adds resources to"],
            "-": ["ventilates from", "removes fuel from"],
            "->": ["advances toward", "attacks"],
            "<-": ["retreats from", "pulls back from"],
            "o": ["rotates crews through", "cycles attack"],
        },
        "templates": [
            "Like how {elem1} {op} {elem2} in firefighting",
            "Similar to when {elem1} {op} {elem2} on scene",
            "Just as {elem1} {op} {elem2} during rescue",
        ],
    },

    "detective": {
        "name": "Police Detective",
        "worlds": {
            "A": ["investigative instinct", "justice drive", "truth essence"],
            "B": ["case theory", "investigation plan", "evidence logic"],
            "C": ["gut feeling", "suspect intuition", "detective sense"],
            "D": ["evidence", "crime scene", "witnesses", "case file"],
        },
        "positions": {
            "1": ["investigative awareness", "case consciousness", "detective mind"],
            "2": ["lead", "breakthrough", "positive evidence"],
            "3": ["dead end", "false lead", "contradicting evidence"],
            "4": ["case momentum", "investigation rhythm", "lead fluctuation"],
            "5": ["witness testimony", "evidence reception", "information intake"],
            "6": ["interrogation", "investigation push", "active pursuit"],
            "7": ["investigation cycle", "follow-up rhythm", "case cadence"],
            "8": ["crime origin", "initial incident", "case start"],
            "9": ["case closed", "arrest made", "investigation result"],
            "10": ["complete case", "whole investigation", "full justice"],
        },
        "operators": {
            "+T": ["builds evidence over time", "accumulates leads"],
            "-T": ["goes cold over time", "loses momentum"],
            "*T": ["corroborates evidence", "strengthens case"],
            "/T": ["narrows suspects", "eliminates possibilities"],
            "+": ["connects to case", "links evidence"],
            "-": ["rules out", "eliminates from"],
            "->": ["points to", "leads to suspect"],
            "<-": ["traces back to", "originates from"],
            "o": ["revisits through review", "cycles through leads"],
        },
        "templates": [
            "Like how {elem1} {op} {elem2} in investigation",
            "Similar to when {elem1} {op} {elem2} on a case",
            "Just as {elem1} {op} {elem2} in detective work",
        ],
    },

    # =========================================================================
    # BUSINESS (3)
    # =========================================================================
    "banker": {
        "name": "Banker/Finance",
        "worlds": {
            "A": ["market spirit", "financial intuition", "wealth essence"],
            "B": ["financial model", "investment strategy", "portfolio theory"],
            "C": ["market sentiment", "risk appetite", "trading feel"],
            "D": ["money", "assets", "accounts", "transactions"],
        },
        "positions": {
            "1": ["market awareness", "financial consciousness", "economic sense"],
            "2": ["profit", "gains", "positive return"],
            "3": ["loss", "deficit", "negative return"],
            "4": ["market volatility", "price oscillation", "rate fluctuation"],
            "5": ["deposits", "capital intake", "investment reception"],
            "6": ["loans", "capital output", "investment projection"],
            "7": ["market cycle", "fiscal rhythm", "trading pattern"],
            "8": ["principal", "initial investment", "capital origin"],
            "9": ["returns", "final balance", "investment result"],
            "10": ["complete portfolio", "whole market", "full financial picture"],
        },
        "operators": {
            "+T": ["compounds over time", "accrues interest"],
            "-T": ["depreciates over time", "loses value"],
            "*T": ["leverages returns", "multiplies gains"],
            "/T": ["diversifies risk", "splits investment"],
            "+": ["adds to portfolio", "invests in"],
            "-": ["divests from", "withdraws from"],
            "->": ["transfers to", "flows to account"],
            "<-": ["receives from", "draws from"],
            "o": ["reinvests through", "cycles capital"],
        },
        "templates": [
            "Like how {elem1} {op} {elem2} in finance",
            "Similar to when {elem1} {op} {elem2} in banking",
            "Just as {elem1} {op} {elem2} in the market",
        ],
    },

    "lawyer": {
        "name": "Lawyer",
        "worlds": {
            "A": ["justice spirit", "legal intuition", "truth essence"],
            "B": ["legal strategy", "case theory", "argument structure"],
            "C": ["courtroom feel", "jury sense", "legal instinct"],
            "D": ["documents", "evidence", "contracts", "court"],
        },
        "positions": {
            "1": ["legal awareness", "case consciousness", "judicial mind"],
            "2": ["favorable ruling", "winning argument", "positive precedent"],
            "3": ["adverse ruling", "opposing argument", "negative precedent"],
            "4": ["case momentum", "trial rhythm", "legal oscillation"],
            "5": ["client intake", "case reception", "evidence admission"],
            "6": ["argument delivery", "cross-examination", "legal projection"],
            "7": ["court schedule", "filing rhythm", "legal process cycle"],
            "8": ["incident", "cause of action", "legal origin"],
            "9": ["verdict", "settlement", "legal outcome"],
            "10": ["complete case", "whole matter", "full legal resolution"],
        },
        "operators": {
            "+T": ["builds case over discovery", "strengthens through depositions"],
            "-T": ["weakens over time", "diminishes through motions"],
            "*T": ["reinforces with precedent", "amplifies argument"],
            "/T": ["limits scope", "narrows claims"],
            "+": ["adds to brief", "supplements motion"],
            "-": ["strikes from record", "dismisses"],
            "->": ["establishes", "proves"],
            "<-": ["derives from statute", "cites from"],
            "o": ["appeals through courts", "cycles through system"],
        },
        "templates": [
            "Like how {elem1} {op} {elem2} in law",
            "Similar to when {elem1} {op} {elem2} in court",
            "Just as {elem1} {op} {elem2} in legal practice",
        ],
    },

    "salesperson": {
        "name": "Salesperson",
        "worlds": {
            "A": ["sales spirit", "closing instinct", "deal essence"],
            "B": ["sales strategy", "pitch plan", "closing technique"],
            "C": ["customer rapport", "buying signal sense", "sales intuition"],
            "D": ["product", "contracts", "commission", "territory"],
        },
        "positions": {
            "1": ["sales awareness", "deal consciousness", "customer read"],
            "2": ["buying signal", "yes momentum", "positive response"],
            "3": ["objection", "no signal", "resistance"],
            "4": ["interest fluctuation", "engagement rhythm", "attention span"],
            "5": ["customer needs", "pain points", "requirement intake"],
            "6": ["pitch delivery", "value proposition", "offer projection"],
            "7": ["sales cycle", "follow-up rhythm", "closing cadence"],
            "8": ["lead", "prospect", "opportunity start"],
            "9": ["closed deal", "signed contract", "sales result"],
            "10": ["complete sale", "whole relationship", "full account"],
        },
        "operators": {
            "+T": ["builds rapport over calls", "nurtures through touchpoints"],
            "-T": ["cools over time", "loses urgency"],
            "*T": ["upsells", "multiplies deal size"],
            "/T": ["discounts", "reduces price"],
            "+": ["bundles with", "adds to proposal"],
            "-": ["removes from offer", "excludes"],
            "->": ["closes toward", "advances to"],
            "<-": ["gets referral from", "receives lead from"],
            "o": ["follows up through", "cycles nurture"],
        },
        "templates": [
            "Like how {elem1} {op} {elem2} in sales",
            "Similar to when {elem1} {op} {elem2} with customers",
            "Just as {elem1} {op} {elem2} in closing deals",
        ],
    },
}

# =============================================================================
# REGULAR TKS VOCABULARY (for mappings)
# =============================================================================

REGULAR_VOCAB = {
    "worlds": {
        "A": ["spiritual realm", "transcendent plane", "spiritual dimension"],
        "B": ["mental realm", "cognitive plane", "mental dimension"],
        "C": ["emotional realm", "affective plane", "emotional dimension"],
        "D": ["physical realm", "material plane", "physical dimension"],
    },
    "positions": {
        "1": ["consciousness", "awareness", "mind principle"],
        "2": ["positive polarity", "attractive force", "expansion"],
        "3": ["negative polarity", "repulsive force", "contraction"],
        "4": ["vibration", "frequency", "oscillation"],
        "5": ["feminine principle", "receptivity", "intake"],
        "6": ["masculine principle", "projection", "output"],
        "7": ["rhythm", "cycle", "periodicity"],
        "8": ["cause", "origin", "source"],
        "9": ["effect", "result", "manifestation"],
        "10": ["totality", "wholeness", "integration"],
    },
    "operators": {
        "+T": ["accumulates temporally with", "builds over time with"],
        "-T": ["diminishes temporally from", "reduces over time from"],
        "*T": ["amplifies temporally with", "resonates over time with"],
        "/T": ["attenuates temporally through", "disperses over time through"],
        "+": ["combines with", "merges with"],
        "-": ["separates from", "differentiates from"],
        "->": ["flows to", "transforms into"],
        "<-": ["derives from", "receives from"],
        "o": ["cycles with", "circulates through"],
    },
    "templates": [
        "The {elem1} {op} the {elem2}",
        "{elem1} {op} {elem2} in this working",
        "This equation shows {elem1} {op} {elem2}",
    ],
}

# =============================================================================
# GENERATORS
# =============================================================================

def generate_element() -> str:
    """Generate random TKS element code."""
    world = random.choice(["A", "B", "C", "D"])
    pos = random.choice(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
    elem = f"{world}{pos}"

    if random.random() < 0.2:
        sense = random.randint(1, 10)
        elem += f"^{sense}"

    if random.random() < 0.1:
        found = random.randint(1, 7)
        elem += f"_d{found}"

    return elem


def generate_chain(length: int = None) -> str:
    """Generate random TKS operator chain."""
    if length is None:
        length = random.randint(2, 4)

    elements = [generate_element() for _ in range(length)]
    ops = [random.choice(OPERATORS) for _ in range(length - 1)]

    chain = elements[0]
    for i, op in enumerate(ops):
        chain += f" {op} {elements[i + 1]}"

    return chain


def get_element_desc(code: str, vocab: dict) -> str:
    """Get element description using vocabulary."""
    world = code[0]
    rest = code[1:]
    if rest.startswith("10"):
        pos = "10"
    else:
        pos = rest[0] if rest else "1"

    world_desc = random.choice(vocab["worlds"].get(world, [WORLDS[world]]))
    pos_desc = random.choice(vocab["positions"].get(pos, [POSITIONS[pos]]))

    return f"{pos_desc} in {world_desc}"


def generate_profession_sample(prof_key: str, prof: dict) -> Dict:
    """Generate one profession analogy sample."""
    chain = generate_chain()
    tokens = chain.split()
    elems = [t for t in tokens if t and t[0] in "ABCD"]
    ops = [t for t in tokens if t in OPERATORS]

    if len(elems) >= 2 and ops:
        elem1_desc = get_element_desc(elems[0], prof)
        elem2_desc = get_element_desc(elems[1], prof)
        op_desc = random.choice(prof["operators"].get(ops[0], [ops[0]]))

        template = random.choice(prof["templates"])
        story = template.format(elem1=elem1_desc, elem2=elem2_desc, op=op_desc)

        return {
            "story": story,
            "equation": chain,
            "source": f"profession_{prof_key}",
            "profession": prof["name"]
        }

    return {"story": f"Working with {chain}", "source": f"profession_{prof_key}"}


def generate_regular_sample(chain: str) -> str:
    """Generate regular TKS description for a chain."""
    tokens = chain.split()
    elems = [t for t in tokens if t and t[0] in "ABCD"]
    ops = [t for t in tokens if t in OPERATORS]

    if len(elems) >= 2 and ops:
        elem1_desc = get_element_desc(elems[0], REGULAR_VOCAB)
        elem2_desc = get_element_desc(elems[1], REGULAR_VOCAB)
        op_desc = random.choice(REGULAR_VOCAB["operators"].get(ops[0], [ops[0]]))

        template = random.choice(REGULAR_VOCAB["templates"])
        return template.format(elem1=elem1_desc, elem2=elem2_desc, op=op_desc)

    return f"The equation {chain}"


def generate_mapping_sample(chain: str, prof_key: str, prof: dict) -> Dict:
    """Generate 1:1 mapping between regular and profession analogy."""
    regular = generate_regular_sample(chain)

    tokens = chain.split()
    elems = [t for t in tokens if t and t[0] in "ABCD"]
    ops = [t for t in tokens if t in OPERATORS]

    if len(elems) >= 2 and ops:
        elem1_desc = get_element_desc(elems[0], prof)
        elem2_desc = get_element_desc(elems[1], prof)
        op_desc = random.choice(prof["operators"].get(ops[0], [ops[0]]))

        template = random.choice(prof["templates"])
        analogy = template.format(elem1=elem1_desc, elem2=elem2_desc, op=op_desc)

        # Randomly choose direction
        if random.random() < 0.5:
            return {
                "input": f"Express in {prof['name']} terms: {regular}",
                "output": analogy,
                "equation": chain,
                "source": f"mapping_{prof_key}"
            }
        else:
            return {
                "input": f"What does this mean in TKS? {analogy}",
                "output": f"{regular} (Equation: {chain})",
                "source": f"mapping_{prof_key}"
            }

    return {"story": regular, "source": f"mapping_{prof_key}"}


def generate_cross_profession_sample(chain: str, prof1_key: str, prof1: dict,
                                      prof2_key: str, prof2: dict) -> Dict:
    """Generate cross-profession mapping."""
    tokens = chain.split()
    elems = [t for t in tokens if t and t[0] in "ABCD"]
    ops = [t for t in tokens if t in OPERATORS]

    if len(elems) >= 2 and ops:
        # Prof 1 version
        elem1_desc_1 = get_element_desc(elems[0], prof1)
        elem2_desc_1 = get_element_desc(elems[1], prof1)
        op_desc_1 = random.choice(prof1["operators"].get(ops[0], [ops[0]]))
        template_1 = random.choice(prof1["templates"])
        analogy_1 = template_1.format(elem1=elem1_desc_1, elem2=elem2_desc_1, op=op_desc_1)

        # Prof 2 version
        elem1_desc_2 = get_element_desc(elems[0], prof2)
        elem2_desc_2 = get_element_desc(elems[1], prof2)
        op_desc_2 = random.choice(prof2["operators"].get(ops[0], [ops[0]]))
        template_2 = random.choice(prof2["templates"])
        analogy_2 = template_2.format(elem1=elem1_desc_2, elem2=elem2_desc_2, op=op_desc_2)

        return {
            "input": f"Translate from {prof1['name']} to {prof2['name']}: {analogy_1}",
            "output": analogy_2,
            "equation": chain,
            "source": f"cross_{prof1_key}_{prof2_key}"
        }

    return {"story": f"Working with {chain}", "source": "cross_profession"}


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate profession analogy samples")
    parser.add_argument("--output", default="output/train_professions.jsonl")
    parser.add_argument("--per-profession", type=int, default=50000)
    parser.add_argument("--mappings-per-profession", type=int, default=40000)
    parser.add_argument("--cross-pairs", type=int, default=10)
    parser.add_argument("--cross-per-pair", type=int, default=25000)
    args = parser.parse_args()

    prof_keys = list(PROFESSIONS.keys())

    print("=" * 60)
    print("TKS PROFESSION ANALOGIES GENERATOR")
    print("=" * 60)
    print(f"\nProfessions: {len(prof_keys)}")
    for i, key in enumerate(prof_keys, 1):
        print(f"  {i}. {PROFESSIONS[key]['name']}")

    print(f"\nTarget samples:")
    print(f"  Profession analogies: {len(prof_keys)} x {args.per_profession:,} = {len(prof_keys) * args.per_profession:,}")
    print(f"  Regular-to-analogy mappings: {len(prof_keys)} x {args.mappings_per_profession:,} = {len(prof_keys) * args.mappings_per_profession:,}")
    print(f"  Cross-profession mappings: {args.cross_pairs} x {args.cross_per_pair:,} = {args.cross_pairs * args.cross_per_pair:,}")

    total_target = (len(prof_keys) * args.per_profession +
                   len(prof_keys) * args.mappings_per_profession +
                   args.cross_pairs * args.cross_per_pair)
    print(f"  TOTAL TARGET: {total_target:,}")
    print()

    all_samples = []

    # 1. Generate profession analogies
    print("Generating profession analogies...")
    for prof_key in prof_keys:
        prof = PROFESSIONS[prof_key]
        print(f"  {prof['name']}...", end=" ", flush=True)
        for _ in range(args.per_profession):
            sample = generate_profession_sample(prof_key, prof)
            all_samples.append(sample)
        print(f"{args.per_profession:,}")

    # 2. Generate regular-to-analogy mappings
    print("\nGenerating regular-to-analogy mappings...")
    for prof_key in prof_keys:
        prof = PROFESSIONS[prof_key]
        print(f"  {prof['name']}...", end=" ", flush=True)
        for _ in range(args.mappings_per_profession):
            chain = generate_chain()
            sample = generate_mapping_sample(chain, prof_key, prof)
            all_samples.append(sample)
        print(f"{args.mappings_per_profession:,}")

    # 3. Generate cross-profession mappings
    print("\nGenerating cross-profession mappings...")
    cross_pairs = []
    for i in range(args.cross_pairs):
        p1, p2 = random.sample(prof_keys, 2)
        cross_pairs.append((p1, p2))

    for p1_key, p2_key in cross_pairs:
        p1 = PROFESSIONS[p1_key]
        p2 = PROFESSIONS[p2_key]
        print(f"  {p1['name']} <-> {p2['name']}...", end=" ", flush=True)
        for _ in range(args.cross_per_pair):
            chain = generate_chain()
            sample = generate_cross_profession_sample(chain, p1_key, p1, p2_key, p2)
            all_samples.append(sample)
        print(f"{args.cross_per_pair:,}")

    # Shuffle and deduplicate
    print("\nShuffling and deduplicating...")
    random.shuffle(all_samples)

    seen = set()
    final = []
    for item in all_samples:
        key = item.get("story", item.get("input", ""))
        if key and key not in seen:
            seen.add(key)
            final.append(item)

    print(f"\nTotal generated: {len(all_samples):,}")
    print(f"Unique samples: {len(final):,}")
    print(f"Uniqueness: {len(final)/len(all_samples)*100:.1f}%")

    # Save
    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for item in final:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\nSaved to: {args.output}")

    # Distribution
    print("\nDistribution by source (top 30):")
    sources = {}
    for item in final:
        src = item.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    for src, cnt in sorted(sources.items(), key=lambda x: -x[1])[:30]:
        pct = cnt / len(final) * 100
        print(f"  {src}: {cnt:,} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
