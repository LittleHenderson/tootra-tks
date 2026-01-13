#!/usr/bin/env python3
"""
TKS Multi-Grade Vocabulary Generator

NO HEBREW. NO KABBALAH. PURE TKS CANON.

Generates the same TKS concepts at different reading/vocabulary levels:
- Grade 5-12 (8 levels)
- College Freshman-Senior (4 levels)
- Master's, PhD (2 levels)
- Street/AAVE (1 level)

Each level gets 100k unique samples.
Total: 15 levels x 100k = 1.5 million samples

Usage:
    python scripts/generate_graded_vocabulary.py
"""

import json
import random
from pathlib import Path
from typing import List, Dict
import argparse

# =============================================================================
# TKS CANON - NO HEBREW WHATSOEVER
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

FOUNDATIONS = ["Unity", "Wisdom", "Understanding", "Companionship", "Power", "Material", "Lust"]

# =============================================================================
# VOCABULARY BY GRADE LEVEL - NO HEBREW
# =============================================================================

VOCAB_LEVELS = {
    "grade5": {
        "name": "5th Grade",
        "worlds": {
            "A": ["spirit world", "soul place", "spiritual stuff"],
            "B": ["thinking world", "mind place", "mental stuff"],
            "C": ["feeling world", "heart place", "emotional stuff"],
            "D": ["real world", "body place", "physical stuff"],
        },
        "positions": {
            "1": ["thinking", "brain power", "mind stuff"],
            "2": ["good vibes", "plus energy", "positive stuff"],
            "3": ["bad vibes", "minus energy", "negative stuff"],
            "4": ["shaking", "buzzing", "vibrating"],
            "5": ["girl energy", "receiving", "taking in"],
            "6": ["boy energy", "giving out", "sending"],
            "7": ["beat", "pattern", "repeating"],
            "8": ["reason why", "starter", "cause"],
            "9": ["what happens", "result", "effect"],
            "10": ["everything", "all of it", "the whole thing"],
        },
        "operators": {
            "+T": ["adds to over time", "joins with slowly", "combines with"],
            "-T": ["takes away over time", "removes slowly", "shrinks from"],
            "*T": ["makes bigger", "grows with", "multiplies"],
            "/T": ["makes smaller", "shrinks with", "divides"],
            "+": ["plus", "and", "together with"],
            "-": ["minus", "without", "takes away"],
            "->": ["goes to", "becomes", "turns into"],
            "<-": ["comes from", "gets from", "receives from"],
            "o": ["goes around with", "cycles with", "loops with"],
        },
        "templates": [
            "The {elem1} {op} the {elem2}",
            "{elem1} {op} {elem2}",
            "When {elem1} {op} {elem2}, stuff happens",
        ],
    },

    "grade6": {
        "name": "6th Grade",
        "worlds": {
            "A": ["spiritual realm", "soul dimension", "spirit space"],
            "B": ["mental realm", "thought dimension", "mind space"],
            "C": ["emotional realm", "feeling dimension", "heart space"],
            "D": ["physical realm", "body dimension", "material space"],
        },
        "positions": {
            "1": ["consciousness", "awareness", "mind"],
            "2": ["positive force", "attraction", "plus polarity"],
            "3": ["negative force", "repulsion", "minus polarity"],
            "4": ["vibration", "frequency", "oscillation"],
            "5": ["feminine force", "receptive energy", "receiving"],
            "6": ["masculine force", "projective energy", "sending"],
            "7": ["rhythm", "cycle", "pattern"],
            "8": ["cause", "origin", "source"],
            "9": ["effect", "result", "outcome"],
            "10": ["totality", "wholeness", "everything"],
        },
        "operators": {
            "+T": ["accumulates with", "builds up with", "gathers with"],
            "-T": ["diminishes from", "reduces from", "weakens from"],
            "*T": ["amplifies with", "strengthens with", "boosts with"],
            "/T": ["attenuates with", "weakens with", "reduces with"],
            "+": ["combines with", "merges with", "joins"],
            "-": ["separates from", "splits from", "divides from"],
            "->": ["flows to", "moves to", "transforms into"],
            "<-": ["draws from", "receives from", "gets from"],
            "o": ["cycles with", "rotates with", "circulates with"],
        },
        "templates": [
            "The {elem1} {op} the {elem2}",
            "{elem1} {op} {elem2} in this working",
            "This pattern shows {elem1} {op} {elem2}",
        ],
    },

    "grade7": {
        "name": "7th Grade",
        "worlds": {
            "A": ["spiritual plane", "transcendent realm", "spirit dimension"],
            "B": ["mental plane", "cognitive realm", "thought dimension"],
            "C": ["emotional plane", "affective realm", "feeling dimension"],
            "D": ["physical plane", "material realm", "tangible dimension"],
        },
        "positions": {
            "1": ["pure consciousness", "mental awareness", "cognitive center"],
            "2": ["positive polarity", "attractive force", "expansive energy"],
            "3": ["negative polarity", "repulsive force", "contractive energy"],
            "4": ["vibrational frequency", "oscillating energy", "wave pattern"],
            "5": ["feminine principle", "receptive quality", "inward flow"],
            "6": ["masculine principle", "projective quality", "outward flow"],
            "7": ["rhythmic pattern", "cyclic motion", "periodic flow"],
            "8": ["causal force", "originating power", "initiating energy"],
            "9": ["effectual result", "consequential outcome", "manifested effect"],
            "10": ["unified totality", "complete whole", "integrated all"],
        },
        "operators": {
            "+T": ["temporally accumulates with", "progressively combines with"],
            "-T": ["temporally diminishes from", "progressively reduces from"],
            "*T": ["temporally amplifies with", "progressively strengthens"],
            "/T": ["temporally attenuates through", "progressively weakens"],
            "+": ["integrates with", "unifies with", "synthesizes with"],
            "-": ["differentiates from", "distinguishes from", "separates from"],
            "->": ["transmits to", "projects toward", "channels into"],
            "<-": ["derives from", "originates from", "sources from"],
            "o": ["circulates through", "revolves with", "oscillates with"],
        },
        "templates": [
            "The {elem1} {op} the {elem2} in this configuration",
            "Working: {elem1} {op} {elem2}",
            "This equation demonstrates {elem1} {op} {elem2}",
        ],
    },

    "grade8": {
        "name": "8th Grade",
        "worlds": {
            "A": ["archetypal spiritual domain", "transcendent consciousness field"],
            "B": ["noetic mental domain", "cognitive processing field"],
            "C": ["psycho-emotional domain", "affective processing field"],
            "D": ["phenomenal physical domain", "material manifestation field"],
        },
        "positions": {
            "1": ["unified consciousness principle", "fundamental awareness"],
            "2": ["positive polar expression", "centrifugal attractive force"],
            "3": ["negative polar expression", "centripetal repulsive force"],
            "4": ["vibratory harmonic principle", "frequency modulation"],
            "5": ["receptive feminine archetype", "passive integrative force"],
            "6": ["projective masculine archetype", "active differentiating force"],
            "7": ["rhythmic temporal principle", "cyclical manifestation pattern"],
            "8": ["primary causal determinant", "initiating agency"],
            "9": ["secondary effectual consequence", "resultant manifestation"],
            "10": ["holistic unified totality", "comprehensive integration"],
        },
        "operators": {
            "+T": ["progressively accumulates and integrates with"],
            "-T": ["progressively diminishes and separates from"],
            "*T": ["progressively amplifies and resonates with"],
            "/T": ["progressively attenuates and disperses through"],
            "+": ["synthesizes and combines with"],
            "-": ["analyzes and separates from"],
            "->": ["transmutes and flows toward"],
            "<-": ["receives derivation from"],
            "o": ["cyclically processes through"],
        },
        "templates": [
            "The {elem1} {op} the {elem2} within this noetic framework",
            "Configuration: {elem1} {op} {elem2}",
            "This working demonstrates how {elem1} {op} {elem2}",
        ],
    },

    "grade9": {
        "name": "9th Grade",
        "worlds": {
            "A": ["transcendent spiritual substrate", "archetypal consciousness matrix"],
            "B": ["abstract mental substrate", "cognitive ideation matrix"],
            "C": ["intermediary emotional substrate", "affective valence matrix"],
            "D": ["concrete physical substrate", "material instantiation matrix"],
        },
        "positions": {
            "1": ["primordial consciousness nexus", "fundamental awareness principle"],
            "2": ["positive polarity vector", "expansive centrifugal dynamics"],
            "3": ["negative polarity vector", "contractive centripetal dynamics"],
            "4": ["harmonic vibration coefficient", "frequency resonance parameter"],
            "5": ["feminine receptive modality", "integrative absorption function"],
            "6": ["masculine projective modality", "differentiative emission function"],
            "7": ["temporal rhythm coefficient", "cyclical periodicity parameter"],
            "8": ["causal origination vector", "deterministic initiation factor"],
            "9": ["effectual termination vector", "consequential manifestation factor"],
            "10": ["holistic integration constant", "unified field totality"],
        },
        "operators": {
            "+T": ["temporally accumulates via progressive integration with"],
            "-T": ["temporally diminishes via progressive separation from"],
            "*T": ["temporally amplifies via harmonic resonance with"],
            "/T": ["temporally attenuates via dispersive modulation through"],
            "+": ["synthesizes via constructive interference with"],
            "-": ["differentiates via destructive interference from"],
            "->": ["transmutes via directed flow toward"],
            "<-": ["derives via receptive channeling from"],
            "o": ["processes via cyclical transformation through"],
        },
        "templates": [
            "The {elem1} {op} the {elem2} per TKS operational semantics",
            "Equation: {elem1} {op} {elem2}",
            "This transformation shows {elem1} {op} {elem2}",
        ],
    },

    "grade10": {
        "name": "10th Grade",
        "worlds": {
            "A": ["transcendent noumenal plane of spiritual origination"],
            "B": ["abstract ideational plane of mental processing"],
            "C": ["intermediary formative plane of emotional dynamics"],
            "D": ["concrete phenomenal plane of physical manifestation"],
        },
        "positions": {
            "1": ["primordial unified consciousness field"],
            "2": ["positive polar charge distribution"],
            "3": ["negative polar charge distribution"],
            "4": ["harmonic oscillation frequency spectrum"],
            "5": ["feminine receptive integration function"],
            "6": ["masculine projective differentiation function"],
            "7": ["temporal periodicity modulation function"],
            "8": ["causal determination origination function"],
            "9": ["effectual manifestation termination function"],
            "10": ["holistic unified field integration"],
        },
        "operators": {
            "+T": ["accumulates via temporal progressive synthesis with"],
            "-T": ["diminishes via temporal progressive analysis from"],
            "*T": ["amplifies via temporal harmonic multiplication with"],
            "/T": ["attenuates via temporal dispersive division through"],
            "+": ["integrates via constructive superposition with"],
            "-": ["separates via destructive interference from"],
            "->": ["transforms via unidirectional transmission to"],
            "<-": ["derives via receptive acquisition from"],
            "o": ["circulates via bidirectional exchange through"],
        },
        "templates": [
            "Per TKS formalism: {elem1} {op} {elem2}",
            "The operator chain {elem1} {op} {elem2} demonstrates this principle",
            "Working equation: {elem1} {op} {elem2}",
        ],
    },

    "grade11": {
        "name": "11th Grade",
        "worlds": {
            "A": ["primary spiritual emanation plane of archetypal potentiality"],
            "B": ["secondary mental creation plane of abstract ideation"],
            "C": ["tertiary emotional formation plane of affective patterning"],
            "D": ["quaternary physical action plane of concrete actualization"],
        },
        "positions": {
            "1": ["unified consciousness monad at the apex of manifestation"],
            "2": ["positive polarity expressing centrifugal expansion dynamics"],
            "3": ["negative polarity expressing centripetal contraction dynamics"],
            "4": ["vibrational frequency expressing harmonic oscillation"],
            "5": ["feminine archetype expressing receptive integration"],
            "6": ["masculine archetype expressing projective differentiation"],
            "7": ["rhythmic principle expressing temporal periodicity"],
            "8": ["causal principle expressing deterministic origination"],
            "9": ["effectual principle expressing consequential termination"],
            "10": ["unified field expressing holistic totality"],
        },
        "operators": {
            "+T": ["undergoes temporal accumulative synthesis with"],
            "-T": ["undergoes temporal diminutive analysis from"],
            "*T": ["undergoes temporal amplificative resonance with"],
            "/T": ["undergoes temporal attenuative dispersion through"],
            "+": ["experiences constructive integrative superposition with"],
            "-": ["experiences destructive differentiative separation from"],
            "->": ["manifests unidirectional transmutative flow toward"],
            "<-": ["receives bidirectional derivational influx from"],
            "o": ["participates in cyclical transformational exchange with"],
        },
        "templates": [
            "According to TKS canonical formalism: {elem1} {op} {elem2}",
            "The noetic equation {elem1} {op} {elem2} expresses this relationship",
            "This working demonstrates: {elem1} {op} {elem2}",
        ],
    },

    "grade12": {
        "name": "12th Grade",
        "worlds": {
            "A": ["the primary plane of archetypal spiritual emanation and transcendent potentiality"],
            "B": ["the secondary plane of abstract mental creation and ideational genesis"],
            "C": ["the tertiary plane of emotional formation and affective pattern crystallization"],
            "D": ["the quaternary plane of concrete physical action and material actualization"],
        },
        "positions": {
            "1": ["the unified consciousness monad representing primordial awareness"],
            "2": ["the positive polar vector expressing centrifugal expansive dynamics"],
            "3": ["the negative polar vector expressing centripetal contractive dynamics"],
            "4": ["the vibrational coefficient expressing harmonic frequency oscillation"],
            "5": ["the feminine receptive modality expressing integrative absorption"],
            "6": ["the masculine projective modality expressing differentiative emission"],
            "7": ["the rhythmic temporal coefficient expressing cyclical periodicity"],
            "8": ["the causal origination factor expressing deterministic initiation"],
            "9": ["the effectual termination factor expressing consequential manifestation"],
            "10": ["the holistic integration constant expressing unified totality"],
        },
        "operators": {
            "+T": ["undergoes progressive temporal accumulation and synthesis with"],
            "-T": ["undergoes progressive temporal diminution and separation from"],
            "*T": ["undergoes progressive temporal amplification and resonance with"],
            "/T": ["undergoes progressive temporal attenuation and dispersion through"],
            "+": ["experiences constructive superposition and integration with"],
            "-": ["experiences destructive interference and differentiation from"],
            "->": ["manifests unidirectional transmutation and projection toward"],
            "<-": ["receives derivational influx and acquisition from"],
            "o": ["participates in cyclical exchange and transformation through"],
        },
        "templates": [
            "Per TKS v7.4 canonical specification: {elem1} {op} {elem2}",
            "The noetic operator chain {elem1} {op} {elem2} instantiates this principle",
            "Working equation per canonical semantics: {elem1} {op} {elem2}",
        ],
    },

    "college1": {
        "name": "College Freshman",
        "worlds": {
            "A": ["the transcendent spiritual substrate constituting the archetypal emanation plane"],
            "B": ["the abstract mental substrate constituting the cognitive ideation plane"],
            "C": ["the intermediary emotional substrate constituting the affective formation plane"],
            "D": ["the concrete physical substrate constituting the material instantiation plane"],
        },
        "positions": {
            "1": ["unified consciousness representing the primordial monad of awareness"],
            "2": ["positive polarity representing centrifugal expansive vector dynamics"],
            "3": ["negative polarity representing centripetal contractive vector dynamics"],
            "4": ["vibrational frequency representing harmonic oscillation coefficients"],
            "5": ["feminine receptivity representing integrative absorption modalities"],
            "6": ["masculine projectivity representing differentiative emission modalities"],
            "7": ["rhythmic periodicity representing temporal cyclical coefficients"],
            "8": ["causal origination representing deterministic initiation vectors"],
            "9": ["effectual manifestation representing consequential termination vectors"],
            "10": ["holistic totality representing unified field integration constants"],
        },
        "operators": {
            "+T": ["undergoes temporally-indexed progressive accumulative synthesis with"],
            "-T": ["undergoes temporally-indexed progressive diminutive separation from"],
            "*T": ["undergoes temporally-indexed progressive amplificative resonance with"],
            "/T": ["undergoes temporally-indexed progressive attenuative dispersion through"],
            "+": ["experiences constructive superposition via integrative combination with"],
            "-": ["experiences destructive interference via differentiative separation from"],
            "->": ["manifests unidirectional transmutative projection toward"],
            "<-": ["receives bidirectional derivational acquisition from"],
            "o": ["participates in cyclical bidirectional transformational exchange with"],
        },
        "templates": [
            "Per TKS canonical formalization: {elem1} {op} {elem2}",
            "The operator sequence {elem1} {op} {elem2} instantiates the specified relationship",
            "Canonical working: {elem1} {op} {elem2}",
        ],
    },

    "college2": {
        "name": "College Sophomore",
        "worlds": {
            "A": ["the primary archetypal emanation substrate of transcendent spiritual potentiality"],
            "B": ["the secondary creative ideation substrate of abstract mental processing"],
            "C": ["the tertiary formative patterning substrate of emotional dynamics"],
            "D": ["the quaternary actualization substrate of concrete physical manifestation"],
        },
        "positions": {
            "1": ["the primordial consciousness monad constituting unified awareness at emanation apex"],
            "2": ["the positive polar charge distribution constituting centrifugal expansion dynamics"],
            "3": ["the negative polar charge distribution constituting centripetal contraction dynamics"],
            "4": ["the harmonic oscillation spectrum constituting vibrational frequency modulation"],
            "5": ["the feminine receptive function constituting integrative absorption processes"],
            "6": ["the masculine projective function constituting differentiative emission processes"],
            "7": ["the temporal periodicity function constituting rhythmic cyclical modulation"],
            "8": ["the causal determination function constituting origination initiation processes"],
            "9": ["the effectual manifestation function constituting termination actualization processes"],
            "10": ["the unified field integration constituting holistic totality synthesis"],
        },
        "operators": {
            "+T": ["undergoes temporally-parameterized progressive accumulative integration with"],
            "-T": ["undergoes temporally-parameterized progressive diminutive differentiation from"],
            "*T": ["undergoes temporally-parameterized progressive amplificative multiplication with"],
            "/T": ["undergoes temporally-parameterized progressive attenuative division through"],
            "+": ["experiences constructive integrative superposition synthesis with"],
            "-": ["experiences destructive differentiative interference analysis from"],
            "->": ["manifests unidirectional transmutative transformation toward"],
            "<-": ["receives receptive derivational transformation from"],
            "o": ["participates in cyclical bidirectional exchange transformation through"],
        },
        "templates": [
            "Per TKS v7.4 formal semantics: {elem1} {op} {elem2}",
            "The canonical operator chain {elem1} {op} {elem2} expresses this noetic relationship",
            "Formal working specification: {elem1} {op} {elem2}",
        ],
    },

    "college3": {
        "name": "College Junior",
        "worlds": {
            "A": ["the primary archetypal substrate of transcendent spiritual emanation and potentiality actualization"],
            "B": ["the secondary creative substrate of abstract mental ideation and cognitive processing instantiation"],
            "C": ["the tertiary formative substrate of emotional patterning and affective dynamics crystallization"],
            "D": ["the quaternary actualization substrate of concrete physical manifestation and material instantiation"],
        },
        "positions": {
            "1": ["the primordial unified consciousness monad at the apex of noetic emanation hierarchy"],
            "2": ["the positive polar vector expressing centrifugal expansive dynamics in field space"],
            "3": ["the negative polar vector expressing centripetal contractive dynamics in field space"],
            "4": ["the vibrational frequency coefficient expressing harmonic oscillation in phase space"],
            "5": ["the feminine receptive modality expressing integrative absorption in transformation space"],
            "6": ["the masculine projective modality expressing differentiative emission in transformation space"],
            "7": ["the rhythmic periodicity coefficient expressing temporal cyclical dynamics"],
            "8": ["the causal origination vector expressing deterministic initiation in causal space"],
            "9": ["the effectual termination vector expressing consequential manifestation in causal space"],
            "10": ["the holistic integration constant expressing unified totality in field space"],
        },
        "operators": {
            "+T": ["undergoes temporally-indexed progressive accumulative synthesis transformation with"],
            "-T": ["undergoes temporally-indexed progressive diminutive analysis transformation from"],
            "*T": ["undergoes temporally-indexed progressive amplificative resonance transformation with"],
            "/T": ["undergoes temporally-indexed progressive attenuative dispersion transformation through"],
            "+": ["experiences constructive superposition integration transformation with"],
            "-": ["experiences destructive interference differentiation transformation from"],
            "->": ["manifests unidirectional transmutative projection transformation toward"],
            "<-": ["receives bidirectional derivational acquisition transformation from"],
            "o": ["participates in cyclical exchange circulation transformation through"],
        },
        "templates": [
            "Per TKS v7.4 canonical formal specification: {elem1} {op} {elem2}",
            "The noetic operator sequence {elem1} {op} {elem2} formally instantiates this relationship",
            "Canonical equation per formal semantics: {elem1} {op} {elem2}",
        ],
    },

    "college4": {
        "name": "College Senior",
        "worlds": {
            "A": ["the primary transcendent archetypal emanation substrate instantiating spiritual potentiality"],
            "B": ["the secondary abstract creative ideation substrate instantiating cognitive mental processing"],
            "C": ["the tertiary intermediary formative patterning substrate instantiating emotional dynamics"],
            "D": ["the quaternary concrete actualization substrate instantiating physical material manifestation"],
        },
        "positions": {
            "1": ["the primordial unified consciousness monad instantiating fundamental awareness at emanation apex"],
            "2": ["the positive polar charge vector instantiating centrifugal expansive field dynamics"],
            "3": ["the negative polar charge vector instantiating centripetal contractive field dynamics"],
            "4": ["the harmonic oscillation frequency coefficient instantiating vibrational resonance parameters"],
            "5": ["the feminine receptive integration modality instantiating absorptive transformation processes"],
            "6": ["the masculine projective differentiation modality instantiating emissive transformation processes"],
            "7": ["the temporal rhythmic periodicity coefficient instantiating cyclical modulation parameters"],
            "8": ["the causal origination determination vector instantiating initiation transformation processes"],
            "9": ["the effectual termination manifestation vector instantiating consequential transformation processes"],
            "10": ["the holistic unified field integration constant instantiating totality synthesis parameters"],
        },
        "operators": {
            "+T": ["undergoes temporally-parameterized progressive accumulative integration synthesis with"],
            "-T": ["undergoes temporally-parameterized progressive diminutive differentiation analysis from"],
            "*T": ["undergoes temporally-parameterized progressive amplificative resonance multiplication with"],
            "/T": ["undergoes temporally-parameterized progressive attenuative dispersion division through"],
            "+": ["experiences constructive integrative superposition combination synthesis with"],
            "-": ["experiences destructive differentiative interference separation analysis from"],
            "->": ["manifests unidirectional transmutative transformation projection toward"],
            "<-": ["receives bidirectional derivational transformation acquisition from"],
            "o": ["participates in cyclical bidirectional transformation exchange circulation through"],
        },
        "templates": [
            "Per TKS v7.4 canonical formal semantic specification: {elem1} {op} {elem2}",
            "The canonical noetic operator chain {elem1} {op} {elem2} formally instantiates this relationship",
            "Formal canonical working equation: {elem1} {op} {elem2}",
        ],
    },

    "masters": {
        "name": "Master's Level",
        "worlds": {
            "A": ["the primary transcendent archetypal emanation substrate constituting the foundational plane of spiritual potentiality actualization"],
            "B": ["the secondary abstract creative ideation substrate constituting the cognitive plane of mental processing instantiation"],
            "C": ["the tertiary intermediary formative patterning substrate constituting the affective plane of emotional dynamics crystallization"],
            "D": ["the quaternary concrete actualization substrate constituting the material plane of physical manifestation instantiation"],
        },
        "positions": {
            "1": ["the primordial unified consciousness monad constituting fundamental noetic awareness at the apex of the emanation hierarchy"],
            "2": ["the positive polar charge distribution vector constituting centrifugal expansive dynamics in noetic field space"],
            "3": ["the negative polar charge distribution vector constituting centripetal contractive dynamics in noetic field space"],
            "4": ["the harmonic oscillation frequency spectrum coefficient constituting vibrational resonance parameters in phase space"],
            "5": ["the feminine receptive integration modality constituting absorptive transformation processes in noetic transformation space"],
            "6": ["the masculine projective differentiation modality constituting emissive transformation processes in noetic transformation space"],
            "7": ["the temporal rhythmic periodicity modulation coefficient constituting cyclical dynamics parameters in temporal space"],
            "8": ["the causal origination determination function vector constituting initiation processes in noetic causal space"],
            "9": ["the effectual termination manifestation function vector constituting consequential processes in noetic causal space"],
            "10": ["the holistic unified field integration totality constant constituting synthesis parameters in noetic field space"],
        },
        "operators": {
            "+T": ["undergoes temporally-parameterized progressive accumulative integration synthesis transformation with"],
            "-T": ["undergoes temporally-parameterized progressive diminutive differentiation analysis transformation from"],
            "*T": ["undergoes temporally-parameterized progressive amplificative resonance multiplication transformation with"],
            "/T": ["undergoes temporally-parameterized progressive attenuative dispersion division transformation through"],
            "+": ["experiences constructive integrative superposition combination synthesis transformation with"],
            "-": ["experiences destructive differentiative interference separation analysis transformation from"],
            "->": ["manifests unidirectional transmutative transformation projection instantiation toward"],
            "<-": ["receives bidirectional derivational transformation acquisition instantiation from"],
            "o": ["participates in cyclical bidirectional transformation exchange circulation instantiation through"],
        },
        "templates": [
            "Per TKS v7.4 canonical formal semantic specification framework: {elem1} {op} {elem2}",
            "The canonical noetic operator sequence {elem1} {op} {elem2} formally instantiates this specified relationship",
            "Formal canonical working equation per semantic framework: {elem1} {op} {elem2}",
        ],
    },

    "phd": {
        "name": "PhD Level",
        "worlds": {
            "A": ["the primary transcendent archetypal emanation substrate constituting the foundational ontological plane of spiritual potentiality actualization and noumenal instantiation"],
            "B": ["the secondary abstract creative ideation substrate constituting the cognitive ontological plane of mental processing instantiation and ideational crystallization"],
            "C": ["the tertiary intermediary formative patterning substrate constituting the affective ontological plane of emotional dynamics crystallization and phenomenal instantiation"],
            "D": ["the quaternary concrete actualization substrate constituting the material ontological plane of physical manifestation instantiation and phenomenal crystallization"],
        },
        "positions": {
            "1": ["the primordial unified consciousness monad constituting fundamental noetic awareness instantiation at the apex of the emanation ontological hierarchy"],
            "2": ["the positive polar charge distribution vector constituting centrifugal expansive dynamics instantiation in noetic field configuration space"],
            "3": ["the negative polar charge distribution vector constituting centripetal contractive dynamics instantiation in noetic field configuration space"],
            "4": ["the harmonic oscillation frequency spectrum coefficient constituting vibrational resonance parameter instantiation in noetic phase configuration space"],
            "5": ["the feminine receptive integration modality constituting absorptive transformation process instantiation in noetic transformation configuration space"],
            "6": ["the masculine projective differentiation modality constituting emissive transformation process instantiation in noetic transformation configuration space"],
            "7": ["the temporal rhythmic periodicity modulation coefficient constituting cyclical dynamics parameter instantiation in noetic temporal configuration space"],
            "8": ["the causal origination determination function vector constituting initiation process instantiation in noetic causal configuration space"],
            "9": ["the effectual termination manifestation function vector constituting consequential process instantiation in noetic causal configuration space"],
            "10": ["the holistic unified field integration totality constant constituting synthesis parameter instantiation in noetic field configuration space"],
        },
        "operators": {
            "+T": ["undergoes temporally-parameterized progressive accumulative integration synthesis transformation instantiation with"],
            "-T": ["undergoes temporally-parameterized progressive diminutive differentiation analysis transformation instantiation from"],
            "*T": ["undergoes temporally-parameterized progressive amplificative resonance multiplication transformation instantiation with"],
            "/T": ["undergoes temporally-parameterized progressive attenuative dispersion division transformation instantiation through"],
            "+": ["experiences constructive integrative superposition combination synthesis transformation instantiation with"],
            "-": ["experiences destructive differentiative interference separation analysis transformation instantiation from"],
            "->": ["manifests unidirectional transmutative transformation projection instantiation crystallization toward"],
            "<-": ["receives bidirectional derivational transformation acquisition instantiation crystallization from"],
            "o": ["participates in cyclical bidirectional transformation exchange circulation instantiation crystallization through"],
        },
        "templates": [
            "Per TKS v7.4 canonical formal semantic specification ontological framework: {elem1} {op} {elem2}",
            "The canonical noetic operator sequence {elem1} {op} {elem2} formally instantiates this specified ontological relationship",
            "Formal canonical working equation per semantic ontological framework specification: {elem1} {op} {elem2}",
        ],
    },

    "street": {
        "name": "Street/AAVE",
        "worlds": {
            "A": ["spirit side", "soul vibes", "higher self stuff"],
            "B": ["mind game", "thought vibes", "mental waves"],
            "C": ["feeling vibes", "heart energy", "mood stuff"],
            "D": ["real world", "physical vibes", "material stuff"],
        },
        "positions": {
            "1": ["consciousness", "awareness", "mind state"],
            "2": ["positive energy", "good vibes", "plus side"],
            "3": ["negative energy", "bad vibes", "minus side"],
            "4": ["vibration", "frequency", "energy waves"],
            "5": ["receiving energy", "feminine flow", "taking in"],
            "6": ["giving energy", "masculine flow", "putting out"],
            "7": ["rhythm", "flow", "cycles"],
            "8": ["cause", "source", "origin"],
            "9": ["effect", "result", "outcome"],
            "10": ["everything", "the whole thing", "all of it"],
        },
        "operators": {
            "+T": ["stacks up with", "builds on", "adds to over time"],
            "-T": ["takes away from", "reduces from", "fades from"],
            "*T": ["amplifies with", "boosts with", "powers up with"],
            "/T": ["weakens through", "dims through", "cuts through"],
            "+": ["links up with", "connects with", "joins with"],
            "-": ["splits from", "breaks from", "separates from"],
            "->": ["flows into", "goes to", "transfers to"],
            "<-": ["pulls from", "draws from", "gets from"],
            "o": ["cycles through", "rotates with", "loops with"],
        },
        "templates": [
            "Yo, {elem1} {op} {elem2}, feel me?",
            "{elem1} {op} {elem2}, that's how it work",
            "Check it: {elem1} {op} {elem2}",
            "So basically {elem1} {op} {elem2}",
            "{elem1} be {op} {elem2}, you know what I'm sayin?",
        ],
    },
}


# =============================================================================
# GENERATORS
# =============================================================================

def generate_element() -> str:
    """Generate random TKS element code."""
    world = random.choice(["A", "B", "C", "D"])
    pos = random.choice(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
    elem = f"{world}{pos}"

    if random.random() < 0.25:
        sense = random.randint(1, 10)
        elem += f"^{sense}"

    if random.random() < 0.15:
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


def get_element_description(code: str, level: dict) -> str:
    """Get element description at specified vocabulary level."""
    world = code[0]

    rest = code[1:]
    if rest.startswith("10"):
        pos = "10"
    else:
        pos = rest[0] if rest else "1"

    world_desc = random.choice(level["worlds"].get(world, [WORLDS[world]]))
    pos_desc = random.choice(level["positions"].get(pos, [POSITIONS[pos]]))

    return f"{pos_desc} in {world_desc}"


def generate_sample(level_key: str, level: dict) -> Dict:
    """Generate one sample at specified level."""
    chain = generate_chain()

    sample_type = random.choice(["working", "interpretation", "template", "translate"])

    if sample_type == "template":
        tokens = chain.split()
        elems = [t for t in tokens if t and t[0] in "ABCD"]
        ops = [t for t in tokens if t in OPERATORS]

        if len(elems) >= 2 and ops:
            elem1_desc = get_element_description(elems[0], level)
            elem2_desc = get_element_description(elems[1], level)
            op_desc = random.choice(level["operators"].get(ops[0], [ops[0]]))

            template = random.choice(level["templates"])
            story = template.format(elem1=elem1_desc, elem2=elem2_desc, op=op_desc)
            return {"story": story, "source": f"graded_{level_key}"}

    elif sample_type == "translate":
        tokens = chain.split()
        elems = [t for t in tokens if t and t[0] in "ABCD"]
        ops = [t for t in tokens if t in OPERATORS]

        parts = []
        for i, elem in enumerate(elems):
            parts.append(get_element_description(elem, level))
            if i < len(ops):
                parts.append(random.choice(level["operators"].get(ops[i], [ops[i]])))

        narrative = " ".join(parts)
        return {
            "input": f"Translate to TKS: {narrative}",
            "output": chain,
            "source": f"graded_{level_key}"
        }

    elif sample_type == "working":
        return {"story": f"Working with {chain}", "source": f"graded_{level_key}"}

    else:
        return {"story": f"Interpretation of {chain}", "source": f"graded_{level_key}"}


def generate_level_samples(level_key: str, count: int) -> List[Dict]:
    """Generate samples for one vocabulary level."""
    level = VOCAB_LEVELS[level_key]
    return [generate_sample(level_key, level) for _ in range(count)]


def main():
    parser = argparse.ArgumentParser(description="Generate multi-grade TKS vocabulary samples")
    parser.add_argument("--output", default="output/train_graded.jsonl", help="Output path")
    parser.add_argument("--per-level", type=int, default=100000, help="Samples per level")
    args = parser.parse_args()

    levels = list(VOCAB_LEVELS.keys())
    total_target = len(levels) * args.per_level

    print(f"Generating {args.per_level:,} samples per level")
    print(f"Levels: {len(levels)}")
    print(f"Total target: {total_target:,}")
    print()

    all_samples = []

    for level_key in levels:
        level_name = VOCAB_LEVELS[level_key]["name"]
        print(f"Generating {level_name}...", end=" ", flush=True)
        samples = generate_level_samples(level_key, args.per_level)
        all_samples.extend(samples)
        print(f"{len(samples):,} samples")

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

    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for item in final:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
