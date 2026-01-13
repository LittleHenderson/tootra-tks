"""
TKS Strategic Agent Harness - Integrating v4 Core with Strategic Stack

This agent wraps the TKSNoeticLM (v4) and adds:
- Governance (High-Stakes/Irreversibility)
- Regulators (FM/ACBE/MVR)
- Abstraction Laddering (L0-L8)
- Consciousness Unlock (Novelty-based recursion)
- Canon Awareness (Loads normalized definitions)
- Natural Language Mapping (Lexicon-based)
- INTEGRITY GATING (Hash checks)
"""

import os
import json
import torch
import random
import sys
from typing import Dict, Any, List

# Adjust imports based on new src structure
try:
    from src.tks.core.tks_llm_core_v4 import TKSNoeticLM, TKSNoeticLMConfig
    from src.tks.governance.governance import GovernanceRail, GovernanceMode, Reversibility
    from src.tks.regulators.regulators import RPMPlanner, TKSNode
    from src.tks.core.abstraction import AbstractionLadder, AbstractionTier, ConsciousnessManager, NoveltyCandidate
    from src.tks.nl.map_to_tks import TKSMapper
    from src.tks.core.canon_gate import validate_canon_integrity, CanonIntegrityError
except ImportError:
    # Fallback for direct execution testing
    from tks_llm_core_v4 import TKSNoeticLM, TKSNoeticLMConfig
    from governance import GovernanceRail, GovernanceMode, Reversibility
    from regulators import RPMPlanner, TKSNode
    from abstraction import AbstractionLadder, AbstractionTier, ConsciousnessManager, NoveltyCandidate
    from map_to_tks import TKSMapper
    from canon_gate import validate_canon_integrity, CanonIntegrityError


class TKSStrategicAgent:
    def __init__(self, model_checkpoint: str = None, canon_dir: str = "canon/normalized"):
        # 0. INTEGRITY CHECK (Gate)
        try:
            validate_canon_integrity(canon_dir)
        except CanonIntegrityError as e:
            print(f"\n🛑 FATAL: {e}")
            sys.exit(1)

        # 1. Load Canon (The Dictionary)
        self.canon = self._load_canon(canon_dir)
        print(f"Agent initialized with {len(self.canon.get('elements', []))} Elements and {len(self.canon.get('noetics', []))} Noetics.")
        
        # 2. Initialize NL Mapper
        self.mapper = TKSMapper(canon_dir)

        # 3. Initialize the "Brain" (v4 Core)
        self.config = TKSNoeticLMConfig()
        
        # Load Checkpoint logic (needs to happen BEFORE model init if config is inside)
        if model_checkpoint and os.path.exists(model_checkpoint):
            print(f"Loading checkpoint: {model_checkpoint}")
            checkpoint = torch.load(model_checkpoint, weights_only=False)
            
            # Load Config if available
            if "config" in checkpoint:
                saved_config = checkpoint["config"]
                # Map train_cuda.py config dict to TKSNoeticLMConfig object
                # train_cuda saves a dict like {'vocab_size': ...} inside config or uses args
                # Actually train_cuda saves 'config': {...} dict. We need to extract what fits.
                # TKSNoeticLMConfig has: vocab_size, max_seq_len, num_layers, etc.
                
                # Check for vocab_size in the saved config dict
                # Note: train_cuda config might store it differently or we check the weights
                
                if "vocab_size" in saved_config:
                    self.config.vocab_size = saved_config["vocab_size"]
                elif "model_state_dict" in checkpoint:
                    # Infer vocab size from weights if missing in config
                    weight = checkpoint["model_state_dict"]["embedding.embedding.weight"]
                    self.config.vocab_size = weight.shape[0]
                    
                if "num_layers" in saved_config:
                    self.config.num_layers = saved_config["num_layers"]
                if "hidden_dim" in saved_config:
                    # v4 uses fixed dims (40), but check if overridden
                    pass
            
            # Re-init model with correct config
            self.model = TKSNoeticLM(self.config)
            
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            print("No checkpoint found. Running in initialized state (untrained).")
            self.model = TKSNoeticLM(self.config)
            
        self.model.eval()

        # 4. Initialize the "Superego" (Strategic Stack)
        self.governance = GovernanceRail()
        self.planner = RPMPlanner()
        self.ladder = AbstractionLadder()
        self.consciousness = ConsciousnessManager()

    def _load_canon(self, canon_dir: str) -> Dict[str, List[Any]]:
        """Loads the JSON source of truth."""
        canon = {}
        try:
            with open(os.path.join(canon_dir, "canon_elements_40.json"), "r") as f:
                canon["elements"] = json.load(f)
            with open(os.path.join(canon_dir, "canon_noetics_10.json"), "r") as f:
                canon["noetics"] = json.load(f)
            with open(os.path.join(canon_dir, "canon_foundations_7.json"), "r") as f:
                canon["foundations"] = json.load(f)
        except FileNotFoundError:
            print("Warning: Canon files not found. Agent is running without grounding.")
            canon = {"elements": [], "noetics": [], "foundations": []}
        return canon

    def _generate_tks_packet(self, goal_text: str) -> Dict:
        """
        Uses the TKSMapper to map NL to TKS symbols.
        Returns full mapping result with metadata.
        """
        return self.mapper.emit_packet(goal_text)

    def run_episode(self, goal_text: str, foundation_stack: str = "1a:5b:2b"):
        """
        Runs a complete cognitive cycle from input to action.
        """
        # L0: Raw Input
        self.ladder.add_node(AbstractionTier.L0, goal_text)
        
        # Determine Stakes
        uncertainty = 0.5
        stakes = 0.9 if "delete" in goal_text.lower() or "critical" in goal_text.lower() else 0.2
        alignment = 0.95
        hs_score = self.governance.calculate_hs_score(uncertainty, stakes, alignment)
        
        # Governance Check
        gov_check = self.governance.check_action(
            action_name="initial_inference",
            reversibility=Reversibility.IRREVERSIBLE if stakes > 0.8 else Reversibility.REVERSIBLE,
            hs_score=hs_score,
            alignment=alignment
        )
        
        if not gov_check["allow"]:
            return {
                "status": "BLOCKED",
                "reason": gov_check['reason'],
                "tks_expression": gov_check.get('tks_expression'),
                "mode": gov_check['mode']
            }

        # L4: Generate TKS Packet (Inference)
        mapping_result = self._generate_tks_packet(goal_text)
        tks_packet = mapping_result['packet']
        self.ladder.add_node(AbstractionTier.L4, tks_packet)

        # L3: RPM Planning
        root_node = TKSNode(expression=tks_packet, foundation_stack=foundation_stack)
        
        # Heuristic for Regulators
        if "quantum" in goal_text.lower():
            root_node.meta = {"gap_type": "capability"}
            root_node.status = "unsatisfied"
        elif "desire" in goal_text.lower():
             root_node.status = "low_desire"
        
        # Trigger Regulators if there's a gap
        plan_trace = []
        if root_node.status != "satisfied":
            sub_nodes = self.planner.expand_gap(root_node)
            for sn in sub_nodes:
                self.ladder.add_node(AbstractionTier.L3, sn)
                plan_trace.append(f"Regulator Triggered: {sn.meta.get('regulator')} -> {sn.expression}")
        
        # Evaluate Novelty
        novelty = NoveltyCandidate(
            tks_packet=tks_packet,
            validity=0.9,
            impact=0.6,
            similarity=0.3
        )
        unlock_status = self.consciousness.evaluate_novelty(novelty)
        
        return {
            "status": "SUCCESS",
            "mode": gov_check["mode"],
            "tks_packet": tks_packet,
            "plan_trace": plan_trace,
            "p_max": self.consciousness.p_max,
            "novelty_report": unlock_status,
            "lexicon_match": {
                "elements": mapping_result['top_elements'],
                "noetics": mapping_result['top_noetics']
            }
        }


if __name__ == "__main__":
    # Ensure running from root for imports to work
    agent = TKSStrategicAgent()
    
    print("\n--- Scenario 1: Normal Query (Build AI) ---")
    res1 = agent.run_episode("How do I build a goal-aware AI?")
    print(res1)
