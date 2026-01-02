"""
TKS Conversation Memory - Multi-turn Conversation Support for TKS Wrapper

This module adds conversation memory capabilities to the TKS wrapper, enabling
multi-turn dialogues where context from previous analyses is maintained and
incorporated into follow-up interactions.

Architecture:
    User Input -> Conversation History -> TKS Analysis -> Store Result -> Output

Features:
    - Maintains full conversation history
    - Automatically includes previous context in prompts
    - Supports explicit follow-up questions referencing last analysis
    - Provides history retrieval and clearing capabilities
    - Preserves all TKS analysis results for reference

Usage:
    from tks_features.conversation_memory import TKSConversation

    # Initialize conversation with provider
    convo = TKSConversation(provider="anthropic", api_key="...")

    # First analysis
    result1 = convo.analyze("I'm torn between my career and family time")
    print(result1.equation.raw)  # B7 +T D3 -T A5

    # Follow-up with automatic context
    result2 = convo.analyze("How can I balance these better?")
    # Previous equation and context automatically included

    # Explicit follow-up to last result
    result3 = convo.follow_up("What specific actions should I take?")

    # Access history
    history = convo.get_history()
    print(f"Total turns: {len(history)}")

    # Get just the last result
    print(convo.last_result.guidance)

    # Clear and start fresh
    convo.clear_history()

Author: TKS-LLM Project
Date: 2025
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

# Import TKS Wrapper
sys.path.insert(0, str(Path(__file__).parent.parent))

from tks_wrapper import (
    TKSWrapper,
    TKSAnalysisResult,
    AnalysisMode
)


# ==============================================================================
# CONVERSATION TURN DATA CLASS
# ==============================================================================

@dataclass
class ConversationTurn:
    """A single turn in a TKS conversation."""

    turn_number: int
    timestamp: datetime
    user_input: str
    result: TKSAnalysisResult
    context_used: Optional[str] = None
    is_follow_up: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "turn_number": self.turn_number,
            "timestamp": self.timestamp.isoformat(),
            "user_input": self.user_input,
            "equation": self.result.equation.raw,
            "interpretation": self.result.interpretation,
            "guidance": self.result.guidance,
            "rpm": {
                "desire": self.result.rpm.desire,
                "wisdom": self.result.rpm.wisdom,
                "power": self.result.rpm.power
            },
            "foundations": self.result.foundations,
            "attractor": self.result.attractor,
            "is_canonical": self.result.is_canonical,
            "context_used": self.context_used,
            "is_follow_up": self.is_follow_up
        }

    def summary(self) -> str:
        """Return a concise summary of this turn."""
        follow_up_marker = " [FOLLOW-UP]" if self.is_follow_up else ""
        return f"""
Turn {self.turn_number}{follow_up_marker} - {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 60}
Input: {self.user_input}
Equation: {self.result.equation.raw}
Guidance: {self.result.guidance[:200]}{'...' if len(self.result.guidance) > 200 else ''}
"""


# ==============================================================================
# TKS CONVERSATION CLASS
# ==============================================================================

class TKSConversation:
    """
    Multi-turn conversation wrapper for TKS analysis.

    This class wraps TKSWrapper and maintains conversation history, automatically
    including context from previous turns to enable coherent multi-turn dialogues.

    Attributes:
        wrapper: The underlying TKSWrapper instance
        history: List of all conversation turns
        max_context_turns: Maximum number of previous turns to include in context
        include_equations_only: If True, only include equations in context, not full analysis
    """

    def __init__(
        self,
        provider: str = "mock",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        strict_validation: bool = True,
        max_context_turns: int = 3,
        include_equations_only: bool = False
    ):
        """
        Initialize TKS conversation wrapper.

        Args:
            provider: LLM provider ("anthropic"/"claude", "google"/"gemini", "mock")
            api_key: API key for the provider
            model: Specific model to use
            strict_validation: Enforce strict canonical validation
            max_context_turns: Maximum previous turns to include in context (default: 3)
            include_equations_only: Only include equations in context, not full text (default: False)
        """
        self.wrapper = TKSWrapper(
            provider=provider,
            api_key=api_key,
            model=model,
            strict_validation=strict_validation
        )

        self.history: List[ConversationTurn] = []
        self.max_context_turns = max_context_turns
        self.include_equations_only = include_equations_only

    def analyze(
        self,
        user_input: str,
        mode: AnalysisMode = AnalysisMode.FULL,
        include_history: bool = True
    ) -> TKSAnalysisResult:
        """
        Analyze user input with conversation context.

        This method performs TKS analysis while automatically including context
        from previous conversation turns. The context helps maintain coherence
        across the conversation.

        Args:
            user_input: The situation/question to analyze
            mode: Analysis mode (full, equation, coaching, diagnostic)
            include_history: Whether to include previous turns as context (default: True)

        Returns:
            TKSAnalysisResult with equation, interpretation, and guidance
        """
        # Build context from previous turns
        context = None
        if include_history and self.history:
            context = self._build_context()

        # Perform analysis with context
        result = self.wrapper.analyze(
            user_input=user_input,
            mode=mode,
            context=context
        )

        # Store in history
        turn = ConversationTurn(
            turn_number=len(self.history) + 1,
            timestamp=datetime.now(),
            user_input=user_input,
            result=result,
            context_used=context,
            is_follow_up=False
        )
        self.history.append(turn)

        return result

    def follow_up(
        self,
        question: str,
        mode: AnalysisMode = AnalysisMode.COACHING
    ) -> TKSAnalysisResult:
        """
        Ask a follow-up question that explicitly references the last analysis.

        This method is designed for questions that directly build on the previous
        result, such as "What should I do next?" or "How can I improve this?".
        It constructs a prompt that clearly references the last equation and analysis.

        Args:
            question: Follow-up question about the previous analysis
            mode: Analysis mode (defaults to COACHING for actionable guidance)

        Returns:
            TKSAnalysisResult with guidance specific to the follow-up

        Raises:
            ValueError: If no previous analysis exists to follow up on
        """
        if not self.history:
            raise ValueError("No previous analysis to follow up on. Use analyze() first.")

        last_turn = self.history[-1]
        last_equation = last_turn.result.equation.raw
        last_interpretation = last_turn.result.interpretation[:300]

        # Build explicit follow-up context
        follow_up_context = f"""
PREVIOUS ANALYSIS:
- User asked: "{last_turn.user_input}"
- TKS Equation: {last_equation}
- Key insight: {last_interpretation}

USER NOW ASKS: {question}

Please provide guidance that builds on the previous analysis. Reference the
previous equation and explain how your recommendations relate to it.
"""

        # Perform analysis with explicit follow-up context
        result = self.wrapper.analyze(
            user_input=follow_up_context,
            mode=mode,
            context=None  # Don't double-add context
        )

        # Store in history as follow-up
        turn = ConversationTurn(
            turn_number=len(self.history) + 1,
            timestamp=datetime.now(),
            user_input=question,
            result=result,
            context_used=follow_up_context,
            is_follow_up=True
        )
        self.history.append(turn)

        return result

    def clear_history(self) -> None:
        """
        Clear conversation history and start fresh.

        This removes all stored turns and resets the conversation to initial state.
        The underlying TKSWrapper instance is preserved.
        """
        self.history.clear()

    def get_history(self) -> List[ConversationTurn]:
        """
        Retrieve all conversation turns.

        Returns:
            List of ConversationTurn objects in chronological order
        """
        return self.history.copy()

    @property
    def last_result(self) -> Optional[TKSAnalysisResult]:
        """
        Get the most recent analysis result.

        Returns:
            The TKSAnalysisResult from the last turn, or None if no history
        """
        if not self.history:
            return None
        return self.history[-1].result

    @property
    def turn_count(self) -> int:
        """Get the number of conversation turns."""
        return len(self.history)

    def _build_context(self) -> str:
        """
        Build context string from recent conversation history.

        Returns:
            Formatted context string with previous turns
        """
        # Get the most recent N turns
        recent_turns = self.history[-self.max_context_turns:]

        if not recent_turns:
            return ""

        if self.include_equations_only:
            # Compact format: just equations
            context_parts = ["PREVIOUS EQUATIONS:"]
            for turn in recent_turns:
                context_parts.append(
                    f"- Turn {turn.turn_number}: {turn.result.equation.raw} "
                    f"(from \"{turn.user_input[:50]}...\")"
                )
        else:
            # Full format: inputs and key insights
            context_parts = ["CONVERSATION HISTORY:"]
            for turn in recent_turns:
                context_parts.append(f"\nTurn {turn.turn_number}:")
                context_parts.append(f"User: {turn.user_input}")
                context_parts.append(f"Equation: {turn.result.equation.raw}")

                # Include RPM summary
                context_parts.append(
                    f"Analysis: Desire={turn.result.rpm.desire[:50]}, "
                    f"Wisdom={turn.result.rpm.wisdom[:50]}, "
                    f"Power={turn.result.rpm.power[:50]}"
                )

        context_parts.append("\nThe user's current question builds on this context.")

        return "\n".join(context_parts)

    def export_history(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Export conversation history to dictionary (and optionally save to file).

        Args:
            file_path: Optional path to save JSON file

        Returns:
            Dictionary containing full conversation history
        """
        export_data = {
            "provider": self.wrapper.provider.name,
            "model": getattr(self.wrapper.provider, 'model', 'unknown'),
            "turn_count": self.turn_count,
            "conversation": [turn.to_dict() for turn in self.history]
        }

        if file_path:
            import json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)

        return export_data

    def print_history(self, full: bool = False) -> None:
        """
        Print conversation history to console.

        Args:
            full: If True, print full analysis for each turn. If False, print summaries.
        """
        if not self.history:
            print("No conversation history.")
            return

        print("=" * 60)
        print(f"TKS CONVERSATION HISTORY ({self.turn_count} turns)")
        print("=" * 60)

        for turn in self.history:
            if full:
                print(turn.result.summary())
                print("-" * 60)
            else:
                print(turn.summary())

        print("=" * 60)


# ==============================================================================
# CLI INTERFACE
# ==============================================================================

def main():
    """Command-line interface for TKS Conversation."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="TKS Conversation - Multi-turn TKS analysis with memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive conversation mode with mock provider
  python conversation_memory.py --interactive --provider mock

  # Interactive with Claude
  python conversation_memory.py --interactive --provider claude

  # Single analysis with context
  python conversation_memory.py "I'm stressed at work" --provider mock

  # Export conversation history
  python conversation_memory.py --interactive --provider mock --export conversation.json

Environment Variables:
  ANTHROPIC_API_KEY  - For Claude provider
  GOOGLE_API_KEY     - For Gemini provider
"""
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Initial situation to analyze"
    )
    parser.add_argument(
        "--provider", "-p",
        choices=["claude", "gemini", "mock"],
        default="mock",
        help="LLM provider"
    )
    parser.add_argument(
        "--model", "-m",
        help="Specific model to use"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive conversation mode"
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=3,
        help="Maximum previous turns to include in context (default: 3)"
    )
    parser.add_argument(
        "--equations-only",
        action="store_true",
        help="Include only equations in context, not full analysis"
    )
    parser.add_argument(
        "--export",
        help="Export conversation history to JSON file"
    )
    parser.add_argument(
        "--api-key", "-k",
        help="API key (or use environment variable)"
    )

    args = parser.parse_args()

    # Initialize conversation
    try:
        convo = TKSConversation(
            provider=args.provider,
            api_key=args.api_key,
            model=args.model,
            max_context_turns=args.max_context,
            include_equations_only=args.equations_only
        )
    except Exception as e:
        print(f"Error initializing conversation: {e}")
        return

    if args.interactive:
        print("=" * 60)
        print("TKS Conversation - Interactive Mode")
        print(f"Provider: {args.provider.upper()}")
        print(f"Max context turns: {args.max_context}")
        print("=" * 60)
        print("Commands:")
        print("  /follow <question>  - Ask follow-up about last analysis")
        print("  /history           - Show conversation history")
        print("  /clear             - Clear conversation history")
        print("  /export <file>     - Export history to JSON")
        print("  /quit              - Exit")
        print("\nEnter situations to analyze:\n")

        while True:
            try:
                user_input = input(f"\n[Turn {convo.turn_count + 1}] You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    cmd_parts = user_input.split(maxsplit=1)
                    cmd = cmd_parts[0].lower()

                    if cmd in ["/quit", "/exit", "/q"]:
                        break

                    elif cmd == "/history":
                        convo.print_history(full=False)
                        continue

                    elif cmd == "/clear":
                        convo.clear_history()
                        print("Conversation history cleared.")
                        continue

                    elif cmd == "/follow":
                        if len(cmd_parts) < 2:
                            print("Usage: /follow <question>")
                            continue

                        if not convo.history:
                            print("No previous analysis to follow up on.")
                            continue

                        question = cmd_parts[1]
                        print("\nAnalyzing follow-up...")
                        result = convo.follow_up(question)
                        print(result.summary())
                        continue

                    elif cmd == "/export":
                        if len(cmd_parts) < 2:
                            print("Usage: /export <filename>")
                            continue

                        filename = cmd_parts[1]
                        convo.export_history(filename)
                        print(f"Conversation exported to {filename}")
                        continue

                    else:
                        print(f"Unknown command: {cmd}")
                        continue

                # Regular analysis
                print("\nAnalyzing through TKS lens...")
                result = convo.analyze(user_input)
                print(result.summary())

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        # Export on exit if requested
        if args.export:
            convo.export_history(args.export)
            print(f"\nConversation exported to {args.export}")

        print(f"\nConversation complete. Total turns: {convo.turn_count}")
        print("Goodbye!")

    elif args.input:
        # Single analysis mode
        try:
            result = convo.analyze(args.input)
            print(result.summary())

            if args.export:
                convo.export_history(args.export)
                print(f"\nExported to {args.export}")
        except Exception as e:
            print(f"Error: {e}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
