"""
TKS Session Logger - Track and archive TKS analyses over time.

This module provides session logging capabilities for TKS analyses, enabling:
- Persistent storage of analyses in JSON and Markdown formats
- Historical search and retrieval by date, tags, or foundations
- Report generation for analysis trends
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class LogEntry:
    """Single logged TKS analysis."""
    timestamp: str
    query: str
    equation: str
    foundations: List[str]
    noetics: Dict[str, Any]
    interpretation: str
    provider: str
    tags: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}


class TKSSessionLogger:
    """
    Logger for TKS analysis sessions.

    Usage:
        logger = TKSSessionLogger(log_dir="./tks_logs")

        # Log an analysis
        logger.log(
            query="My life situation",
            equation="A3 +T B7 -T C2 → D5",
            foundations=["FIRE", "WATER"],
            noetics={"positive": True},
            interpretation="The analysis shows...",
            provider="gemini"
        )

        # Search logs
        entries = logger.search(foundation="FIRE", days=30)

        # Generate report
        report = logger.generate_report()
    """

    def __init__(self, log_dir: str = "./tks_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.json_file = self.log_dir / "session_log.jsonl"
        self.markdown_dir = self.log_dir / "markdown"
        self.markdown_dir.mkdir(exist_ok=True)

    def log(
        self,
        query: str,
        equation: str,
        foundations: List[str],
        noetics: Dict[str, Any],
        interpretation: str,
        provider: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LogEntry:
        """
        Log a TKS analysis.

        Args:
            query: The user's input query
            equation: Generated TKS equation
            foundations: Identified Foundations
            noetics: Noetic elements breakdown
            interpretation: Analysis interpretation
            provider: Provider used (claude/gemini)
            tags: Optional tags for categorization
            metadata: Optional additional metadata

        Returns:
            LogEntry object
        """
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            query=query,
            equation=equation,
            foundations=foundations,
            noetics=noetics,
            interpretation=interpretation,
            provider=provider,
            tags=tags or [],
            metadata=metadata or {}
        )

        # Save to JSON Lines
        self._append_json(entry)

        # Save to Markdown
        self._save_markdown(entry)

        return entry

    def _append_json(self, entry: LogEntry) -> None:
        """Append entry to JSONL file."""
        with open(self.json_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(entry)) + '\n')

    def _save_markdown(self, entry: LogEntry) -> None:
        """Save entry as markdown file."""
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        md_file = self.markdown_dir / f"{date_str}.md"

        content = f"""# TKS Analysis - {entry.timestamp}

## Query
{entry.query}

## Equation
```
{entry.equation}
```

## Foundations
{', '.join(entry.foundations)}

## Noetics
```json
{json.dumps(entry.noetics, indent=2)}
```

## Interpretation
{entry.interpretation}

## Provider
{entry.provider}

## Tags
{', '.join(entry.tags) if entry.tags else 'None'}
"""

        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(content)

    def load_all(self) -> List[LogEntry]:
        """Load all log entries."""
        entries = []

        if not self.json_file.exists():
            return entries

        with open(self.json_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    entries.append(LogEntry(**data))

        return entries

    def search(
        self,
        foundation: Optional[str] = None,
        tag: Optional[str] = None,
        provider: Optional[str] = None,
        days: Optional[int] = None,
        query_contains: Optional[str] = None
    ) -> List[LogEntry]:
        """
        Search log entries by criteria.

        Args:
            foundation: Filter by Foundation
            tag: Filter by tag
            provider: Filter by provider
            days: Only entries from last N days
            query_contains: Filter by query text

        Returns:
            List of matching LogEntry objects
        """
        entries = self.load_all()
        results = []

        cutoff_date = None
        if days:
            from datetime import timedelta
            cutoff_date = datetime.now() - timedelta(days=days)

        for entry in entries:
            # Check foundation
            if foundation and foundation not in entry.foundations:
                continue

            # Check tag
            if tag and tag not in entry.tags:
                continue

            # Check provider
            if provider and entry.provider != provider:
                continue

            # Check date
            if cutoff_date:
                entry_date = datetime.fromisoformat(entry.timestamp)
                if entry_date < cutoff_date:
                    continue

            # Check query text
            if query_contains and query_contains.lower() not in entry.query.lower():
                continue

            results.append(entry)

        return results

    def generate_report(self, days: Optional[int] = None) -> str:
        """
        Generate analysis report.

        Args:
            days: Only include entries from last N days

        Returns:
            Formatted report string
        """
        entries = self.search(days=days) if days else self.load_all()

        if not entries:
            return "No entries found."

        # Foundation frequency
        foundation_counts = {}
        for entry in entries:
            for f in entry.foundations:
                foundation_counts[f] = foundation_counts.get(f, 0) + 1

        # Provider usage
        provider_counts = {}
        for entry in entries:
            provider_counts[entry.provider] = provider_counts.get(entry.provider, 0) + 1

        # Tag frequency
        tag_counts = {}
        for entry in entries:
            for t in entry.tags:
                tag_counts[t] = tag_counts.get(t, 0) + 1

        report = f"""# TKS Session Report
Generated: {datetime.now().isoformat()}
Total Entries: {len(entries)}

## Foundation Distribution
"""
        for f, count in sorted(foundation_counts.items(), key=lambda x: -x[1]):
            report += f"- {f}: {count}\n"

        report += "\n## Provider Usage\n"
        for p, count in sorted(provider_counts.items(), key=lambda x: -x[1]):
            report += f"- {p}: {count}\n"

        if tag_counts:
            report += "\n## Tags\n"
            for t, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
                report += f"- {t}: {count}\n"

        return report

    def get_patterns(self, min_occurrences: int = 3) -> Dict[str, Any]:
        """
        Identify patterns in logged analyses.

        Args:
            min_occurrences: Minimum occurrences to report

        Returns:
            Dict with pattern analysis
        """
        entries = self.load_all()

        patterns = {
            "common_foundation_pairs": {},
            "frequent_noetic_combinations": {},
            "query_themes": []
        }

        # Foundation pair analysis
        for entry in entries:
            if len(entry.foundations) >= 2:
                pair = tuple(sorted(entry.foundations[:2]))
                patterns["common_foundation_pairs"][pair] = \
                    patterns["common_foundation_pairs"].get(pair, 0) + 1

        # Filter by min_occurrences
        patterns["common_foundation_pairs"] = {
            k: v for k, v in patterns["common_foundation_pairs"].items()
            if v >= min_occurrences
        }

        return patterns


def test_logger():
    """Test the session logger."""
    logger = TKSSessionLogger(log_dir="./test_logs")

    # Log some entries
    logger.log(
        query="Feeling stuck in my career",
        equation="A3 +T B7 -T C2 → D5",
        foundations=["FIRE", "AIR"],
        noetics={"positive": True, "energy_level": 7},
        interpretation="Need to balance mental and physical energy",
        provider="gemini",
        tags=["career", "balance"]
    )

    logger.log(
        query="Relationship issues",
        equation="A5 +T B3 -T C8 → D2",
        foundations=["WATER", "EARTH"],
        noetics={"emotional": True, "grounded": False},
        interpretation="Focus on emotional stability",
        provider="claude",
        tags=["relationship", "emotions"]
    )

    # Search
    results = logger.search(foundation="FIRE")
    print(f"Found {len(results)} entries with FIRE foundation")

    # Report
    report = logger.generate_report()
    print(report)


if __name__ == "__main__":
    test_logger()
