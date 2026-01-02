"""
TKS Scenario Generator - Generate ideal TKS equations for goals.

This module provides goal-to-equation generation:
- Input a goal/desired outcome
- Generate the ideal TKS equation for achieving it
- Show what elements to cultivate or diminish
"""
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from tks_rules.foundations import FOUNDATIONS
from tks_rules.noetics import NOETICS
GOAL_KEYWORDS = {'career': ['FIRE', 'EARTH'], 'job': ['FIRE', 'EARTH'], 'success': ['FIRE', 'AIR'], 'money': ['EARTH', 'FIRE'], 'wealth': ['EARTH', 'FIRE'], 'relationship': ['WATER', 'FIRE'], 'love': ['WATER', 'FIRE'], 'health': ['LIFE', 'EARTH'], 'healing': ['WATER', 'LIFE'], 'creativity': ['FIRE', 'AIR'], 'spiritual': ['AETHER', 'VOID'], 'meditation': ['AETHER', 'WATER'], 'clarity': ['AIR', 'AETHER'], 'focus': ['AIR', 'FIRE'], 'confidence': ['FIRE', 'EARTH'], 'peace': ['WATER', 'AETHER'], 'balance': ['AETHER', 'EARTH'], 'growth': ['LIFE', 'FIRE'], 'learning': ['AIR', 'FIRE'], 'communication': ['AIR', 'WATER'], 'protection': ['EARTH', 'FIRE'], 'transformation': ['VOID', 'FIRE'], 'abundance': ['EARTH', 'LIFE'], 'intuition': ['WATER', 'AETHER'], 'strength': ['FIRE', 'EARTH'], 'wisdom': ['AIR', 'AETHER'], 'motivation': ['FIRE', 'LIFE'], 'release': ['WATER', 'VOID'], 'manifestation': ['FIRE', 'EARTH', 'VOID']}

@dataclass
class ScenarioResult:
    """Generated scenario equation result."""
    goal: str
    equation: str
    foundations: List[str]
    cultivate: List[str]
    diminish: List[str]
    explanation: str
    recommendations: List[str]

class TKSScenarioGenerator:
    """
    Generate ideal TKS equations for given goals.

    Usage:
        generator = TKSScenarioGenerator()

        result = generator.generate("I want to succeed in my career")
        print(result.equation)
        print(result.recommendations)
    """

    def __init__(self):
        self.foundations = FOUNDATIONS
        self.noetics = NOETICS
        self.goal_keywords = GOAL_KEYWORDS

    def generate(self, goal: str) -> ScenarioResult:
        """
        Generate ideal TKS equation for a goal.

        Args:
            goal: Natural language goal description

        Returns:
            ScenarioResult with equation and recommendations
        """
        detected_foundations = self._detect_foundations(goal)
        if not detected_foundations:
            detected_foundations = ['FIRE', 'EARTH']
        equation, cultivate, diminish = self._build_equation(detected_foundations, goal)
        explanation = self._generate_explanation(goal, detected_foundations, cultivate, diminish)
        recommendations = self._generate_recommendations(detected_foundations, cultivate, diminish)
        return ScenarioResult(goal=goal, equation=equation, foundations=detected_foundations, cultivate=cultivate, diminish=diminish, explanation=explanation, recommendations=recommendations)

    def _detect_foundations(self, goal: str) -> List[str]:
        """Detect relevant Foundations from goal text."""
        goal_lower = goal.lower()
        found = set()
        for keyword, foundations in self.goal_keywords.items():
            if keyword in goal_lower:
                for f in foundations:
                    found.add(f)
        for foundation in self.foundations.keys():
            if foundation.lower() in goal_lower:
                found.add(foundation)
        return list(found)[:3]

    def _build_equation(self, foundations: List[str], goal: str) -> Tuple[str, List[str], List[str]]:
        """Build TKS equation for given Foundations."""
        cultivate = []
        diminish = []
        terms = []
        for i, foundation in enumerate(foundations):
            f_data = self.foundations.get(foundation, {})
            world = f_data.get('world', 'A')
            positive = f_data.get('positive_noetics', [1, 3])
            negative = f_data.get('negative_noetics', [4])
            primary_n = positive[0] if positive else 1
            terms.append(f'{world}{primary_n}')
            cultivate.append(f"{world}{primary_n} ({self.noetics.get(primary_n, 'IDEA')})")
            if len(positive) > 1:
                secondary_n = positive[1]
                op = '+T' if i < len(foundations) - 1 else '→'
                terms.append(op)
                terms.append(f'{world}{secondary_n}')
                cultivate.append(f"{world}{secondary_n} ({self.noetics.get(secondary_n, 'ORDER')})")
            if negative:
                neg_n = negative[0]
                diminish.append(f"{world}{neg_n} ({self.noetics.get(neg_n, 'CHAOS')})")
        if len(terms) > 0 and '→' not in terms:
            terms.append('→')
            terms.append('D7')
        equation = ' '.join(terms)
        return (equation, cultivate, diminish)

    def _generate_explanation(self, goal: str, foundations: List[str], cultivate: List[str], diminish: List[str]) -> str:
        """Generate explanation of the equation."""
        f_names = ', '.join(foundations)
        cult_list = ', '.join(cultivate[:3])
        dim_list = ', '.join(diminish[:2]) if diminish else 'none specifically'
        return f'For the goal: "{goal}"\n\nThe primary Foundations are: {f_names}\n\nCultivate these noetic energies: {cult_list}\nDiminish or release: {dim_list}\n\nThis equation represents the ideal energy flow pattern for achieving your goal.\nThe transformations (+T) indicate where energy needs to shift between states.'

    def _generate_recommendations(self, foundations: List[str], cultivate: List[str], diminish: List[str]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        for foundation in foundations:
            f_data = self.foundations.get(foundation, {})
            qualities = f_data.get('qualities', [])
            if qualities:
                rec = f"Cultivate {foundation} energy through: {', '.join(qualities[:3])}"
                recommendations.append(rec)
        if 'FIRE' in foundations:
            recommendations.append("Take decisive action - don't wait for perfect conditions")
        if 'WATER' in foundations:
            recommendations.append('Trust your intuition and allow emotions to flow')
        if 'AIR' in foundations:
            recommendations.append('Clarify your thoughts through writing or discussion')
        if 'EARTH' in foundations:
            recommendations.append('Ground your plans in practical, measurable steps')
        if 'AETHER' in foundations:
            recommendations.append('Connect with higher purpose through meditation')
        if 'LIFE' in foundations:
            recommendations.append('Focus on growth and nurturing what matters')
        return recommendations

    def generate_inverse(self, scenario_result: ScenarioResult) -> ScenarioResult:
        """
        Generate the inverse scenario (what to avoid).

        Args:
            scenario_result: Original scenario

        Returns:
            Inverted ScenarioResult
        """
        inverse_equation = scenario_result.equation.replace('+T', '-T')
        return ScenarioResult(goal=f'INVERSE: Avoiding {scenario_result.goal}', equation=inverse_equation, foundations=scenario_result.foundations, cultivate=scenario_result.diminish, diminish=scenario_result.cultivate, explanation=f'This is what to AVOID: {scenario_result.explanation}', recommendations=[f'AVOID: {r}' for r in scenario_result.recommendations])

    def suggest_adjustments(self, current_equation: str, target_goal: str) -> Dict[str, Any]:
        """
        Suggest adjustments to an existing equation for a new goal.

        Args:
            current_equation: Current TKS equation
            target_goal: New goal to achieve

        Returns:
            Dict with adjustment suggestions
        """
        ideal = self.generate(target_goal)
        current_elements = re.findall('[A-D]\\d+', current_equation)
        ideal_elements = re.findall('[A-D]\\d+', ideal.equation)
        to_add = [e for e in ideal_elements if e not in current_elements]
        to_remove = [e for e in current_elements if e not in ideal_elements]
        to_keep = [e for e in current_elements if e in ideal_elements]
        return {'current': current_equation, 'ideal': ideal.equation, 'add': to_add, 'remove': to_remove, 'keep': to_keep, 'foundations_needed': ideal.foundations, 'explanation': f"To achieve '{target_goal}', adjust your equation by adding {to_add} and removing {to_remove}"}

def test_generator():
    """Test the scenario generator."""
    generator = TKSScenarioGenerator()
    goals = ['I want to succeed in my career', 'Find true love and a lasting relationship', 'Improve my health and vitality', 'Gain spiritual clarity and peace', 'Manifest abundance and wealth']
    for goal in goals:
        print(f"\n{'=' * 60}")
        result = generator.generate(goal)
        print(f'Goal: {goal}')
        print(f'Equation: {result.equation}')
        print(f'Foundations: {result.foundations}')
        print(f'Cultivate: {result.cultivate}')
        print(f'Diminish: {result.diminish}')
        print(f'\nRecommendations:')
        for rec in result.recommendations:
            print(f'  - {rec}')
if __name__ == '__main__':
    test_generator()