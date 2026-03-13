"""Unit tests and demo helpers for custom portfolio weights."""


def calculate_weighted_score(agent_scores, weights):
    """
    Calculate weighted final score.
    This is the same logic as _blend_scores in portfolio_orchestrator.py
    """
    total_score = 0
    total_weight = 0
    
    for agent, weight in weights.items():
        score = agent_scores.get(agent, 50)
        total_score += score * weight
        total_weight += weight
    
    return total_score / total_weight if total_weight > 0 else 50


AGENT_SCORES = {
    'value_agent': 80,
    'growth_momentum_agent': 40,
    'macro_regime_agent': 60,
    'risk_agent': 70,
    'sentiment_agent': 50
}

EQUAL_WEIGHTS = {
    'value_agent': 1.0,
    'growth_momentum_agent': 1.0,
    'macro_regime_agent': 1.0,
    'risk_agent': 1.0,
    'sentiment_agent': 1.0
}

VALUE_FOCUSED_WEIGHTS = {
    'value_agent': 2.0,  # Double weight on high-scoring Value
    'growth_momentum_agent': 1.0,
    'macro_regime_agent': 1.0,
    'risk_agent': 1.0,
    'sentiment_agent': 1.0
}

GROWTH_DEEMPHASIZED_WEIGHTS = {
    'value_agent': 1.0,
    'growth_momentum_agent': 0.5,  # Reduce weight on low-scoring Growth
    'macro_regime_agent': 1.0,
    'risk_agent': 1.0,
    'sentiment_agent': 1.0
}

EXTREME_VALUE_WEIGHTS = {
    'value_agent': 2.0,
    'growth_momentum_agent': 0.1,
    'macro_regime_agent': 0.1,
    'risk_agent': 0.1,
    'sentiment_agent': 0.1
}


def test_equal_weights_produce_simple_average():
    assert calculate_weighted_score(AGENT_SCORES, EQUAL_WEIGHTS) == 60


def test_emphasizing_high_scoring_agent_raises_final_score():
    baseline = calculate_weighted_score(AGENT_SCORES, EQUAL_WEIGHTS)
    adjusted = calculate_weighted_score(AGENT_SCORES, VALUE_FOCUSED_WEIGHTS)
    assert adjusted > baseline


def test_deemphasizing_low_scoring_agent_raises_final_score():
    baseline = calculate_weighted_score(AGENT_SCORES, EQUAL_WEIGHTS)
    adjusted = calculate_weighted_score(AGENT_SCORES, GROWTH_DEEMPHASIZED_WEIGHTS)
    assert adjusted > baseline


def test_extreme_weighting_moves_score_toward_dominant_agent():
    adjusted = calculate_weighted_score(AGENT_SCORES, EXTREME_VALUE_WEIGHTS)
    assert adjusted > 75


if __name__ == '__main__':
    final_score_equal = calculate_weighted_score(AGENT_SCORES, EQUAL_WEIGHTS)
    final_score_value = calculate_weighted_score(AGENT_SCORES, VALUE_FOCUSED_WEIGHTS)
    final_score_no_growth = calculate_weighted_score(AGENT_SCORES, GROWTH_DEEMPHASIZED_WEIGHTS)
    final_score_extreme = calculate_weighted_score(AGENT_SCORES, EXTREME_VALUE_WEIGHTS)

    print("=" * 60)
    print("CUSTOM WEIGHTS TEST - Demonstrating Weight Influence")
    print("=" * 60)
    print(f"Equal weights: {final_score_equal:.2f}/100")
    print(f"Emphasize value: {final_score_value:.2f}/100")
    print(f"De-emphasize growth: {final_score_no_growth:.2f}/100")
    print(f"Extreme value focus: {final_score_extreme:.2f}/100")
