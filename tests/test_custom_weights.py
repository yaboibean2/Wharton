"""
Test script to verify custom weights work correctly.
This demonstrates that higher weights give more influence to agent scores.
"""

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


# Example test case
print("=" * 60)
print("CUSTOM WEIGHTS TEST - Demonstrating Weight Influence")
print("=" * 60)

# Agent scores (these are independent of weights)
agent_scores = {
    'value_agent': 80,
    'growth_momentum_agent': 40,
    'macro_regime_agent': 60,
    'risk_agent': 70,
    'sentiment_agent': 50
}

print("\n📊 Agent Scores (independent of weights):")
for agent, score in agent_scores.items():
    print(f"   {agent.replace('_agent', '').replace('_', ' ').title()}: {score}/100")

# Test 1: Equal weights (baseline)
print("\n" + "=" * 60)
print("TEST 1: Equal Weights (1.0 for all agents)")
print("=" * 60)

equal_weights = {
    'value_agent': 1.0,
    'growth_momentum_agent': 1.0,
    'macro_regime_agent': 1.0,
    'risk_agent': 1.0,
    'sentiment_agent': 1.0
}

final_score_equal = calculate_weighted_score(agent_scores, equal_weights)
print(f"\n⚖️ Weights: All agents at 1.0x")
print(f"📈 Final Score: {final_score_equal:.2f}/100")
print(f"\nCalculation: (80×1.0 + 40×1.0 + 60×1.0 + 70×1.0 + 50×1.0) / (1.0+1.0+1.0+1.0+1.0)")
print(f"           = {sum(agent_scores.values()):.2f} / 5.0 = {final_score_equal:.2f}")

# Test 2: Emphasize Value (high-scoring agent)
print("\n" + "=" * 60)
print("TEST 2: Emphasize Value Agent (2.0x weight)")
print("=" * 60)

value_focused_weights = {
    'value_agent': 2.0,  # Double weight on high-scoring Value
    'growth_momentum_agent': 1.0,
    'macro_regime_agent': 1.0,
    'risk_agent': 1.0,
    'sentiment_agent': 1.0
}

final_score_value = calculate_weighted_score(agent_scores, value_focused_weights)
print(f"\n⚖️ Weights: Value=2.0x, Others=1.0x")
print(f"📈 Final Score: {final_score_value:.2f}/100")
print(f"📊 Change from baseline: {final_score_value - final_score_equal:+.2f} points")
print(f"\nCalculation: (80×2.0 + 40×1.0 + 60×1.0 + 70×1.0 + 50×1.0) / (2.0+1.0+1.0+1.0+1.0)")
print(f"           = {80*2.0 + 40 + 60 + 70 + 50:.2f} / 6.0 = {final_score_value:.2f}")
print(f"\n💡 Value agent (80 score) has MORE influence → Final score INCREASED")

# Test 3: De-emphasize Growth (low-scoring agent)
print("\n" + "=" * 60)
print("TEST 3: De-emphasize Growth Agent (0.5x weight)")
print("=" * 60)

growth_deemphasized_weights = {
    'value_agent': 1.0,
    'growth_momentum_agent': 0.5,  # Reduce weight on low-scoring Growth
    'macro_regime_agent': 1.0,
    'risk_agent': 1.0,
    'sentiment_agent': 1.0
}

final_score_no_growth = calculate_weighted_score(agent_scores, growth_deemphasized_weights)
print(f"\n⚖️ Weights: Growth=0.5x, Others=1.0x")
print(f"📈 Final Score: {final_score_no_growth:.2f}/100")
print(f"📊 Change from baseline: {final_score_no_growth - final_score_equal:+.2f} points")
print(f"\nCalculation: (80×1.0 + 40×0.5 + 60×1.0 + 70×1.0 + 50×1.0) / (1.0+0.5+1.0+1.0+1.0)")
print(f"           = {80 + 40*0.5 + 60 + 70 + 50:.2f} / 4.5 = {final_score_no_growth:.2f}")
print(f"\n💡 Growth agent (40 score) has LESS influence → Final score INCREASED")

# Test 4: Extreme case - Only Value matters
print("\n" + "=" * 60)
print("TEST 4: Extreme - Only Value (2.0x), Others minimal (0.1x)")
print("=" * 60)

extreme_value_weights = {
    'value_agent': 2.0,
    'growth_momentum_agent': 0.1,
    'macro_regime_agent': 0.1,
    'risk_agent': 0.1,
    'sentiment_agent': 0.1
}

final_score_extreme = calculate_weighted_score(agent_scores, extreme_value_weights)
print(f"\n⚖️ Weights: Value=2.0x, Others=0.1x")
print(f"📈 Final Score: {final_score_extreme:.2f}/100")
print(f"📊 Change from baseline: {final_score_extreme - final_score_equal:+.2f} points")
print(f"\n💡 Final score is very close to Value agent's score (80)")
print(f"   This shows Value agent dominates the final score when heavily weighted!")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
✅ Equal weights:              {final_score_equal:.2f}/100 (baseline)
📈 Emphasize Value (high):     {final_score_value:.2f}/100 ({final_score_value - final_score_equal:+.2f})
📉 De-emphasize Growth (low):  {final_score_no_growth:.2f}/100 ({final_score_no_growth - final_score_equal:+.2f})
🎯 Extreme Value focus:        {final_score_extreme:.2f}/100 ({final_score_extreme - final_score_equal:+.2f})

KEY INSIGHTS:
• Higher weight = MORE influence on final score ✅
• Agents score independently (Value always scores 80, Growth always scores 40)
• Weights only affect how scores are COMBINED into final score
• You can emphasize agents you trust more by increasing their weight
• You can de-emphasize agents by reducing their weight
""")

print("=" * 60)
