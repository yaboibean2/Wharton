"""
Test to demonstrate the upside-focused scoring changes.
Shows before/after comparison for different stock profiles.
"""

# Example Stock Profile: High-Growth Tech Stock
high_growth_stock = {
    'ticker': 'EXAMPLE-GROWTH',
    'fundamentals': {
        'earnings_growth': 0.35,  # 35% EPS growth
        'revenue_growth': 0.28,   # 28% revenue growth
        'pe_ratio': 38,           # Premium valuation
    },
    'agent_scores': {
        'value_agent': 60,        # Decent (not cheap but fair)
        'growth_momentum_agent': 88,  # Excellent growth
        'macro_regime_agent': 72, # Good macro
        'risk_agent': 48,         # Higher risk
        'sentiment_agent': 76     # Positive sentiment
    }
}

# OLD SYSTEM (Balanced weights)
old_weights = {
    'value': 0.25,
    'growth': 0.20,
    'macro': 0.15,
    'risk': 0.25,
    'sentiment': 0.15
}

old_score = (
    60 * 0.25 +    # Value: 15.0
    88 * 0.20 +    # Growth: 17.6
    72 * 0.15 +    # Macro: 10.8
    48 * 0.25 +    # Risk: 12.0
    76 * 0.15      # Sentiment: 11.4
)
# Old Score: 66.8 → HOLD

print("=" * 80)
print("UPSIDE POTENTIAL OPTIMIZATION - BEFORE vs AFTER")
print("=" * 80)
print()
print("📊 HIGH-GROWTH TECH STOCK EXAMPLE")
print("-" * 80)
print("Profile:")
print("  - EPS Growth: 35% (exceptional)")
print("  - Revenue Growth: 28% (explosive)")
print("  - P/E Ratio: 38 (premium but reasonable)")
print("  - Risk: Moderate-High")
print("  - Sentiment: Positive")
print()
print("Agent Scores:")
print(f"  - Value Agent: {high_growth_stock['agent_scores']['value_agent']}/100")
print(f"  - Growth Agent: {high_growth_stock['agent_scores']['growth_momentum_agent']}/100")
print(f"  - Macro Agent: {high_growth_stock['agent_scores']['macro_regime_agent']}/100")
print(f"  - Risk Agent: {high_growth_stock['agent_scores']['risk_agent']}/100")
print(f"  - Sentiment Agent: {high_growth_stock['agent_scores']['sentiment_agent']}/100")
print()
print("=" * 80)
print("BEFORE (Balanced System):")
print("=" * 80)
print("Agent Weights:")
print("  - Value: 25%")
print("  - Growth/Momentum: 20%")
print("  - Macro: 15%")
print("  - Risk: 25%")
print("  - Sentiment: 15%")
print()
print("Score Calculation:")
print(f"  60 × 0.25 (Value)    = 15.0")
print(f"  88 × 0.20 (Growth)   = 17.6")
print(f"  72 × 0.15 (Macro)    = 10.8")
print(f"  48 × 0.25 (Risk)     = 12.0")
print(f"  76 × 0.15 (Sentiment) = 11.4")
print("  " + "-" * 35)
print(f"  Base Score: {old_score:.1f}")
print(f"  Multiplier: 1.00x (none)")
print(f"  Final Score: {old_score:.1f}")
print()
print(f"  Recommendation: HOLD (score {old_score:.1f} < 70)")
print()
print("=" * 80)
print("AFTER (UPSIDE-FOCUSED System):")
print("=" * 80)
print("Agent Weights:")
print("  - Value: 20% ⬇️")
print("  - Growth/Momentum: 40% ⬆️ (DOUBLED!)")
print("  - Macro: 10% ⬇️")
print("  - Risk: 15% ⬇️")
print("  - Sentiment: 15%")
print()
print("Score Calculation:")
new_base = (
    60 * 0.20 +    # Value: 12.0
    88 * 0.40 +    # Growth: 35.2 🚀
    72 * 0.10 +    # Macro: 7.2
    48 * 0.15 +    # Risk: 7.2
    76 * 0.15      # Sentiment: 11.4
)
print(f"  60 × 0.20 (Value)    = 12.0")
print(f"  88 × 0.40 (Growth)   = 35.2 🚀")
print(f"  72 × 0.10 (Macro)    = 7.2")
print(f"  48 × 0.15 (Risk)     = 7.2")
print(f"  76 × 0.15 (Sentiment) = 11.4")
print("  " + "-" * 35)
print(f"  Base Score: {new_base:.1f}")
print()
print("Upside Multiplier Calculation:")
print("  - Strong growth (88/100) → +10% boost")
print("  - Very positive sentiment (76/100) → +8% boost")
print("  - Total multiplier: 1.18x")
print()
multiplier = 1.18
final_score = new_base * multiplier
print(f"  Final Score: {new_base:.1f} × {multiplier:.2f} = {final_score:.1f}")
print()
print(f"  Recommendation: STRONG BUY ✅ (score {final_score:.1f} >= 80)")
print()
print("=" * 80)
print("IMPACT SUMMARY")
print("=" * 80)
improvement = final_score - old_score
improvement_pct = (improvement / old_score) * 100
print(f"Score Change: {old_score:.1f} → {final_score:.1f}")
print(f"Improvement: +{improvement:.1f} points ({improvement_pct:.1f}% increase)")
print(f"Recommendation: HOLD → STRONG BUY")
print()
print("Why this matters:")
print("  ✅ High-growth stocks now get the recognition they deserve")
print("  ✅ Upside potential is THE PRIMARY consideration")
print("  ✅ Growth-oriented portfolios will outperform")
print("  ✅ System captures momentum and earnings acceleration")
print()
print("=" * 80)
