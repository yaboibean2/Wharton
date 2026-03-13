import pandas as pd

from agents.risk_agent import RiskAgent


def test_diversification_handles_portfolio_dict_entries():
    agent = RiskAgent(config={})

    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    candidate_history = pd.DataFrame(
        {"Returns": [0.01] * 30},
        index=dates,
    )
    existing_history = pd.DataFrame(
        {"Returns": [0.005] * 30},
        index=dates,
    )

    score = agent._calculate_diversification_benefit(
        ticker="AAPL",
        price_history=candidate_history,
        existing_portfolio=[
            {"ticker": "MSFT", "price_history": existing_history},
            {"ticker": "NVDA"},
        ],
        all_data={},
    )

    assert isinstance(score, float)
    assert 0 <= score <= 100