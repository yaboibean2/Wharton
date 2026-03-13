from agents.sentiment_agent import SentimentAgent


def test_select_display_articles_prefers_source_variety():
    agent = SentimentAgent(config={"sentiment_agent": {"display_articles": 5}}, openai_client=object())

    primary = [
        {"title": "MSFT layoffs rumor", "source": "ET", "url": "https://economictimes.example/a"},
        {"title": "MSFT restructuring", "source": "ET", "url": "https://economictimes.example/b"},
        {"title": "MSFT earnings call", "source": "Microsoft", "url": "https://microsoft.example/c"},
        {"title": "MSFT webcast", "source": "Microsoft", "url": "https://microsoft.example/d"},
    ]

    fallback = [
        {"title": "MSFT analyst update", "source": "Reuters", "url": "https://reuters.example/e"},
        {"title": "MSFT cloud outlook", "source": "Bloomberg", "url": "https://bloomberg.example/f"},
        {"title": "MSFT valuation note", "source": "WSJ", "url": "https://wsj.example/g"},
    ]

    selected = agent._select_display_articles(primary, fallback)
    assert len(selected) == 5

    sources = {a.get("source") for a in selected}
    assert "Reuters" in sources
    assert "Bloomberg" in sources
    assert "WSJ" in sources
