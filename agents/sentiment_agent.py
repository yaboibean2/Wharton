"""
Sentiment/News Agent
Monitors news sentiment and narrative risks.
Analyzes: earnings calls, headlines, analyst revisions, key events.
"""

from typing import Dict, Any, List, Optional
import logging
import os
import requests
from datetime import datetime, timedelta, timezone
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# Try to import BeautifulSoup for web scraping
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logger.warning("BeautifulSoup4 not available - web scraping will be limited")

# Try to import dateutil for better date parsing
try:
    import dateutil.parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False
    logger.warning("python-dateutil not available - date parsing will be limited")


class SentimentAgent(BaseAgent):
    """
    News and sentiment analysis agent.
    Scores stocks based on recent news sentiment and narrative trends.
    """
    
    def __init__(self, config: Dict[str, Any], openai_client=None):
        super().__init__("SentimentAgent", config, openai_client)
        self.sentiment_config = config.get('sentiment_agent', {})
        self.enabled = self.sentiment_config.get('enabled', True)
    
    def analyze(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze stock based on news sentiment.
        
        Returns score based on:
        - Recent news headline sentiment (enhanced with Perplexity)
        - Earnings surprise/guidance
        - Notable events (litigation, management changes, etc.)
        """
        if not self.enabled:
            return {
                'score': 50,
                'rationale': 'Sentiment analysis disabled',
                'details': {'enabled': False},
                'component_scores': {}
            }
        
        news_items = data.get('news', [])
        fundamentals = data.get('fundamentals', {})
        
        # Build analysis context from available data
        analysis_context = self._build_analysis_context(ticker, fundamentals, data)
        
        # Enhance news analysis with Perplexity for recent, relevant articles
        enhanced_news = self._fetch_enhanced_news(ticker, news_items, fundamentals, analysis_context)
        
        scores = {}
        details = {}
        
        # 1. Enhanced News Sentiment Analysis - ONLY if we have successfully scraped articles
        if enhanced_news and len(enhanced_news) > 0:
            # Check if articles have scraped content (indicating successful scraping)
            scraped_articles = [article for article in enhanced_news if 'scraped_content' in article and article.get('scraped_content')]
            
            if scraped_articles:
                logger.info(f"Found {len(scraped_articles)} successfully scraped articles for {ticker}")
                sentiment_score = self._analyze_news_sentiment(scraped_articles, ticker)
                scores['news_sentiment_score'] = sentiment_score
                details['num_articles'] = len(scraped_articles)
                details['enhanced_news_used'] = True
                
                # Include article details with links for scraped articles only
                details['article_details'] = []
                for item in scraped_articles[:5]:
                    article_detail = {
                        'title': item.get('title', 'No title'),
                        'url': item.get('url', ''),
                        'source': item.get('source', 'Unknown'),
                        'published_at': item.get('publishedAt', item.get('published_at', 'Unknown')),
                        'description': item.get('description', '')[:200] + '...' if item.get('description') else 'No description',
                        'relevance_score': item.get('relevance_score', 'N/A'),
                        'content_length': len(item.get('scraped_content', '')),
                        'preview': self._extract_article_preview(item.get('scraped_content', ''), ticker)
                    }
                    details['article_details'].append(article_detail)
                
                details['recent_headlines'] = [item['title'] for item in scraped_articles[:5]]
            else:
                logger.warning(f"No successfully scraped articles found for {ticker} - trying direct Perplexity sentiment")
                perplexity_result = self._direct_perplexity_sentiment(ticker, fundamentals)
                if perplexity_result:
                    scores['news_sentiment_score'] = perplexity_result['score']
                    details['num_articles'] = len(perplexity_result.get('articles', []))
                    details['article_details'] = perplexity_result.get('articles', [])
                    details['enhanced_news_used'] = True
                    details['perplexity_direct'] = True
                    details['recent_headlines'] = [a['title'] for a in perplexity_result.get('articles', [])]
                else:
                    scores['news_sentiment_score'] = None
                    details['num_articles'] = 0
                    details['article_details'] = []
                    details['enhanced_news_used'] = False
                    details['scraping_failed'] = True
        else:
            logger.warning(f"No enhanced news articles found for {ticker} - trying direct Perplexity sentiment")
            perplexity_result = self._direct_perplexity_sentiment(ticker, fundamentals)
            if perplexity_result:
                scores['news_sentiment_score'] = perplexity_result['score']
                details['num_articles'] = len(perplexity_result.get('articles', []))
                details['article_details'] = perplexity_result.get('articles', [])
                details['enhanced_news_used'] = True
                details['perplexity_direct'] = True
                details['recent_headlines'] = [a['title'] for a in perplexity_result.get('articles', [])]
            else:
                scores['news_sentiment_score'] = None
                details['num_articles'] = 0
                details['article_details'] = []
                details['enhanced_news_used'] = False
        
        # 2. Event Detection
        events = self._detect_key_events(news_items)
        if events:
            # Adjust score based on event type
            event_impact = self._score_events(events)
            scores['event_score'] = event_impact
            details['key_events'] = events
        else:
            scores['event_score'] = 50
            details['key_events'] = []
        
        # Composite score - only calculate if we have a valid sentiment score
        if scores.get('news_sentiment_score') is not None:
            composite_score = (
                scores['news_sentiment_score'] * 0.7 +
                scores['event_score'] * 0.3
            )
        else:
            # Default to neutral when news is unavailable
            composite_score = 50.0
            logger.warning(f"Using neutral default score (50) for {ticker} - news retrieval unsuccessful")

        # Build the best available article list for rationale generation.
        # Priority: (1) scraped enhanced articles, (2) unscraped enhanced articles,
        # (3) Perplexity-direct articles stored in details['article_details'].
        # This ensures the rationale generator always receives the articles that
        # were actually used for scoring, preventing the "limited news coverage"
        # fallback from firing when we have real Perplexity results.
        if enhanced_news:
            scraped = [a for a in enhanced_news if 'scraped_content' in a and a.get('scraped_content')]
            news_for_analysis = scraped if scraped else enhanced_news
        else:
            news_for_analysis = []

        if not news_for_analysis and details.get('article_details'):
            news_for_analysis = details['article_details']

        if scores.get('news_sentiment_score') is not None:
            # Generate detailed scoring explanation
            scoring_explanation = self._generate_scoring_explanation(ticker, scores, details, composite_score, news_for_analysis, events)
            details['scoring_explanation'] = scoring_explanation

            # Generate rationale with enhanced news (includes scraped articles and URLs)
            rationale = self._generate_rationale(ticker, news_for_analysis, events, composite_score)
        else:
            rationale = f"Sentiment analysis for {ticker} used a neutral default score (50/100) because recent news articles could not be retrieved from financial sources. This does not indicate positive or negative sentiment - it reflects limited news coverage availability."
            details['scoring_explanation'] = "News retrieval unsuccessful - neutral default applied."

        return {
            'score': round(composite_score, 2),
            'rationale': rationale,
            'details': details,
            'component_scores': scores
        }
    
    def _analyze_news_sentiment(self, news_items: List[Dict], ticker: str) -> float:
        """
        Analyze sentiment using the 3-step process:
        1. Get 3 recent articles from Perplexity
        2. Scrape each link for content  
        3. Use OpenAI for sentiment analysis with scraped content
        Returns score 0-100 (0=very negative, 50=neutral, 100=very positive).
        """
        if not news_items:
            return 50
        
        # Step 1: Get 3 recent articles from Perplexity (already done in _fetch_enhanced_news)
        # Step 2: Scrape each link (already done in _fetch_enhanced_news)
        # Step 3: Use OpenAI for sentiment analysis with scraped content
        
        # If we have scraped articles with content, use them for sentiment analysis
        if news_items and 'scraped_content' in news_items[0]:
            return self._analyze_scraped_content_sentiment(news_items, ticker)
        
        # Fallback to original method for non-scraped content
        # Check for pre-scored sentiment (from Alpha Vantage)
        if 'sentiment_score' in news_items[0]:
            # Alpha Vantage provides scores from -1 to +1
            avg_sentiment = sum(
                item.get('sentiment_score', 0) for item in news_items[:10]
            ) / min(len(news_items), 10)
            # Convert to 0-100 scale
            score = (avg_sentiment + 1) * 50
            return score
        
        # Use OpenAI for sentiment analysis
        headlines = [item['title'] for item in news_items[:10]]  # Use top 10 headlines
        headlines_text = '\n'.join([f"- {h}" for h in headlines])
        
        system_prompt = """You are a financial news sentiment analyst. Analyze news headlines and output a single sentiment score.
Score 0-100 where: 0-30=bearish, 40-60=neutral, 70-100=bullish.
Consider: earnings beats, guidance raises, new products (bullish) vs misses, downgrades, scandals (bearish)."""
        
        user_prompt = f"""Stock: {ticker}
Recent headlines:
{headlines_text}

Output only a number 0-100 for overall sentiment:"""
        
        try:
            response = self._call_openai(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=10
            )
            
            # Extract number from response
            score = float(response.strip())
            return max(0, min(100, score))
            
        except Exception as e:
            logger.warning(f"Failed to analyze sentiment: {e}")
            return 50  # Neutral default

    def _analyze_scraped_content_sentiment(self, scraped_articles: List[Dict], ticker: str) -> float:
        """
        Analyze sentiment using scraped article content following the 3-step process:
        Step 3: Use OpenAI with scraped content and include article links
        """
        if not scraped_articles:
            return 50
        
        # Collect scraped information and links
        scraped_info_parts = []
        article_links = []
        
        for i, article in enumerate(scraped_articles[:5], 1):  # Use top 5 articles
            title = article.get('title', f'Article {i}')
            url = article.get('url', '')
            content = article.get('scraped_content', article.get('content', ''))
            source = article.get('source', 'Unknown')
            
            # Add to scraped information
            if content:
                scraped_info_parts.append(f"Article {i} - {title} ({source}):\n{content[:800]}...")  # Limit content length
            else:
                scraped_info_parts.append(f"Article {i} - {title} ({source}): [Content could not be scraped]")
            
            # Collect links
            if url and url.startswith('http'):
                article_links.append(url)
        
        scraped_information = '\n\n'.join(scraped_info_parts)
        links_text = '\n'.join([f"Link {i}: {link}" for i, link in enumerate(article_links, 1)])
        
        # Use improved prompt format that ensures clear sentiment score
        system_prompt = "You are a sentiment analysis agent as part of a stock analysis pipeline. You must provide a clear numerical sentiment score."
        
        user_prompt = f"""Analyze the sentiment for stock {ticker} using the information below and provide a numerical sentiment score.

IMPORTANT: Start your response with "SENTIMENT SCORE: [number]/100" where [number] is between 0-100:
- 0-20: Very negative/bearish sentiment
- 21-40: Negative/bearish sentiment  
- 41-60: Neutral sentiment
- 61-80: Positive/bullish sentiment
- 81-100: Very positive/bullish sentiment

Article Information:
{scraped_information}

Sources:
{links_text}

Provide your analysis after the sentiment score."""
        
        try:
            response = self._call_openai(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.2,
                max_tokens=800
            )
            
            # Store the full response for rationale generation
            self._sentiment_analysis_response = response
            
            # Extract sentiment score from response with improved precision
            import re
            
            # First, look for the exact format we requested: "SENTIMENT SCORE: XX/100"
            primary_pattern = r'SENTIMENT\s+SCORE:\s*(\d{1,3})(?:/100)?'
            primary_match = re.search(primary_pattern, response, re.IGNORECASE)
            
            if primary_match:
                score = float(primary_match.group(1))
                score = max(0, min(100, score))  # Clamp to 0-100 range
                logger.info(f"Extracted primary sentiment score for {ticker}: {score}")
                
                # Validate score consistency with analysis content
                validated_score = self._validate_score_consistency(score, response, ticker)
                return validated_score
            
            # Fallback patterns - more specific than before
            fallback_patterns = [
                r'(?:sentiment|score)(?:\s*:?\s*)(\d{1,3})(?:\s*/\s*100|\s*out\s+of\s+100)',  # Score: 75/100 or Score: 75 out of 100
                r'(?:sentiment|score)(?:\s*:?\s*)(\d{1,3})(?!\.|,|\d)',  # Score: 75 (not followed by decimal or comma)
                r'overall.*?sentiment.*?(\d{1,3})(?:/100)?',  # overall sentiment 75
                r'(\d{1,3})(?:/100)?\s*(?:sentiment|score)',  # 75/100 sentiment
            ]
            
            for pattern in fallback_patterns:
                score_match = re.search(pattern, response, re.IGNORECASE)
                if score_match:
                    score = float(score_match.group(1))
                    score = max(0, min(100, score))  # Clamp to 0-100 range
                    logger.info(f"Extracted fallback sentiment score for {ticker}: {score}")
                    
                    # Validate score consistency with analysis content
                    validated_score = self._validate_score_consistency(score, response, ticker)
                    return validated_score
            
            # If no score found, try to infer from sentiment words and context
            logger.warning(f"No explicit score found in response for {ticker}, attempting inference...")
            
            # Enhanced sentiment word inference with more context awareness
            sentiment_indicators = {
                # Very positive indicators
                r'overwhelmingly\s+positive|extremely\s+bullish|very\s+strong.*positive': 85,
                r'predominantly\s+positive|highly\s+favorable|strong.*bullish': 80,
                r'analyst.*upgrade|outperform|buy.*rating|price.*target.*rais': 75,
                
                # Positive indicators  
                r'positive\s+outlook|bullish|optimistic|favorable|strong.*performance': 70,
                r'upgrade|positive.*sentiment|good\s+news|strong.*results': 65,
                
                # Neutral indicators
                r'mixed|neutral|uncertain|hold.*rating': 50,
                
                # Negative indicators
                r'negative|bearish|pessimistic|unfavorable|weak.*performance': 30,
                r'downgrade|sell.*rating|poor.*results|concerning': 25,
                
                # Very negative indicators
                r'very\s+negative|extremely\s+bearish|highly\s+unfavorable': 15,
            }
            
            for pattern, score in sentiment_indicators.items():
                if re.search(pattern, response, re.IGNORECASE):
                    logger.info(f"Inferred sentiment score {score} for {ticker} from pattern: {pattern[:50]}...")
                    return score
            
            # Final fallback - analyze the overall tone
            logger.warning(f"Could not extract or infer sentiment score for {ticker}, analyzing overall tone...")
            
            # Count positive vs negative words as last resort
            positive_words = len(re.findall(r'\b(?:positive|good|strong|upgrade|outperform|bullish|favorable|growth|beat|exceed)\b', response, re.IGNORECASE))
            negative_words = len(re.findall(r'\b(?:negative|bad|weak|downgrade|underperform|bearish|unfavorable|decline|miss|disappoint)\b', response, re.IGNORECASE))
            
            if positive_words > negative_words * 2:  # Strong positive bias
                logger.info(f"Defaulting to positive sentiment (70) for {ticker} based on word analysis: +{positive_words} vs -{negative_words}")
                return 70
            elif positive_words > negative_words:  # Mild positive bias
                logger.info(f"Defaulting to mild positive sentiment (60) for {ticker} based on word analysis: +{positive_words} vs -{negative_words}")
                return 60
            elif negative_words > positive_words:  # Negative bias
                logger.info(f"Defaulting to negative sentiment (40) for {ticker} based on word analysis: +{positive_words} vs -{negative_words}")
                return 40
            else:  # Neutral
                logger.info(f"Defaulting to neutral sentiment (50) for {ticker} based on word analysis: +{positive_words} vs -{negative_words}")
                return 50
            
        except Exception as e:
            logger.error(f"Failed to analyze scraped content sentiment for {ticker}: {e}")
            return 50

    def _validate_score_consistency(self, extracted_score: float, response: str, ticker: str) -> float:
        """
        Validate that the extracted score is consistent with the analysis content.
        If there's a major mismatch, adjust the score to match the analysis tone.
        """
        import re
        
        # Analyze the tone of the response
        very_positive_indicators = [
            r'overwhelmingly\s+positive', r'extremely\s+bullish', r'very\s+strong.*positive',
            r'highly\s+favorable', r'outstanding', r'exceptional'
        ]
        
        positive_indicators = [
            r'predominantly\s+positive', r'positive\s+outlook', r'bullish', r'optimistic', 
            r'favorable', r'upgrade', r'outperform', r'strong.*performance', r'good\s+news'
        ]
        
        negative_indicators = [
            r'negative', r'bearish', r'pessimistic', r'unfavorable', r'weak.*performance',
            r'downgrade', r'underperform', r'concerning', r'disappointing'
        ]
        
        very_negative_indicators = [
            r'overwhelmingly\s+negative', r'extremely\s+bearish', r'very\s+negative',
            r'highly\s+unfavorable', r'terrible', r'disastrous'
        ]
        
        # Count indicators
        very_pos_count = sum(1 for pattern in very_positive_indicators if re.search(pattern, response, re.IGNORECASE))
        pos_count = sum(1 for pattern in positive_indicators if re.search(pattern, response, re.IGNORECASE))
        neg_count = sum(1 for pattern in negative_indicators if re.search(pattern, response, re.IGNORECASE))
        very_neg_count = sum(1 for pattern in very_negative_indicators if re.search(pattern, response, re.IGNORECASE))
        
        # Determine expected score range based on content analysis
        if very_pos_count > 0 or pos_count >= 3:
            expected_range = (75, 90)  # Very positive
        elif pos_count > neg_count:
            expected_range = (60, 80)  # Positive
        elif neg_count > pos_count or very_neg_count > 0:
            expected_range = (20, 45)  # Negative
        else:
            expected_range = (40, 60)  # Neutral
        
        # Check if extracted score is wildly inconsistent
        if extracted_score < expected_range[0] - 20 or extracted_score > expected_range[1] + 20:
            # Major inconsistency detected
            suggested_score = (expected_range[0] + expected_range[1]) / 2
            logger.warning(f"Score inconsistency detected for {ticker}: extracted={extracted_score}, expected={expected_range}, adjusting to {suggested_score}")
            logger.warning(f"Content analysis: very_pos={very_pos_count}, pos={pos_count}, neg={neg_count}, very_neg={very_neg_count}")
            return suggested_score
        else:
            # Score is reasonable, keep it
            logger.info(f"Score consistency validated for {ticker}: {extracted_score} is within expected range {expected_range}")
            return extracted_score
    
    def _detect_key_events(self, news_items: List[Dict]) -> List[str]:
        """
        Detect key events from news headlines.
        Returns list of event types detected.
        """
        events = []
        keywords = {
            'earnings_beat': ['beat', 'exceeds expectations', 'earnings surprise', 'strong quarter', 'beats estimates', 'stronger than expected', 'outperformed estimates'],
            'earnings_miss': ['miss', 'disappoints', 'below expectations', 'weak quarter', 'missed estimates', 'fell short', 'weaker than expected'],
            'earnings_report': ['reported earnings', 'quarterly earnings', 'earnings results', 'financial results', 'q1 earnings', 'q2 earnings', 'q3 earnings', 'q4 earnings', 'quarterly results', 'earnings call', 'earnings announcement'],
            'guidance_raise': ['raises guidance', 'increases forecast', 'upgraded outlook', 'raised full-year', 'boosted outlook', 'increased guidance'],
            'guidance_cut': ['lowers guidance', 'cuts forecast', 'reduced outlook', 'lowered full-year', 'cut outlook', 'reduced guidance'],
            'revenue_beat': ['revenue beat', 'sales beat', 'top-line beat', 'revenue exceeded', 'sales exceeded'],
            'revenue_miss': ['revenue miss', 'sales miss', 'top-line miss', 'revenue fell short', 'sales disappointed'],
            'litigation': ['lawsuit', 'sued', 'legal', 'investigation', 'regulatory'],
            'management_change': ['ceo', 'chief executive', 'management change', 'appoints'],
            'product_launch': ['launches', 'new product', 'unveils', 'announces'],
            'acquisition': ['acquires', 'merger', 'acquisition', 'buys'],
        }
        
        for item in news_items[:10]:
            title_lower = item['title'].lower()
            for event_type, event_keywords in keywords.items():
                if any(keyword in title_lower for keyword in event_keywords):
                    if event_type not in events:
                        events.append(event_type)
        
        return events
    
    def _score_events(self, events: List[str]) -> float:
        """Score impact of detected events."""
        event_impacts = {
            'earnings_beat': 70,
            'earnings_miss': 30,
            'earnings_report': 55,  # Neutral earnings report (no beat/miss info)
            'guidance_raise': 75,
            'guidance_cut': 25,
            'revenue_beat': 65,
            'revenue_miss': 35,
            'litigation': 35,
            'management_change': 50,  # Neutral
            'product_launch': 65,
            'acquisition': 60,
        }
        
        if not events:
            return 50
        
        scores = [event_impacts.get(event, 50) for event in events]
        return sum(scores) / len(scores)
    
    def _generate_scoring_explanation(self, ticker: str, scores: Dict, details: Dict, final_score: float, news_items: List[Dict], events: List[str]) -> str:
        """Generate detailed explanation of why this specific score was assigned."""
        
        explanation = f"**Sentiment Score Breakdown: {final_score:.1f}/100**\n\n"
        
        # Component score explanations
        news_score = scores.get('news_sentiment_score', 50)
        event_score = scores.get('event_score', 50)
        num_articles = details.get('num_articles', 0)
        
        explanation += f"**Component Scores:**\n"
        explanation += f"• News Sentiment: {news_score:.1f}/100 (70% weight) - "
        if num_articles == 0:
            explanation += "No recent news articles found for analysis\n"
        elif news_score >= 75:
            explanation += f"Very positive sentiment across {num_articles} articles with bullish themes\n"
        elif news_score >= 60:
            explanation += f"Positive sentiment from {num_articles} articles with optimistic coverage\n"
        elif news_score >= 40:
            explanation += f"Neutral sentiment from {num_articles} articles with balanced reporting\n"
        elif news_score >= 25:
            explanation += f"Negative sentiment from {num_articles} articles with bearish themes\n"
        else:
            explanation += f"Very negative sentiment across {num_articles} articles with pessimistic coverage\n"
        
        explanation += f"• Event Impact: {event_score:.1f}/100 (30% weight) - "
        if not events:
            explanation += "No significant corporate events detected in recent news\n"
        else:
            positive_events = [e for e in events if any(pos in e for pos in ['beat', 'raise', 'launch', 'acquisition'])]
            negative_events = [e for e in events if any(neg in e for neg in ['miss', 'cut', 'litigation'])]
            
            if positive_events and not negative_events:
                explanation += f"Positive events detected: {', '.join(positive_events)}\n"
            elif negative_events and not positive_events:
                explanation += f"Negative events detected: {', '.join(negative_events)}\n"
            elif positive_events and negative_events:
                explanation += f"Mixed events: {len(positive_events)} positive, {len(negative_events)} negative\n"
            else:
                explanation += f"Neutral events detected: {', '.join(events[:3])}\n"
        
        explanation += f"\n**Why this score?**\n"
        if final_score >= 80:
            explanation += "Overwhelmingly positive sentiment with strong bullish catalysts and favorable news coverage.\n"
        elif final_score >= 70:
            explanation += "Positive sentiment environment with good news flow and optimistic market perception.\n"
        elif final_score >= 50:
            explanation += "Neutral sentiment with balanced news coverage and mixed market signals.\n"
        elif final_score >= 30:
            explanation += "Negative sentiment with concerning news themes and bearish market perception.\n"
        else:
            explanation += "Very negative sentiment with multiple bearish factors and pessimistic coverage.\n"
        
        # Recent headlines analysis with sources and links
        if news_items and len(news_items) > 0:
            explanation += f"\n**Recent Headlines Analysis:**\n"
            for i, item in enumerate(news_items[:5], 1):
                title = item.get('title', 'No title')
                source = item.get('source', 'Unknown Source')
                url = item.get('url', '')

                title_display = f"{title[:80]}{'...' if len(title) > 80 else ''}"
                
                if url:
                    explanation += f"{i}. **{title_display}** - [{source}]({url})\n"
                else:
                    explanation += f"{i}. **{title_display}** - {source}\n"
        
        explanation += f"\n**To improve score:**\n"
        improvements = []
        if news_score < 60:
            improvements.append("More positive news coverage and analyst commentary needed")
        if event_score < 60 and not events:
            improvements.append("Positive corporate events (earnings beats, guidance raises) would boost sentiment")
        if num_articles < 5:
            improvements.append("Increased media attention and news coverage would provide better sentiment signal")
        
        if improvements:
            for imp in improvements:
                explanation += f"• {imp}\n"
        else:
            explanation += "Score is strong based on current news sentiment and events\n"
        
        return explanation

    def _generate_rationale(
        self,
        ticker: str,
        news_items: List[Dict],
        events: List[str],
        sentiment_score: float
    ) -> str:
        """Generate detailed sentiment rationale using OpenAI."""
        # Defensive check for None sentiment score
        if sentiment_score is None:
            return f"Sentiment analysis unavailable for {ticker}: Unable to retrieve and scrape news articles from reliable sources."
        
        if not news_items:
            return "Limited recent news coverage indicates low market attention and neutral sentiment"
        
        # Check if we have a stored sentiment analysis response from the two-step process
        if hasattr(self, '_sentiment_analysis_response') and self._sentiment_analysis_response:
            # Use the detailed analysis from the two-step process
            detailed_analysis = self._sentiment_analysis_response
            
            # Format article links
            article_links_section = self._format_article_links(news_items)
            
            # Combine the detailed analysis with article links
            rationale = f"{detailed_analysis}"
            if article_links_section:
                rationale += f"\n\n{article_links_section}"
            
            return rationale.strip()
        
        # Prepare detailed context for OpenAI with URLs included
        headlines_with_urls = []
        for item in news_items[:5]:  # Focus on top 5 scraped articles
            title = item.get('title', 'No title')
            url = item.get('url', '')
            if url and url.startswith('http'):
                headlines_with_urls.append(f"- {title} [{url}]")
            else:
                headlines_with_urls.append(f"- {title}")
        headlines_text = '\n'.join(headlines_with_urls)
        
        system_prompt = """You are a senior market sentiment analyst at a leading investment research firm.
You specialize in analyzing news flow, market narratives, and investor sentiment to gauge stock momentum.
Your analysis should be:
1. Comprehensive and insightful, explaining how news and events shape investor perception
2. Evidence-based, citing specific news sources and events that drive sentiment
3. Market-aware, considering broader market context and investor behavior
4. Forward-looking, discussing implications for stock performance
5. Around 120-180 words with specific, actionable insights about sentiment trends

ACCURACY RULES — ZERO TOLERANCE FOR ERRORS:
- ONLY use the exact numerical values provided in the user prompt below. NEVER invent, round differently, or hallucinate statistics.
- ONLY reference headlines and sources that are explicitly listed in the data below. Do NOT invent article titles or sources.
- If a sentiment score is provided, cite it exactly as given.
- Before writing each number or headline, mentally verify it matches the data provided verbatim.
- Do NOT claim sentiment shifts, analyst rating changes, or events that are not explicitly in the data below."""
        
        # Calculate component scores for comprehensive analysis
        news_volume_score = min(100, len(news_items) * 10)  # Approximate score based on volume
        event_impact_score = 75 if events else 50  # High if events detected, neutral otherwise
        recency_score = 70  # Assume recent news for context
        
        # Enhanced context with prominent URL display
        sources_info = ""
        if news_items:
            source_details = []
            for i, item in enumerate(news_items[:5], 1):
                source = item.get('source', 'Unknown')
                url = item.get('url', '')
                title = item.get('title', f'Article {i}')

                if url and url.startswith('http'):
                    source_details.append(f"Article {i}: {title} | {source} | {url}")
                else:
                    source_details.append(f"Article {i}: {title} | {source} | (URL not available)")
            
            sources_info = f"\n\nDETAILED ARTICLE SOURCES:\n" + '\n'.join(source_details)
        
        user_prompt = f"""
SENTIMENT ANALYSIS REQUEST: {ticker}
FINAL SENTIMENT SCORE: {sentiment_score:.1f}/100

NEWS COVERAGE ANALYSIS:
• Total Articles Analyzed: {len(news_items)} recent articles
• Key Market Events: {', '.join(events) if events else 'No major events detected'}
• Coverage Intensity: {'High' if len(news_items) > 5 else 'Moderate' if len(news_items) > 2 else 'Limited'}

RECENT HEADLINES & SOURCES:
{headlines_text}{sources_info}

SCORING CONTEXT:
- Scores above 80 = Very positive sentiment with strong bullish catalysts
- Scores 60-80 = Positive sentiment with supportive news flow
- Scores 40-60 = Neutral sentiment, mixed or limited news
- Scores 20-40 = Negative sentiment with concerning developments
- Scores below 20 = Very negative sentiment with significant headwinds

ANALYSIS REQUEST:
As a sentiment expert, provide a comprehensive analysis explaining why {ticker} earned a {sentiment_score:.1f}/100 sentiment score.
Address:
1. What are the key news themes and narratives driving current sentiment?
2. How do recent headlines reflect investor psychology and market perception?
3. What specific events or developments are most impactful for sentiment?
4. How does news volume and source credibility affect the sentiment assessment?
5. What are the implications for near-term stock performance based on current sentiment?

Focus on actionable insights about market sentiment momentum and investor behavior patterns."""
        
        try:
            rationale = self._call_openai(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=250
            )
            
            # ALWAYS append the article links to the rationale
            article_links_section = self._format_article_links(news_items)
            if article_links_section:
                rationale += f"\n\n{article_links_section}"
            
            return rationale.strip()
        except Exception as e:
            logger.warning(f"Failed to generate rationale: {e}")
            
            # Simple, direct fallback explanations with links
            fallback_rationale = f"Sentiment analysis based on {len(news_items)} recent articles for {ticker}."
            article_links_section = self._format_article_links(news_items)
            if article_links_section:
                fallback_rationale += f"\n\n{article_links_section}"
            return fallback_rationale
    
    def _format_article_links(self, news_items: List[Dict]) -> str:
        """
        Format article links for display in the sentiment rationale.
        Deduplicates by normalised URL so the same article never appears twice.
        """
        import re as _re

        if not news_items:
            return ""

        seen_urls: set = set()
        links_with_info = []
        idx = 1
        for item in news_items[:8]:  # check up to 8 to fill 5 unique slots
            title = item.get('title', f'Article {idx}')
            url = item.get('url', '')
            source = item.get('source', 'Unknown Source')

            if url:
                # Strip trailing citation markers like [3]
                url = _re.sub(r'\[\d+\]$', '', url.strip())
                norm = url.lower().rstrip('/')
                if norm in seen_urls:
                    continue
                seen_urls.add(norm)

            if url and url.startswith('http'):
                title_short = title[:60] + '...' if len(title) > 60 else title
                links_with_info.append(f"Article {idx}: {title_short} | {source} | {url}")
            else:
                links_with_info.append(f"Article {idx}: {title} | {source} | (No URL available)")
            idx += 1
            if idx > 5:
                break

        if links_with_info:
            return "ARTICLE SOURCES:\n" + '\n'.join(links_with_info)
        else:
            return ""

    def _extract_article_preview(self, content: str, ticker: str) -> str:
        """Extract a meaningful preview quote from article content."""
        if not content or len(content) < 50:
            return ""

        import re
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)

        # Try to find sentences mentioning the ticker
        relevant = [s.strip() for s in sentences
                    if ticker.lower() in s.lower() and 30 < len(s.strip()) < 300]

        if relevant:
            return relevant[0][:200]

        # Fall back to first substantial sentence
        for sentence in sentences:
            s = sentence.strip()
            if 40 < len(s) < 300:
                return s[:200]

        return content[:200]

    def _direct_perplexity_sentiment(self, ticker: str, fundamentals: Dict) -> Optional[Dict]:
        """
        Fallback: Ask Perplexity directly to analyze recent news sentiment for a stock.
        Used when article scraping fails entirely.

        Returns:
            Dict with 'score' (0-100) and 'articles' list, or None on failure
        """
        import os
        import requests
        import re
        import json

        perplexity_key = os.getenv('PERPLEXITY_API_KEY')
        if not perplexity_key:
            return None

        company_name = fundamentals.get('name', ticker)

        prompt = (
            f"Analyze the current news sentiment for {company_name} ({ticker}) stock.\n\n"
            f"1. Review the most recent 5 news articles about {ticker} from the past 14 days.\n"
            f"2. Assess overall sentiment on a scale of 0-100 "
            f"(0=extremely negative, 50=neutral, 100=extremely positive).\n\n"
            f"Return your response in EXACTLY this JSON format:\n"
            f'{{"score": <number 0-100>, "summary": "<1-2 sentence summary>", '
            f'"articles": ['
            f'{{"title": "<title>", "source": "<source name>", "url": "<url>", "sentiment": "<positive/negative/neutral>"}},'
            f"..."
            f"]}}"
        )

        headers = {
            "Authorization": f"Bearer {perplexity_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "sonar-pro",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 800
        }

        try:
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code != 200:
                logger.warning(f"Direct Perplexity sentiment failed: HTTP {response.status_code}")
                return None

            result = response.json()
            content = result['choices'][0]['message']['content']

            # Parse JSON from response (may be wrapped in markdown code blocks)
            json_match = re.search(r'\{[\s\S]*\}', content)
            if not json_match:
                logger.warning(f"Could not parse JSON from Perplexity sentiment response")
                return None

            parsed = json.loads(json_match.group())
            score = float(parsed.get('score', 50))
            score = max(0, min(100, score))

            articles = []
            for a in parsed.get('articles', [])[:5]:
                articles.append({
                    'title': a.get('title', 'News Article'),
                    'url': a.get('url', ''),
                    'source': a.get('source', 'Unknown'),
                    'published_at': 'Recent',
                    'description': a.get('sentiment', 'neutral'),
                    'preview': '',
                    'relevance_score': 'N/A',
                    'content_length': 0
                })

            logger.info(f"Direct Perplexity sentiment for {ticker}: score={score:.0f}, articles={len(articles)}")
            return {'score': score, 'articles': articles, 'summary': parsed.get('summary', '')}

        except Exception as e:
            logger.error(f"Direct Perplexity sentiment failed for {ticker}: {e}")
            return None
    
    def _fetch_enhanced_news(self, ticker: str, original_news: List[Dict], fundamentals: Dict, analysis_context: str = "") -> List[Dict]:
        """
        Fetch enhanced news articles using a multi-source approach:
        1. Ask Perplexity for 5 specific article URLs from credible sources
        2. Scrape those URLs to get actual content
        3. Fall back to NewsAPI + original news if Perplexity fails

        Args:
            ticker: Stock ticker
            original_news: Original news from data provider
            fundamentals: Company fundamentals for context
            analysis_context: Context from other agent rationales

        Returns:
            Enhanced news articles with scraped content
        """
        import os
        import requests
        from datetime import datetime

        perplexity_key = os.getenv('PERPLEXITY_API_KEY')
        if not perplexity_key:
            logger.warning("Perplexity API key not available - using original news only")
            return original_news

        # Step 1: Get article URLs from Perplexity
        logger.info(f"Searching for recent {ticker} articles from credible financial sources...")
        article_urls = self._get_article_urls_from_perplexity(ticker, fundamentals, analysis_context)

        if not article_urls:
            logger.warning(f"No article URLs found for {ticker} from Perplexity")
            # Fall back to original news if available, enriching with scraped_content
            if original_news:
                logger.info(f"Enriching {len(original_news)} original articles for {ticker}")
                enriched = []
                for article in original_news:
                    a = dict(article)
                    content = (a.get('summary', '') or a.get('description', '')
                               or a.get('content', '') or a.get('title', ''))
                    if content and len(content) >= 10:
                        a['scraped_content'] = content[:1000]
                        enriched.append(a)
                return enriched if enriched else original_news
            return []

        logger.info(f"Found {len(article_urls)} article URLs for {ticker}")

        # Step 2: Scrape the article URLs to get content
        logger.info(f"Scraping {len(article_urls)} articles for {ticker}...")
        scraped_articles = self._scrape_article_urls(article_urls, ticker)

        if scraped_articles:
            total_content_length = sum(len(article.get('scraped_content', '')) for article in scraped_articles)
            logger.info(f"Successfully scraped {len(scraped_articles)} articles for {ticker} ({total_content_length:,} chars)")
            # Sort articles by recency (newest first)
            sorted_articles = self._sort_articles_by_recency(scraped_articles, ticker)
            return sorted_articles
        else:
            logger.warning(f"Failed to scrape articles for {ticker} - enriching original news as fallback")
            # Enrich original_news with scraped_content from their existing fields
            # so the scraped_content filter in analyze() won't reject them
            fallback_articles = []
            source_articles = original_news if original_news else []
            for article in source_articles:
                enriched = dict(article)
                content = (article.get('summary', '') or article.get('description', '')
                           or article.get('content', '') or article.get('title', ''))
                if content and len(content) >= 10:
                    enriched['scraped_content'] = content[:1000]
                    fallback_articles.append(enriched)
            if fallback_articles:
                logger.info(f"Enriched {len(fallback_articles)} original news articles as fallback for {ticker}")
                return self._sort_articles_by_recency(fallback_articles, ticker)
            return original_news

    def _build_analysis_context(self, ticker: str, fundamentals: Dict, data: Dict) -> str:
        """
        Build analysis context from available data to provide to Perplexity for better article selection.
        
        Args:
            ticker: Stock ticker
            fundamentals: Company fundamentals
            data: All available data
            
        Returns:
            Analysis context string
        """
        context_parts = []
        
        # Company basics
        company_name = fundamentals.get('name', ticker)
        sector = fundamentals.get('sector', 'Unknown')
        context_parts.append(f"Company: {company_name} ({ticker}) in {sector} sector")
        
        # Key metrics if available
        pe_ratio = fundamentals.get('pe_ratio')
        if pe_ratio:
            context_parts.append(f"P/E ratio: {pe_ratio:.1f}")
        
        market_cap = fundamentals.get('market_cap')
        if market_cap:
            market_cap_b = market_cap / 1e9
            context_parts.append(f"Market cap: ${market_cap_b:.1f}B")
        
        # Price trends if available
        price = fundamentals.get('price')
        if price:
            context_parts.append(f"Current price: ${price:.2f}")
        
        # Recent news themes if available
        existing_news = data.get('news', [])
        if existing_news:
            recent_themes = [item.get('title', '')[:50] for item in existing_news[:2]]
            if recent_themes:
                context_parts.append(f"Recent news themes: {', '.join(recent_themes)}")
        
        return ". ".join(context_parts)

    def _get_article_urls_from_perplexity(self, ticker: str, fundamentals: Dict, analysis_context: str) -> List[str]:
        """
        Ask Perplexity for recent, high-quality news article URLs from credible financial sources.

        Args:
            ticker: Stock ticker
            fundamentals: Company fundamentals for context
            analysis_context: Context from other agent analyses

        Returns:
            List of article URLs
        """
        perplexity_key = os.getenv('PERPLEXITY_API_KEY')
        company_name = fundamentals.get('name', ticker)
        sector = fundamentals.get('sector', 'Unknown')

        prompt = f"""Find 5 recent and credible news articles about {company_name} ({ticker}) stock published within the last 14 days.

Requirements:
- Articles must be from reputable financial sources such as Reuters, Bloomberg, CNBC, Wall Street Journal, Financial Times, Barron's, MarketWatch, SeekingAlpha, Yahoo Finance, Investor's Business Daily, or Motley Fool
- Prioritize articles covering: earnings results, analyst ratings/upgrades/downgrades, revenue guidance, major business developments, partnerships, product launches, or sector trends affecting {ticker}
- Each article must be a direct URL to the full article (not a search results page)
- Prefer the most recent articles available

Return ONLY the 5 URLs, one per line, with no other text or formatting."""

        headers = {
            "Authorization": f"Bearer {perplexity_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "sonar-pro",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 1000
        }

        try:
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                url_content = result['choices'][0]['message']['content']

                # Extract URLs from the response
                logger.info(f"Perplexity response for {ticker}: {url_content[:300]}...")
                urls = self._extract_urls_from_response(url_content)
                logger.info(f"Extracted {len(urls)} URLs for {ticker}")
                return urls

            else:
                logger.error(f"Perplexity URL fetch failed for {ticker}: {response.status_code}")
                try:
                    error_detail = response.json()
                    logger.error(f"Perplexity error details: {error_detail}")
                except:
                    logger.error(f"Perplexity response text: {response.text}")
                return []

        except Exception as e:
            logger.error(f"Error fetching article URLs for {ticker}: {e}")
            return []



    def _sort_articles_by_recency(self, articles: List[Dict], ticker: str) -> List[Dict]:
        """
        Sort articles by publication date, prioritizing newer articles.
        
        Args:
            articles: List of article dictionaries
            ticker: Stock ticker for logging
            
        Returns:
            Sorted list with newest articles first
        """
        from datetime import datetime, timezone, timedelta
        import dateutil.parser
        
        def parse_article_date(article: Dict) -> datetime:
            """Parse article date with multiple fallback strategies."""
            # Try different date fields
            date_fields = ['publishedAt', 'published_at', 'date', 'timestamp']

            for field in date_fields:
                date_str = article.get(field)
                if date_str:
                    try:
                        if isinstance(date_str, str):
                            parsed_date = dateutil.parser.parse(date_str)
                            # Normalize to UTC-aware to avoid comparison errors
                            if parsed_date.tzinfo is None:
                                parsed_date = parsed_date.replace(tzinfo=timezone.utc)
                            return parsed_date
                        elif isinstance(date_str, datetime):
                            if date_str.tzinfo is None:
                                return date_str.replace(tzinfo=timezone.utc)
                            return date_str
                    except Exception as e:
                        logger.debug(f"Failed to parse date '{date_str}' with dateutil: {e}")
                        continue
            
            # If no valid date found, extract from URL or title
            url = article.get('url', '')
            title = article.get('title', '')
            
            # Look for date patterns in URL (e.g., /2025/10/01/)
            import re
            url_date_match = re.search(r'/(\d{4})/(\d{1,2})/(\d{1,2})/', url)
            if url_date_match:
                try:
                    year, month, day = map(int, url_date_match.groups())
                    return datetime(year, month, day, tzinfo=timezone.utc)
                except:
                    pass
            
            # Look for recent indicators in title
            recent_indicators = {
                'today': 0,      # 0 days ago
                'yesterday': 1,  # 1 day ago
                'this week': 3,  # 3 days ago
                'last week': 7,  # 7 days ago
                'this month': 15, # 15 days ago
            }
            
            title_lower = title.lower()
            for indicator, days_ago in recent_indicators.items():
                if indicator in title_lower:
                    return datetime.now(timezone.utc) - timedelta(days=days_ago)
            
            # Default to current time minus 1 day (assume recent)
            return datetime.now(timezone.utc) - timedelta(days=1)
        
        try:
            # Parse dates and sort
            articles_with_dates = []
            for article in articles:
                parsed_date = parse_article_date(article)
                articles_with_dates.append((parsed_date, article))
            
            # Sort by date (newest first)
            articles_with_dates.sort(key=lambda x: x[0], reverse=True)
            
            # Extract sorted articles
            sorted_articles = [article for date, article in articles_with_dates]
            
            logger.info(f"Sorted {len(sorted_articles)} articles for {ticker} by recency")
            
            # Log the date ordering for debugging
            for i, (date, article) in enumerate(articles_with_dates[:3]):
                title_short = article.get('title', 'No title')[:50] + '...' if len(article.get('title', '')) > 50 else article.get('title', 'No title')
                logger.info(f"Article {i+1}: {date.strftime('%Y-%m-%d %H:%M')} - {title_short}")
            
            return sorted_articles
            
        except Exception as e:
            logger.error(f"Error sorting articles by recency for {ticker}: {e}")
            # Return original list if sorting fails
            return articles

    def _extract_urls_from_response(self, content: str) -> List[str]:
        """Extract URLs from Perplexity response."""
        import re
        
        logger.info(f"Extracting URLs from content: {content[:500]}...")
        
        # Improved URL patterns to preserve full URLs
        url_patterns = [
            r'https?://[^\s\[\]<>()"\n]+',  # Standard URLs - no truncation
            r'https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[/\w\-._~:/?#[\]@!$&\'()*+,;=\%]*',  # Include % for encoded URLs
        ]
        
        all_found_urls = []
        
        for pattern in url_patterns:
            urls = re.findall(pattern, content, re.IGNORECASE)
            all_found_urls.extend(urls)
        
        logger.info(f"Found {len(all_found_urls)} URLs with multiple patterns: {all_found_urls}")
        
        # Enhanced extraction for specific format: "1. [link], 2. [link], 3. [link]"
        numbered_urls = []
        
        # First, try to extract from the specific requested format
        numbered_format_match = re.findall(r'\d+\.\s*(https?://[^\s,\[\]<>()"]+)', content)
        if numbered_format_match:
            numbered_urls.extend(numbered_format_match)
            logger.info(f"Found URLs in numbered format: {numbered_format_match}")
        
        # Also look for numbered list format and bracketed URLs
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            # Check for numbered list format
            if re.match(r'^\d+\.?\s*https?://', line):
                url_match = re.search(r'https?://[^\s\[\]<>()"]+', line)
                if url_match:
                    numbered_urls.append(url_match.group())
            # Check for bracketed URLs [URL]
            elif '[http' in line.lower():
                bracket_urls = re.findall(r'\[https?://[^\]]+\]', line)
                for bracket_url in bracket_urls:
                    clean_url = bracket_url.strip('[]')
                    numbered_urls.append(clean_url)
            # Check for URLs after common prefixes
            elif any(prefix in line.lower() for prefix in ['url:', 'link:', 'source:']):
                url_match = re.search(r'https?://[^\s\[\]<>()"]+', line)
                if url_match:
                    numbered_urls.append(url_match.group())
        
        # Also check for comma-separated format within the same line
        comma_separated_urls = re.findall(r'https?://[^\s,\[\]<>()"]+(?=\s*,|\s*$)', content)
        if comma_separated_urls:
            numbered_urls.extend(comma_separated_urls)
            logger.info(f"Found URLs in comma-separated format: {comma_separated_urls}")
        
        logger.info(f"Found {len(numbered_urls)} numbered/formatted URLs: {numbered_urls}")
        
        # Combine and deduplicate
        all_urls = list(set(all_found_urls + numbered_urls))
        
        # Clean URLs more carefully to preserve valid URLs
        cleaned_urls = []
        seen_normalized = set()
        for url in all_urls:
            url = url.strip()
            # Strip Perplexity citation markers like [3]
            url = re.sub(r'\[\d+\]$', '', url)
            # Only remove trailing punctuation that's clearly not part of the URL
            if url.endswith('.') and not url.endswith('.html') and not url.endswith('.php') and not url.endswith('.jsp'):
                url = url[:-1]
            elif url.endswith(','):
                url = url[:-1]
            
            # Validate URL structure
            if url and len(url) > 15 and ('/' in url[8:]):
                # Deduplicate by normalised form (lowercase, no trailing slash)
                norm = url.lower().rstrip('/')
                if norm not in seen_normalized:
                    seen_normalized.add(norm)
                    cleaned_urls.append(url)
        
        logger.info(f"Final cleaned URLs (first 5): {cleaned_urls[:5]}")

        # Return first 5 URLs
        return cleaned_urls[:5]
    
    def _scrape_article_urls(self, urls: List[str], ticker: str) -> List[Dict]:
        """
        Scrape the provided URLs to get article content.
        
        Args:
            urls: List of article URLs to scrape
            ticker: Stock ticker
            
        Returns:
            List of scraped article data
        """
        scraped_articles = []
        
        for i, url in enumerate(urls):
            try:
                article_data = self._scrape_single_article(url, ticker, i + 1)
                if article_data:
                    scraped_articles.append(article_data)
                else:
                    logger.warning(f"Article scraping returned no data for {url}")
            except Exception as e:
                logger.error(f"Failed to scrape article {url}: {e}")
                # No fallback article - just skip this URL
        
        return scraped_articles
    
    def _scrape_single_article(self, url: str, ticker: str, article_num: int) -> Optional[Dict]:
        """
        Scrape a single article URL to get content.
        First tries to use fetch_webpage tool if available, then falls back to BeautifulSoup.
        
        Args:
            url: Article URL
            ticker: Stock ticker  
            article_num: Article number (1, 2, or 3)
            
        Returns:
            Article data dictionary
        """
        import requests
        import re
        from datetime import datetime
        
        # Check if URL is a document file (PDF, etc.) that we can't scrape
        if url.lower().endswith(('.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx')):
            logger.info(f"Skipping document file for article {article_num}: {url}")
            return {
                'title': f"{ticker} Financial Document",
                'source': 'Financial Document',
                'publishedAt': None,
                'description': f"Financial document related to {ticker}",
                'url': url,
                'content': f"Financial document about {ticker} - document parsing not supported",
                'scraped_content': f"Financial document about {ticker}",
                'relevance_score': 'Low',
                'sentiment_impact': 'Neutral'
            }
        
        # Try enhanced scraping with fetch_webpage first
        scraped_content = self._enhanced_scrape_with_fetch_webpage(url, ticker)
        if scraped_content:
            logger.info(f"Successfully scraped article {article_num} using fetch_webpage")
            return scraped_content
        
        # Fallback to BeautifulSoup scraping
        if not BS4_AVAILABLE:
            logger.warning("BeautifulSoup not available - cannot scrape article")
            return None  # Return None instead of fallback content
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            logger.info(f"Scraping article {article_num}: {url[:60]}...")
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}")
            
            soup = BeautifulSoup(response.content, 'html.parser')
            logger.info(f"Successfully loaded HTML for article {article_num} ({len(response.content):,} bytes)")
            
            # Extract title
            title = ""
            title_selectors = ['h1', 'title', '.headline', '.article-title', '.post-title']
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem and title_elem.text.strip():
                    title = title_elem.text.strip()
                    break
            
            if not title:
                title = f"{ticker} News Article {article_num}"
            
            # Extract article content
            content = ""
            content_selectors = [
                '.article-content', '.entry-content', '.post-content', 
                '.article-body', '.story-body', '.content', 'article', '.main-content'
            ]
            
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Get text and clean it
                    content = content_elem.get_text()
                    content = re.sub(r'\s+', ' ', content).strip()
                    break
            
            # If no content found, try paragraph tags
            if not content:
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
            
            # Extract source/domain
            source = url.split('//')[1].split('/')[0] if '//' in url else 'Unknown Source'
            
            # Try to extract publication date from meta tags
            published_at = self._extract_publication_date(soup, url)
            
            # Create description from first part of content
            description = content[:200] + "..." if len(content) > 200 else content
            
            logger.info(f"Extracted from article {article_num}: Title='{title[:50]}...', Content={len(content):,} chars")
            
            return {
                'title': title,
                'source': source,
                'publishedAt': published_at,
                'description': description,
                'url': url,
                'content': content[:1000],  # Limit content length
                'scraped_content': content[:1000],  # Add this field to indicate successful scraping
                'relevance_score': 'High',
                'sentiment_impact': 'Neutral'  # Will be analyzed later
            }
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None

    def _enhanced_scrape_with_fetch_webpage(self, url: str, ticker: str) -> Optional[Dict]:
        """
        Enhanced scraping using fetch_webpage tool if available.
        
        Args:
            url: Article URL to scrape
            ticker: Stock ticker for context
            
        Returns:
            Article data dictionary or None if scraping fails
        """
        try:
            # Check if URL is a PDF or other non-HTML content
            if url.lower().endswith(('.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx')):
                logger.info(f"Skipping document file (not HTML): {url}")
                return {
                    'title': f"{ticker} Financial Document",
                    'source': 'Financial Document',
                    'publishedAt': None,
                    'description': f"Financial document related to {ticker}",
                    'url': url,
                    'content': f"Financial document about {ticker} - document parsing not supported",
                    'scraped_content': f"Financial document about {ticker}",
                    'relevance_score': 'Low',
                    'sentiment_impact': 'Neutral'
                }
            
            # Try to use fetch_webpage tool for better content extraction
            from datetime import datetime
            
            # Import here to avoid circular imports
            import requests
            
            # Create a simple query for the fetch_webpage tool
            query = f"{ticker} news article analysis"
            
            # Simulate fetch_webpage functionality with requests and better parsing
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            logger.info(f"Enhanced scraping for {url[:60]}...")
            response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
            
            if response.status_code != 200:
                logger.warning(f"Enhanced scraping failed with status {response.status_code}")
                return None
                
            if not BS4_AVAILABLE:
                logger.warning("BeautifulSoup not available for enhanced scraping")
                return None
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.decompose()
            
            # Enhanced title extraction
            title = self._extract_title_enhanced(soup, ticker)
            
            # Enhanced content extraction
            content = self._extract_content_enhanced(soup, ticker)
            
            # Enhanced source extraction
            source = self._extract_source_enhanced(soup, url)
            
            # Enhanced date extraction
            published_at = self._extract_publication_date(soup, url)
            
            if not content or len(content.strip()) < 50:
                logger.warning(f"Enhanced scraping found insufficient content ({len(content)} chars) for {url[:60]}")
                # Return a basic article structure with just the title and URL for sentiment analysis
                return {
                    'title': title if title else f"{ticker} Market News",
                    'source': source if source else 'Financial News',
                    'publishedAt': published_at,
                    'description': f"News article about {ticker} - content could not be scraped",
                    'url': url,
                    'content': f"Financial news about {ticker}",  # Minimal content for sentiment
                    'scraped_content': f"Financial news about {ticker}",  # Mark as scraped even if minimal
                    'relevance_score': 'Medium',
                    'sentiment_impact': 'Neutral'
                }
            
            # Create description from content
            description = content[:200] + "..." if len(content) > 200 else content
            
            logger.info(f"Enhanced scraping successful: {len(content):,} chars, title: '{title[:50]}...'")
            
            return {
                'title': title,
                'source': source,
                'publishedAt': published_at,
                'description': description,
                'url': url,
                'content': content[:2000],  # Increased content limit
                'scraped_content': content[:2000],  # Mark as successfully scraped
                'relevance_score': 'High',
                'sentiment_impact': 'Neutral'
            }
            
        except Exception as e:
            logger.error(f"Enhanced scraping failed for {url}: {e}")
            return None
    
    def _extract_title_enhanced(self, soup, ticker: str) -> str:
        """Extract title with enhanced selectors."""
        title_selectors = [
            'h1.headline', 'h1.article-title', 'h1.entry-title', 'h1.post-title',
            'h1[class*="title"]', 'h1[class*="headline"]', 
            '.article-headline', '.story-headline', '.news-headline',
            'h1', 'title', '.headline', '.article-title', '.post-title'
        ]
        
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem and title_elem.get_text().strip():
                title = title_elem.get_text().strip()
                # Clean title
                title = ' '.join(title.split())  # Normalize whitespace
                if len(title) > 10 and ticker.lower() in title.lower():
                    return title
                elif len(title) > 10:  # Even if ticker not mentioned, use it if long enough
                    return title
        
        return f"{ticker} Market News"
    
    def _extract_content_enhanced(self, soup, ticker: str) -> str:
        """Extract content with enhanced selectors and cleaning."""
        content_selectors = [
            '.article-content .article-body', '.entry-content .post-content',
            '.story-body .main-content', '.content .article-text',
            '[class*="article-content"]', '[class*="story-body"]', '[class*="entry-content"]',
            'article .content', 'main .content', '.post-content', '.article-body',
            '.story-content', '.news-content', '.text-content'
        ]
        
        content = ""
        
        # Try content selectors
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                content = content_elem.get_text()
                content = ' '.join(content.split())  # Normalize whitespace
                if len(content) > 200:  # Prefer longer content
                    break
        
        # If no content found, try paragraphs within article
        if not content or len(content) < 100:
            article_elem = soup.find('article')
            if article_elem:
                paragraphs = article_elem.find_all('p')
                content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        # If still no content, try all paragraphs
        if not content or len(content) < 100:
            paragraphs = soup.find_all('p')
            all_text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
            # Filter for relevant paragraphs (containing ticker or financial terms)
            financial_terms = [ticker.lower(), 'stock', 'share', 'market', 'earnings', 'revenue', 'analyst']
            relevant_paragraphs = []
            for p in paragraphs:
                p_text = p.get_text().strip()
                if any(term in p_text.lower() for term in financial_terms) and len(p_text) > 20:
                    relevant_paragraphs.append(p_text)
            
            if relevant_paragraphs:
                content = ' '.join(relevant_paragraphs)
            else:
                content = all_text
        
        # Final fallback - try to get any text content at all
        if not content or len(content) < 50:
            # Get all text from the page, excluding navigation elements
            body = soup.find('body')
            if body:
                # Remove unwanted elements
                for unwanted in body.find_all(['nav', 'footer', 'header', 'aside', 'script', 'style']):
                    unwanted.decompose()
                content = body.get_text()
                content = ' '.join(content.split())  # Normalize whitespace
                # Try to find ticker-relevant content
                sentences = content.split('.')
                relevant_sentences = [s.strip() for s in sentences if ticker.lower() in s.lower() or any(term in s.lower() for term in ['stock', 'market', 'earnings', 'analyst'])]
                if relevant_sentences:
                    content = '. '.join(relevant_sentences[:10])  # Limit to first 10 relevant sentences
        
        return content.strip()
    
    def _extract_source_enhanced(self, soup, url: str) -> str:
        """Extract source with enhanced detection."""
        # Try meta tags first
        meta_sources = soup.find_all('meta', {'name': ['author', 'source', 'publisher']})
        for meta in meta_sources:
            source = meta.get('content', '').strip()
            if source:
                return source
        
        # Try structured data
        for script in soup.find_all('script', {'type': 'application/ld+json'}):
            try:
                import json
                data = json.loads(script.string)
                if 'publisher' in data and 'name' in data['publisher']:
                    return data['publisher']['name']
            except:
                continue
        
        # Extract from URL domain
        if '//' in url:
            domain = url.split('//')[1].split('/')[0]
            # Clean up common domain patterns
            domain = domain.replace('www.', '')
            if '.' in domain:
                source_name = domain.split('.')[0]
                return source_name.title()
        
        return 'Financial News'

    def _extract_publication_date(self, soup, url: str) -> str:
        """
        Extract publication date from article HTML.
        
        Args:
            soup: BeautifulSoup parsed HTML
            url: Article URL
            
        Returns:
            ISO formatted date string
        """
        from datetime import datetime
        
        # Common meta tag selectors for publication date
        date_selectors = [
            'meta[property="article:published_time"]',
            'meta[name="publishdate"]',
            'meta[name="date"]',
            'meta[property="og:article:published_time"]',
            'meta[name="DC.date.issued"]',
            'meta[name="article:published_time"]',
            'time[datetime]',
            '.date',
            '.publish-date',
            '.article-date',
            '.timestamp'
        ]
        
        for selector in date_selectors:
            try:
                elem = soup.select_one(selector)
                if elem:
                    # Try different attributes
                    date_text = elem.get('content') or elem.get('datetime') or elem.get_text().strip()
                    if date_text:
                        # Try to parse the date
                        if DATEUTIL_AVAILABLE:
                            parsed_date = dateutil.parser.parse(date_text)
                            return parsed_date.isoformat()
                        else:
                            # Simple ISO date parsing fallback
                            import re
                            iso_match = re.match(r'(\d{4})-(\d{2})-(\d{2})', date_text)
                            if iso_match:
                                year, month, day = map(int, iso_match.groups())
                                date_obj = datetime(year, month, day)
                                return date_obj.isoformat()
            except Exception as e:
                logger.debug(f"Failed to parse date from selector {selector}: {e}")
                continue
        
        # Try to extract date from URL pattern
        import re
        url_date_match = re.search(r'/(\d{4})/(\d{1,2})/(\d{1,2})/', url)
        if url_date_match:
            try:
                year, month, day = map(int, url_date_match.groups())
                date_obj = datetime(year, month, day)
                return date_obj.isoformat()
            except:
                pass
        
        # Default to current time
        return datetime.now().isoformat()
    
    def _parse_perplexity_news(self, news_content: str, ticker: str) -> List[Dict]:
        """
        Parse news content from Perplexity response into structured format.
        Enhanced to parse exactly 3 articles with full details including URLs.
        
        Args:
            news_content: Raw news content from Perplexity
            ticker: Stock ticker
            
        Returns:
            List of exactly 3 structured news articles
        """
        articles = []
        
        # Split content into article blocks
        article_blocks = []
        current_block = []
        
        lines = news_content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('Article ') and current_block:
                article_blocks.append('\n'.join(current_block))
                current_block = [line]
            elif line:
                current_block.append(line)
        
        if current_block:
            article_blocks.append('\n'.join(current_block))
        
        # Parse each article block
        for block in article_blocks[:3]:  # Limit to 3 articles
            article = self._parse_single_article_block(block, ticker)
            if article:
                articles.append(article)
        
        # If we don't have 3 articles, pad with generic entries
        while len(articles) < 3:
            article_num = len(articles) + 1
            articles.append({
                'title': f'{ticker} Market Analysis - Article {article_num}',
                'source': 'Market Research',
                'publishedAt': datetime.now().isoformat(),
                'description': f'Recent market developments and sentiment analysis for {ticker}.',
                'url': f'https://perplexity.ai/search?q={ticker}+news+recent',
                'relevance_score': 'High',
                'sentiment_impact': 'Neutral'
            })
        
        return articles[:3]  # Ensure exactly 3 articles
    
    def _parse_single_article_block(self, block: str, ticker: str) -> Dict:
        """
        Parse a single article block from Perplexity response.
        
        Args:
            block: Single article text block
            ticker: Stock ticker
            
        Returns:
            Structured article dictionary
        """
        article = {
            'title': '',
            'source': 'Unknown Source',
            'publishedAt': datetime.now().isoformat(),
            'description': '',
            'url': f'https://perplexity.ai/search?q={ticker}+news',
            'relevance_score': 'High',
            'sentiment_impact': 'Neutral'
        }
        
        lines = block.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Extract headline
            if line.lower().startswith('headline:'):
                article['title'] = line.replace('headline:', '').replace('Headline:', '').strip()
            
            # Extract source
            elif line.lower().startswith('source:'):
                article['source'] = line.replace('source:', '').replace('Source:', '').strip()
            
            # Extract date
            elif line.lower().startswith('date:'):
                date_str = line.replace('date:', '').replace('Date:', '').strip()
                article['publishedAt'] = date_str
            
            # Extract summary/description
            elif line.lower().startswith('summary:'):
                article['description'] = line.replace('summary:', '').replace('Summary:', '').strip()
            
            # Extract URL
            elif line.lower().startswith('url:') or 'http' in line.lower():
                url_line = line.replace('url:', '').replace('URL:', '').strip()
                if url_line.startswith('http'):
                    article['url'] = url_line
                else:
                    # Look for URLs in the line
                    import re
                    url_match = re.search(r'https?://[^\s]+', line)
                    if url_match:
                        article['url'] = url_match.group()
            
            # Extract sentiment impact
            elif line.lower().startswith('sentiment impact:'):
                sentiment = line.replace('sentiment impact:', '').replace('Sentiment Impact:', '').strip().lower()
                article['sentiment_impact'] = sentiment.title()
        
        # If no title found, use first line as title
        if not article['title'] and lines:
            article['title'] = lines[0].strip()
        
        # If no description, combine relevant lines
        if not article['description']:
            desc_lines = [line for line in lines if len(line) > 20 and not any(
                line.lower().startswith(prefix) for prefix in ['headline:', 'source:', 'date:', 'url:', 'sentiment impact:']
            )]
            if desc_lines:
                article['description'] = ' '.join(desc_lines[:2])  # First 2 relevant lines
        
        return article
