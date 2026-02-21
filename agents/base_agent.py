"""
Base Agent Class
Abstract base class for all investment analysis agents.
Defines common interface and OpenAI integration.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from openai import OpenAI
import logging
from utils.logger import get_disclosure_logger

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    Each agent analyzes stocks from a specific perspective.
    """
    
    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        openai_client: Optional[OpenAI] = None
    ):
        self.name = name
        self.config = config
        self.disclosure_logger = get_disclosure_logger()
        
        # Initialize OpenAI client
        if openai_client:
            self.openai = openai_client
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            
            self.openai = OpenAI(api_key=api_key)
        
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        logger.info(f"Initialized {self.name}")
    
    @abstractmethod
    def analyze(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a stock and return a score + rationale.
        
        Args:
            ticker: Stock ticker symbol
            data: Dictionary containing all relevant data for analysis
        
        Returns:
            {
                'score': float (0-100),
                'rationale': str (one-line explanation),
                'details': dict (supporting metrics)
            }
        """
        pass
    
    def _call_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 500
    ) -> str:
        """
        Call OpenAI API with disclosure logging.
        
        Args:
            system_prompt: System instructions
            user_prompt: User query
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
        
        Returns:
            Response text
        """
        try:
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            result = response.choices[0].message.content
            tokens = response.usage.total_tokens
            
            # Estimate cost (gpt-4o-mini: $0.15/$0.60 per 1M tokens)
            cost = (response.usage.prompt_tokens * 0.15 + response.usage.completion_tokens * 0.60) / 1_000_000
            
            # Log for disclosure
            self.disclosure_logger.log_ai_usage(
                tool=f"OpenAI-{self.model}",
                purpose=f"{self.name} analysis",
                prompt_summary=user_prompt[:100] + "..." if len(user_prompt) > 100 else user_prompt,
                output_summary=result[:100] + "..." if len(result) > 100 else result,
                tokens_used=tokens,
                cost_usd=cost
            )
            
            return result
            
        except Exception as e:
            logger.error(f"OpenAI API error in {self.name}: {e}")
            raise
    
    def _normalize_score(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize a value to 0-100 scale."""
        if max_val == min_val:
            return 50.0
        normalized = ((value - min_val) / (max_val - min_val)) * 100
        return max(0.0, min(100.0, normalized))
    
    def _safe_get(self, data: Dict, key: str, default: Any = None) -> Any:
        """Safely get value from dict with default."""
        return data.get(key, default)

    def _fetch_supporting_articles(self, ticker: str, domain_query: str, num_articles: int = 2) -> List[Dict]:
        """
        Fetch domain-specific supporting articles using Perplexity.

        Uses Perplexity's native citations (grounded URLs the model actually referenced)
        as the primary source, then validates every URL with an HTTP HEAD check to ensure
        links are live before including them in the output.

        Args:
            ticker: Stock ticker symbol
            domain_query: Domain-specific search query (e.g., "valuation P/E analysis")
            num_articles: Number of articles to fetch (default: 2)

        Returns:
            List of article dicts with 'title', 'url', 'source', 'verified' keys
        """
        import requests
        import re
        from concurrent.futures import ThreadPoolExecutor, as_completed

        perplexity_key = os.getenv('PERPLEXITY_API_KEY')
        if not perplexity_key:
            logger.debug(f"No Perplexity API key - skipping article fetch for {self.name}")
            return []

        prompt = (
            f"Find {num_articles} recent credible articles specifically about "
            f"{domain_query} for stock ticker {ticker}.\n\n"
            f"Requirements:\n"
            f"- From reputable financial sources (Reuters, Bloomberg, CNBC, WSJ, "
            f"Barron's, MarketWatch, SeekingAlpha, Yahoo Finance, Motley Fool)\n"
            f"- Directly relevant to {domain_query} for {ticker}\n"
            f"- Published within the last 30 days\n"
            f"- Return ONLY in this exact format, one per line:\n"
            f"TITLE: [article title] | URL: [full URL]"
        )

        headers = {
            "Authorization": f"Bearer {perplexity_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "sonar",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 400
        }

        try:
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload,
                timeout=15
            )

            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']

                # --- Primary source: Perplexity native citations ---
                # These are grounded URLs that the model actually referenced
                native_citations = result.get('citations', [])
                if native_citations:
                    logger.info(
                        f"{self.name}: Perplexity returned {len(native_citations)} "
                        f"native citations for {ticker}"
                    )

                # Parse model-generated article text as secondary source
                parsed_articles = self._parse_article_citations(content, num_articles + 2)

                # Cross-reference: prefer parsed titles matched to native citation URLs
                articles = self._merge_citations(parsed_articles, native_citations, num_articles)

                # --- Validate every URL with HTTP HEAD ---
                articles = self._validate_article_urls(articles)

                logger.info(
                    f"{self.name}: {len(articles)} verified articles for {ticker}"
                )
                return articles[:num_articles]
            else:
                logger.warning(f"{self.name}: Perplexity article fetch failed with status {response.status_code}")
                return []
        except Exception as e:
            logger.warning(f"{self.name}: Failed to fetch supporting articles for {ticker}: {e}")
            return []

    def _merge_citations(
        self,
        parsed_articles: List[Dict],
        native_citations: List[str],
        num_articles: int,
    ) -> List[Dict]:
        """
        Merge parsed article text with Perplexity's native citation URLs.

        Priority:
        1. Parsed articles whose URL matches a native citation (highest confidence)
        2. Parsed articles with non-citation URLs (model generated, lower confidence)
        3. Native citations not matched to any parsed article (use domain as title)
        """
        merged: List[Dict] = []
        used_native: set = set()
        used_parsed: set = set()

        # Pass 1 — match parsed articles to native citations by domain overlap
        for i, art in enumerate(parsed_articles):
            art_domain = art.get('source', '').lower()
            art_url = art.get('url', '').lower()
            for j, cite_url in enumerate(native_citations):
                if j in used_native:
                    continue
                cite_lower = cite_url.lower()
                # Match if domains overlap or one URL is a prefix of the other
                if (art_domain and art_domain in cite_lower) or art_url == cite_lower:
                    merged.append({
                        'title': art['title'],
                        'url': cite_url,  # prefer native citation URL (grounded)
                        'source': art['source'],
                        'citation_backed': True,
                    })
                    used_native.add(j)
                    used_parsed.add(i)
                    break

        # Pass 2 — remaining parsed articles
        for i, art in enumerate(parsed_articles):
            if i in used_parsed:
                continue
            merged.append({**art, 'citation_backed': False})

        # Pass 3 — leftover native citations (no matching parsed article)
        for j, cite_url in enumerate(native_citations):
            if j in used_native:
                continue
            source = cite_url.split('//')[1].split('/')[0].replace('www.', '') if '//' in cite_url else 'Unknown'
            merged.append({
                'title': f"Financial analysis from {source}",
                'url': cite_url,
                'source': source,
                'citation_backed': True,
            })

        # Sort: citation-backed first, then by original order
        merged.sort(key=lambda a: (not a.get('citation_backed', False)))
        return merged[:num_articles + 2]  # keep extras in case some fail validation

    def _validate_article_urls(self, articles: List[Dict]) -> List[Dict]:
        """
        Validate article URLs with concurrent HTTP HEAD requests.
        Drops articles whose URLs are unreachable (non-2xx or timeout).
        """
        import requests as _req
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not articles:
            return []

        def _check_url(article: Dict) -> Optional[Dict]:
            url = article.get('url', '')
            if not url:
                return None
            try:
                # HEAD first (fast); fall back to GET with stream for sites
                # that block HEAD (e.g. some news paywalls)
                resp = _req.head(
                    url,
                    timeout=4,
                    allow_redirects=True,
                    headers={'User-Agent': 'Mozilla/5.0 (compatible; InvestmentBot/1.0)'},
                )
                if resp.status_code < 400:
                    article['verified'] = True
                    return article

                # Retry with GET (some servers reject HEAD)
                resp = _req.get(
                    url,
                    timeout=4,
                    allow_redirects=True,
                    stream=True,  # don't download body
                    headers={'User-Agent': 'Mozilla/5.0 (compatible; InvestmentBot/1.0)'},
                )
                resp.close()
                if resp.status_code < 400:
                    article['verified'] = True
                    return article

                logger.debug(f"{self.name}: URL returned {resp.status_code}: {url}")
                return None
            except Exception as e:
                logger.debug(f"{self.name}: URL unreachable ({e}): {url}")
                return None

        verified: List[Dict] = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_check_url, art): art for art in articles}
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None:
                    verified.append(result)

        # Maintain original order
        url_order = {art['url']: idx for idx, art in enumerate(articles)}
        verified.sort(key=lambda a: url_order.get(a.get('url', ''), 999))
        return verified

    def _parse_article_citations(self, content: str, max_articles: int = 2) -> List[Dict]:
        """Parse article citations from Perplexity response."""
        import re

        articles = []

        def _clean_url(u: str) -> str:
            """Strip trailing punctuation and Perplexity citation markers like [3]."""
            u = u.strip()
            u = re.sub(r'\[\d+\]$', '', u)       # remove trailing [N]
            u = u.rstrip('.,;)')                   # trailing punctuation
            return u

        seen_urls: set = set()

        def _add_article(title: str, url: str):
            url = _clean_url(url)
            norm = url.lower().rstrip('/')
            if norm in seen_urls:
                return
            seen_urls.add(norm)
            source = url.split('//')[1].split('/')[0].replace('www.', '') if '//' in url else 'Unknown'
            articles.append({'title': title.strip()[:100], 'url': url, 'source': source})

        # Try TITLE: ... | URL: ... pattern first
        pattern = r'TITLE:\s*(.+?)\s*\|\s*URL:\s*(https?://[^\s\]]+(?:\[\d+\])?)'
        matches = re.findall(pattern, content, re.IGNORECASE)

        for title, url in matches:
            _add_article(title, url)

        # Fallback: extract URLs and nearby text
        if not articles:
            url_pattern = r'(https?://[^\s\[\]<>()"]+(?:\[\d+\])?)'
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                url_match = re.search(url_pattern, line)
                if url_match:
                    url = url_match.group(1)
                    title_part = line[:url_match.start()].strip()
                    title_part = re.sub(r'^[\d\.\)\-\*]+\s*', '', title_part).strip()
                    title_part = title_part.rstrip('|:- ').strip()
                    if not title_part or len(title_part) < 5:
                        title_part = "Financial Analysis Article"
                    _add_article(title_part, url)

        return articles[:max_articles]

    def _format_article_references(self, articles: List[Dict]) -> str:
        """Format article references for inclusion in rationale."""
        if not articles:
            return ""

        refs = "\n\nSources:\n"
        for i, article in enumerate(articles, 1):
            title = article.get('title', 'Article')
            url = article.get('url', '')
            source = article.get('source', 'Unknown')
            verified = article.get('verified', False)
            badge = " ✓" if verified else ""
            if url:
                refs += f"{i}. {title} | {source} | {url}{badge}\n"
            else:
                refs += f"{i}. {title} | {source}\n"
        return refs
