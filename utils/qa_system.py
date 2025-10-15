"""
Quality Assurance System for Investment Recommendations
Tracks recommendation performance and enables model self-improvement
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass, asdict
from enum import Enum

class RecommendationType(Enum):
    """Types of recommendations the system can make."""
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"

@dataclass
class StockAnalysis:
    """Data structure for a complete stock analysis with all agent rationales."""
    analysis_id: str  # Unique identifier for this analysis
    ticker: str
    timestamp: datetime
    price_at_analysis: float
    recommendation: RecommendationType
    confidence_score: float  # 0-100
    final_rationale: str  # Combined rationale
    agent_scores: Dict[str, float]
    agent_rationales: Dict[str, str]  # Full rationales from each agent
    key_factors: List[str]
    fundamentals: Dict[str, any]  # All fundamental data
    market_data: Dict[str, any]  # Market context data
    expected_target_price: Optional[float] = None
    expected_timeframe: Optional[str] = None  # e.g., "3-6 months"
    sector: Optional[str] = None
    market_cap: Optional[float] = None

@dataclass
class StockRecommendation:
    """Data structure for a stock recommendation (for QA tracking)."""
    ticker: str
    timestamp: datetime
    price_at_recommendation: float
    recommendation: RecommendationType
    confidence_score: float  # 0-100
    rationale: str
    agent_scores: Dict[str, float]
    key_factors: List[str]
    expected_target_price: Optional[float] = None
    expected_timeframe: Optional[str] = None  # e.g., "3-6 months"
    sector: Optional[str] = None
    market_cap: Optional[float] = None
    analysis_id: Optional[str] = None  # Link to full analysis

@dataclass
class PerformanceReview:
    """Data structure for performance review of a recommendation."""
    ticker: str
    original_recommendation: StockRecommendation
    review_date: datetime
    current_price: float
    price_change_pct: float
    price_change_absolute: float
    performance_vs_prediction: str  # "better", "as_expected", "worse"
    analysis_accuracy: str  # "accurate", "partially_accurate", "inaccurate"
    lessons_learned: List[str]
    unforeseen_factors: List[str]
    improvement_suggestions: List[str]
    next_review_date: datetime

class QASystem:
    """Quality Assurance System for tracking and improving recommendations."""
    
    def __init__(self, data_dir: str = "data/qa_system"):
        """Initialize the QA system with data storage directory."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.recommendations_file = self.data_dir / "recommendations.json"
        self.reviews_file = self.data_dir / "performance_reviews.json"
        self.insights_file = self.data_dir / "learning_insights.json"
        self.analyses_file = self.data_dir / "all_analyses.json"  # New: Complete analysis archive
        
        # Load existing data
        self.recommendations = self._load_recommendations()
        self.reviews = self._load_reviews()
        self.insights = self._load_insights()
        self.all_analyses = self._load_all_analyses()  # New: Load complete analysis archive
    
    def _load_recommendations(self) -> Dict[str, StockRecommendation]:
        """Load recommendations from storage."""
        if not self.recommendations_file.exists():
            return {}
        
        try:
            with open(self.recommendations_file, 'r') as f:
                data = json.load(f)
            
            recommendations = {}
            for ticker, rec_data in data.items():
                rec_data['timestamp'] = datetime.fromisoformat(rec_data['timestamp'])
                rec_data['recommendation'] = RecommendationType(rec_data['recommendation'])
                recommendations[ticker] = StockRecommendation(**rec_data)
            
            return recommendations
        except Exception as e:
            print(f"Error loading recommendations: {e}")
            return {}
    
    def _load_reviews(self) -> Dict[str, List[PerformanceReview]]:
        """Load performance reviews from storage."""
        if not self.reviews_file.exists():
            return {}
        
        try:
            with open(self.reviews_file, 'r') as f:
                data = json.load(f)
            
            reviews = {}
            for ticker, review_list in data.items():
                reviews[ticker] = []
                for review_data in review_list:
                    # Reconstruct original recommendation
                    orig_rec_data = review_data['original_recommendation']
                    orig_rec_data['timestamp'] = datetime.fromisoformat(orig_rec_data['timestamp'])
                    orig_rec_data['recommendation'] = RecommendationType(orig_rec_data['recommendation'])
                    original_rec = StockRecommendation(**orig_rec_data)
                    
                    # Reconstruct review
                    review_data['original_recommendation'] = original_rec
                    review_data['review_date'] = datetime.fromisoformat(review_data['review_date'])
                    review_data['next_review_date'] = datetime.fromisoformat(review_data['next_review_date'])
                    
                    reviews[ticker].append(PerformanceReview(**review_data))
            
            return reviews
        except Exception as e:
            print(f"Error loading reviews: {e}")
            return {}
    
    def _load_insights(self) -> Dict[str, any]:
        """Load learning insights from storage."""
        if not self.insights_file.exists():
            return {
                'common_mistakes': [],
                'improvement_patterns': [],
                'successful_strategies': [],
                'market_lessons': [],
                'model_adjustments': []
            }
        
        try:
            with open(self.insights_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading insights: {e}")
            return {
                'common_mistakes': [],
                'improvement_patterns': [],
                'successful_strategies': [],
                'market_lessons': [],
                'model_adjustments': []
            }
    
    def _load_all_analyses(self) -> Dict[str, StockAnalysis]:
        """Load all analyses from storage."""
        if not self.analyses_file.exists():
            return {}
        
        try:
            with open(self.analyses_file, 'r') as f:
                data = json.load(f)
            
            analyses = {}
            for analysis_id, analysis_data in data.items():
                analysis_data['timestamp'] = datetime.fromisoformat(analysis_data['timestamp'])
                analysis_data['recommendation'] = RecommendationType(analysis_data['recommendation'])
                analyses[analysis_id] = StockAnalysis(**analysis_data)
            
            return analyses
        except Exception as e:
            print(f"Error loading analyses: {e}")
            return {}
    
    def _save_recommendations(self):
        """Save recommendations to storage."""
        try:
            data = {}
            for ticker, rec in self.recommendations.items():
                rec_dict = asdict(rec)
                rec_dict['timestamp'] = rec.timestamp.isoformat()
                rec_dict['recommendation'] = rec.recommendation.value
                data[ticker] = rec_dict
            
            print(f"Saving {len(data)} recommendations to {self.recommendations_file}")
            with open(self.recommendations_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            print("Recommendations saved successfully")
            return True
        except Exception as e:
            print(f"Error saving recommendations: {e}")
            return False
    
    def _save_reviews(self):
        """Save performance reviews to storage."""
        try:
            data = {}
            for ticker, review_list in self.reviews.items():
                data[ticker] = []
                for review in review_list:
                    review_dict = asdict(review)
                    # Handle original recommendation
                    orig_rec_dict = asdict(review.original_recommendation)
                    orig_rec_dict['timestamp'] = review.original_recommendation.timestamp.isoformat()
                    orig_rec_dict['recommendation'] = review.original_recommendation.recommendation.value
                    review_dict['original_recommendation'] = orig_rec_dict
                    
                    review_dict['review_date'] = review.review_date.isoformat()
                    review_dict['next_review_date'] = review.next_review_date.isoformat()
                    data[ticker].append(review_dict)
            
            with open(self.reviews_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving reviews: {e}")
    
    def _save_insights(self):
        """Save learning insights to storage."""
        try:
            with open(self.insights_file, 'w') as f:
                json.dump(self.insights, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving insights: {e}")
    
    def _save_all_analyses(self):
        """Save all analyses to storage."""
        try:
            data = {}
            for analysis_id, analysis in self.all_analyses.items():
                analysis_dict = asdict(analysis)
                analysis_dict['timestamp'] = analysis.timestamp.isoformat()
                analysis_dict['recommendation'] = analysis.recommendation.value
                data[analysis_id] = analysis_dict
            
            with open(self.analyses_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving analyses: {e}")
    
    def log_recommendation(self, 
                          ticker: str,
                          price: float,
                          recommendation: RecommendationType,
                          confidence_score: float,
                          rationale: str,
                          agent_scores: Dict[str, float],
                          key_factors: List[str],
                          expected_target_price: Optional[float] = None,
                          expected_timeframe: Optional[str] = None,
                          sector: Optional[str] = None,
                          market_cap: Optional[float] = None) -> bool:
        """Log a new recommendation (replaces existing if same ticker)."""
        
        try:
            recommendation_obj = StockRecommendation(
                ticker=ticker,
                timestamp=datetime.now(),
                price_at_recommendation=price,
                recommendation=recommendation,
                confidence_score=confidence_score,
                rationale=rationale,
                agent_scores=agent_scores,
                key_factors=key_factors,
                expected_target_price=expected_target_price,
                expected_timeframe=expected_timeframe,
                sector=sector,
                market_cap=market_cap
            )
            
            # Replace existing recommendation if present
            if ticker in self.recommendations:
                print(f"Replacing existing recommendation for {ticker}")
            else:
                print(f"Adding new recommendation for {ticker}")
            
            self.recommendations[ticker] = recommendation_obj
            saved_success = self._save_recommendations()
            
            print(f"Successfully logged recommendation for {ticker}")
            print(f"Total recommendations now: {len(self.recommendations)}")
            print(f"Recommendations file: {self.recommendations_file}")
            print(f"File save result: {saved_success}")
            
            # Verify the data was actually saved
            reloaded_recs = self._load_recommendations()
            print(f"Verification - reloaded recommendations count: {len(reloaded_recs)}")
            if ticker in reloaded_recs:
                print(f"✅ Verified: {ticker} found in reloaded data")
            else:
                print(f"❌ Error: {ticker} NOT found in reloaded data")
            
            return True
            
        except Exception as e:
            print(f"Error logging recommendation for {ticker}: {e}")
            return False
    
    def log_complete_analysis(self,
                            ticker: str,
                            price: float,
                            recommendation: RecommendationType,
                            confidence_score: float,
                            final_rationale: str,
                            agent_scores: Dict[str, float],
                            agent_rationales: Dict[str, str],
                            key_factors: List[str],
                            fundamentals: Dict[str, any],
                            market_data: Dict[str, any] = None,
                            expected_target_price: Optional[float] = None,
                            expected_timeframe: Optional[str] = None,
                            sector: Optional[str] = None,
                            market_cap: Optional[float] = None) -> str:
        """Log a complete analysis with all details. Returns analysis ID."""
        
        try:
            # Generate unique analysis ID
            timestamp = datetime.now()
            analysis_id = f"{ticker}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            analysis_obj = StockAnalysis(
                analysis_id=analysis_id,
                ticker=ticker,
                timestamp=timestamp,
                price_at_analysis=price,
                recommendation=recommendation,
                confidence_score=confidence_score,
                final_rationale=final_rationale,
                agent_scores=agent_scores,
                agent_rationales=agent_rationales,
                key_factors=key_factors,
                fundamentals=fundamentals or {},
                market_data=market_data or {},
                expected_target_price=expected_target_price,
                expected_timeframe=expected_timeframe,
                sector=sector,
                market_cap=market_cap
            )
            
            self.all_analyses[analysis_id] = analysis_obj
            self._save_all_analyses()
            
            print(f"Successfully logged complete analysis for {ticker} with ID: {analysis_id}")
            return analysis_id
            
        except Exception as e:
            print(f"Error logging complete analysis for {ticker}: {e}")
            return ""
    
    def get_stocks_due_for_review(self) -> List[str]:
        """Get list of stocks that are due for weekly review."""
        due_for_review = []
        current_time = datetime.now()
        
        for ticker, recommendation in self.recommendations.items():
            # Check if it's been at least a week since last review
            last_review_date = recommendation.timestamp
            
            # If there are existing reviews, use the most recent review date
            if ticker in self.reviews and self.reviews[ticker]:
                last_review_date = max(review.review_date for review in self.reviews[ticker])
            
            days_since_review = (current_time - last_review_date).days
            
            if days_since_review >= 7:
                due_for_review.append(ticker)
        
        return due_for_review
    
    def conduct_performance_review(self, 
                                 ticker: str, 
                                 current_price: float,
                                 openai_client=None) -> Optional[PerformanceReview]:
        """Conduct a performance review for a stock."""
        
        if ticker not in self.recommendations:
            print(f"No recommendation found for {ticker}")
            return None
        
        original_rec = self.recommendations[ticker]
        
        # Calculate performance metrics
        price_change_absolute = current_price - original_rec.price_at_recommendation
        price_change_pct = (price_change_absolute / original_rec.price_at_recommendation) * 100
        
        # Determine performance vs prediction
        performance_vs_prediction = self._assess_performance_vs_prediction(
            original_rec, price_change_pct
        )
        
        # Generate analysis using LLM if available
        analysis_results = self._generate_performance_analysis(
            original_rec, current_price, price_change_pct, openai_client
        )
        
        # Create performance review
        review = PerformanceReview(
            ticker=ticker,
            original_recommendation=original_rec,
            review_date=datetime.now(),
            current_price=current_price,
            price_change_pct=price_change_pct,
            price_change_absolute=price_change_absolute,
            performance_vs_prediction=performance_vs_prediction,
            analysis_accuracy=analysis_results['accuracy'],
            lessons_learned=analysis_results['lessons_learned'],
            unforeseen_factors=analysis_results['unforeseen_factors'],
            improvement_suggestions=analysis_results['improvement_suggestions'],
            next_review_date=datetime.now() + timedelta(days=7)
        )
        
        # Store the review
        if ticker not in self.reviews:
            self.reviews[ticker] = []
        self.reviews[ticker].append(review)
        
        # Update insights
        self._update_insights(review)
        
        # Save data
        self._save_reviews()
        self._save_insights()
        
        return review
    
    def _assess_performance_vs_prediction(self, 
                                        original_rec: StockRecommendation, 
                                        price_change_pct: float) -> str:
        """Assess how the stock performed vs the original prediction."""
        
        if original_rec.recommendation in [RecommendationType.BUY, RecommendationType.STRONG_BUY]:
            if price_change_pct > 5:
                return "better"
            elif price_change_pct > -2:
                return "as_expected"
            else:
                return "worse"
        elif original_rec.recommendation in [RecommendationType.SELL, RecommendationType.STRONG_SELL]:
            if price_change_pct < -5:
                return "better"
            elif price_change_pct < 2:
                return "as_expected"
            else:
                return "worse"
        else:  # HOLD
            if abs(price_change_pct) < 3:
                return "as_expected"
            elif price_change_pct > 0:
                return "better"
            else:
                return "worse"
    
    def _generate_performance_analysis(self, 
                                     original_rec: StockRecommendation,
                                     current_price: float,
                                     price_change_pct: float,
                                     openai_client=None) -> Dict[str, any]:
        """Generate comprehensive performance analysis using LLM."""
        
        # Default analysis if no LLM available
        default_analysis = {
            'accuracy': 'partially_accurate',
            'lessons_learned': [f'Stock moved {price_change_pct:.1f}% from original prediction'],
            'unforeseen_factors': ['Market volatility'],
            'improvement_suggestions': ['Monitor market conditions more closely']
        }
        
        if not openai_client:
            return default_analysis
        
        try:
            # Prepare comprehensive analysis prompt
            prompt = f"""You are an expert investment analyst conducting a performance review of a stock recommendation.

ORIGINAL RECOMMENDATION DETAILS:
- Ticker: {original_rec.ticker}
- Date: {original_rec.timestamp.strftime('%Y-%m-%d')}
- Recommendation: {original_rec.recommendation.value.upper()}
- Price at Recommendation: ${original_rec.price_at_recommendation:.2f}
- Confidence Score: {original_rec.confidence_score}/100
- Rationale: {original_rec.rationale}
- Key Factors: {', '.join(original_rec.key_factors)}
- Agent Scores: {original_rec.agent_scores}

PERFORMANCE RESULTS:
- Current Price: ${current_price:.2f}
- Price Change: {price_change_pct:.1f}%
- Days Since Recommendation: {(datetime.now() - original_rec.timestamp).days}

ANALYSIS REQUIREMENTS:
1. Assess the accuracy of the original analysis (accurate/partially_accurate/inaccurate)
2. Identify key lessons learned from this performance
3. Identify factors that were unforeseeable at the time of recommendation
4. Provide specific suggestions for improving future analysis

Consider factors like:
- Market conditions during the period
- Company-specific developments
- Sector performance
- Economic events
- Whether the original rationale still holds

OUTPUT FORMAT (JSON):
{{
    "accuracy": "<accurate|partially_accurate|inaccurate>",
    "lessons_learned": [
        "Specific lesson 1",
        "Specific lesson 2"
    ],
    "unforeseen_factors": [
        "Factor that couldn't have been predicted 1",
        "Factor that couldn't have been predicted 2"
    ],
    "improvement_suggestions": [
        "Specific improvement suggestion 1",
        "Specific improvement suggestion 2"
    ]
}}

Provide detailed, actionable insights that will help improve future analysis."""

            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500
            )
            
            import json as json_lib
            analysis_text = response.choices[0].message.content
            
            # Extract JSON from response
            if '```json' in analysis_text:
                json_start = analysis_text.find('```json') + 7
                json_end = analysis_text.find('```', json_start)
                analysis_text = analysis_text[json_start:json_end]
            elif '{' in analysis_text:
                json_start = analysis_text.find('{')
                json_end = analysis_text.rfind('}') + 1
                analysis_text = analysis_text[json_start:json_end]
            
            analysis = json_lib.loads(analysis_text)
            
            # Validate and return
            return {
                'accuracy': analysis.get('accuracy', 'partially_accurate'),
                'lessons_learned': analysis.get('lessons_learned', []),
                'unforeseen_factors': analysis.get('unforeseen_factors', []),
                'improvement_suggestions': analysis.get('improvement_suggestions', [])
            }
            
        except Exception as e:
            print(f"Error generating LLM analysis: {e}")
            return default_analysis
    
    def _update_insights(self, review: PerformanceReview):
        """Update learning insights based on performance review."""
        
        # Add lessons learned
        for lesson in review.lessons_learned:
            if lesson not in self.insights['market_lessons']:
                self.insights['market_lessons'].append(lesson)
        
        # Add improvement suggestions
        for suggestion in review.improvement_suggestions:
            if suggestion not in self.insights['improvement_patterns']:
                self.insights['improvement_patterns'].append(suggestion)
        
        # Track successful strategies
        if review.performance_vs_prediction == "better":
            strategy = f"Successful {review.original_recommendation.recommendation.value} recommendation for {review.original_recommendation.sector or 'unknown'} sector"
            if strategy not in self.insights['successful_strategies']:
                self.insights['successful_strategies'].append(strategy)
        
        # Track common mistakes
        if review.performance_vs_prediction == "worse":
            mistake = f"Inaccurate {review.original_recommendation.recommendation.value} recommendation - {review.analysis_accuracy}"
            if mistake not in self.insights['common_mistakes']:
                self.insights['common_mistakes'].append(mistake)
    
    def get_qa_summary(self) -> Dict[str, any]:
        """Get comprehensive QA system summary for display."""
        
        total_recommendations = len(self.recommendations)
        total_reviews = sum(len(reviews) for reviews in self.reviews.values())
        
        # Calculate performance metrics
        performance_stats = {
            'better': 0,
            'as_expected': 0,
            'worse': 0
        }
        
        accuracy_stats = {
            'accurate': 0,
            'partially_accurate': 0,
            'inaccurate': 0
        }
        
        for ticker_reviews in self.reviews.values():
            for review in ticker_reviews:
                performance_stats[review.performance_vs_prediction] += 1
                accuracy_stats[review.analysis_accuracy] += 1
        
        return {
            'total_recommendations': total_recommendations,
            'total_reviews': total_reviews,
            'performance_stats': performance_stats,
            'accuracy_stats': accuracy_stats,
            'insights': self.insights,
            'stocks_due_for_review': self.get_stocks_due_for_review(),
            'latest_reviews': self._get_latest_reviews(5)
        }
    
    def _get_latest_reviews(self, limit: int = 5) -> List[Dict[str, any]]:
        """Get the latest performance reviews for display."""
        
        all_reviews = []
        for ticker, reviews in self.reviews.items():
            for review in reviews:
                all_reviews.append({
                    'ticker': ticker,
                    'review_date': review.review_date,
                    'price_change_pct': review.price_change_pct,
                    'performance_vs_prediction': review.performance_vs_prediction,
                    'analysis_accuracy': review.analysis_accuracy
                })
        
        # Sort by review date (most recent first)
        all_reviews.sort(key=lambda x: x['review_date'], reverse=True)
        
        return all_reviews[:limit]
    
    def get_learning_insights_for_analysis(self) -> str:
        """Get formatted learning insights to include in future analyses."""
        
        insights_text = []
        
        if self.insights['common_mistakes']:
            insights_text.append("COMMON MISTAKES TO AVOID:")
            for mistake in self.insights['common_mistakes'][-5:]:  # Last 5 mistakes
                insights_text.append(f"- {mistake}")
        
        if self.insights['successful_strategies']:
            insights_text.append("\nSUCCESSFUL STRATEGIES:")
            for strategy in self.insights['successful_strategies'][-5:]:  # Last 5 successes
                insights_text.append(f"- {strategy}")
        
        if self.insights['improvement_patterns']:
            insights_text.append("\nKEY IMPROVEMENTS TO IMPLEMENT:")
            for improvement in self.insights['improvement_patterns'][-3:]:  # Last 3 improvements
                insights_text.append(f"- {improvement}")
        
        return '\n'.join(insights_text) if insights_text else "No historical insights available yet."
    
    def get_tracked_tickers(self) -> List[str]:
        """Get list of all tickers currently being tracked in QA system."""
        return list(self.recommendations.keys())
    
    def get_analysis_archive(self) -> Dict[str, List[StockAnalysis]]:
        """Get complete analysis archive organized by ticker."""
        archive = {}
        for analysis in self.all_analyses.values():
            ticker = analysis.ticker
            if ticker not in archive:
                archive[ticker] = []
            archive[ticker].append(analysis)
        
        # Sort each ticker's analyses by timestamp (most recent first)
        for ticker in archive:
            archive[ticker].sort(key=lambda x: x.timestamp, reverse=True)
        
        return archive
    
    def get_analysis_stats(self) -> Dict[str, any]:
        """Get comprehensive statistics about all analyses."""
        total_analyses = len(self.all_analyses)
        unique_tickers = len(set(analysis.ticker for analysis in self.all_analyses.values()))
        
        # Recommendation breakdown
        recommendation_counts = {}
        for analysis in self.all_analyses.values():
            rec_type = analysis.recommendation.value
            recommendation_counts[rec_type] = recommendation_counts.get(rec_type, 0) + 1
        
        # Sector breakdown
        sector_counts = {}
        for analysis in self.all_analyses.values():
            sector = analysis.sector or 'Unknown'
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        # Get recent activity (last 30 days)
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_analyses = [
            analysis for analysis in self.all_analyses.values()
            if analysis.timestamp >= thirty_days_ago
        ]
        
        return {
            'total_analyses': total_analyses,
            'unique_tickers': unique_tickers,
            'recommendation_breakdown': recommendation_counts,
            'sector_breakdown': sector_counts,
            'recent_activity_count': len(recent_analyses),
            'avg_confidence_score': sum(a.confidence_score for a in self.all_analyses.values()) / total_analyses if total_analyses > 0 else 0
        }
    
    def delete_analysis(self, ticker: str, timestamp: datetime) -> bool:
        """Delete a specific analysis by ticker and timestamp.
        
        Args:
            ticker: The stock ticker
            timestamp: The timestamp of the analysis to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            # Find and remove the analysis with matching ticker and timestamp
            analysis_ids_to_delete = [
                analysis_id for analysis_id, analysis in self.all_analyses.items()
                if analysis.ticker == ticker and analysis.timestamp == timestamp
            ]
            
            if not analysis_ids_to_delete:
                print(f"No analysis found for {ticker} at {timestamp}")
                return False
            
            for analysis_id in analysis_ids_to_delete:
                del self.all_analyses[analysis_id]
            
            # Save updated data
            self._save_all_analyses()
            print(f"Deleted {len(analysis_ids_to_delete)} analysis(es) for {ticker}")
            return True
            
        except Exception as e:
            print(f"Error deleting analysis: {e}")
            return False
    
    def delete_all_analyses_for_ticker(self, ticker: str) -> bool:
        """Delete all analyses for a specific ticker.
        
        Args:
            ticker: The stock ticker
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            # Find all analysis IDs for this ticker
            analysis_ids_to_delete = [
                analysis_id for analysis_id, analysis in self.all_analyses.items()
                if analysis.ticker == ticker
            ]
            
            if not analysis_ids_to_delete:
                print(f"No analyses found for {ticker}")
                return False
            
            # Delete all matching analyses
            for analysis_id in analysis_ids_to_delete:
                del self.all_analyses[analysis_id]
            
            # Save updated data
            self._save_all_analyses()
            print(f"Deleted {len(analysis_ids_to_delete)} analysis(es) for {ticker}")
            return True
            
        except Exception as e:
            print(f"Error deleting analyses for {ticker}: {e}")
            return False