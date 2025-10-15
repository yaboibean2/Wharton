"""
Performance Analysis Engine - Advanced Learning System
Analyzes stocks that moved significantly (up/down), determines root causes,
and provides actionable model improvement recommendations.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class StockMovement:
    """Data structure for a significant stock price movement."""
    ticker: str
    start_date: str
    end_date: str
    start_price: float
    end_price: float
    price_change_pct: float
    price_change_abs: float
    direction: str  # "up" or "down"
    magnitude: str  # "significant", "major", "extreme"
    sector: Optional[str] = None
    market_cap: Optional[float] = None
    volume_change_pct: Optional[float] = None


@dataclass
class NewsArticle:
    """Data structure for a news article."""
    title: str
    description: str
    url: str
    published_date: str
    source: str
    sentiment: Optional[str] = None
    keywords: Optional[List[str]] = None


@dataclass
class MovementAnalysis:
    """Comprehensive analysis of why a stock moved."""
    ticker: str
    movement: StockMovement
    news_articles: List[NewsArticle]
    root_causes: List[str]  # Primary reasons for the movement
    confidence: float  # 0-100 confidence in the analysis
    earnings_related: bool
    news_driven: bool
    market_driven: bool
    sector_driven: bool
    fundamental_change: bool
    technical_breakout: bool
    catalyst_summary: str  # AI-generated summary
    agent_relevance: Dict[str, str]  # Which agents should have caught this
    model_gaps: List[str]  # What the model missed
    

@dataclass
class ModelAdjustmentRecommendation:
    """Specific recommendation for adjusting the model."""
    recommendation_id: str
    priority: str  # "critical", "high", "medium", "low"
    category: str  # "agent_weight", "feature_focus", "data_source", "threshold"
    specific_change: str  # Exact change to make
    rationale: str  # Why this change is recommended
    expected_impact: str  # What improvement to expect
    supporting_evidence: List[str]  # Examples that support this recommendation
    affected_agents: List[str]
    implementation_steps: List[str]
    confidence: float  # 0-100


class PerformanceAnalysisEngine:
    """
    Advanced engine for analyzing stock performance and generating model improvements.
    
    This engine:
    1. Identifies stocks that moved significantly (up or down)
    2. Fetches relevant news and events
    3. Uses AI to determine root causes
    4. Generates specific, actionable model adjustment recommendations
    5. Tracks improvements over time
    """
    
    def __init__(self, data_provider, openai_client=None, perplexity_client=None):
        """Initialize the performance analysis engine."""
        self.data_provider = data_provider
        self.openai_client = openai_client
        self.perplexity_client = perplexity_client
        
        # Storage
        self.storage_dir = Path("data/performance_analysis")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.movements_file = self.storage_dir / "significant_movements.json"
        self.analyses_file = self.storage_dir / "movement_analyses.json"
        self.recommendations_file = self.storage_dir / "model_recommendations.json"
        self.tracking_file = self.storage_dir / "improvement_tracking.json"
        
        # Configuration
        self.movement_thresholds = {
            'significant': 5.0,  # 5% move
            'major': 10.0,       # 10% move
            'extreme': 20.0      # 20% move
        }
        
        # Load existing data
        self.movement_history = self._load_json(self.movements_file, [])
        self.analysis_history = self._load_json(self.analyses_file, [])
        self.recommendations_history = self._load_json(self.recommendations_file, [])
        self.improvement_tracking = self._load_json(self.tracking_file, {})
    
    def _load_json(self, file_path: Path, default):
        """Safely load JSON file."""
        if not file_path.exists():
            return default
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return default
    
    def _save_json(self, file_path: Path, data):
        """Safely save JSON file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Saved data to {file_path}")
        except Exception as e:
            logger.error(f"Error saving {file_path}: {e}")
    
    def analyze_performance_period(
        self,
        start_date: str,
        end_date: str,
        tickers: Optional[List[str]] = None,
        qa_system=None,
        sheets_integration=None,
        min_threshold: float = 15.0
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of stock performance over a period.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            tickers: Optional list of tickers to analyze (otherwise analyzes all tracked stocks)
            qa_system: Optional QA system instance to get recommendations
            sheets_integration: Optional Google Sheets integration to get percent changes
            min_threshold: Minimum percent change threshold (default: 15.0%)
        
        Returns:
            Complete performance analysis with model recommendations
        """
        logger.info(f"Starting performance analysis from {start_date} to {end_date}")
        logger.info(f"Analysis mode: {'ALL stocks' if tickers is None else f'{len(tickers)} specific tickers'} with ≥{min_threshold}% movement")
        
        # Step 1: Identify significant movements
        movements = self._identify_significant_movements(
            start_date, 
            end_date, 
            tickers, 
            qa_system, 
            sheets_integration,
            min_threshold=min_threshold
        )
        
        if not movements:
            logger.info("No significant movements found in this period")
            return {
                'status': 'no_movements',
                'period': {'start': start_date, 'end': end_date},
                'message': 'No significant stock movements detected in this period. Try expanding the date range or analyzing more stocks.',
                'executive_summary': 'No significant movements detected',
                'summary': {
                    'total_movements': 0,
                    'analyses_completed': 0,
                    'recommendations_generated': 0,
                    'top_gainers_count': 0,
                    'top_losers_count': 0,
                    'critical_recommendations': 0,
                    'high_priority_recommendations': 0
                },
                'top_gainers': [],
                'top_losers': [],
                'patterns': {},
                'analyses': [],
                'recommendations': []
            }
        
        logger.info(f"Found {len(movements)} significant movements (before deduplication)")
        
        # OPTIMIZATION: Deduplicate by ticker (keep the largest absolute movement)
        unique_movements = {}
        for movement in movements:
            if movement.ticker not in unique_movements:
                unique_movements[movement.ticker] = movement
            else:
                # Keep the movement with larger absolute change
                existing = unique_movements[movement.ticker]
                if abs(movement.price_change_pct) > abs(existing.price_change_pct):
                    unique_movements[movement.ticker] = movement
        
        movements = list(unique_movements.values())
        logger.info(f"🎯 After deduplication: {len(movements)} unique tickers to analyze")
        
        # Step 2: Analyze each movement to determine root causes
        analyses = []
        for movement in movements:
            try:
                analysis = self._analyze_movement(movement)
                if analysis:
                    analyses.append(analysis)
            except Exception as e:
                logger.error(f"Error analyzing {movement.ticker}: {e}")
        
        logger.info(f"Completed {len(analyses)} movement analyses")
        
        # Step 3: Generate model adjustment recommendations
        recommendations = self._generate_model_recommendations(analyses, start_date, end_date)
        
        logger.info(f"Generated {len(recommendations)} model recommendations")
        
        # Step 4: AUTONOMOUS ADJUSTMENT - Actually apply the recommendations
        adjustment_result = self.apply_autonomous_adjustments(
            analyses, recommendations, auto_apply=True
        )
        
        # Step 5: Create comprehensive report (include adjustment results)
        report = self._create_performance_report(
            movements, analyses, recommendations, start_date, end_date
        )
        report['autonomous_adjustments'] = adjustment_result
        
        # Step 6: Save results
        self._save_analysis_results(movements, analyses, recommendations)
        
        return report
    
    def _get_google_sheets_data(self, sheets_integration) -> Optional[pd.DataFrame]:
        """
        Fetch data from Google Sheets if available.
        Looks for 'Historical Price Analysis' worksheet.
        
        Returns:
            DataFrame with columns including 'Ticker', 'Percent Change', 'Price at Analysis', 'Price', etc.
        """
        try:
            if not sheets_integration or not hasattr(sheets_integration, 'sheet'):
                logger.warning("No sheets_integration or sheet attribute")
                return None
            
            sheet = sheets_integration.sheet
            if not sheet:
                logger.warning("sheets_integration.sheet is None")
                return None
            
            # Try to get the 'Historical Price Analysis' worksheet
            worksheet = None
            worksheet_names = ['Historical Price Analysis', 'Portfolio Analysis', 'Price Analysis']
            
            for ws_name in worksheet_names:
                try:
                    worksheet = sheet.worksheet(ws_name)
                    logger.info(f"✅ Found worksheet: '{ws_name}'")
                    break
                except Exception as e:
                    logger.debug(f"Worksheet '{ws_name}' not found: {e}")
                    continue
            
            if worksheet is None:
                logger.error(f"❌ Could not find any of the expected worksheets: {worksheet_names}")
                logger.error(f"Available worksheets: {[ws.title for ws in sheet.worksheets()]}")
                return None
            
            # Get all data from worksheet
            try:
                data = worksheet.get_all_records()
            except Exception as e:
                if "duplicates" in str(e):
                    logger.error(f"❌ Worksheet '{worksheet.title}' has duplicate/empty column headers")
                    logger.error(f"Please ensure your worksheet has unique, non-empty column names in the first row")
                    logger.error(f"Available worksheets: {[ws.title for ws in sheet.worksheets()]}")
                else:
                    logger.error(f"Error reading worksheet data: {e}")
                return None
            
            if not data:
                logger.warning(f"No data returned from worksheet '{worksheet.title}'")
                return None
            
            df = pd.DataFrame(data)
            
            # Clean column names (strip whitespace, normalize)
            df.columns = df.columns.str.strip()
            
            logger.info(f"✅ Fetched {len(df)} rows from Google Sheets (worksheet: '{worksheet.title}')")
            logger.info(f"📋 Columns available (cleaned): {list(df.columns)}")
            
            # Verify we have the required columns
            required_cols = ['Ticker']
            has_pct_change = 'Percent Change' in df.columns
            has_prices = 'Price at Analysis' in df.columns and 'Price' in df.columns
            
            if not any(col in df.columns for col in required_cols):
                logger.error(f"❌ Missing required 'Ticker' column!")
                logger.error(f"Available columns: {list(df.columns)}")
                return None
            
            if not has_pct_change and not has_prices:
                logger.error(f"❌ Missing required data columns!")
                logger.error(f"Need either: 'Percent Change' OR ('Price at Analysis' + 'Price')")
                logger.error(f"Available columns: {list(df.columns)}")
                return None
            
            # Log sample data for debugging
            if len(df) > 0:
                # Check for percent change column
                has_pct_change = any(col in df.columns for col in ['Percent Change', 'Price Change %', '% Change', 'Percent_Change'])
                has_prices = any(col in df.columns for col in ['Price at Analysis', 'Price_at_Analysis']) and \
                            any(col in df.columns for col in ['Price', 'Current Price'])
                
                logger.info(f"📊 Data source: {'Percent Change column' if has_pct_change else 'Will calculate from prices' if has_prices else 'MISSING DATA'}")
                
                if has_pct_change and 'Percent Change' in df.columns:
                    sample_values = df['Percent Change'].head(10).tolist()
                    logger.info(f"   Sample Percent Change values: {sample_values}")
                
                if has_prices:
                    price_col = next((c for c in ['Price at Analysis', 'Price_at_Analysis'] if c in df.columns), None)
                    current_col = next((c for c in ['Price', 'Current Price'] if c in df.columns), None)
                    if price_col and current_col:
                        logger.info(f"   Price columns: '{price_col}' and '{current_col}'")
                        logger.info(f"   Sample {price_col}: {df[price_col].head(5).tolist()}")
                        logger.info(f"   Sample {current_col}: {df[current_col].head(5).tolist()}")
                
                logger.info(f"🔍 First row example: Ticker={df.iloc[0].get('Ticker', 'N/A')}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Google Sheets data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _identify_movements_from_sheets(
        self,
        sheets_df: pd.DataFrame,
        tickers: Optional[List[str]],
        start_date: str,
        end_date: str,
        min_threshold: float = 15.0
    ) -> List[StockMovement]:
        """
        Identify significant movements from Google Sheets data.
        Uses 'Percent Change' column which shows change since analysis.
        
        Args:
            sheets_df: DataFrame with Google Sheets data
            tickers: Optional list to filter by specific tickers (None = analyze all)
            start_date: Start date for analysis
            end_date: End date for analysis
            min_threshold: Minimum percent change threshold (default: 15.0%)
        """
        movements = []
        
        # Filter for requested tickers if provided, otherwise analyze ALL
        if tickers is not None and len(tickers) > 0:
            sheets_df = sheets_df[sheets_df['Ticker'].isin(tickers)]
            logger.info(f"Filtering for {len(tickers)} specific tickers")
        else:
            logger.info(f"Analyzing ALL {len(sheets_df)} stocks from Google Sheets")
        
        logger.info(f"Processing {len(sheets_df)} rows with min_threshold={min_threshold}%")
        
        # DEBUG: Show all stocks and their percent changes with detailed type info
        debug_data = []
        for idx, row in sheets_df.iterrows():
            ticker = row.get('Ticker', '')
            pct_change_raw = row.get('Percent Change', 'N/A')
            debug_data.append(f"{ticker}: {pct_change_raw} (type: {type(pct_change_raw).__name__})")
        logger.info(f"📊 RAW DATA FROM SHEETS (first 20): {', '.join(debug_data[:20])}")
        
        # Show column names exactly as they appear
        logger.info(f"🔍 EXACT COLUMN NAMES: {list(sheets_df.columns)}")
        logger.info(f"🔍 'Percent Change' in columns: {'Percent Change' in sheets_df.columns}")
        
        for idx, row in sheets_df.iterrows():
            try:
                ticker = row.get('Ticker', '').strip()
                if not ticker:
                    continue
                
                # Get percent change - try different column name variations
                price_change_pct = None
                raw_value = None
                for col_name in ['Percent Change', 'Price Change %', '% Change', 'Percent_Change']:
                    if col_name in row:
                        raw_value = row[col_name]
                        logger.debug(f"{ticker}: Checking {col_name}='{raw_value}' (type: {type(raw_value).__name__})")
                        
                        # Skip if None or truly empty
                        if raw_value is None or (isinstance(raw_value, str) and raw_value.strip() == ''):
                            continue
                        
                        try:
                            # If it's already a number, use it directly
                            if isinstance(raw_value, (int, float)):
                                price_change_pct = float(raw_value)
                                logger.info(f"{ticker}: Direct numeric value from {col_name}: {price_change_pct}%")
                                break
                            
                            # Handle string percentage (e.g., "5.25%", "-18.5%", "58.97")
                            val = str(raw_value).strip().replace('%', '').replace(',', '').replace(' ', '')
                            
                            # Skip empty, N/A, or placeholder values
                            if val and val != 'N/A' and val != '-' and val != '' and val.lower() != 'nan':
                                price_change_pct = float(val)
                                logger.info(f"{ticker}: Parsed string '{raw_value}' from {col_name} -> {price_change_pct}%")
                                break
                            else:
                                logger.debug(f"{ticker}: Skipping empty/invalid value in {col_name}: '{raw_value}'")
                        except (ValueError, AttributeError, TypeError) as e:
                            logger.warning(f"{ticker}: Failed to parse {col_name}='{raw_value}' (type: {type(raw_value).__name__}): {e}")
                            continue
                
                # If no percent change column found, calculate from prices
                # This is the "percent change since analysis" calculation
                if price_change_pct is None:
                    # Try multiple column name variations for both prices
                    price_at_analysis_raw = None
                    current_price_raw = None
                    
                    # Try to find "Price at Analysis" (baseline price)
                    for col in ['Price at Analysis', 'Price_at_Analysis', 'Initial Price', 'Starting Price', 'Analysis Price']:
                        if col in row and row[col] is not None:
                            price_at_analysis_raw = row[col]
                            break
                    
                    # Try to find current/latest price
                    for col in ['Price', 'Current Price', 'Latest Price', 'Current_Price']:
                        if col in row and row[col] is not None:
                            current_price_raw = row[col]
                            break
                    
                    # Only proceed if we have both prices
                    if price_at_analysis_raw is not None and current_price_raw is not None:
                        try:
                            # Clean and parse price at analysis
                            price_at_analysis_str = str(price_at_analysis_raw).strip().replace(',', '').replace('$', '').replace(' ', '')
                            
                            # Clean and parse current price
                            current_price_str = str(current_price_raw).strip().replace(',', '').replace('$', '').replace(' ', '')
                            
                            # Validate not empty or invalid
                            if not price_at_analysis_str or not current_price_str:
                                logger.debug(f"{ticker}: Empty price strings")
                                continue
                            
                            if price_at_analysis_str.lower() in ['nan', 'n/a', '-', 'none', ''] or \
                               current_price_str.lower() in ['nan', 'n/a', '-', 'none', '']:
                                logger.debug(f"{ticker}: Invalid price values: '{price_at_analysis_str}' / '{current_price_str}'")
                                continue
                            
                            # Convert to float
                            price_at_analysis = float(price_at_analysis_str)
                            current_price = float(current_price_str)
                            
                            # Validate prices are positive and reasonable
                            if price_at_analysis <= 0:
                                logger.debug(f"{ticker}: Invalid baseline price: {price_at_analysis}")
                                continue
                            
                            if current_price <= 0:
                                logger.debug(f"{ticker}: Invalid current price: {current_price}")
                                continue
                            
                            # Calculate percent change since analysis
                            # Formula: ((Current - Baseline) / Baseline) * 100
                            price_change_pct = ((current_price - price_at_analysis) / price_at_analysis) * 100
                            
                            # Log the calculation for verification
                            logger.info(f"📊 {ticker}: Calculated change since analysis")
                            logger.info(f"   Price at Analysis: ${price_at_analysis:.2f}")
                            logger.info(f"   Current Price: ${current_price:.2f}")
                            logger.info(f"   Change: ${current_price - price_at_analysis:.2f} ({price_change_pct:+.2f}%)")
                            
                        except (ValueError, TypeError) as e:
                            logger.warning(f"{ticker}: Failed to calculate from prices '{price_at_analysis_raw}' / '{current_price_raw}': {e}")
                            continue
                        except ZeroDivisionError:
                            logger.warning(f"{ticker}: Division by zero - price at analysis is 0")
                            continue
                    else:
                        logger.debug(f"{ticker}: Missing price data (baseline={price_at_analysis_raw}, current={current_price_raw})")
                        continue
                
                if price_change_pct is None:
                    logger.warning(f"{ticker}: ❌ No percent change data available after trying all methods - SKIPPED")
                    logger.warning(f"   Raw 'Percent Change' value: {row.get('Percent Change', 'NOT FOUND')}")
                    logger.warning(f"   Available columns: {list(row.index)}")
                    continue
                
                logger.info(f"{ticker}: ✓ Successfully parsed percent change: {price_change_pct}%")
                
                # Validate percent change is reasonable (not NaN, infinity, or extreme outlier)
                if not isinstance(price_change_pct, (int, float)):
                    logger.warning(f"{ticker}: Invalid percent change type: {type(price_change_pct)}")
                    continue
                
                if abs(price_change_pct) > 1000:  # More than 1000% change is likely a data error
                    logger.warning(f"{ticker}: Extreme percent change {price_change_pct:+.2f}% - likely data error, skipping")
                    continue
                
                # Determine if significant (use min_threshold parameter)
                abs_change = abs(price_change_pct)
                
                if abs_change < min_threshold:
                    logger.debug(f"⚪ {ticker}: {price_change_pct:+.2f}% (below {min_threshold}% threshold)")
                    continue  # Not significant enough (must be >= min_threshold)
                
                logger.info(f"✅ {ticker}: {price_change_pct:+.2f}% - QUALIFIED (threshold={min_threshold}%)")
                
                # Classify magnitude
                if abs_change >= self.movement_thresholds['extreme']:
                    magnitude = "extreme"
                elif abs_change >= self.movement_thresholds['major']:
                    magnitude = "major"
                else:
                    magnitude = "significant"
                
                # Determine direction
                direction = "up" if price_change_pct > 0 else "down"
                
                # Extract prices robustly - try multiple column names
                start_price = 0.0
                end_price = 0.0
                
                # Try to get baseline price (price at analysis)
                for col in ['Price at Analysis', 'Price_at_Analysis', 'Initial Price', 'Starting Price', 'Analysis Price']:
                    if col in row and row[col] is not None:
                        try:
                            val_str = str(row[col]).strip().replace(',', '').replace('$', '').replace(' ', '')
                            if val_str and val_str.lower() not in ['nan', 'n/a', '-', 'none', '']:
                                start_price = float(val_str)
                                if start_price > 0:
                                    break
                        except (ValueError, TypeError):
                            continue
                
                # Try to get current price
                for col in ['Price', 'Current Price', 'Latest Price', 'Current_Price']:
                    if col in row and row[col] is not None:
                        try:
                            val_str = str(row[col]).strip().replace(',', '').replace('$', '').replace(' ', '')
                            if val_str and val_str.lower() not in ['nan', 'n/a', '-', 'none', '']:
                                end_price = float(val_str)
                                if end_price > 0:
                                    break
                        except (ValueError, TypeError):
                            continue
                
                # If we don't have both prices, estimate from percent change
                if start_price == 0 or end_price == 0:
                    if end_price > 0 and start_price == 0:
                        # Have current price, calculate baseline from percent change
                        # current = baseline * (1 + pct/100)
                        # baseline = current / (1 + pct/100)
                        start_price = end_price / (1 + price_change_pct / 100)
                    elif start_price > 0 and end_price == 0:
                        # Have baseline price, calculate current from percent change
                        end_price = start_price * (1 + price_change_pct / 100)
                    else:
                        # Have neither - use arbitrary baseline of 100
                        start_price = 100.0
                        end_price = start_price * (1 + price_change_pct / 100)
                
                # Calculate absolute price change
                price_change_abs = end_price - start_price
                
                # Verify calculation matches percent change
                calculated_pct = ((end_price - start_price) / start_price) * 100 if start_price > 0 else 0
                if abs(calculated_pct - price_change_pct) > 0.01:  # Allow small rounding difference
                    logger.debug(f"{ticker}: Verification - reported: {price_change_pct:.2f}%, calculated: {calculated_pct:.2f}%")
                
                # Get sector and market cap
                sector = row.get('Sector', None)
                market_cap = row.get('Market Cap', None)
                if market_cap:
                    try:
                        market_cap = float(str(market_cap).replace(',', ''))
                    except (ValueError, AttributeError, TypeError):
                        market_cap = None
                
                # Get analysis date if available
                analysis_date = row.get('Analysis Date', start_date)
                
                movement = StockMovement(
                    ticker=ticker,
                    start_date=str(analysis_date) if analysis_date else start_date,
                    end_date=end_date,
                    start_price=start_price,
                    end_price=end_price,
                    price_change_pct=price_change_pct,
                    price_change_abs=price_change_abs,
                    direction=direction,
                    magnitude=magnitude,
                    sector=sector,
                    market_cap=market_cap,
                    volume_change_pct=None  # Not available from sheets
                )
                
                movements.append(movement)
                logger.info(f"Found {magnitude} {direction} movement from sheets: {ticker} {price_change_pct:+.2f}%")
                
            except Exception as e:
                logger.error(f"Error processing sheets row for {row.get('Ticker', 'unknown')}: {e}")
                continue
        
        # Sort by absolute price change (most significant first)
        movements.sort(key=lambda x: abs(x.price_change_pct), reverse=True)
        
        # Count up and down movements for better visibility
        up_movements = [m for m in movements if m.direction == "up"]
        down_movements = [m for m in movements if m.direction == "down"]
        
        logger.info(f"🎯 ANALYSIS COMPLETE: Identified {len(movements)} significant movements from {len(sheets_df)} stocks (threshold: {min_threshold}%)")
        logger.info(f"   📈 {len(up_movements)} stocks moved UP  |  📉 {len(down_movements)} stocks moved DOWN")
        
        if len(movements) > 0:
            top_gainers = [m for m in movements if m.direction == "up"][:3]
            top_losers = [m for m in movements if m.direction == "down"][:3]
            
            if top_gainers:
                logger.info(f"✅ Top GAINERS: {', '.join([f'{m.ticker} (+{m.price_change_pct:.1f}%)' for m in top_gainers])}")
            if top_losers:
                logger.info(f"❌ Top LOSERS: {', '.join([f'{m.ticker} ({m.price_change_pct:.1f}%)' for m in top_losers])}")
        else:
            logger.warning(f"⚠️ NO MOVEMENTS FOUND meeting {min_threshold}% threshold")
            logger.warning(f"   - Analyzed {len(sheets_df)} stocks from Google Sheets")
            logger.warning(f"   - Try lowering threshold or check if 'Percent Change' column has valid data")
            logger.warning(f"   - Check logs above for RAW DATA to see actual values")
        
        return movements
    
    def _identify_significant_movements(
        self,
        start_date: str,
        end_date: str,
        tickers: Optional[List[str]] = None,
        qa_system=None,
        sheets_integration=None,
        min_threshold: float = 15.0
    ) -> List[StockMovement]:
        """
        Identify stocks with significant price movements.
        Prefers using Google Sheets 'Percent Change' data when available.
        
        Args:
            start_date: Start date
            end_date: End date
            tickers: Optional list of tickers (None = analyze ALL from sheets)
            qa_system: QA system for getting tracked stocks (unused if tickers=None)
            sheets_integration: Google Sheets integration
            min_threshold: Minimum percent change to be considered significant (default: 15.0%)
        """
        movements = []
        
        # Try to use Google Sheets data first
        sheets_df = self._get_google_sheets_data(sheets_integration)
        if sheets_df is not None and 'Ticker' in sheets_df.columns:
            logger.info("Using Google Sheets data for movement detection")
            # Pass tickers=None to analyze ALL stocks, or specific tickers to filter
            movements_from_sheets = self._identify_movements_from_sheets(
                sheets_df, 
                tickers, 
                start_date, 
                end_date, 
                min_threshold
            )
            if movements_from_sheets:
                return movements_from_sheets
            logger.warning("No significant movements found in Google Sheets")
        
        # Fallback: only works if specific tickers provided
        if tickers is None:
            # Get from QA system recommendations if available
            if qa_system and hasattr(qa_system, 'recommendations'):
                tickers = list(qa_system.recommendations.keys())
            else:
                logger.warning("No tickers provided, no QA system, and no sheets data - cannot analyze")
                return movements
        
        if not tickers:
            logger.warning("Empty tickers list and no sheets data available")
            return movements
        
        logger.info(f"Falling back to price history for {len(tickers)} tickers")
        
        # Fallback to fetching price history
        logger.info("Using price history for movement detection")
        
        for ticker in tickers:
            try:
                # Get price history
                price_data = self.data_provider.get_price_history(
                    ticker, start_date, end_date, cache_hours=1
                )
                
                if price_data is None or price_data.empty or len(price_data) < 2:
                    continue
                
                start_price = float(price_data['Close'].iloc[0])
                end_price = float(price_data['Close'].iloc[-1])
                
                # Calculate change
                price_change_pct = ((end_price - start_price) / start_price) * 100
                price_change_abs = end_price - start_price
                
                # Determine if significant
                abs_change = abs(price_change_pct)
                
                if abs_change < self.movement_thresholds['significant']:
                    continue  # Not significant enough
                
                # Classify magnitude
                if abs_change >= self.movement_thresholds['extreme']:
                    magnitude = "extreme"
                elif abs_change >= self.movement_thresholds['major']:
                    magnitude = "major"
                else:
                    magnitude = "significant"
                
                # Determine direction
                direction = "up" if price_change_pct > 0 else "down"
                
                # Calculate volume change if available
                volume_change_pct = None
                if 'Volume' in price_data.columns:
                    avg_volume_before = price_data['Volume'].iloc[:len(price_data)//2].mean()
                    avg_volume_after = price_data['Volume'].iloc[len(price_data)//2:].mean()
                    if avg_volume_before > 0:
                        volume_change_pct = ((avg_volume_after - avg_volume_before) / avg_volume_before) * 100
                
                # Get fundamentals for sector/market cap (if available)
                sector = None
                market_cap = None
                try:
                    if hasattr(self.data_provider, 'get_fundamentals'):
                        fundamentals = self.data_provider.get_fundamentals(ticker, cache_hours=24)
                        sector = fundamentals.get('sector') if fundamentals else None
                        market_cap = fundamentals.get('market_cap') if fundamentals else None
                except Exception:
                    pass  # Not critical, continue without fundamentals
                
                movement = StockMovement(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    start_price=start_price,
                    end_price=end_price,
                    price_change_pct=price_change_pct,
                    price_change_abs=price_change_abs,
                    direction=direction,
                    magnitude=magnitude,
                    sector=sector,
                    market_cap=market_cap,
                    volume_change_pct=volume_change_pct
                )
                
                movements.append(movement)
                logger.info(f"Found {magnitude} {direction} movement: {ticker} {price_change_pct:+.2f}%")
                
            except Exception as e:
                logger.error(f"Error analyzing {ticker}: {e}")
                continue
        
        # Sort by absolute price change (most significant first)
        movements.sort(key=lambda x: abs(x.price_change_pct), reverse=True)
        
        return movements
    
    def _fetch_news_for_stock(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> List[NewsArticle]:
        """
        OPTIMIZED: Fast news fetching with focus on recent, relevant articles.
        Uses parallel strategies and smart filtering for speed.
        """
        from datetime import datetime, timedelta
        news_articles = []
        
        # Calculate recent date range (prioritize last 14 days from end_date)
        try:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            recent_start = (end_dt - timedelta(days=14)).strftime('%Y-%m-%d')
        except:
            recent_start = start_date
        
        try:
            # FAST STRATEGY 1: Polygon API (fastest, most reliable for recent news)
            if hasattr(self.data_provider, 'polygon_key') and self.data_provider.polygon_key:
                try:
                    import requests
                    api_key = self.data_provider.polygon_key
                    logger.info(f"📰 Fetching recent news for {ticker} via Polygon.io...")
                    # Focus on last 14 days for speed and relevance
                    url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&published_utc.gte={recent_start}&published_utc.lte={end_date}&limit=10&order=desc&apiKey={api_key}"
                    response = requests.get(url, timeout=10)  # Reduced timeout
                    if response.status_code == 200:
                        data = response.json()
                        polygon_articles = data.get('results', [])
                        for article in polygon_articles[:10]:
                            news_articles.append(NewsArticle(
                                title=article.get('title', ''),
                                description=article.get('description', '')[:300],  # Truncate for speed
                                url=article.get('article_url', ''),
                                published_date=article.get('published_utc', ''),
                                source=article.get('publisher', {}).get('name', 'Polygon News') if isinstance(article.get('publisher'), dict) else 'Polygon News',
                                keywords=article.get('keywords', [])
                            ))
                        logger.info(f"✅ Found {len(polygon_articles)} recent articles via Polygon.io")
                except Exception as e:
                    logger.debug(f"Polygon API error: {e}")
            
            # FAST STRATEGY 2: Only if we have <5 articles, try get_news_with_sources (slower)
            if len(news_articles) < 5 and hasattr(self.data_provider, 'get_news_with_sources'):
                logger.info(f"📰 Supplementing with get_news_with_sources for {ticker}...")
                news_data = self.data_provider.get_news_with_sources(ticker, limit=8)  # Reduced limit
                
                if news_data and isinstance(news_data, list):
                    for article in news_data[:8]:
                        news_articles.append(NewsArticle(
                            title=article.get('title', article.get('headline', '')),
                            description=article.get('description', article.get('summary', article.get('snippet', '')))[:300],
                            url=article.get('url', article.get('link', '')),
                            published_date=article.get('published_date', article.get('date', '')),
                            source=article.get('source', 'Unknown'),
                            keywords=article.get('keywords', [])
                        ))
                    logger.info(f"✅ Found {len(news_data)} supplemental articles")
            
            # FAST STRATEGY 3: Perplexity ONLY if critical (very low coverage)
            if len(news_articles) < 3 and hasattr(self.data_provider, 'perplexity_key') and self.data_provider.perplexity_key:
                try:
                    logger.info(f"📰 Using Perplexity for critical {ticker} research...")
                    perplexity_news = self._perplexity_news_search_fast(ticker, recent_start, end_date)
                    news_articles.extend(perplexity_news)
                    logger.info(f"✅ Found {len(perplexity_news)} insights via Perplexity")
                except Exception as e:
                    logger.debug(f"Perplexity error: {e}")
            
            # OPTIMIZATION: Deduplicate and sort by date (most recent first)
            seen_titles = set()
            unique_articles = []
            
            # Sort by published date (recent first)
            news_articles.sort(key=lambda x: x.published_date if x.published_date else '', reverse=True)
            
            for article in news_articles:
                title_key = article.title.lower().strip()[:80]  # Shorter key for better dedup
                if title_key and title_key not in seen_titles and article.title:
                    seen_titles.add(title_key)
                    unique_articles.append(article)
            
            # Keep top 10 most recent, relevant articles
            news_articles = unique_articles[:10]
            
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
        
        logger.info(f"📊 Total: {len(news_articles)} recent, unique articles for {ticker}")
        return news_articles
    
    def _perplexity_news_search(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> List[NewsArticle]:
        """
        Use Perplexity AI to search the web for comprehensive news and analysis.
        """
        news_articles = []
        
        try:
            import requests
            from datetime import datetime
            
            # Format dates for query
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                date_range = f"{start_dt.strftime('%B %d, %Y')} to {end_dt.strftime('%B %d, %Y')}"
            except:
                date_range = f"{start_date} to {end_date}"
            
            query = f"""Search the web comprehensively for ALL significant news, developments, and catalysts for stock ticker {ticker} from {date_range}.

SEARCH REQUIREMENTS:
1. Financial results (earnings, revenue, guidance, analyst reactions)
2. Business developments (partnerships, product launches, contracts, regulatory approvals)
3. Management/corporate actions (leadership changes, buybacks, dividends, restructuring)
4. Analyst coverage (upgrades, downgrades, price targets, initiations)
5. Market sentiment shifts (short seller reports, hedge fund positions, institutional buying/selling)
6. Sector/competitive dynamics affecting {ticker}
7. Macroeconomic events with direct impact on {ticker}

For EACH relevant news item found, provide:
**HEADLINE:** [Exact title]
**DATE:** [Specific date]
**SOURCE:** [Publication name with URL if available]
**KEY IMPACT:** [1-2 sentences on how this specifically affects {ticker} stock value]
**SENTIMENT:** [Bullish/Bearish/Neutral]

Find at least 10-15 relevant news items. Be comprehensive - this is for investment analysis."""

            url = "https://api.perplexity.ai/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.data_provider.perplexity_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "sonar-pro",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a financial research analyst conducting comprehensive due diligence on stocks. Search extensively across all major financial news sources, regulatory filings, analyst reports, and business news to find every relevant development."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 3000
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                
                # Parse the response to extract news items
                import re
                
                # Split by headline markers
                sections = re.split(r'\*\*HEADLINE:\*\*', content)
                
                for section in sections[1:]:  # Skip first empty section
                    try:
                        # Extract components
                        title_match = re.search(r'^([^\n]+)', section)
                        date_match = re.search(r'\*\*DATE:\*\*\s*([^\n]+)', section)
                        source_match = re.search(r'\*\*SOURCE:\*\*\s*([^\n]+)', section)
                        impact_match = re.search(r'\*\*KEY IMPACT:\*\*\s*([^\n]+(?:\n(?!\*\*)[^\n]+)*)', section)
                        
                        if title_match:
                            title = title_match.group(1).strip()
                            date = date_match.group(1).strip() if date_match else end_date
                            source = source_match.group(1).strip() if source_match else "Web Search"
                            description = impact_match.group(1).strip() if impact_match else ""
                            
                            # Extract URL if present in source
                            url_match = re.search(r'https?://[^\s\)]+', source)
                            article_url = url_match.group(0) if url_match else ""
                            
                            news_articles.append(NewsArticle(
                                title=title,
                                description=description[:500],
                                url=article_url,
                                published_date=date,
                                source=source.split('(')[0].strip(),  # Remove URL from source name
                                keywords=[ticker]
                            ))
                    except Exception as e:
                        logger.debug(f"Could not parse news section: {e}")
                        continue
                
        except Exception as e:
            logger.warning(f"Perplexity news search error: {e}")
        
        return news_articles
    
    def _perplexity_news_search_fast(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> List[NewsArticle]:
        """
        FAST version: Streamlined Perplexity search focused on most recent catalysts only.
        """
        news_articles = []
        
        try:
            import requests
            from datetime import datetime
            
            # Format dates for query
            try:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                date_range = f"last 14 days (ending {end_dt.strftime('%B %d, %Y')})"
            except:
                date_range = f"from {start_date} to {end_date}"
            
            # STREAMLINED QUERY - focused on key catalysts only
            query = f"""Find the TOP 5 most important recent news/catalysts for {ticker} stock in the {date_range}.

Focus ONLY on major price-moving events:
1. Earnings reports with specific numbers (EPS, revenue vs estimates)
2. Analyst upgrades/downgrades with price targets
3. Major business announcements (deals, products, approvals)
4. Significant corporate actions (management, buybacks, acquisitions)

For each item provide CONCISELY:
**HEADLINE:** [Title]
**DATE:** [MM/DD/YYYY]
**IMPACT:** [1 sentence: what happened and stock impact]

Only include news that directly impacts {ticker} stock price. Be specific with numbers and dates."""

            url = "https://api.perplexity.ai/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.data_provider.perplexity_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "sonar-pro",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a fast financial news analyst. Provide only the most critical, price-moving catalysts with specific details."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 1000  # Reduced for speed
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)  # Faster timeout
            
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                
                # Quick parse
                import re
                sections = re.split(r'\*\*HEADLINE:\*\*', content)
                
                for section in sections[1:6]:  # Max 5 items
                    try:
                        title_match = re.search(r'^([^\n]+)', section)
                        date_match = re.search(r'\*\*DATE:\*\*\s*([^\n]+)', section)
                        impact_match = re.search(r'\*\*IMPACT:\*\*\s*([^\n]+)', section)
                        
                        if title_match and impact_match:
                            title = title_match.group(1).strip()
                            date = date_match.group(1).strip() if date_match else end_date
                            description = impact_match.group(1).strip()
                            
                            news_articles.append(NewsArticle(
                                title=title,
                                description=description,
                                url="",
                                published_date=date,
                                source="Perplexity Research",
                                keywords=[ticker]
                            ))
                    except:
                        continue
                
        except Exception as e:
            logger.debug(f"Fast Perplexity search error: {e}")
        
        return news_articles
    
    def _analyze_movement(self, movement: StockMovement) -> Optional[MovementAnalysis]:
        """
        Deep analysis of why a stock moved using AI and available data.
        """
        logger.info(f"Analyzing movement for {movement.ticker}")
        
        # Fetch news articles
        news_articles = self._fetch_news_for_stock(
            movement.ticker,
            movement.start_date,
            movement.end_date
        )
        
        # Get fundamentals and additional context (if available)
        fundamentals = {}
        try:
            if hasattr(self.data_provider, 'get_fundamentals'):
                fundamentals = self.data_provider.get_fundamentals(movement.ticker, cache_hours=24) or {}
        except Exception as e:
            logger.debug(f"Could not fetch fundamentals for {movement.ticker}: {e}")
            fundamentals = {}
        
        # Use AI to analyze root causes
        analysis_result = self._ai_analyze_root_causes(
            movement, news_articles, fundamentals
        )
        
        if not analysis_result:
            logger.warning(f"Could not complete AI analysis for {movement.ticker}")
            return None
        
        # Create comprehensive analysis
        analysis = MovementAnalysis(
            ticker=movement.ticker,
            movement=movement,
            news_articles=news_articles,
            root_causes=analysis_result.get('root_causes', []),
            confidence=analysis_result.get('confidence', 50.0),
            earnings_related=analysis_result.get('earnings_related', False),
            news_driven=analysis_result.get('news_driven', False),
            market_driven=analysis_result.get('market_driven', False),
            sector_driven=analysis_result.get('sector_driven', False),
            fundamental_change=analysis_result.get('fundamental_change', False),
            technical_breakout=analysis_result.get('technical_breakout', False),
            catalyst_summary=analysis_result.get('catalyst_summary', ''),
            agent_relevance=analysis_result.get('agent_relevance', {}),
            model_gaps=analysis_result.get('model_gaps', [])
        )
        
        logger.info(f"Completed analysis for {movement.ticker}: {len(analysis.root_causes)} root causes identified")
        
        return analysis
    
    def _ai_analyze_root_causes(
        self,
        movement: StockMovement,
        news_articles: List[NewsArticle],
        fundamentals: Optional[Dict]
    ) -> Optional[Dict[str, Any]]:
        """
        Use AI (OpenAI or Perplexity) to analyze root causes of stock movement.
        """
        # Choose AI client
        ai_client = self.openai_client or self.perplexity_client
        
        if not ai_client:
            logger.warning("No AI client available for root cause analysis")
            return self._fallback_analysis(movement, news_articles, fundamentals)
        
        # OPTIMIZED: Prepare concise context - top 8 articles only
        news_summary = "\n".join([
            f"- [{article.published_date}] {article.title}"
            for article in news_articles[:8]
        ]) if news_articles else "No specific news found"
        
        # Concise fundamentals
        fundamentals_summary = ""
        if fundamentals:
            fundamentals_summary = f"Sector: {fundamentals.get('sector', 'N/A')} | P/E: {fundamentals.get('pe_ratio', 'N/A')}"
        
        # Format volume change safely
        volume_change_str = f"{movement.volume_change_pct:+.2f}%" if movement.volume_change_pct is not None else 'N/A'
        
        # Determine if we need web search
        has_news = len(news_articles) >= 3
        search_instruction = "" if has_news else "\nCRITICAL: Search web for recent catalysts - earnings, analyst actions, news."
        
        # Create comprehensive AI prompt with research requirements
        prompt = f"""COMPREHENSIVE STOCK MOVEMENT ANALYSIS REQUIRED

Stock: {movement.ticker}
Movement: {movement.direction.upper()} {abs(movement.price_change_pct):.2f}% 
Period: {movement.start_date} to {movement.end_date}
Magnitude: {movement.magnitude.upper()}

Price Data:
- Start Price: ${movement.start_price:.2f}
- End Price: ${movement.end_price:.2f}
- Change: {movement.price_change_pct:+.2f}%
- Volume Change: {volume_change_str}

Context:
{fundamentals_summary if fundamentals_summary else "N/A"}

Recent News Headlines:
{news_summary}{search_instruction}

TASK: Find SPECIFIC catalysts (with dates/numbers):
1. Earnings: reports, guidance, analyst reactions
2. Analyst actions: upgrades/downgrades, price targets
3. Business: products, deals, contracts, approvals
4. Corporate: management, buybacks, dividends
5. Sector: industry trends, competitors
6. Technical: short squeezes, momentum
7. Macro: rates, economy, geopolitics

Search last 14 days. Be SPECIFIC with dates and numbers.

JSON OUTPUT (must include dates/numbers):
{{
  "root_causes": ["Reason 1 with DATE and NUMBERS", "Reason 2...", "..." ],
  "confidence": 0-100,
  "earnings_related": true/false,
  "news_driven": true/false,
  "market_driven": true/false,
  "sector_driven": true/false,
  "fundamental_change": true/false,
  "technical_breakout": true/false,
  "catalyst_summary": "2-3 sentences with SPECIFIC dates, names, numbers explaining the move",
  "agent_relevance": {{
    "value": "Should value agent have caught this? Why/why not?",
    "growth_momentum": "Should growth agent have caught this? Why/why not?",
    "sentiment": "Should sentiment agent have caught this? Why/why not?",
    "macro_regime": "Should macro agent have caught this? Why/why not?",
    "risk": "Should risk agent have caught this? Why/why not?"
  }},
  "model_gaps": ["Specific gap 1", "Specific gap 2", "Specific gap 3"]
}}

Be SPECIFIC. Include dates and numbers in every root cause."""
        
        try:
            # OPTIMIZED: Choose model and configure for speed
            if self.openai_client:
                model = "gpt-4"
                system_msg = "Financial analyst: Find specific stock movement catalysts with dates and numbers. JSON only."
                max_tok = 1500
            else:
                model = "sonar-pro"  # Perplexity with web search
                system_msg = "Financial analyst with web search: Find specific recent catalysts for stock moves. Include dates, numbers, analyst names. JSON only."
                max_tok = 1500
            
            response = ai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Very low for speed and precision
                max_tokens=max_tok  # Optimized for speed
            )
            
            content = response.choices[0].message.content
            if not content:
                logger.error("Empty response from AI")
                return self._fallback_analysis(movement, news_articles, fundamentals)
            
            content = content.strip()
            
            # Extract JSON
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            # Find JSON object
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_content = content[start_idx:end_idx]
                result = json.loads(json_content)
                return result
            else:
                logger.error("No JSON found in AI response")
                return self._fallback_analysis(movement, news_articles, fundamentals)
                
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            return self._fallback_analysis(movement, news_articles, fundamentals)
    
    def _fallback_analysis(
        self,
        movement: StockMovement,
        news_articles: List[NewsArticle],
        fundamentals: Optional[Dict]
    ) -> Dict[str, Any]:
        """Fallback analysis when AI is not available."""
        root_causes = []
        
        # Basic heuristics
        if news_articles:
            root_causes.append("News-driven movement (multiple articles published)")
        
        if abs(movement.price_change_pct) > 15:
            root_causes.append("Significant price movement suggests major catalyst")
        
        if movement.volume_change_pct and movement.volume_change_pct > 50:
            root_causes.append("High volume increase indicates strong interest")
        
        return {
            'root_causes': root_causes if root_causes else ["Unknown - requires further investigation"],
            'confidence': 30.0,
            'earnings_related': False,
            'news_driven': len(news_articles) > 0,
            'market_driven': False,
            'sector_driven': False,
            'fundamental_change': False,
            'technical_breakout': abs(movement.price_change_pct) > 10,
            'catalyst_summary': f"{movement.ticker} moved {movement.direction} by {abs(movement.price_change_pct):.2f}%. Further analysis needed.",
            'agent_relevance': {},
            'model_gaps': ["AI analysis unavailable"]
        }
    
    def _generate_model_recommendations(
        self,
        analyses: List[MovementAnalysis],
        start_date: str,
        end_date: str
    ) -> List[ModelAdjustmentRecommendation]:
        """
        Generate specific, actionable model adjustment recommendations based on analyses.
        """
        recommendations = []
        
        # Return empty if no analyses completed
        if not analyses:
            logger.warning("No analyses completed - cannot generate recommendations")
            return recommendations
        
        # Analyze patterns across all movements
        patterns = self._identify_patterns(analyses)
        
        # Generate recommendations based on patterns
        rec_id = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Recommendation 1: Earnings-related improvements
        if patterns.get('earnings_frequency', 0) > 0.3:  # More than 30% earnings-related
            recommendations.append(ModelAdjustmentRecommendation(
                recommendation_id=f"{rec_id}_earnings",
                priority="high",
                category="feature_focus",
                specific_change="Increase weight on earnings analysis and expectations",
                rationale=f"{patterns.get('earnings_frequency', 0)*100:.0f}% of significant movements were earnings-related. Model should focus more on earnings quality, surprises, and guidance.",
                expected_impact="Better prediction of earnings-driven moves; improved timing around earnings dates",
                supporting_evidence=[
                    f"{a.ticker}: {a.catalyst_summary}" 
                    for a in analyses if a.earnings_related
                ][:3],
                affected_agents=["value", "growth_momentum", "sentiment"],
                implementation_steps=[
                    "Add earnings date tracking to all analyses",
                    "Increase value agent weight by 0.15x during earnings season",
                    "Add earnings surprise factor to growth momentum agent",
                    "Monitor analyst estimate revisions more closely"
                ],
                confidence=85.0
            ))
        
        # Recommendation 2: News/Sentiment improvements
        if patterns.get('news_driven_frequency', 0) > 0.4:
            recommendations.append(ModelAdjustmentRecommendation(
                recommendation_id=f"{rec_id}_news",
                priority="high",
                category="agent_weight",
                specific_change="Increase sentiment agent weight by 0.2x",
                rationale=f"{patterns.get('news_driven_frequency', 0)*100:.0f}% of movements were news-driven. Sentiment agent should have stronger influence.",
                expected_impact="Faster reaction to breaking news; better capture of sentiment shifts",
                supporting_evidence=[
                    f"{a.ticker}: {len(a.news_articles)} articles, {a.catalyst_summary[:100]}"
                    for a in analyses if a.news_driven
                ][:3],
                affected_agents=["sentiment"],
                implementation_steps=[
                    "Increase sentiment agent weight from 1.0 to 1.2",
                    "Add real-time news monitoring for tracked stocks",
                    "Implement breaking news alerts",
                    "Add social media sentiment tracking"
                ],
                confidence=80.0
            ))
        
        # Recommendation 3: Sector/Market correlation
        if patterns.get('sector_driven_frequency', 0) > 0.25:
            recommendations.append(ModelAdjustmentRecommendation(
                recommendation_id=f"{rec_id}_sector",
                priority="medium",
                category="feature_focus",
                specific_change="Enhance sector rotation analysis in macro agent",
                rationale=f"{patterns.get('sector_driven_frequency', 0)*100:.0f}% of movements were sector-driven. Need better sector trend analysis.",
                expected_impact="Better capture of sector rotation trends; improved relative sector positioning",
                supporting_evidence=[
                    f"{a.ticker} ({a.movement.sector}): {a.catalyst_summary[:100]}"
                    for a in analyses if a.sector_driven
                ][:3],
                affected_agents=["macro_regime"],
                implementation_steps=[
                    "Add sector ETF performance tracking",
                    "Monitor sector rotation indicators",
                    "Compare individual stock to sector performance",
                    "Add sector momentum scoring"
                ],
                confidence=75.0
            ))
        
        # Recommendation 4: Risk management
        extreme_moves = [a for a in analyses if a.movement.magnitude == "extreme"]
        if extreme_moves:
            recommendations.append(ModelAdjustmentRecommendation(
                recommendation_id=f"{rec_id}_risk",
                priority="critical",
                category="threshold",
                specific_change="Tighten risk controls for high-volatility stocks",
                rationale=f"{len(extreme_moves)} stocks had extreme movements (>20%). Risk agent needs to better identify high-volatility situations.",
                expected_impact="Reduced exposure to extreme volatility; better downside protection",
                supporting_evidence=[
                    f"{a.ticker}: {a.movement.price_change_pct:+.2f}% move"
                    for a in extreme_moves
                ][:3],
                affected_agents=["risk"],
                implementation_steps=[
                    "Add implied volatility screening",
                    "Implement position size limits for high-beta stocks",
                    "Monitor options market for volatility signals",
                    "Add stop-loss recommendations"
                ],
                confidence=90.0
            ))
        
        # Recommendation 5: Model gaps analysis
        common_gaps = self._analyze_common_gaps(analyses)
        if common_gaps:
            for gap_type, gap_data in common_gaps.items():
                if gap_data['frequency'] > 0.2:  # More than 20% of cases
                    recommendations.append(ModelAdjustmentRecommendation(
                        recommendation_id=f"{rec_id}_{gap_type}",
                        priority="medium",
                        category="data_source",
                        specific_change=f"Add {gap_type} data to analysis pipeline",
                        rationale=f"Model missed {gap_type} in {gap_data['frequency']*100:.0f}% of cases",
                        expected_impact=f"Better capture of {gap_type}-driven movements",
                        supporting_evidence=gap_data['examples'][:3],
                        affected_agents=gap_data['affected_agents'],
                        implementation_steps=[
                            f"Identify data source for {gap_type}",
                            f"Integrate {gap_type} into relevant agents",
                            f"Test {gap_type} feature importance",
                            "Monitor improvement in predictions"
                        ],
                        confidence=70.0
                    ))
        
        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda x: (priority_order.get(x.priority, 4), -x.confidence))
        
        logger.info(f"Generated {len(recommendations)} model recommendations")
        
        return recommendations
    
    def _identify_patterns(self, analyses: List[MovementAnalysis]) -> Dict[str, Any]:
        """Identify patterns across multiple analyses."""
        if not analyses:
            return {}
        
        total = len(analyses)
        
        return {
            'earnings_frequency': sum(1 for a in analyses if a.earnings_related) / total,
            'news_driven_frequency': sum(1 for a in analyses if a.news_driven) / total,
            'market_driven_frequency': sum(1 for a in analyses if a.market_driven) / total,
            'sector_driven_frequency': sum(1 for a in analyses if a.sector_driven) / total,
            'fundamental_change_frequency': sum(1 for a in analyses if a.fundamental_change) / total,
            'technical_breakout_frequency': sum(1 for a in analyses if a.technical_breakout) / total,
            'avg_confidence': sum(a.confidence for a in analyses) / total,
            'up_movements': sum(1 for a in analyses if a.movement.direction == "up"),
            'down_movements': sum(1 for a in analyses if a.movement.direction == "down")
        }
    
    def _analyze_common_gaps(self, analyses: List[MovementAnalysis]) -> Dict[str, Dict]:
        """Analyze common gaps across analyses."""
        # Use a more explicit structure
        gap_counter: Dict[str, Dict[str, Any]] = {}
        
        for analysis in analyses:
            for gap in analysis.model_gaps:
                gap_key = gap.lower().replace(' ', '_')[:50]  # Normalize
                
                # Initialize if not exists
                if gap_key not in gap_counter:
                    gap_counter[gap_key] = {
                        'count': 0,
                        'examples': [],
                        'affected_agents': set()
                    }
                
                gap_counter[gap_key]['count'] += 1
                gap_counter[gap_key]['examples'].append(f"{analysis.ticker}: {gap}")
                
                # Determine affected agents based on gap content
                gap_lower = gap.lower()
                if 'valuation' in gap_lower or 'price' in gap_lower:
                    gap_counter[gap_key]['affected_agents'].add('value')
                if 'growth' in gap_lower or 'momentum' in gap_lower:
                    gap_counter[gap_key]['affected_agents'].add('growth_momentum')
                if 'sentiment' in gap_lower or 'news' in gap_lower:
                    gap_counter[gap_key]['affected_agents'].add('sentiment')
                if 'macro' in gap_lower or 'market' in gap_lower:
                    gap_counter[gap_key]['affected_agents'].add('macro_regime')
                if 'risk' in gap_lower or 'volatility' in gap_lower:
                    gap_counter[gap_key]['affected_agents'].add('risk')
        
        # Convert to proper format
        result = {}
        total_analyses = len(analyses)
        for gap_key, data in gap_counter.items():
            result[gap_key] = {
                'frequency': data['count'] / total_analyses,
                'examples': data['examples'][:5],
                'affected_agents': list(data['affected_agents'])
            }
        
        return result
    
    def _create_performance_report(
        self,
        movements: List[StockMovement],
        analyses: List[MovementAnalysis],
        recommendations: List[ModelAdjustmentRecommendation],
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """Create comprehensive performance report."""
        patterns = self._identify_patterns(analyses) if analyses else {}
        
        # Top movers
        top_gainers = sorted(
            [m for m in movements if m.direction == "up"],
            key=lambda x: x.price_change_pct,
            reverse=True
        )[:10]
        
        top_losers = sorted(
            [m for m in movements if m.direction == "down"],
            key=lambda x: x.price_change_pct
        )[:10]
        
        report = {
            'report_id': datetime.now().strftime("%Y%m%d%H%M%S"),
            'generated_at': datetime.now().isoformat(),
            'period': {
                'start_date': start_date,
                'end_date': end_date,
                'days': (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days
            },
            'summary': {
                'total_movements': len(movements),
                'analyses_completed': len(analyses),
                'recommendations_generated': len(recommendations),
                'top_gainers_count': len(top_gainers),
                'top_losers_count': len(top_losers),
                'critical_recommendations': len([r for r in recommendations if r.priority == "critical"]),
                'high_priority_recommendations': len([r for r in recommendations if r.priority == "high"])
            },
            'top_gainers': [asdict(m) for m in top_gainers],
            'top_losers': [asdict(m) for m in top_losers],
            'patterns': patterns,
            'analyses': [asdict(a) for a in analyses],
            'recommendations': [asdict(r) for r in recommendations],
            'executive_summary': self._generate_executive_summary(
                movements, analyses, recommendations, patterns
            )
        }
        
        return report
    
    def _generate_executive_summary(
        self,
        movements: List[StockMovement],
        analyses: List[MovementAnalysis],
        recommendations: List[ModelAdjustmentRecommendation],
        patterns: Dict[str, Any]
    ) -> str:
        """Generate executive summary of findings."""
        if not movements:
            return "No significant movements detected in this period."
        
        summary_parts = []
        
        # Movement summary
        up_count = len([m for m in movements if m.direction == "up"])
        down_count = len([m for m in movements if m.direction == "down"])
        summary_parts.append(
            f"Analyzed {len(movements)} significant movements: {up_count} up, {down_count} down."
        )
        
        # Pattern summary
        if patterns:
            dominant_pattern = max(
                [(k, v) for k, v in patterns.items() if k.endswith('_frequency')],
                key=lambda x: x[1]
            )
            pattern_name = dominant_pattern[0].replace('_frequency', '').replace('_', ' ')
            pattern_pct = dominant_pattern[1] * 100
            summary_parts.append(
                f"Dominant pattern: {pattern_name} ({pattern_pct:.0f}% of movements)."
            )
        
        # Recommendation summary
        if recommendations:
            critical = len([r for r in recommendations if r.priority == "critical"])
            high = len([r for r in recommendations if r.priority == "high"])
            if critical > 0:
                summary_parts.append(f"{critical} CRITICAL recommendations require immediate action.")
            if high > 0:
                summary_parts.append(f"{high} high-priority improvements identified.")
            
            # Highlight top recommendation
            top_rec = recommendations[0]
            summary_parts.append(
                f"Top recommendation: {top_rec.specific_change} ({top_rec.confidence:.0f}% confidence)."
            )
        
        return " ".join(summary_parts)
    
    def _save_analysis_results(
        self,
        movements: List[StockMovement],
        analyses: List[MovementAnalysis],
        recommendations: List[ModelAdjustmentRecommendation]
    ):
        """Save analysis results to storage."""
        # Append to history (keep last 1000 entries)
        self.movement_history.extend([asdict(m) for m in movements])
        self.movement_history = self.movement_history[-1000:]
        self._save_json(self.movements_file, self.movement_history)
        
        self.analysis_history.extend([asdict(a) for a in analyses])
        self.analysis_history = self.analysis_history[-1000:]
        self._save_json(self.analyses_file, self.analysis_history)
        
        self.recommendations_history.extend([asdict(r) for r in recommendations])
        self.recommendations_history = self.recommendations_history[-500:]
        self._save_json(self.recommendations_file, self.recommendations_history)
        
        logger.info("Saved analysis results to storage")
    
    def get_latest_recommendations(self, limit: int = 10) -> List[Dict]:
        """Get the most recent model recommendations."""
        return self.recommendations_history[-limit:]
    
    def get_improvement_tracking(self) -> Dict:
        """Get tracking of improvements over time."""
        return self.improvement_tracking
    
    def mark_recommendation_implemented(self, recommendation_id: str, notes: str = ""):
        """Mark a recommendation as implemented and start tracking its impact."""
        implementation_record = {
            'recommendation_id': recommendation_id,
            'implemented_at': datetime.now().isoformat(),
            'notes': notes,
            'performance_before': {},  # Will be populated with pre-implementation metrics
            'performance_after': {}   # Will be tracked post-implementation
        }
        
        if 'implemented_recommendations' not in self.improvement_tracking:
            self.improvement_tracking['implemented_recommendations'] = []
        
        self.improvement_tracking['implemented_recommendations'].append(implementation_record)
        self._save_json(self.tracking_file, self.improvement_tracking)
        
        logger.info(f"Marked recommendation {recommendation_id} as implemented")
    
    def apply_autonomous_adjustments(
        self,
        analyses: List[MovementAnalysis],
        recommendations: List[ModelAdjustmentRecommendation],
        auto_apply: bool = True
    ) -> Dict[str, Any]:
        """
        AUTONOMOUS: Automatically apply model adjustments based on analysis.
        This method actually modifies agent weights and configurations.
        """
        if not auto_apply:
            return {'status': 'disabled', 'adjustments': []}
        
        adjustments_made = []
        patterns = self._identify_patterns(analyses) if analyses else {}
        
        logger.info("🤖 AUTONOMOUS ADJUSTMENT: Analyzing performance patterns...")
        logger.info(f"   Patterns detected: {patterns}")
        
        # Adjustment 1: Agent Weight Adjustments
        agent_weight_changes = self._calculate_agent_weight_adjustments(analyses, patterns)
        if agent_weight_changes:
            success = self._apply_agent_weight_changes(agent_weight_changes)
            if success:
                adjustments_made.append({
                    'type': 'agent_weights',
                    'changes': agent_weight_changes,
                    'timestamp': datetime.now().isoformat()
                })
                logger.info(f"✅ Applied agent weight adjustments: {agent_weight_changes}")
        
        # Adjustment 2: Threshold Adjustments
        threshold_changes = self._calculate_threshold_adjustments(analyses, patterns)
        if threshold_changes:
            success = self._apply_threshold_changes(threshold_changes)
            if success:
                adjustments_made.append({
                    'type': 'thresholds',
                    'changes': threshold_changes,
                    'timestamp': datetime.now().isoformat()
                })
                logger.info(f"✅ Applied threshold adjustments: {threshold_changes}")
        
        # Adjustment 3: Feature Focus Adjustments
        feature_changes = self._calculate_feature_adjustments(analyses, patterns)
        if feature_changes:
            adjustments_made.append({
                'type': 'feature_focus',
                'changes': feature_changes,
                'timestamp': datetime.now().isoformat()
            })
            logger.info(f"✅ Identified feature focus changes: {feature_changes}")
        
        # Save adjustment history
        if adjustments_made:
            self._save_adjustment_history(adjustments_made)
            logger.info(f"🎯 AUTONOMOUS ADJUSTMENT COMPLETE: {len(adjustments_made)} adjustments applied")
        else:
            logger.info("ℹ️  No adjustments needed - model performing adequately")
        
        return {
            'status': 'completed',
            'adjustments': adjustments_made,
            'patterns': patterns,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_agent_weight_adjustments(
        self,
        analyses: List[MovementAnalysis],
        patterns: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate optimal agent weight adjustments based on which agents should have caught the movements.
        """
        weight_changes = {}
        
        # Analyze agent_relevance from analyses to see which agents missed opportunities
        agent_misses = {
            'value': 0,
            'growth_momentum': 0,
            'sentiment': 0,
            'macro_regime': 0,
            'risk': 0
        }
        
        for analysis in analyses:
            relevance = analysis.agent_relevance
            for agent, feedback in relevance.items():
                if feedback and ('should have' in feedback.lower() or 'missed' in feedback.lower()):
                    agent_misses[agent] = agent_misses.get(agent, 0) + 1
        
        total_analyses = len(analyses)
        
        # Calculate weight adjustments (increase weight for agents that missed opportunities)
        for agent, miss_count in agent_misses.items():
            if miss_count > 0:
                miss_rate = miss_count / total_analyses
                if miss_rate > 0.3:  # Missed more than 30% of relevant moves
                    # Increase weight by 10-25% depending on severity
                    weight_increase = min(0.25, 0.10 + (miss_rate - 0.3) * 0.5)
                    weight_changes[agent] = round(1.0 + weight_increase, 2)
        
        # Pattern-based adjustments
        if patterns.get('earnings_frequency', 0) > 0.4:
            weight_changes['value'] = max(weight_changes.get('value', 1.0), 1.15)
            weight_changes['growth_momentum'] = max(weight_changes.get('growth_momentum', 1.0), 1.10)
        
        if patterns.get('news_driven_frequency', 0) > 0.4:
            weight_changes['sentiment'] = max(weight_changes.get('sentiment', 1.0), 1.20)
        
        if patterns.get('sector_driven_frequency', 0) > 0.3:
            weight_changes['macro_regime'] = max(weight_changes.get('macro_regime', 1.0), 1.15)
        
        # Risk agent adjustment based on extreme moves
        extreme_moves = sum(1 for a in analyses if a.movement.magnitude == "extreme")
        if extreme_moves > total_analyses * 0.2:  # More than 20% extreme moves
            weight_changes['risk'] = max(weight_changes.get('risk', 1.0), 1.10)
        
        return weight_changes
    
    def _calculate_threshold_adjustments(
        self,
        analyses: List[MovementAnalysis],
        patterns: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate threshold adjustments for scoring."""
        threshold_changes = {}
        
        # If many high-confidence analyses, we can be more aggressive with thresholds
        avg_confidence = patterns.get('avg_confidence', 50)
        
        if avg_confidence > 75:
            threshold_changes['upside_minimum'] = 0.12  # Lower from 0.15 (more aggressive)
            threshold_changes['conviction_threshold'] = 65  # Lower from 70 (more aggressive)
        elif avg_confidence < 50:
            threshold_changes['upside_minimum'] = 0.20  # Raise from 0.15 (more conservative)
            threshold_changes['conviction_threshold'] = 75  # Raise from 70 (more conservative)
        
        return threshold_changes
    
    def _calculate_feature_adjustments(
        self,
        analyses: List[MovementAnalysis],
        patterns: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate feature importance adjustments."""
        feature_changes = {}
        
        if patterns.get('earnings_frequency', 0) > 0.4:
            feature_changes['earnings_monitoring'] = 'high_priority'
            feature_changes['earnings_surprise_weight'] = 1.5
        
        if patterns.get('news_driven_frequency', 0) > 0.4:
            feature_changes['news_sentiment_weight'] = 1.3
            feature_changes['breaking_news_alerts'] = 'enabled'
        
        if patterns.get('technical_breakout_frequency', 0) > 0.3:
            feature_changes['momentum_indicators'] = 'enhanced'
            feature_changes['volume_analysis_weight'] = 1.2
        
        return feature_changes
    
    def _apply_agent_weight_changes(self, weight_changes: Dict[str, float]) -> bool:
        """Actually apply agent weight changes to the system."""
        try:
            import yaml
            from pathlib import Path
            
            # Load current model config
            config_path = Path("config/model.yaml")
            if not config_path.exists():
                logger.warning("model.yaml not found - cannot apply weight changes")
                return False
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Update agent weights
            if 'agent_weights' not in config:
                config['agent_weights'] = {}
            
            for agent, new_weight in weight_changes.items():
                old_weight = config['agent_weights'].get(agent, 1.0)
                config['agent_weights'][agent] = new_weight
                logger.info(f"   {agent}: {old_weight:.2f} → {new_weight:.2f} ({(new_weight/old_weight-1)*100:+.0f}%)")
            
            # Save updated config with backup
            backup_path = Path(f"config/model.yaml.backup.{datetime.now().strftime('%Y%m%d%H%M%S')}")
            import shutil
            shutil.copy(config_path, backup_path)
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"✅ Updated agent weights in {config_path} (backup: {backup_path})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply agent weight changes: {e}")
            return False
    
    def _apply_threshold_changes(self, threshold_changes: Dict[str, float]) -> bool:
        """Actually apply threshold changes to the system."""
        try:
            import yaml
            from pathlib import Path
            
            config_path = Path("config/model.yaml")
            if not config_path.exists():
                return False
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if 'thresholds' not in config:
                config['thresholds'] = {}
            
            for threshold, new_value in threshold_changes.items():
                old_value = config['thresholds'].get(threshold, 'N/A')
                config['thresholds'][threshold] = new_value
                logger.info(f"   {threshold}: {old_value} → {new_value}")
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply threshold changes: {e}")
            return False
    
    def _save_adjustment_history(self, adjustments: List[Dict]):
        """Save adjustment history for tracking."""
        history_file = self.storage_dir / "adjustment_history.json"
        
        try:
            history = self._load_json(history_file, default=[])
            history.extend(adjustments)
            history = history[-100:]  # Keep last 100
            self._save_json(history_file, history)
        except Exception as e:
            logger.error(f"Failed to save adjustment history: {e}")
