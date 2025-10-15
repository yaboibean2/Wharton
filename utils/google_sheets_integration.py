"""
Google Sheets Integration for Automatic Portfolio Updates
Pushes analysis results to Google Sheets automatically
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class GoogleSheetsIntegration:
    """Manages automatic updates to Google Sheets."""
    
    def __init__(self):
        """Initialize Google Sheets integration."""
        self.client = None
        self.sheet = None
        self.enabled = False
        self._setup_client()
    
    def _setup_client(self):
        """Set up Google Sheets API client."""
        try:
            import gspread
            from google.oauth2.service_account import Credentials
            
            # Check for credentials file
            creds_path = os.getenv('GOOGLE_SHEETS_CREDENTIALS_PATH', 'google_credentials.json')
            
            if not os.path.exists(creds_path):
                logger.info("Google Sheets credentials not found. Integration disabled.")
                return
            
            # Define the required scopes
            SCOPES = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            
            # Authenticate using service account
            credentials = Credentials.from_service_account_file(creds_path, scopes=SCOPES)
            self.client = gspread.authorize(credentials)
            self.enabled = True
            
            logger.info("✅ Google Sheets integration enabled")
            
        except ImportError:
            logger.warning("gspread not installed. Run: pip install gspread google-auth")
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to setup Google Sheets: {e}")
            self.enabled = False
    
    def connect_to_sheet(self, sheet_id: str) -> bool:
        """
        Connect to a specific Google Sheet.
        
        Args:
            sheet_id: The Google Sheet ID from the URL
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            self.sheet = self.client.open_by_key(sheet_id)
            logger.info(f"✅ Connected to Google Sheet: {self.sheet.title}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to sheet: {e}")
            return False
    
    def update_portfolio_analysis(self, 
                                  portfolio_data: List[Dict[str, Any]], 
                                  worksheet_name: str = "Portfolio Analysis") -> bool:
        """
        Update portfolio analysis worksheet.
        
        Args:
            portfolio_data: List of portfolio holdings with analysis
            worksheet_name: Name of the worksheet to update
            
        Returns:
            True if successful, False otherwise
        """
        if not self.sheet:
            logger.warning("No sheet connected. Call connect_to_sheet() first.")
            return False
        
        try:
            # Get or create worksheet
            try:
                worksheet = self.sheet.worksheet(worksheet_name)
            except:
                worksheet = self.sheet.add_worksheet(title=worksheet_name, rows=1000, cols=26)
            
            # Prepare data for sheet
            df = pd.DataFrame(portfolio_data)
            
            # Add timestamp
            df.insert(0, 'Last Updated', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            # Replace NaN and inf with None (Google Sheets will show as empty)
            df = df.replace([float('inf'), float('-inf')], None)
            
            # Convert to list of lists, preserving numeric types
            headers = df.columns.values.tolist()
            
            # Convert DataFrame to list of lists, keeping numeric types
            data_rows = []
            for _, row in df.iterrows():
                row_data = []
                for val in row:
                    if pd.isna(val):
                        row_data.append(None)  # Empty cell in Google Sheets
                    else:
                        row_data.append(val)  # Keep original type (number or string)
                data_rows.append(row_data)
            
            # Clear only old data rows (preserves conditional formatting)
            try:
                existing_rows = len(worksheet.get_all_values())
                if existing_rows > len(data_rows) + 1:  # +1 for header
                    clear_range = f'A{len(data_rows) + 2}:ZZ{existing_rows}'
                    worksheet.batch_clear([clear_range])
            except:
                pass  # Continue if this fails
            
            # Update data without clearing formatting
            all_data = [headers] + data_rows
            worksheet.update(all_data, value_input_option='USER_ENTERED')
            
            # Format header row (comment out to preserve existing header formatting)
            try:
                worksheet.format('A1:Z1', {
                    'textFormat': {'bold': True},
                    'backgroundColor': {'red': 0.2, 'green': 0.2, 'blue': 0.8}
                })
            except:
                pass
            
            logger.info(f"✅ Updated {worksheet_name} with {len(df)} rows")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update portfolio analysis: {e}")
            return False
    
    def update_qa_analyses(self, 
                          qa_data: List[Dict[str, Any]], 
                          worksheet_name: str = "QA Analyses",
                          column_order: List[str] = None) -> bool:
        """
        Update QA analyses worksheet.
        
        Args:
            qa_data: List of QA analysis records
            worksheet_name: Name of the worksheet to update
            column_order: Optional list specifying exact column order
            
        Returns:
            True if successful, False otherwise
        """
        if not self.sheet:
            logger.warning("No sheet connected. Call connect_to_sheet() first.")
            return False
        
        try:
            # Get or create worksheet
            try:
                worksheet = self.sheet.worksheet(worksheet_name)
            except:
                # Create with more columns to accommodate all data
                worksheet = self.sheet.add_worksheet(title=worksheet_name, rows=10000, cols=40)
            
            # Prepare data for sheet
            df = pd.DataFrame(qa_data)
            
            # If column_order specified, reorder columns to match
            if column_order:
                # Only use columns that exist in the dataframe
                existing_cols = [col for col in column_order if col in df.columns]
                df = df[existing_cols]
            
            # Add export timestamp at the end
            df['Export Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Replace NaN and inf with None (Google Sheets will show as empty)
            # This preserves numeric types instead of converting to strings
            df = df.replace([float('inf'), float('-inf')], None)
            
            # Convert to list of lists, preserving numeric types
            # None values will become empty cells in Google Sheets
            headers = df.columns.values.tolist()
            
            # Convert DataFrame to list of lists, keeping numeric types
            data_rows = []
            for _, row in df.iterrows():
                row_data = []
                for val in row:
                    if pd.isna(val):
                        row_data.append(None)  # Empty cell in Google Sheets
                    else:
                        row_data.append(val)  # Keep original type (number or string)
                data_rows.append(row_data)
            
            # Clear only old data rows (preserves conditional formatting)
            # Get current row count to clear old data beyond new data
            try:
                existing_rows = len(worksheet.get_all_values())
                if existing_rows > len(data_rows) + 1:  # +1 for header
                    # Clear rows beyond our new data
                    clear_range = f'A{len(data_rows) + 2}:ZZ{existing_rows}'
                    worksheet.batch_clear([clear_range])
            except:
                pass  # If this fails, just continue - we'll overwrite old data anyway
            
            # Update data using USER_ENTERED to preserve formatting
            # This does NOT clear conditional formatting or cell formatting
            all_data = [headers] + data_rows
            worksheet.update(all_data, value_input_option='USER_ENTERED')
            
            # Only format header row on first setup (skip if already formatted)
            # Comment out these lines if you want to preserve ALL formatting including headers
            try:
                worksheet.format('A1:Z1', {
                    'textFormat': {'bold': True},
                    'backgroundColor': {'red': 0.2, 'green': 0.8, 'blue': 0.2}
                })
            except:
                pass  # Formatting may already exist
                
            # Format rationale columns for proper text wrapping
            try:
                # Find rationale columns and apply text wrapping
                rationale_col_indices = []
                for i, col_name in enumerate(headers):
                    if 'Rationale' in str(col_name) or 'Analysis' in str(col_name):
                        rationale_col_indices.append(i)
                
                if rationale_col_indices:
                    # Apply text wrapping to all data rows for rationale columns
                    for col_index in rationale_col_indices:
                        # Convert 0-based index to column letter
                        def col_index_to_letter(index):
                            result = ""
                            while index >= 0:
                                result = chr(ord('A') + index % 26) + result
                                index = index // 26 - 1
                            return result
                        
                        col_letter = col_index_to_letter(col_index)
                        
                        # Format entire column for text wrapping
                        col_range = f'{col_letter}:{col_letter}'
                        worksheet.format(col_range, {
                            'wrapStrategy': 'WRAP',
                            'verticalAlignment': 'TOP',
                            'textFormat': {'fontSize': 9}
                        })
                        
                    logger.info(f"Applied text wrapping to {len(rationale_col_indices)} rationale columns")
                        
            except Exception as e:
                logger.warning(f"Could not format rationale columns: {e}")
                pass
            
            # Auto-resize non-rationale columns
            try:
                worksheet.columns_auto_resize(0, len(df.columns))
            except:
                pass
            
            logger.info(f"✅ Updated {worksheet_name} with {len(df)} rows")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update QA analyses: {e}")
            return False
    
    def append_portfolio_history(self, 
                                 portfolio_data: Dict[str, Any], 
                                 worksheet_name: str = "Portfolio History") -> bool:
        """
        Append portfolio snapshot to history (doesn't overwrite).
        
        Args:
            portfolio_data: Portfolio data to append
            worksheet_name: Name of the worksheet to append to
            
        Returns:
            True if successful, False otherwise
        """
        if not self.sheet:
            logger.warning("No sheet connected. Call connect_to_sheet() first.")
            return False
        
        try:
            # Get or create worksheet
            try:
                worksheet = self.sheet.worksheet(worksheet_name)
            except:
                worksheet = self.sheet.add_worksheet(title=worksheet_name, rows=10000, cols=26)
                # Add headers
                headers = ['Timestamp', 'Ticker', 'Weight %', 'Price', 'Score', 'Recommendation']
                worksheet.append_row(headers)
            
            # Prepare rows to append
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            rows_to_append = []
            for holding in portfolio_data.get('portfolio', []):
                row = [
                    timestamp,
                    holding.get('ticker', ''),
                    holding.get('target_weight_pct', 0),
                    holding.get('price', 0),
                    holding.get('final_score', 0),
                    holding.get('recommendation', '')
                ]
                rows_to_append.append(row)
            
            # Append all rows
            if rows_to_append:
                worksheet.append_rows(rows_to_append)
                logger.info(f"✅ Appended {len(rows_to_append)} holdings to history")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to append portfolio history: {e}")
            return False
    
    def get_sheet_url(self) -> Optional[str]:
        """Get the URL of the connected sheet."""
        if self.sheet:
            return self.sheet.url
        return None


# Global instance
_sheets_integration = None


def get_sheets_integration() -> GoogleSheetsIntegration:
    """Get or create the global sheets integration instance."""
    global _sheets_integration
    if _sheets_integration is None:
        _sheets_integration = GoogleSheetsIntegration()
    return _sheets_integration
