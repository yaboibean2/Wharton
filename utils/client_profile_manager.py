"""
Client Profile Manager
Utility for managing client profiles and converting them to IPS configurations.
"""

from typing import Dict, Any
import yaml
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ClientProfileManager:
    """
    Manages client profiles and converts them to structured IPS configurations.
    Supports both text-based profiles and structured input.
    """
    
    def __init__(self, profiles_dir: str = "profiles"):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(exist_ok=True)
    
    def save_client_profile(self, client_name: str, profile_text: str, metadata: Dict = None):
        """Save client profile text for future reference."""
        profile_data = {
            'client_name': client_name,
            'profile_text': profile_text,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat()
        }
        
        filename = f"{client_name.lower().replace(' ', '_')}_profile.yaml"
        filepath = self.profiles_dir / filename
        
        with open(filepath, 'w') as f:
            yaml.dump(profile_data, f, default_flow_style=False)
        
        logger.info(f"Client profile saved: {filepath}")
    
    def load_client_profile(self, client_name: str) -> Dict:
        """Load saved client profile from profile files or IPS config."""
        # First try to load from profiles directory
        filename = f"{client_name.lower().replace(' ', '_')}_profile.yaml"
        filepath = self.profiles_dir / filename
        
        if filepath.exists():
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        
        # If not found, try loading from IPS config
        try:
            ips_config_path = Path("config/ips.yaml")
            if ips_config_path.exists():
                with open(ips_config_path, 'r') as f:
                    ips_data = yaml.safe_load(f)
                    if 'client' in ips_data and ips_data['client'].get('name') == client_name:
                        # Convert IPS client data to profile format
                        client_data = ips_data['client']
                        profile_text = self._generate_profile_text_from_ips(client_data)
                        return {
                            'client_name': client_name,
                            'profile_text': profile_text,
                            'ips_data': client_data,
                            'source': 'ips_config'
                        }
        except Exception as e:
            logger.warning(f"Could not load from IPS config: {e}")
        
        return {}
    
    def list_client_profiles(self) -> list:
        """List all saved client profiles, including from IPS config."""
        profiles = []
        
        # First, check for profiles in the profiles directory
        for file in self.profiles_dir.glob("*_profile.yaml"):
            try:
                with open(file, 'r') as f:
                    data = yaml.safe_load(f)
                    profiles.append({
                        'filename': file.name,
                        'client_name': data.get('client_name', 'Unknown'),
                        'created_at': data.get('created_at', 'Unknown'),
                        'source': 'profile_file'
                    })
            except Exception as e:
                logger.warning(f"Could not load profile {file}: {e}")
        
        # Also check for client data in IPS config
        try:
            ips_config_path = Path("config/ips.yaml")
            if ips_config_path.exists():
                with open(ips_config_path, 'r') as f:
                    ips_data = yaml.safe_load(f)
                    if 'client' in ips_data and 'name' in ips_data['client']:
                        client_name = ips_data['client']['name']
                        profiles.append({
                            'filename': 'ips.yaml',
                            'client_name': client_name,
                            'created_at': 'From IPS Config',
                            'source': 'ips_config'
                        })
                        logger.info(f"Found client profile in IPS config: {client_name}")
        except Exception as e:
            logger.warning(f"Could not load IPS config: {e}")
        
        return profiles
    
    def extract_key_constraints(self, profile_text: str) -> Dict[str, Any]:
        """
        Extract key constraints from profile text using simple keyword matching.
        This is a fallback if AI parsing is not available.
        """
        text_lower = profile_text.lower()
        constraints = {}
        
        # Risk tolerance
        if any(word in text_lower for word in ['conservative', 'low risk', 'risk-averse']):
            constraints['risk_tolerance'] = 'low'
        elif any(word in text_lower for word in ['aggressive', 'high risk', 'growth']):
            constraints['risk_tolerance'] = 'high'
        else:
            constraints['risk_tolerance'] = 'moderate'
        
        # Exclusions
        exclusions = []
        if 'tobacco' in text_lower:
            exclusions.append('tobacco')
        if any(word in text_lower for word in ['weapons', 'defense', 'military']):
            exclusions.append('weapons')
        if any(word in text_lower for word in ['fossil fuel', 'oil', 'coal']):
            exclusions.append('fossil_fuels')
        
        constraints['exclusions'] = exclusions
        
        # Time horizon (look for patterns like "5 year", "10-year", etc.)
        import re
        time_match = re.search(r'(\d+)[-\s]?year', text_lower)
        if time_match:
            constraints['time_horizon_years'] = int(time_match.group(1))
        
        # Position limits (look for percentages)
        position_match = re.search(r'(?:position|holding).*?(\d+)%', text_lower)
        if position_match:
            constraints['max_position_pct'] = int(position_match.group(1))
        
        sector_match = re.search(r'sector.*?(\d+)%', text_lower)
        if sector_match:
            constraints['max_sector_pct'] = int(sector_match.group(1))
        
        return constraints
    
    def _generate_profile_text_from_ips(self, client_data: Dict) -> str:
        """Generate detailed profile text from IPS client data with specific metrics."""
        profile_parts = []
        
        # Basic client information with specific details
        profile_parts.append(f"Client: {client_data.get('name', 'Unknown')}")
        if 'organization' in client_data:
            profile_parts.append(f"Organization: {client_data['organization']}")
        if 'background' in client_data:
            profile_parts.append(f"Background: {client_data['background']}")
        
        # Financial specifics with exact numbers
        if 'initial_capital' in client_data:
            profile_parts.append(f"Initial Capital: ${client_data['initial_capital']:,}")
        if 'target_value' in client_data:
            profile_parts.append(f"Target Portfolio Value: ${client_data['target_value']:,}")
        if 'required_growth_rate' in client_data:
            profile_parts.append(f"Required Annual Growth Rate: {client_data['required_growth_rate']}%")
        
        # Risk tolerance with specific metrics
        risk_tolerance = client_data.get('risk_tolerance', 'moderate')
        profile_parts.append(f"Risk Tolerance: {risk_tolerance}")
        
        # Time horizon specifics
        if 'time_horizon_years' in client_data:
            profile_parts.append(f"Investment Time Horizon: {client_data['time_horizon_years']} years")
        
        # Drawdown specifics
        if 'drawdown_start_year' in client_data and 'annual_drawdown' in client_data:
            profile_parts.append(f"Annual Drawdowns: ${client_data['annual_drawdown']:,} starting year {client_data['drawdown_start_year']}")
        if 'drawdown_growth_rate' in client_data:
            profile_parts.append(f"Drawdown Growth Rate: {client_data['drawdown_growth_rate']}% annually")
        
        # Mission and values specifics
        if 'foundation_mission' in client_data:
            profile_parts.append(f"Mission: {client_data['foundation_mission']}")
        if 'focus_areas' in client_data:
            focus_areas = ', '.join(client_data['focus_areas'])
            profile_parts.append(f"Focus Areas: {focus_areas}")
        if 'track_record' in client_data:
            profile_parts.append(f"Track Record: {client_data['track_record']}")
        if 'final_project_target' in client_data:
            profile_parts.append(f"Final Project Goal: {client_data['final_project_target']}")
        
        # Cash management specifics
        if 'cash_buffer_pct' in client_data:
            profile_parts.append(f"Required Cash Buffer: {client_data['cash_buffer_pct']}%")
        
        return '\n'.join(profile_parts)
    
    def create_sample_profiles(self):
        """Create sample client profiles for testing."""
        
        # Conservative Retiree
        conservative_profile = """
        Client: Margaret Johnson (65 years old, retired teacher)
        
        Investment Objectives:
        - Capital preservation with modest growth
        - Generate steady income for retirement
        - Low volatility tolerance
        
        Constraints:
        - 5-year investment horizon
        - Conservative risk tolerance
        - Maximum 5% in any single position
        - Maximum 25% in any sector
        - Minimum $5 stock price
        - Exclude tobacco and weapons companies
        - Prefer dividend-paying stocks
        - Target 10% cash buffer
        - Beta range: 0.6 to 1.0
        
        Preferences:
        - Favor utilities, consumer staples, healthcare
        - Avoid high-growth technology stocks
        - ESG considerations important
        """
        
        # Aggressive Growth Investor
        aggressive_profile = """
        Client: David Chen (28 years old, software engineer)
        
        Investment Objectives:
        - Aggressive capital appreciation
        - Long-term wealth building
        - High growth potential
        
        Constraints:
        - 15-year investment horizon
        - High risk tolerance
        - Maximum 10% in any single position
        - Maximum 40% in any sector
        - No sector exclusions
        - Comfortable with high volatility
        - Target 5% cash buffer
        - Beta range: 0.8 to 1.5
        
        Preferences:
        - Focus on technology, healthcare, emerging sectors
        - Growth over value
        - International exposure acceptable
        - ESG not a primary concern
        """
        
        # Balanced Family Investor
        balanced_profile = """
        Client: The Rodriguez Family (dual income, 2 children)
        
        Investment Objectives:
        - Balanced growth and income
        - Education funding for children
        - Long-term wealth accumulation
        
        Constraints:
        - 10-year investment horizon
        - Moderate risk tolerance
        - Maximum 8% in any single position
        - Maximum 30% in any sector
        - Exclude tobacco, weapons, and fossil fuels
        - Minimum $3 stock price
        - Target 8% cash buffer
        - Beta range: 0.7 to 1.2
        
        Preferences:
        - Diversified across sectors
        - Some ESG considerations
        - Prefer established companies
        - Mix of growth and dividend stocks
        """
        
        # Save sample profiles
        self.save_client_profile("Conservative Retiree", conservative_profile)
        self.save_client_profile("Aggressive Growth", aggressive_profile)
        self.save_client_profile("Balanced Family", balanced_profile)
        
        logger.info("Sample client profiles created")


# Example usage and templates
PROFILE_TEMPLATES = {
    "Conservative": {
        "description": "Low risk, income-focused, capital preservation",
        "template": """
Client Information:
- Age: [age]
- Risk Tolerance: Conservative/Low
- Investment Horizon: [X] years
- Primary Objective: Capital preservation and income

Constraints:
- Maximum [X]% per position
- Maximum [X]% per sector  
- Minimum $[X] stock price
- Exclude: [sectors/companies]
- Cash buffer: [X]%
- Beta range: [X] to [X]

Preferences:
- Dividend-paying stocks preferred
- Stable, established companies
- [Any specific sector preferences]
        """
    },
    
    "Moderate": {
        "description": "Balanced growth and income, moderate risk",
        "template": """
Client Information:
- Age: [age]
- Risk Tolerance: Moderate
- Investment Horizon: [X] years
- Primary Objective: Balanced growth and income

Constraints:
- Maximum [X]% per position
- Maximum [X]% per sector
- Any exclusions: [list]
- Cash buffer: [X]%
- Beta range: [X] to [X]

Preferences:
- Mix of growth and value stocks
- Diversified across sectors
- [Any ESG considerations]
        """
    },
    
    "Aggressive": {
        "description": "High growth, high risk tolerance",
        "template": """
Client Information:
- Age: [age]
- Risk Tolerance: High/Aggressive
- Investment Horizon: [X] years
- Primary Objective: Capital appreciation/growth

Constraints:
- Maximum [X]% per position
- Maximum [X]% per sector
- Comfortable with volatility
- Cash buffer: [X]%
- Beta range: [X] to [X]

Preferences:
- Growth stocks preferred
- Technology/innovation focus acceptable
- [Sector preferences]
        """
    }
}