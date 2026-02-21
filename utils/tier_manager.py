"""
API Key Manager for the Investment Analysis Platform.

Resolves API keys from multiple sources with a clear precedence:
  session-provided > Streamlit Secrets > environment variables.
"""

import os
import streamlit as st
from typing import Optional


class TierManager:
    """Resolves API keys from session state, Streamlit Secrets, or env vars."""

    def __init__(self):
        if 'user_api_keys' not in st.session_state:
            st.session_state.user_api_keys = {}

    def get_api_key(self, key_name: str) -> Optional[str]:
        """Get API key with precedence: session-provided > Streamlit Secrets > env.

        Args:
            key_name: Name of the API key (e.g. 'OPENAI_API_KEY').

        Returns:
            The resolved key string, or None if not found anywhere.
        """
        # 1. Session-provided keys (e.g. set programmatically)
        user_keys = st.session_state.get('user_api_keys', {})
        if user_keys.get(key_name):
            return user_keys[key_name]

        # 2. Streamlit Secrets (secrets.toml or Streamlit Cloud)
        try:
            return st.secrets[key_name]
        except (FileNotFoundError, KeyError, AttributeError):
            pass

        # 3. Environment variable (.env via python-dotenv)
        return os.getenv(key_name)
