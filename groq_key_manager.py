#!/usr/bin/env python3
"""
Groq API Key Manager with Round-Robin Rotation

Provides thread-safe round-robin key rotation with rate limit tracking.
"""

import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

import requests
from dotenv import load_dotenv


@dataclass
class KeyState:
    """Tracks state and rate limits for a single API key."""
    key: str
    masked_key: str = ""
    requests_remaining: Optional[int] = None
    requests_limit: Optional[int] = None
    tokens_remaining: Optional[int] = None
    tokens_limit: Optional[int] = None
    reset_requests: Optional[datetime] = None
    reset_tokens: Optional[datetime] = None
    last_used: Optional[datetime] = None
    is_exhausted: bool = False
    exhaustion_until: Optional[datetime] = None
    
    def __post_init__(self):
        # Create masked version for display (e.g., gsk_***abc123)
        if self.key and not self.masked_key:
            if len(self.key) > 12:
                self.masked_key = self.key[:8] + "***" + self.key[-6:]
            else:
                self.masked_key = self.key[:4] + "***"


class RoundRobinKeyManager:
    """
    Thread-safe round-robin API key manager with rate limit tracking.
    
    Usage:
        manager = RoundRobinKeyManager()
        api_key = manager.get_key()
        # ... use key for API call ...
        manager.report_rate_limit_response(api_key, response.headers)
        if response.status_code == 429:
            manager.on_rate_limit(api_key)
    """
    
    GROQ_API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
    
    def __init__(self, api_keys: Optional[List[str]] = None):
        """
        Initialize the key manager.
        
        Args:
            api_keys: List of API keys. If None, loads from .env
        """
        self._lock = threading.Lock()
        self._keys: List[str] = []
        self._key_states: Dict[str, KeyState] = {}
        self._current_index = 0
        
        if api_keys:
            self._keys = api_keys
        else:
            self._load_keys_from_env()
        
        # Initialize state for each key
        for key in self._keys:
            self._key_states[key] = KeyState(key=key)
    
    def _load_keys_from_env(self) -> None:
        """Load API keys from .env file."""
        load_dotenv()
        
        # Check for multiple keys first
        keys_str = os.getenv("GROQ_API_KEYS", "")
        if keys_str:
            self._keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        
        # Fall back to single key (backward compatible)
        if not self._keys:
            single_key = os.getenv("GROQ_API_KEY", "")
            if single_key:
                self._keys = [single_key]
        
        if not self._keys:
            raise ValueError("No Groq API keys found in .env file. Set GROQ_API_KEYS or GROQ_API_KEY")
    
    @property
    def keys(self) -> List[str]:
        """Get list of all API keys."""
        return self._keys.copy()
    
    @property
    def num_keys(self) -> int:
        """Get number of configured API keys."""
        return len(self._keys)
    
    def get_key(self) -> Optional[str]:
        """
        Get the next API key in round-robin fashion.
        
        Returns:
            API key string, or None if all keys are exhausted.
        """
        with self._lock:
            if not self._keys:
                return None
            
            # Find next available (non-exhausted) key
            attempts = 0
            start_index = self._current_index
            
            while attempts < len(self._keys):
                key = self._keys[self._current_index]
                state = self._key_states[key]
                
                # Check if key is exhausted
                if state.is_exhausted and state.exhaustion_until:
                    if datetime.now(timezone.utc) >= state.exhaustion_until:
                        # Cooldown period has passed, reset exhaustion
                        state.is_exhausted = False
                        state.exhaustion_until = None
                    else:
                        # Key is still in cooldown, skip it
                        self._current_index = (self._current_index + 1) % len(self._keys)
                        attempts += 1
                        continue
                
                # Key is available
                state.last_used = datetime.now(timezone.utc)
                return key
            
            # All keys are exhausted
            return None
    
    def get_next_key(self) -> Optional[str]:
        """
        Advance to next key without getting current one.
        Useful for when you want to skip to next key immediately.
        
        Returns:
            Next available API key.
        """
        with self._lock:
            if not self._keys:
                return None
            
            # Advance index
            self._current_index = (self._current_index + 1) % len(self._keys)
            
            # Find next available key starting from new index
            attempts = 0
            start_index = self._current_index
            
            while attempts < len(self._keys):
                key = self._keys[self._current_index]
                state = self._key_states[key]
                
                # Check if key is exhausted
                if state.is_exhausted and state.exhaustion_until:
                    if datetime.now(timezone.utc) >= state.exhaustion_until:
                        state.is_exhausted = False
                        state.exhaustion_until = None
                    else:
                        self._current_index = (self._current_index + 1) % len(self._keys)
                        attempts += 1
                        continue
                
                state.last_used = datetime.now(timezone.utc)
                return key
            
            return None
    
    def report_rate_limit_response(self, key: str, headers: Dict[str, str]) -> None:
        """
        Update rate limit information from API response headers.
        
        Args:
            key: The API key that was used
            headers: Response headers from the API call
        """
        if key not in self._key_states:
            return
        
        state = self._key_states[key]
        
        # Parse rate limit headers
        if "x-ratelimit-remaining-requests" in headers:
            try:
                state.requests_remaining = int(headers["x-ratelimit-remaining-requests"])
            except (ValueError, TypeError):
                pass
        
        if "x-ratelimit-limit-requests" in headers:
            try:
                state.requests_limit = int(headers["x-ratelimit-limit-requests"])
            except (ValueError, TypeError):
                pass
        
        if "x-ratelimit-remaining-tokens" in headers:
            try:
                state.tokens_remaining = int(headers["x-ratelimit-remaining-tokens"])
            except (ValueError, TypeError):
                pass
        
        if "x-ratelimit-limit-tokens" in headers:
            try:
                state.tokens_limit = int(headers["x-ratelimit-limit-tokens"])
            except (ValueError, TypeError):
                pass
        
        if "x-ratelimit-reset-requests" in headers:
            state.reset_requests = self._parse_reset_time(headers["x-ratelimit-reset-requests"])
        
        if "x-ratelimit-reset-tokens" in headers:
            state.reset_tokens = self._parse_reset_time(headers["x-ratelimit-reset-tokens"])
    
    def _parse_reset_time(self, reset_str: str) -> Optional[datetime]:
        """
        Parse rate limit reset time string.
        
        Handles formats like:
        - "2h59.56s" (hours, minutes, seconds)
        - "59.56s" (seconds only)
        - "2m59.56s" (minutes and seconds)
        """
        if not reset_str:
            return None
        
        try:
            # Handle "XhYm.Zs" format
            reset_str = reset_str.strip()
            hours = 0
            minutes = 0
            seconds = 0.0
            
            if "h" in reset_str:
                parts = reset_str.split("h")
                hours = int(parts[0])
                reset_str = parts[1] if len(parts) > 1 else "0s"
            
            if "m" in reset_str:
                parts = reset_str.split("m")
                minutes = int(parts[0])
                reset_str = parts[1] if len(parts) > 1 else "0s"
            
            if "s" in reset_str:
                seconds = float(reset_str.replace("s", ""))
            
            total_seconds = hours * 3600 + minutes * 60 + seconds
            return datetime.now(timezone.utc).replace(microsecond=0).__add__(
                __import__("datetime").timedelta(seconds=total_seconds)
            )
        except (ValueError, AttributeError):
            return None
    
    def on_rate_limit(self, key: str, retry_after_seconds: Optional[int] = None) -> None:
        """
        Mark a key as rate-limited and schedule exhaustion.
        
        Args:
            key: The API key that was rate limited
            retry_after_seconds: Optional retry-after value from response
        """
        if key not in self._key_states:
            return
        
        state = self._key_states[key]
        
        # Use provided retry-after or default to 60 seconds
        wait_seconds = retry_after_seconds or 60
        
        state.is_exhausted = True
        state.exhaustion_until = datetime.now(timezone.utc).replace(microsecond=0)
        state.exhaustion_until = state.exhaustion_until.__add__(
            __import__("datetime").timedelta(seconds=wait_seconds)
        )
    
    def mark_key_exhausted(self, key: str, until: datetime) -> None:
        """
        Mark a key as exhausted until a specific time.
        
        Args:
            key: The API key
            until: When the exhaustion period ends
        """
        if key not in self._key_states:
            return
        
        state = self._key_states[key]
        state.is_exhausted = True
        state.exhaustion_until = until
    
    def reset_key(self, key: str) -> None:
        """
        Reset exhaustion state for a key.
        
        Args:
            key: The API key to reset
        """
        if key not in self._key_states:
            return
        
        state = self._key_states[key]
        state.is_exhausted = False
        state.exhaustion_until = None
    
    def get_quota_info(self) -> Dict[str, Any]:
        """
        Get quota information for all keys.
        
        Returns:
            Dictionary with quota info for each key
        """
        with self._lock:
            result = {
                "num_keys": len(self._keys),
                "keys": []
            }
            
            for key in self._keys:
                state = self._key_states[key]
                
                key_info = {
                    "key": state.masked_key,
                    "full_key": key,
                    "requests_remaining": state.requests_remaining,
                    "requests_limit": state.requests_limit,
                    "tokens_remaining": state.tokens_remaining,
                    "tokens_limit": state.tokens_limit,
                    "reset_requests": state.reset_requests.isoformat() if state.reset_requests else None,
                    "reset_tokens": state.reset_tokens.isoformat() if state.reset_tokens else None,
                    "is_exhausted": state.is_exhausted,
                    "exhaustion_until": state.exhaustion_until.isoformat() if state.exhaustion_until else None,
                    "last_used": state.last_used.isoformat() if state.last_used else None,
                }
                result["keys"].append(key_info)
            
            return result
    
    def check_quota_by_api_call(self) -> Dict[str, Any]:
        """
        Check quota for all keys by making a minimal API call.
        
        Uses the chat completions endpoint with a minimal request.
        
        Returns:
            Dictionary with quota info updated from API response headers
        """
        for key in self._keys:
            try:
                # Use a minimal API call to get rate limit headers
                # We'll use the models list endpoint which is lightweight
                response = requests.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=10
                )
                
                self.report_rate_limit_response(key, dict(response.headers))
                
            except requests.exceptions.RequestException:
                pass
        
        return self.get_quota_info()


def get_key_manager() -> RoundRobinKeyManager:
    """
    Factory function to create a key manager from .env.
    
    Returns:
        Configured RoundRobinKeyManager instance
    """
    return RoundRobinKeyManager()


if __name__ == "__main__":
    # Test basic functionality
    try:
        manager = RoundRobinKeyManager()
        print(f"Loaded {manager.num_keys} API keys")
        
        # Get first key
        key = manager.get_key()
        print(f"Got key: {manager._key_states[key].masked_key}")
        
        # Get quota info
        info = manager.get_quota_info()
        print(f"Quota info: {info}")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Please ensure you have GROQ_API_KEYS or GROQ_API_KEY set in your .env file")
