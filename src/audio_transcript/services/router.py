"""Provider routing and key pool management."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Dict, List, Optional

from ..domain.models import ProviderQuotaState
from ..infra.runtime_state import RuntimeState


def mask_key(key: str) -> str:
    """Mask an API key for status surfaces."""
    if len(key) > 12:
        return f"{key[:8]}***{key[-6:]}"
    return f"{key[:4]}***"


@dataclass
class KeyState:
    """Runtime state for a provider key."""

    raw_key: str
    key_id: str
    cooldown_until: Optional[datetime] = None
    last_error: Optional[str] = None

    @property
    def available(self) -> bool:
        if self.cooldown_until is None:
            return True
        return datetime.now(timezone.utc) >= self.cooldown_until


class ProviderKeyPool:
    """Round-robin keys within a provider."""

    def __init__(self, provider_name: str, keys: List[str], runtime_state: Optional[RuntimeState] = None):
        self.provider_name = provider_name
        self.runtime_state = runtime_state
        self._lock = Lock()
        self._index = 0
        self._states = [KeyState(raw_key=key, key_id=mask_key(key)) for key in keys]

    def acquire(self) -> Optional[KeyState]:
        with self._lock:
            if not self._states:
                return None
            for _ in range(len(self._states)):
                state = self._states[self._index]
                self._index = (self._index + 1) % len(self._states)
                if self.runtime_state:
                    cooldown_until = self.runtime_state.get_provider_cooldown(self.provider_name, state.key_id)
                    state.cooldown_until = cooldown_until
                    state.last_error = self.runtime_state.get_provider_error(self.provider_name, state.key_id)
                if state.available:
                    return state
            return None

    def cooldown(self, key_id: str, seconds: int, error: str) -> None:
        for state in self._states:
            if state.key_id == key_id:
                state.cooldown_until = datetime.now(timezone.utc) + timedelta(seconds=seconds)
                state.last_error = error
                if self.runtime_state:
                    self.runtime_state.set_provider_cooldown(self.provider_name, key_id, seconds, error)
                return

    def mark_error(self, key_id: str, error: str) -> None:
        for state in self._states:
            if state.key_id == key_id:
                state.last_error = error
                return

    def status(self) -> List[ProviderQuotaState]:
        if self.runtime_state:
            for state in self._states:
                state.cooldown_until = self.runtime_state.get_provider_cooldown(self.provider_name, state.key_id)
                state.last_error = self.runtime_state.get_provider_error(self.provider_name, state.key_id)
        return [
            ProviderQuotaState(
                provider=self.provider_name,
                key_id=state.key_id,
                available=state.available,
                cooldown_until=state.cooldown_until,
                last_error=state.last_error,
            )
            for state in self._states
        ]


class ProviderRouter:
    """Route across remote providers first and local fallback last."""

    def __init__(self, remote_provider_names: List[str]):
        self.remote_provider_names = remote_provider_names
        self._index = 0
        self._lock = Lock()

    def select_remote_order(self, available: Dict[str, bool]) -> List[str]:
        with self._lock:
            if not self.remote_provider_names:
                return []
            start = self._index
            self._index = (self._index + 1) % len(self.remote_provider_names)
            ordered = []
            for offset in range(len(self.remote_provider_names)):
                provider_name = self.remote_provider_names[(start + offset) % len(self.remote_provider_names)]
                if available.get(provider_name):
                    ordered.append(provider_name)
            return ordered
