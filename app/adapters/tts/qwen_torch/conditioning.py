"""In-memory LRU of Qwen3-TTS clone prompts, used only on the inference thread by the torch runtime."""

from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import Any


class VoiceConditioningCache:
    """Derived, disposable conditioning keyed by (voice_version_id, tts_backend, checkpoint)."""

    def __init__(self, max_entries: int) -> None:
        """Start empty with a fixed capacity."""
        self._max_entries = max_entries
        self._entries: OrderedDict[Hashable, Any] = OrderedDict()

    def get_or_create(self, key: Hashable, factory: Callable[[], Any]) -> Any:
        """Return the cached value for key, building and caching it on a miss."""
        if key in self._entries:
            self._entries.move_to_end(key)
            return self._entries[key]
        value = factory()
        self._entries[key] = value
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
        return value

    def clear(self) -> None:
        """Drop every entry."""
        self._entries.clear()

    def __len__(self) -> int:
        """Return the number of cached entries."""
        return len(self._entries)
