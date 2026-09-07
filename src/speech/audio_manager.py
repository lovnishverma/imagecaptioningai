import threading
import os
from collections import deque
from typing import Tuple, Optional

class AudioQueue:
    """Thread-safe FIFO audio queue with interruption support."""

    def __init__(self, max_size: int = 3):
        self._queue: deque[Tuple[str, str]] = deque()  # (text, audio_path)
        self._current: Optional[str] = None
        self._lock = threading.Lock()
        self._counter = 0
        self._max_size = max_size

    def enqueue(self, text: str, audio_path: str) -> Optional[str]:
        """Add audio to queue. Returns the path to play (or None if queue full)."""
        with self._lock:
            if len(self._queue) >= self._max_size:
                oldest = self._queue.popleft()
                self._safe_delete(oldest[1])
            self._queue.append((text, audio_path))
            self._counter += 1
            return audio_path

    def dequeue(self) -> Optional[Tuple[str, str]]:
        """Get next audio item."""
        with self._lock:
            if self._queue:
                item = self._queue.popleft()
                self._current = item[1]
                return item
            return None

    def clear(self):
        """Clear all queued audio and delete files."""
        with self._lock:
            for _, path in self._queue:
                self._safe_delete(path)
            self._queue.clear()
            self._current = None

    def interrupt(self):
        """Interrupt current and clear queue."""
        self.clear()

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return len(self._queue) == 0

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._queue)

    @staticmethod
    def _safe_delete(path: str):
        try:
            if path and os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass
