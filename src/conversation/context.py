from dataclasses import dataclass, field
import threading
import time
from typing import List, Dict, Any, Optional, Tuple

@dataclass
class SessionContext:
    """Thread-safe application state and conversational context."""

    # Scene hashing
    last_hash: Optional[bytes] = None
    last_task: str = ""
    last_text: str = ""
    last_audio: Optional[str] = None

    # Realtime
    realtime_active: bool = False
    last_capture_time: float = 0.0

    # History
    history: List[Dict[str, Any]] = field(default_factory=list)
    max_history: int = 50

    # Lock
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, hash_val: bytes, task: str, text: str, audio: Optional[str]):
        """Update state with new capture results."""
        with self._lock:
            self.last_hash = hash_val
            self.last_task = task
            self.last_text = text
            self.last_audio = audio
            
            # Add to history
            self.history.insert(0, {
                "time": time.strftime("%H:%M:%S"),
                "task": task,
                "text": text,
            })
            if len(self.history) > self.max_history:
                self.history = self.history[: self.max_history]

    def is_duplicate(self, hash_val: bytes, task: str) -> bool:
        """Check if this hash+task combination was already processed."""
        with self._lock:
            return (
                self.last_hash is not None
                and self.last_hash == hash_val
                and self.last_task == task
                and self.last_text != ""
            )

    def get_last(self) -> Tuple[str, Optional[str]]:
        """Get last description text and audio."""
        with self._lock:
            return self.last_text, self.last_audio

CONTEXT = SessionContext()
