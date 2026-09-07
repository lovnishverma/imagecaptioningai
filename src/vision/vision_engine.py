from abc import ABC, abstractmethod
from typing import Dict, Any
from PIL import Image

class VisionEngine(ABC):
    """Abstract interface for vision models."""

    @abstractmethod
    def load(self):
        """Load the vision model into memory."""
        pass

    @abstractmethod
    def describe_scene(self, image: Image.Image, detailed: bool = False) -> str:
        """Generate a description of the scene."""
        pass

    @abstractmethod
    def read_text(self, image: Image.Image) -> str:
        """Perform OCR and return formatted text."""
        pass

    @abstractmethod
    def analyze(self, image: Image.Image, task: str) -> str:
        """Run a raw generic analysis task on the image."""
        pass
