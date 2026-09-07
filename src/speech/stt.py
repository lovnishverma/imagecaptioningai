class STTEngine:
    """Abstract interface for Speech-to-Text."""
    
    def load(self):
        """Initialize the STT model/provider."""
        pass
        
    def transcribe(self, audio_file: str) -> str:
        """Transcribe an audio file into text."""
        return ""

class MockSTT(STTEngine):
    """Fallback mock STT since we are focusing on UI/Vision first."""
    def transcribe(self, audio_file: str) -> str:
        return "describe the scene"

# Future integration with faster-whisper goes here.
