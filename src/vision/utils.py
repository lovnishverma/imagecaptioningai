from PIL import Image, ImageEnhance
from src.config import CONFIG

def preprocess_image(image: Image.Image) -> Image.Image:
    """Resize image for inference while preserving aspect ratio."""
    w, h = image.size
    if max(w, h) <= CONFIG.MAX_DIM:
        return image
    scale = CONFIG.MAX_DIM / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    return image.resize(new_size, Image.LANCZOS)

def auto_enhance(image: Image.Image) -> Image.Image:
    """Auto-enhance image for better vision model performance."""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(1.1)
