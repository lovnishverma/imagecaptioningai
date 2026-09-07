from PIL import Image
from typing import Optional

def compute_hash(image: Image.Image, size: int = 16) -> bytes:
    """Compute difference hash (dHash) for scene change detection."""
    gray = image.resize((size + 1, size), Image.LANCZOS).convert("L")
    pixels = list(gray.getdata())
    return bytes(
        1 if pixels[y * (size + 1) + x] > pixels[y * (size + 1) + x + 1] else 0
        for y in range(size)
        for x in range(size)
    )

def hash_distance(a: Optional[bytes], b: Optional[bytes]) -> float:
    """Compute normalized Hamming distance between two hashes."""
    if a is None or b is None:
        return 1.0
    if len(a) != len(b):
        return 1.0
    return sum(x != y for x, y in zip(a, b)) / len(a)
