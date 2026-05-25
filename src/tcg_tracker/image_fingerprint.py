"""Perceptual-hash fingerprint cache for card / sealed-box images.

When a user uploads a photo, we compute a perceptual hash (dHash by default)
and look it up in `card_image_fingerprints`. If we find a sufficiently close
match (Hamming distance ≤ threshold) we return the stored product metadata
immediately and skip the slow OCR + vision LLM pipeline entirely.

dHash chosen because:
  - Pure Python + Pillow + numpy (already deps), no heavyweight CV stack
  - ~5ms per image, 64-bit hash → 16 hex chars stored as TEXT
  - Robust to scaling / compression — fine for matching phone-camera photos
    against clean product thumbnails
  - Strict identity matching threshold (Hamming ≤ 5) works well in production
    for TCG/sneaker product detection (per benhoyt.com/writings/duplicate-image-detection)
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# Empirical thresholds from research:
#   ≤ 5 → strict identity (same printing, same crop)
#   ≤ 8 → variant tolerance (same product, different angle/lighting)
#   ≥ 10 → likely different product
HAMMING_STRICT = 5
HAMMING_RELAXED = 8
DEFAULT_ALGO = "dhash"


@dataclass(frozen=True, slots=True)
class FingerprintMatch:
    record: "CardImageFingerprint"
    hamming: int
    confidence: str  # "strict" / "relaxed"


def compute_dhash(image_bytes: bytes | None = None, *, image_path: Path | None = None) -> str | None:
    """Return a 16-hex-char dHash string. Raises nothing — None on failure."""
    import imagehash
    from PIL import Image
    try:
        if image_bytes is not None:
            img = Image.open(io.BytesIO(image_bytes))
        elif image_path is not None:
            img = Image.open(image_path)
        else:
            return None
        # dhash returns an ImageHash object whose str() is the hex form
        h = imagehash.dhash(img.convert("RGB"))
        return str(h)
    except Exception as exc:
        logger.warning("compute_dhash failed: %s", exc)
        return None


def hamming_distance(a: str, b: str) -> int:
    """Hamming distance between two equal-length hex strings.
    Counts bit differences. Returns the max int if lengths differ."""
    if len(a) != len(b):
        return 64  # treat mismatched-length as "no match"
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except ValueError:
        return 64


def nearest_fingerprints(
    candidates: Iterable["CardImageFingerprint"],
    target_hash: str,
    *,
    hamming_max: int = HAMMING_STRICT,
    limit: int = 3,
) -> list[FingerprintMatch]:
    """Brute-force scan + Hamming distance compare. Fast enough for <100k
    rows (each comparison is a 16-char int XOR + popcount); BK-tree only
    needed past that scale."""
    if not target_hash:
        return []
    matches: list[FingerprintMatch] = []
    for cand in candidates:
        if not cand.perceptual_hash:
            continue
        d = hamming_distance(cand.perceptual_hash, target_hash)
        if d <= hamming_max:
            confidence = "strict" if d <= HAMMING_STRICT else "relaxed"
            matches.append(FingerprintMatch(record=cand, hamming=d, confidence=confidence))
    matches.sort(key=lambda m: m.hamming)
    return matches[:limit]


# Re-export for callers (avoids circular import on the dataclass)
from market_monitor.models import CardImageFingerprint  # noqa: E402
