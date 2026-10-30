"""Names that must never appear in data or answers (employers and clients).

They are stored as SHA-256 hashes of the normalized word, so this public repository does not
contain the very names it protects. Java analogy: storing a password hash, not the password.
The check normalizes text the same way (lowercase, no accents) and hashes every word.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable

PRIVATE_NAME_HASHES: frozenset[str] = frozenset(
    {
        "0eeb42a84a8f0e22a20a18ad6aa94e6cb091d4bd81972b7c2041522388c6de8f",
        "b90f4994c4675589617d546a2364d97a26dd97a57ca5983598b2b68d7899bbcd",
        "d0e2ece166bdef23adedfaa3f1c47c9f969270f0fc519185514d58b4c7f6d199",
        "19c0d6808ba1641553b8ad0a65754000ed715bf61b3b4592364046900f2cdfac",
        "cdefdfc1b5f1c676af2d6d9eae8b2bc6092ebc55d8c32ee8825ccce80cacfca8",
        "6d1dd69c34a729841e89654b309814455deeba1fa5cca5db8e1220d52064d2cb",
        "7d106deafe004a5d01d4b4d0c7098d337de8f766937b52bc81125c5002b80b05",
        "389a86fc39c88061089e8a04efd5b327ed5139a266b36713299cebfc13262280",
        "dd3f10c884b075a1c9a71d8c63239ca10c0e9f982c15b906aefb9536190927f0",
    }
)


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()


def word_hash(word: str) -> str:
    return hashlib.sha256(normalize(word).encode()).hexdigest()


def private_names_in(text: str, hashes: Iterable[str] = PRIVATE_NAME_HASHES) -> list[str]:
    """Words of `text` whose hash is in the private list (empty list = clean)."""
    blocked = set(hashes)
    return sorted({w for w in re.findall(r"[a-z0-9]+", normalize(text)) if word_hash(w) in blocked})
