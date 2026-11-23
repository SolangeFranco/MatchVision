"""Appearance-based identity association."""

from matchvision.reidentification.appearance import PersistentIdentityAssociator
from matchvision.reidentification.base import IdentifiedObject, IdentityAssigner

__all__ = [
    "IdentifiedObject",
    "IdentityAssigner",
    "PersistentIdentityAssociator",
]
