"""SAFI-Net: semantic-guided asynchronous infrared-visible video fusion."""

from .types import AsyncOutput, SOURCE_KEYS, TemporalInputs
from .models.model import SAFINet

__all__ = ['SAFINet', 'AsyncOutput', 'SOURCE_KEYS', 'TemporalInputs']
__version__ = '0.1.0'
