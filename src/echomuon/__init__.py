"""EchoMuon: Muon with a per-direction cross-timescale trust gate and an
optional memorization-gap controller.

Experimental research code. The regime of benefit is narrow and audited, and
several claims made by earlier versions of this package have since been
withdrawn. Read the README before using it."""
from .optimizer import EchoMuon, MemorizationGapController, newton_schulz5

__version__ = "0.3.0"
__all__ = ["EchoMuon", "MemorizationGapController", "newton_schulz5", "__version__"]
