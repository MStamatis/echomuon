"""EchoMuon: Muon with a per-direction temporal trust gate and a
memorization-gap controller. Better than scheduled Muon wherever data are
imperfect; ties it everywhere else."""
from .optimizer import EchoMuon, MemorizationGapController, newton_schulz5

__version__ = "0.1.0"
__all__ = ["EchoMuon", "MemorizationGapController", "newton_schulz5", "__version__"]
