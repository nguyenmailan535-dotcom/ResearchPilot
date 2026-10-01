"""ResearchPilot public Python API."""

__version__ = "0.1.4.post6"
__logo__ = "🐈"

from nanobot.nanobot import Nanobot, RunResult

# Product-facing alias. ``Nanobot`` remains available for Python compatibility.
ResearchPilot = Nanobot

__all__ = ["Nanobot", "ResearchPilot", "RunResult"]
