"""Memory system for persistent agent memory.

For backward compatibility, this file continues to export the original
MemoryStore class. The implementation has been moved to the memory/
subpackage for better organization.

New code should import from nanobot.agent.memory (the package) instead
of this file, but both will continue to work.
"""

from nanobot.agent.memory.base import MemoryStore as _MemoryStoreABC
from nanobot.agent.memory.store import LegacyMemoryStore
from nanobot.agent.memory.store import _SAVE_MEMORY_TOOL

# Backward compatibility: LegacyMemoryStore is exported as MemoryStore
# This ensures existing code continues to work without modification
MemoryStore = LegacyMemoryStore

__all__ = ["MemoryStore", "LegacyMemoryStore", "_SAVE_MEMORY_TOOL"]
