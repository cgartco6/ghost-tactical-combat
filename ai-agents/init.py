from .core.coordinator import coordinator

# Import all agents
from .character_creation.agent import CharacterCreationAgent

# Create agent instances
character_agent = CharacterCreationAgent()

# Register agents with coordinator
coordinator.register_agent('character', character_agent)

__all__ = ['coordinator', 'character_agent']
