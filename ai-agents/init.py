from .core.coordinator import coordinator

# Import all agents
from .character_creation.agent import CharacterCreationAgent
from .environment_creation.agent import EnvironmentCreationAgent
from .voice_communication.agent import VoiceCommunicationAgent
from .weapon_creation.agent import WeaponCreationAgent

# Create agent instances
character_agent = CharacterCreationAgent()
environment_agent = EnvironmentCreationAgent()
voice_agent = VoiceCommunicationAgent()
weapon_agent = WeaponCreationAgent()

# Register agents with coordinator
coordinator.register_agent('character', character_agent)
coordinator.register_agent('environment', environment_agent)
coordinator.register_agent('voice', voice_agent)
coordinator.register_agent('weapon', weapon_agent)

__all__ = [
    'coordinator', 
    'character_agent', 
    'environment_agent', 
    'voice_agent', 
    'weapon_agent'
]
