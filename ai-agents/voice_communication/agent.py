import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import soundfile as sf
import numpy as np
from typing import Dict, Any, List, Optional
import asyncio
import aiohttp
import json
from datetime import datetime
import logging
from pathlib import Path
import re

class VoiceCommunicationAgent:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        self.current_task = None
        self.voice_db = {}
        self.logger = self.setup_logging()
        
        # Initialize models
        self.setup_models()
    
    def setup_logging(self):
        """Setup logging for the agent"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/voice_agent.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('VoiceCommunicationAgent')
    
    def setup_models(self):
        """Initialize AI models for voice generation"""
        try:
            self.logger.info("Loading voice communication models...")
            
            # Load text generation model
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.lm_model = GPT2LMHeadModel.from_pretrained("gpt2")
            self.lm_model.to(self.device)
            
            # Load text-to-speech model
            self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            self.tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            
            # Move models to device
            self.tts_model.to(self.device)
            self.vocoder.to(self.device)
            
            self.model_loaded = True
            self.logger.info("Voice communication models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            self.model_loaded = False
    
    async def initialize(self):
        """Initialize the agent"""
        self.logger.info("Initializing VoiceCommunicationAgent...")
        
        # Load existing voice data from database
        await self.load_voice_database()
        
        # Load voice profiles
        await self.load_voice_profiles()
        
        self.logger.info("VoiceCommunicationAgent initialized successfully")
    
    async def load_voice_database(self):
        """Load voice database from file"""
        try:
            db_path = Path("data/voice/voice_db.json")
            if db_path.exists():
                with open(db_path, 'r') as f:
                    self.voice_db = json.load(f)
                self.logger.info(f"Loaded {len(self.voice_db)} voice entries from database")
            else:
                self.logger.info("No existing voice database found")
        except Exception as e:
            self.logger.error(f"Error loading voice database: {str(e)}")
    
    async def load_voice_profiles(self):
        """Load voice profiles from file"""
        try:
            profiles_path = Path("data/voice/voice_profiles.json")
            if profiles_path.exists():
                with open(profiles_path, 'r') as f:
                    self.voice_profiles = json.load(f)
                self.logger.info(f"Loaded {len(self.voice_profiles)} voice profiles")
            else:
                # Create default voice profiles
                self.voice_profiles = self.create_default_voice_profiles()
                await self.save_voice_profiles()
        except Exception as e:
            self.logger.error(f"Error loading voice profiles: {str(e)}")
            self.voice_profiles = self.create_default_voice_profiles()
    
    def create_default_voice_profiles(self) -> Dict[str, Any]:
        """Create default voice profiles"""
        return {
            'deep_assertive': {
                'name': 'Deep Assertive',
                'description': 'Confident and commanding voice',
                'pitch': 0.8,
                'speed': 1.0,
                'energy': 0.9,
                'emotion': 'confident'
            },
            'calm_collected': {
                'name': 'Calm Collected',
                'description': 'Calm and measured voice',
                'pitch': 0.6,
                'speed': 0.9,
                'energy': 0.7,
                'emotion': 'calm'
            },
            'articulate_precise': {
                'name': 'Articulate Precise',
                'description': 'Clear and precise voice',
                'pitch': 0.7,
                'speed': 1.1,
                'energy': 0.8,
                'emotion': 'professional'
            },
            'neutral_authoritative': {
                'name': 'Neutral Authoritative',
                'description': 'Balanced and authoritative voice',
                'pitch': 0.7,
                'speed': 1.0,
                'energy': 0.8,
                'emotion': 'neutral'
            }
        }
    
    async def save_voice_database(self):
        """Save voice database to file"""
        try:
            db_path = Path("data/voice/voice_db.json")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(db_path, 'w') as f:
                json.dump(self.voice_db, f, indent=2)
            
            self.logger.info("Voice database saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving voice database: {str(e)}")
    
    async def save_voice_profiles(self):
        """Save voice profiles to file"""
        try:
            profiles_path = Path("data/voice/voice_profiles.json")
            profiles_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(profiles_path, 'w') as f:
                json.dump(self.voice_profiles, f, indent=2)
            
            self.logger.info("Voice profiles saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving voice profiles: {str(e)}")
    
    async def generate_dialogue(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate dialogue for a character"""
        self.current_task = "generate_dialogue"
        
        try:
            character_id = params['character_id']
            personality = params.get('personality', {})
            situation = params.get('situation', 'combat')
            voice_type = params.get('voice_type', 'neutral_authoritative')
            
            # Generate dialogue lines
            dialogue_lines = await self.generate_dialogue_lines(character_id, personality, situation)
            
            # Generate voice audio for each line
            voice_files = []
            for i, line in enumerate(dialogue_lines):
                audio_path = await self.generate_voice_audio(line, voice_type, character_id, i)
                voice_files.append({
                    'text': line,
                    'audio_path': audio_path,
                    'situation': situation,
                    'voice_type': voice_type
                })
            
            # Save to database
            dialogue_id = self.generate_dialogue_id(character_id, situation)
            self.voice_db[dialogue_id] = {
                'character_id': character_id,
                'situation': situation,
                'voice_type': voice_type,
                'lines': voice_files,
                'created_at': datetime.now().isoformat()
            }
            
            await self.save_voice_database()
            
            return {
                'success': True,
                'dialogue_id': dialogue_id,
                'dialogue': voice_files
            }
            
        except Exception as e:
            self.logger.error(f"Error generating dialogue: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            self.current_task = None
    
    async def generate_dialogue_lines(self, character_id: str, personality: Dict[str, Any], situation: str) -> List[str]:
        """Generate dialogue lines based on character personality and situation"""
        # Define dialogue templates based on situation
        dialogue_templates = {
            'combat': [
                "Enemy spotted at {location}!",
                "Taking fire! Need support!",
                "Target eliminated.",
                "Reloading!",
                "Moving to {location}.",
                "Cover me!",
                "Throwing grenade!",
                "Enemy down!",
                "I'm hit!",
                "Medic! I need a medic!"
            ],
            'stealth': [
                "Area clear, moving forward.",
                "I've got visual on the target.",
                "Stay quiet, enemies nearby.",
                "Taking the silent approach.",
                "Target acquired, awaiting orders.",
                "No detection so far.",
                "I'll take the lead.",
                "Watch my back.",
                "Going dark.",
                "Maintaining radio silence."
            ],
            'casual': [
                "What's the plan, team?",
                "I've seen worse situations.",
                "Remember our training.",
                "Stay focused, team.",
                "Let's get this done.",
                "Copy that.",
                "Affirmative.",
                "Negative.",
                "Roger that.",
                "Standing by."
            ],
            'mission_briefing': [
                "Team, listen up. Our objective is to {objective}.",
                "Intel suggests the target is located at {location}.",
                "We expect heavy resistance in the {area} area.",
                "Extraction will be at {extraction_point}.",
                "Remember, {important_instruction}.",
                "Any questions?",
                "Let's move out!"
            ]
        }
        
        # Get template for the current situation
        templates = dialogue_templates.get(situation, dialogue_templates['casual'])
        
        # Generate specific lines using language model
        generated_lines = []
        for template in templates:
            # Fill in template variables
            line = template.format(
                location=self.generate_location(),
                target=self.generate_target(),
                objective=self.generate_objective(situation),
                area=self.generate_area(),
                extraction_point=self.generate_extraction_point(),
                important_instruction=self.generate_important_instruction()
            )
            
            # Add personality flavor using language model
            personality_prompt = f"A {personality.get('description', 'special forces operator')} says: {line}"
            
            inputs = self.tokenizer.encode(personality_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.lm_model.generate(
                    inputs, 
                    max_length=50, 
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_line = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the dialogue part
            if "says:" in generated_line:
                generated_line = generated_line.split("says:")[1].strip()
            
            generated_lines.append(generated_line)
        
        return generated_lines
    
    def generate_location(self) -> str:
        """Generate a random location reference"""
        locations = [
            "my position", "12 o'clock", "the north side", 
            "the compound", "the building", "the treeline",
            "the ridge", "the valley", "the riverbank",
            "the west entrance", "the east wing", "the command center"
        ]
        return random.choice(locations)
    
    def generate_target(self) -> str:
        """Generate a random target reference"""
        targets = [
            "the objective", "hostages", "the package",
            "the leader", "the device", "the intel",
            "the weapons cache", "the vehicle", "the communications array"
        ]
        return random.choice(targets)
    
    def generate_objective(self, situation: str) -> str:
        """Generate a mission objective"""
        objectives = {
            'combat': "neutralize all hostile forces in the area",
            'stealth': "infiltrate the facility without detection",
            'extraction': "extract the VIP from the hot zone",
            'defense': "defend the position until reinforcements arrive",
            'recon': "gather intelligence on enemy movements"
        }
        return objectives.get(situation, "complete the mission")
    
    def generate_area(self) -> str:
        """Generate a random area reference"""
        areas = [
            "northern", "southern", "eastern", "western",
            "central", "upper", "lower", "main",
            "secondary", "primary", "restricted", "secure"
        ]
        return random.choice(areas) + " " + random.choice(["sector", "zone", "area", "compound"])
    
    def generate_extraction_point(self) -> str:
        """Generate a random extraction point"""
        points = [
            "LZ Alpha", "LZ Bravo", "RV Point Delta",
            "the extraction zone", "the rally point",
            "the safe house", "the fallback position"
        ]
        return random.choice(points)
    
    def generate_important_instruction(self) -> str:
        """Generate an important instruction"""
        instructions = [
            "minimize collateral damage",
            "avoid civilian casualties",
            "maintain radio silence",
            "watch for booby traps",
            "check your fire",
            "stay in formation",
            "conserve ammunition",
            "prioritize the mission objective"
        ]
        return random.choice(instructions)
    
    async def generate_voice_audio(self, text: str, voice_type: str, character_id: str, index: int) -> str:
        """Generate voice audio for the given text"""
        if not self.model_loaded:
            raise Exception("Models not loaded")
        
        # Preprocess text
        inputs = self.processor(text=text, return_tensors="pt")
        
        # Get voice profile
        voice_profile = self.voice_profiles.get(voice_type, self.voice_profiles['neutral_authoritative'])
        
        # Load speaker embeddings (in a real scenario, this would be character-specific)
        # Here we use a random embedding for demonstration
        speaker_embeddings = torch.randn((1, 512)).to(self.device)
        
        # Adjust embeddings based on voice profile
        speaker_embeddings = self.adjust_embeddings_for_profile(speaker_embeddings, voice_profile)
        
        # Generate speech
        with torch.no_grad():
            speech = self.tts_model.generate_speech(
                inputs["input_ids"].to(self.device), 
                speaker_embeddings, 
                vocoder=self.vocoder
            )
        
        # Convert to numpy array and save as WAV file
        speech = speech.cpu().numpy()
        audio_path = f"assets/audio/voice/{character_id}/line_{index}.wav"
        
        # Ensure directory exists
        Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save audio file
        sf.write(audio_path, speech, 22050)
        
        return audio_path
    
    def adjust_embeddings_for_profile(self, embeddings: torch.Tensor, profile: Dict[str, Any]) -> torch.Tensor:
        """Adjust speaker embeddings based on voice profile"""
        # Adjust pitch
        pitch_factor = profile.get('pitch', 0.7)
        embeddings = embeddings * (0.5 + pitch_factor)
        
        # Adjust speed (would affect generation parameters, not embeddings directly)
        # For now, we'll just return the modified embeddings
        return embeddings
    
    async def generate_mission_briefing(self, mission_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate voice audio for mission briefing"""
        briefing_text = self.create_briefing_text(mission_data)
        audio_path = await self.generate_voice_audio(briefing_text, "neutral_authoritative", "command", 0)
        
        # Save to database
        briefing_id = f"briefing_{int(datetime.now().timestamp())}"
        self.voice_db[briefing_id] = {
            'type': 'mission_briefing',
            'mission_data': mission_data,
            'audio_path': audio_path,
            'created_at': datetime.now().isoformat()
        }
        
        await self.save_voice_database()
        
        return {
            'success': True,
            'briefing_id': briefing_id,
            'audio_path': audio_path
        }
    
    def create_briefing_text(self, mission_data: Dict[str, Any]) -> str:
        """Create mission briefing text"""
        mission_type = mission_data.get('type', 'infiltration')
        location = mission_data.get('location', 'unknown location')
        difficulty = mission_data.get('difficulty', 'medium')
        
        briefing_templates = {
            'infiltration': f"Team, your mission is to infiltrate the {location} and gather intelligence. Stay undetected and complete objectives silently. This is a {difficulty} difficulty operation.",
            'hostage_rescue': f"Team, we have hostages at {location}. Your mission is to rescue them with minimal casualties. Exercise extreme caution. This is a {difficulty} difficulty operation.",
            'assault': f"Team, your mission is to assault the {location} and neutralize all hostile forces. Use maximum force but watch for civilians. This is a {difficulty} difficulty operation.",
            'destruction': f"Team, your mission is to destroy the target at {location}. Use explosives and ensure complete destruction of the objective. This is a {difficulty} difficulty operation.",
            'extraction': f"Team, your mission is to extract the VIP from {location}. Get in, get the package, and get out. This is a {difficulty} difficulty operation."
        }
        
        return briefing_templates.get(mission_type, briefing_templates['infiltration'])
    
    def generate_dialogue_id(self, character_id: str, situation: str) -> str:
        """Generate a unique ID for dialogue"""
        timestamp = int(datetime.now().timestamp())
        return f"DLG_{character_id}_{situation}_{timestamp}"
    
    async def create_voice_profile(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a custom voice profile"""
        profile_id = params.get('id', f"profile_{len(self.voice_profiles) + 1}")
        
        voice_profile = {
            'name': params.get('name', 'Custom Voice'),
            'description': params.get('description', 'Custom voice profile'),
            'pitch': params.get('pitch', 0.7),
            'speed': params.get('speed', 1.0),
            'energy': params.get('energy', 0.8),
            'emotion': params.get('emotion', 'neutral')
        }
        
        self.voice_profiles[profile_id] = voice_profile
        await self.save_voice_profiles()
        
        return {
            'success': True,
            'profile_id': profile_id,
            'profile': voice_profile
        }
    
    async def get_voice_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """Get voice profile by ID"""
        return self.voice_profiles.get(profile_id)
    
    async def list_voice_profiles(self) -> Dict[str, Any]:
        """List all voice profiles"""
        return self.voice_profiles
    
    def get_status(self) -> str:
        """Get current agent status"""
        if not self.model_loaded:
            return "offline"
        return "busy" if self.current_task else "idle"
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities"""
        return [
            "dialogue_generation",
            "voice_synthesis",
            "mission_briefing_creation",
            "voice_profile_management"
        ]
    
    async def shutdown(self):
        """Cleanup resources"""
        self.logger.info("Shutting down VoiceCommunicationAgent...")
        
        # Clear model resources
        if hasattr(self, 'lm_model'):
            del self.lm_model
        if hasattr(self, 'tts_model'):
            del self.tts_model
        if hasattr(self, 'vocoder'):
            del self.vocoder
        
        torch.cuda.empty_cache()
        self.logger.info("VoiceCommunicationAgent shutdown complete")
