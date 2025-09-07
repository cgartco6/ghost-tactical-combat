import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import soundfile as sf
import numpy as np
from typing import Dict, Any, List
import asyncio

class VoiceAgent:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        self.current_task = None
        self.setup_models()
    
    def setup_models(self):
        """Initialize AI models for voice generation"""
        try:
            # Load text generation model
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.lm_model = GPT2LMHeadModel.from_pretrained("gpt2")
            
            # Load text-to-speech model
            self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            self.tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            
            # Move models to device
            self.tts_model.to(self.device)
            self.vocoder.to(self.device)
            
            self.model_loaded = True
            print("Voice generation models loaded successfully")
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            self.model_loaded = False
    
    async def generate_dialogue(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate dialogue for a character"""
        self.current_task = "generate_dialogue"
        
        try:
            character_id = params['character_id']
            personality = params.get('personality', {})
            situation = params.get('situation', 'combat')
            
            # Generate dialogue lines
            dialogue_lines = await self.generate_dialogue_lines(character_id, personality, situation)
            
            # Generate voice audio for each line
            voice_files = []
            for i, line in enumerate(dialogue_lines):
                audio_path = await self.generate_voice_audio(line, character_id, i)
                voice_files.append({
                    'text': line,
                    'audio_path': audio_path,
                    'situation': situation
                })
            
            return {
                'success': True,
                'character_id': character_id,
                'dialogue': voice_files
            }
            
        except Exception as e:
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
                "Moving to {location}."
            ],
            'stealth': [
                "Area clear, moving forward.",
                "I've got visual on the target.",
                "Stay quiet, enemies nearby.",
                "Taking the silent approach.",
                "Target acquired, awaiting orders."
            ],
            'casual': [
                "What's the plan, team?",
                "I've seen worse situations.",
                "Remember our training.",
                "Stay focused, team.",
                "Let's get this done."
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
                target=self.generate_target()
            )
            
            # Add personality flavor using language model
            personality_prompt = f"A {personality.get('description', 'special forces operator')} says: {line}"
            
            inputs = self.tokenizer.encode(personality_prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.lm_model.generate(
                    inputs, 
                    max_length=50, 
                    num_return_sequences=1,
                    temperature=0.7,
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
            "the compound", "the building", "the treeline"
        ]
        return np.random.choice(locations)
    
    def generate_target(self) -> str:
        """Generate a random target reference"""
        targets = [
            "the objective", "hostages", "the package",
            "the leader", "the device"
        ]
        return np.random.choice(targets)
    
    async def generate_voice_audio(self, text: str, character_id: str, index: int) -> str:
        """Generate voice audio for the given text"""
        if not self.model_loaded:
            raise Exception("Models not loaded")
        
        # Preprocess text
        inputs = self.processor(text=text, return_tensors="pt")
        
        # Load speaker embeddings (in a real scenario, this would be character-specific)
        # Here we use a random embedding for demonstration
        speaker_embeddings = torch.randn((1, 512))
        
        # Generate speech
        with torch.no_grad():
            speech = self.tts_model.generate_speech(
                inputs["input_ids"].to(self.device), 
                speaker_embeddings.to(self.device), 
                vocoder=self.vocoder
            )
        
        # Convert to numpy array and save as WAV file
        speech = speech.cpu().numpy()
        audio_path = f"assets/audio/voice/{character_id}/line_{index}.wav"
        
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        
        # Save audio file
        sf.write(audio_path, speech, 22050)
        
        return audio_path
    
    async def generate_mission_briefing(self, mission_data: Dict[str, Any]) -> str:
        """Generate voice audio for mission briefing"""
        briefing_text = self.create_briefing_text(mission_data)
        audio_path = await self.generate_voice_audio(briefing_text, "command", 0)
        return audio_path
    
    def create_briefing_text(self, mission_data: Dict[str, Any]) -> str:
        """Create mission briefing text"""
        mission_type = mission_data.get('type', 'infiltration')
        location = mission_data.get('location', 'unknown')
        
        briefing_templates = {
            'infiltration': f"Team, your mission is to infiltrate the {location} and gather intelligence. Stay undetected and complete objectives silently.",
            'hostage_rescue': f"Team, we have hostages at {location}. Your mission is to rescue them with minimal casualties. Exercise extreme caution.",
            'assault': f"Team, your mission is to assault the {location} and neutralize all hostile forces. Use maximum force but watch for civilians.",
            'destruction': f"Team, your mission is to destroy the target at {location}. Use explosives and ensure complete destruction of the objective."
        }
        
        return briefing_templates.get(mission_type, briefing_templates['infiltration'])
    
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
            "character_voice_modeling"
        ]
    
    async def shutdown(self):
        """Cleanup resources"""
        if hasattr(self, 'lm_model'):
            del self.lm_model
        if hasattr(self, 'tts_model'):
            del self.tts_model
        if hasattr(self, 'vocoder'):
            del self.vocoder
        torch.cuda.empty_cache()
