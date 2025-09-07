import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from typing import Dict, Any, List
import asyncio

class CharacterCreationAgent:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        self.current_task = None
        
        # Initialize models
        self.setup_models()
    
    def setup_models(self):
        """Initialize AI models for character creation"""
        try:
            # Load Stable Diffusion for visual generation
            self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            # Load language model for personality generation
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.lm_model = GPT2LMHeadModel.from_pretrained("gpt2")
            
            self.model_loaded = True
            print("Character creation models loaded successfully")
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            self.model_loaded = False
    
    async def create_character(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new character with the given parameters"""
        self.current_task = "create_character"
        
        try:
            # Generate character concept
            concept = await self.generate_character_concept(params)
            
            # Generate visual appearance
            appearance = await self.generate_character_visual(concept)
            
            # Generate personality traits
            personality = await self.generate_personality(concept)
            
            # Generate backstory
            backstory = await self.generate_backstory(concept, personality)
            
            # Compile character data
            character_data = {
                'concept': concept,
                'appearance': appearance,
                'personality': personality,
                'backstory': backstory,
                'skills': self.generate_skills(concept, personality),
                'success': True,
                'character_id': f"char_{hash(str(concept))}_{int(asyncio.get_event_loop().time())}"
            }
            
            return character_data
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            self.current_task = None
    
    async def generate_character_concept(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate character concept based on parameters"""
        # Extract parameters
        character_type = params.get('type', 'special_forces')
        specialization = params.get('specialization', 'assault')
        
        # Generate concept using AI
        prompt = f"A {specialization} {character_type} operator, tactical gear, highly detailed, 4k, photorealistic"
        
        return {
            'type': character_type,
            'specialization': specialization,
            'prompt': prompt,
            'attributes': {
                'strength': np.random.randint(70, 100),
                'agility': np.random.randint(70, 100),
                'intelligence': np.random.randint(70, 100),
                'charisma': np.random.randint(50, 90)
            }
        }
    
    async def generate_character_visual(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Generate character visual appearance"""
        if not self.model_loaded:
            return {'error': 'Models not loaded'}
        
        # Generate image using Stable Diffusion
        with torch.no_grad():
            image = self.sd_pipeline(concept['prompt']).images[0]
        
        # Save image and return path
        image_path = f"assets/characters/{concept['type']}_{concept['specialization']}_{hash(concept['prompt'])}.png"
        image.save(image_path)
        
        return {
            'image_path': image_path,
            'model_data': await self.generate_3d_model(concept)
        }
    
    async def generate_3d_model(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Generate 3D model data for the character"""
        # This would interface with a 3D model generation AI
        # For now, return placeholder data
        return {
            'mesh': f"models/characters/{concept['type']}_{concept['specialization']}.fbx",
            'textures': [
                f"textures/characters/{concept['type']}_{concept['specialization']}_diffuse.png",
                f"textures/characters/{concept['type']}_{concept['specialization']}_normal.png",
                f"textures/characters/{concept['type']}_{concept['specialization']}_specular.png"
            ],
            'animations': {
                'idle': "animations/idle.fbx",
                'walk': "animations/walk.fbx",
                'run': "animations/run.fbx",
                'combat': "animations/combat.fbx"
            }
        }
    
    async def generate_personality(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Generate personality traits for the character"""
        # Use language model to generate personality
        prompt = f"Personality traits for a {concept['specialization']} {concept['type']} operator:"
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.lm_model.generate(
                inputs, 
                max_length=100, 
                num_return_sequences=1,
                temperature=0.7
            )
        
        personality_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract traits from generated text
        traits = self.extract_personality_traits(personality_text)
        
        return {
            'description': personality_text,
            'traits': traits,
            'voice_type': self.determine_voice_type(traits)
        }
    
    def extract_personality_traits(self, text: str) -> List[str]:
        """Extract personality traits from generated text"""
        # Simple implementation - would be more sophisticated in production
        traits = []
        trait_keywords = {
            'confident': ['confident', 'self-assured', 'bold'],
            'aggressive': ['aggressive', 'assertive', 'forceful'],
            'calm': ['calm', 'composed', 'steady'],
            'intelligent': ['intelligent', 'smart', 'clever'],
            'loyal': ['loyal', 'dedicated', 'faithful']
        }
        
        text_lower = text.lower()
        for trait, keywords in trait_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                traits.append(trait)
        
        return traits if traits else ['confident', 'loyal']
    
    def determine_voice_type(self, traits: List[str]) -> str:
        """Determine appropriate voice type based on personality traits"""
        if 'aggressive' in traits:
            return 'deep_assertive'
        elif 'calm' in traits:
            return 'calm_collected'
        elif 'intelligent' in traits:
            return 'articulate_precise'
        else:
            return 'neutral_authoritative'
    
    async def generate_backstory(self, concept: Dict[str, Any], personality: Dict[str, Any]) -> str:
        """Generate character backstory"""
        prompt = f"Backstory for a {concept['specialization']} {concept['type']} operator with personality: {personality['description']}"
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.lm_model.generate(
                inputs, 
                max_length=200, 
                num_return_sequences=1,
                temperature=0.7
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def generate_skills(self, concept: Dict[str, Any], personality: Dict[str, Any]) -> Dict[str, int]:
        """Generate skills based on character concept and personality"""
        base_skills = {
            'marksmanship': np.random.randint(70, 100),
            'stealth': np.random.randint(60, 95),
            'tactics': np.random.randint(65, 95),
            'endurance': np.random.randint(75, 100),
            'technical': np.random.randint(50, 90)
        }
        
        # Adjust based on specialization
        if concept['specialization'] == 'sniper':
            base_skills['marksmanship'] = min(100, base_skills['marksmanship'] + 15)
            base_skills['stealth'] = min(100, base_skills['stealth'] + 10)
        elif concept['specialization'] == 'engineer':
            base_skills['technical'] = min(100, base_skills['technical'] + 20)
        
        return base_skills
    
    def get_status(self) -> str:
        """Get current agent status"""
        if not self.model_loaded:
            return "offline"
        return "busy" if self.current_task else "idle"
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities"""
        return [
            "character_concept_generation",
            "visual_appearance_creation",
            "personality_generation",
            "backstory_creation",
            "skill_assignment"
        ]
    
    async def shutdown(self):
        """Cleanup resources"""
        if hasattr(self, 'sd_pipeline'):
            del self.sd_pipeline
        if hasattr(self, 'lm_model'):
            del self.lm_model
        torch.cuda.empty_cache()
