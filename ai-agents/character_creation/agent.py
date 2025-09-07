import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from typing import Dict, Any, List, Optional
import asyncio
import aiohttp
import json
from datetime import datetime
import logging
from pathlib import Path

class CharacterCreationAgent:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        self.current_task = None
        self.character_db = {}
        self.logger = self.setup_logging()
        
        # Initialize models
        self.setup_models()
    
    def setup_logging(self):
        """Setup logging for the agent"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/character_agent.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('CharacterCreationAgent')
    
    def setup_models(self):
        """Initialize AI models for character creation"""
        try:
            self.logger.info("Loading character creation models...")
            
            # Load Stable Diffusion for visual generation
            self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None  # Disable for more creative generations
            ).to(self.device)
            
            # Load language model for personality generation
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.lm_model = GPT2LMHeadModel.from_pretrained("gpt2")
            self.lm_model.to(self.device)
            
            self.model_loaded = True
            self.logger.info("Character creation models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            self.model_loaded = False
    
    async def initialize(self):
        """Initialize the agent"""
        self.logger.info("Initializing CharacterCreationAgent...")
        
        # Load existing characters from database
        await self.load_character_database()
        
        self.logger.info("CharacterCreationAgent initialized successfully")
    
    async def load_character_database(self):
        """Load character database from file"""
        try:
            db_path = Path("data/characters/character_db.json")
            if db_path.exists():
                with open(db_path, 'r') as f:
                    self.character_db = json.load(f)
                self.logger.info(f"Loaded {len(self.character_db)} characters from database")
            else:
                self.logger.info("No existing character database found")
        except Exception as e:
            self.logger.error(f"Error loading character database: {str(e)}")
    
    async def save_character_database(self):
        """Save character database to file"""
        try:
            db_path = Path("data/characters/character_db.json")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(db_path, 'w') as f:
                json.dump(self.character_db, f, indent=2)
            
            self.logger.info("Character database saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving character database: {str(e)}")
    
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
            
            # Generate skills
            skills = self.generate_skills(concept, personality)
            
            # Compile character data
            character_id = self.generate_character_id(concept)
            character_data = {
                'id': character_id,
                'concept': concept,
                'appearance': appearance,
                'personality': personality,
                'backstory': backstory,
                'skills': skills,
                'created_at': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            # Save to database
            self.character_db[character_id] = character_data
            await self.save_character_database()
            
            # Export to game format
            await self.export_character(character_data)
            
            return {
                'success': True,
                'character_id': character_id,
                'character_data': character_data
            }
            
        except Exception as e:
            self.logger.error(f"Error creating character: {str(e)}")
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
        faction = params.get('faction', 'ghosts')
        gender = params.get('gender', 'male')
        
        # Generate concept using AI
        prompt = f"A {specialization} {character_type} operator, {gender}, {faction}, tactical gear, highly detailed, 4k, photorealistic, unreal engine 5"
        
        # Generate attributes
        attributes = {
            'strength': np.random.randint(70, 100),
            'agility': np.random.randint(70, 100),
            'intelligence': np.random.randint(70, 100),
            'charisma': np.random.randint(50, 90),
            'endurance': np.random.randint(80, 100),
            'perception': np.random.randint(70, 95)
        }
        
        # Adjust attributes based on specialization
        if specialization == 'sniper':
            attributes['perception'] = min(100, attributes['perception'] + 10)
            attributes['agility'] = max(50, attributes['agility'] - 5)
        elif specialization == 'engineer':
            attributes['intelligence'] = min(100, attributes['intelligence'] + 15)
            attributes['strength'] = max(50, attributes['strength'] - 5)
        elif specialization == 'medic':
            attributes['intelligence'] = min(100, attributes['intelligence'] + 10)
            attributes['charisma'] = min(100, attributes['charisma'] + 5)
        
        return {
            'type': character_type,
            'specialization': specialization,
            'faction': faction,
            'gender': gender,
            'prompt': prompt,
            'attributes': attributes
        }
    
    async def generate_character_visual(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Generate character visual appearance"""
        if not self.model_loaded:
            raise Exception("Models not loaded")
        
        # Generate image using Stable Diffusion
        with torch.no_grad():
            image = self.sd_pipeline(
                concept['prompt'],
                num_inference_steps=30,
                guidance_scale=7.5
            ).images[0]
        
        # Save image and return path
        char_id = self.generate_character_id(concept)
        image_path = f"assets/characters/{char_id}/textures/diffuse.png"
        
        # Ensure directory exists
        Path(image_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path)
        
        return {
            'image_path': image_path,
            'model_data': await self.generate_3d_model(concept, char_id)
        }
    
    async def generate_3d_model(self, concept: Dict[str, Any], character_id: str) -> Dict[str, Any]:
        """Generate 3D model data for the character"""
        # This would interface with a 3D model generation AI or use pre-made templates
        # For now, we'll create placeholder data and file structure
        
        model_path = f"assets/characters/{character_id}/models"
        
        # Create directory structure
        Path(model_path).mkdir(parents=True, exist_ok=True)
        
        # Create placeholder model files
        model_files = {
            'mesh': f"{model_path}/character.fbx",
            'skeleton': f"{model_path}/skeleton.fbx",
            'rig': f"{model_path}/rig.fbx",
            'materials': f"{model_path}/materials.json"
        }
        
        # Create placeholder material file
        materials = {
            'skin': {
                'texture': f"assets/characters/{character_id}/textures/diffuse.png",
                'normal_map': f"assets/characters/{character_id}/textures/normal.png",
                'roughness': 0.7,
                'metallic': 0.1
            },
            'equipment': {
                'texture': f"assets/characters/{character_id}/textures/equipment_diffuse.png",
                'normal_map': f"assets/characters/{character_id}/textures/equipment_normal.png",
                'roughness': 0.5,
                'metallic': 0.3
            }
        }
        
        with open(model_files['materials'], 'w') as f:
            json.dump(materials, f, indent=2)
        
        # Create animation references
        animations = {
            'idle': "animations/base/idle.fbx",
            'walk': "animations/base/walk.fbx",
            'run': "animations/base/run.fbx",
            'combat_idle': "animations/combat/idle.fbx",
            'combat_walk': "animations/combat/walk.fbx",
            'reload': "animations/weapons/reload.fbx",
            'melee_attack': "animations/combat/melee_attack.fbx"
        }
        
        # Add specialization-specific animations
        if concept['specialization'] == 'sniper':
            animations['aim'] = "animations/weapons/sniper_aim.fbx"
            animations['prone'] = "animations/combat/prone.fbx"
        elif concept['specialization'] == 'engineer':
            animations['hack'] = "animations/skills/hack.fbx"
            animations['repair'] = "animations/skills/repair.fbx"
        
        model_files['animations'] = animations
        
        return model_files
    
    async def generate_personality(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Generate personality traits for the character"""
        # Use language model to generate personality
        prompt = f"Personality traits for a {concept['specialization']} {concept['type']} operator in a tactical combat team:"
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.lm_model.generate(
                inputs, 
                max_length=100, 
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        personality_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract traits from generated text
        traits = self.extract_personality_traits(personality_text)
        
        return {
            'description': personality_text,
            'traits': traits,
            'voice_type': self.determine_voice_type(traits),
            'behavior_patterns': self.generate_behavior_patterns(traits, concept['specialization'])
        }
    
    def extract_personality_traits(self, text: str) -> List[str]:
        """Extract personality traits from generated text"""
        # Simple implementation - would be more sophisticated in production
        traits = []
        trait_keywords = {
            'confident': ['confident', 'self-assured', 'bold', 'assertive'],
            'aggressive': ['aggressive', 'assertive', 'forceful', 'dominant'],
            'calm': ['calm', 'composed', 'steady', 'level-headed'],
            'intelligent': ['intelligent', 'smart', 'clever', 'strategic'],
            'loyal': ['loyal', 'dedicated', 'faithful', 'devoted'],
            'cautious': ['cautious', 'careful', 'prudent', 'wary'],
            'reckless': ['reckless', 'daring', 'bold', 'impulsive'],
            'stoic': ['stoic', 'reserved', 'unemotional', 'disciplined']
        }
        
        text_lower = text.lower()
        for trait, keywords in trait_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                traits.append(trait)
        
        # Ensure at least 3 traits
        while len(traits) < 3:
            additional_traits = list(trait_keywords.keys())
            np.random.shuffle(additional_traits)
            for trait in additional_traits:
                if trait not in traits:
                    traits.append(trait)
                    break
        
        return traits[:5]  # Limit to 5 traits
    
    def determine_voice_type(self, traits: List[str]) -> str:
        """Determine appropriate voice type based on personality traits"""
        if 'aggressive' in traits:
            return 'deep_assertive'
        elif 'calm' in traits:
            return 'calm_collected'
        elif 'intelligent' in traits:
            return 'articulate_precise'
        elif 'stoic' in traits:
            return 'neutral_reserved'
        else:
            return 'neutral_authoritative'
    
    def generate_behavior_patterns(self, traits: List[str], specialization: str) -> Dict[str, Any]:
        """Generate behavior patterns based on personality and specialization"""
        patterns = {
            'combat_style': 'balanced',
            'risk_tolerance': 0.5,
            'teamwork': 0.7,
            'initiative': 0.6
        }
        
        # Adjust based on traits
        if 'aggressive' in traits:
            patterns['combat_style'] = 'aggressive'
            patterns['risk_tolerance'] = 0.8
        elif 'cautious' in traits:
            patterns['combat_style'] = 'defensive'
            patterns['risk_tolerance'] = 0.3
        
        if 'reckless' in traits:
            patterns['risk_tolerance'] = 0.9
        
        # Adjust based on specialization
        if specialization == 'sniper':
            patterns['combat_style'] = 'stealth'
            patterns['risk_tolerance'] = max(0.2, patterns['risk_tolerance'] - 0.2)
        elif specialization == 'engineer':
            patterns['teamwork'] = 0.8
        elif specialization == 'medic':
            patterns['teamwork'] = 0.9
            patterns['risk_tolerance'] = min(0.7, patterns['risk_tolerance'] + 0.1)
        
        return patterns
    
    async def generate_backstory(self, concept: Dict[str, Any], personality: Dict[str, Any]) -> str:
        """Generate character backstory"""
        prompt = f"Backstory for a {concept['specialization']} {concept['type']} operator with personality: {personality['description']}"
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.lm_model.generate(
                inputs, 
                max_length=200, 
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def generate_skills(self, concept: Dict[str, Any], personality: Dict[str, Any]) -> Dict[str, int]:
        """Generate skills based on character concept and personality"""
        base_skills = {
            'marksmanship': np.random.randint(70, 100),
            'stealth': np.random.randint(60, 95),
            'tactics': np.random.randint(65, 95),
            'endurance': np.random.randint(75, 100),
            'technical': np.random.randint(50, 90),
            'medical': np.random.randint(40, 80),
            'explosives': np.random.randint(50, 85),
            'leadership': np.random.randint(40, 90)
        }
        
        # Adjust based on specialization
        if concept['specialization'] == 'sniper':
            base_skills['marksmanship'] = min(100, base_skills['marksmanship'] + 15)
            base_skills['stealth'] = min(100, base_skills['stealth'] + 10)
            base_skills['technical'] = max(40, base_skills['technical'] - 10)
        elif concept['specialization'] == 'engineer':
            base_skills['technical'] = min(100, base_skills['technical'] + 20)
            base_skills['explosives'] = min(100, base_skills['explosives'] + 10)
            base_skills['marksmanship'] = max(50, base_skills['marksmanship'] - 10)
        elif concept['specialization'] == 'medic':
            base_skills['medical'] = min(100, base_skills['medical'] + 20)
            base_skills['leadership'] = min(100, base_skills['leadership'] + 10)
            base_skills['explosives'] = max(30, base_skills['explosives'] - 10)
        
        # Adjust based on personality traits
        traits = personality.get('traits', [])
        if 'intelligent' in traits:
            base_skills['tactics'] = min(100, base_skills['tactics'] + 10)
            base_skills['technical'] = min(100, base_skills['technical'] + 5)
        
        if 'aggressive' in traits:
            base_skills['marksmanship'] = min(100, base_skills['marksmanship'] + 5)
            base_skills['explosives'] = min(100, base_skills['explosives'] + 5)
        
        return base_skills
    
    def generate_character_id(self, concept: Dict[str, Any]) -> str:
        """Generate a unique ID for the character"""
        timestamp = int(datetime.now().timestamp())
        type_abbr = concept['type'][:3].upper()
        spec_abbr = concept['specialization'][:3].upper()
        random_suffix = np.random.randint(1000, 9999)
        
        return f"CHAR_{type_abbr}_{spec_abbr}_{timestamp}_{random_suffix}"
    
    async def export_character(self, character_data: Dict[str, Any]):
        """Export character data to game format"""
        try:
            char_id = character_data['id']
            export_path = f"exports/characters/{char_id}"
            
            # Create export directory
            Path(export_path).mkdir(parents=True, exist_ok=True)
            
            # Export character data as JSON
            with open(f"{export_path}/character.json", 'w') as f:
                json.dump(character_data, f, indent=2)
            
            # Export for Unity
            await self.export_unity_character(character_data, export_path)
            
            # Export for Unreal Engine
            await self.export_unreal_character(character_data, export_path)
            
            self.logger.info(f"Exported character {char_id} to {export_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting character: {str(e)}")
    
    async def export_unity_character(self, character_data: Dict[str, Any], export_path: str):
        """Export character data for Unity game engine"""
        unity_data = {
            'name': character_data['id'],
            'prefabPath': f"Characters/{character_data['id']}/Prefabs/Character.prefab",
            'attributes': character_data['concept']['attributes'],
            'skills': character_data['skills'],
            'textures': {
                'diffuse': character_data['appearance']['image_path'],
                'normal': character_data['appearance']['image_path'].replace('diffuse', 'normal')
            },
            'animations': character_data['appearance']['model_data']['animations']
        }
        
        unity_path = f"{export_path}/unity"
        Path(unity_path).mkdir(parents=True, exist_ok=True)
        
        with open(f"{unity_path}/character_data.json", 'w') as f:
            json.dump(unity_data, f, indent=2)
    
    async def export_unreal_character(self, character_data: Dict[str, Any], export_path: str):
        """Export character data for Unreal Engine"""
        unreal_data = {
            'name': character_data['id'],
            'blueprintPath': f"/Game/Characters/{character_data['id']}/BP_Character.BP_Character",
            'attributes': character_data['concept']['attributes'],
            'skills': character_data['skills'],
            'materials': [
                {
                    'name': 'CharacterMaterial',
                    'textures': {
                        'baseColor': character_data['appearance']['image_path'],
                        'normal': character_data['appearance']['image_path'].replace('diffuse', 'normal')
                    }
                }
            ],
            'animations': character_data['appearance']['model_data']['animations']
        }
        
        unreal_path = f"{export_path}/unreal"
        Path(unreal_path).mkdir(parents=True, exist_ok=True)
        
        with open(f"{unreal_path}/character_data.json", 'w') as f:
            json.dump(unreal_data, f, indent=2)
    
    async def update_character(self, character_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing character"""
        if character_id not in self.character_db:
            return {'success': False, 'error': 'Character not found'}
        
        try:
            # Update character data
            self.character_db[character_id].update(updates)
            self.character_db[character_id]['updated_at'] = datetime.now().isoformat()
            self.character_db[character_id]['version'] = str(float(self.character_db[character_id]['version']) + 0.1)
            
            # Save to database
            await self.save_character_database()
            
            # Re-export character
            await self.export_character(self.character_db[character_id])
            
            return {
                'success': True,
                'character_id': character_id,
                'updated_data': self.character_db[character_id]
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def get_character(self, character_id: str) -> Optional[Dict[str, Any]]:
        """Get character data by ID"""
        return self.character_db.get(character_id)
    
    async def list_characters(self, filter_type: str = None) -> List[Dict[str, Any]]:
        """List all characters, optionally filtered by type"""
        characters = list(self.character_db.values())
        
        if filter_type:
            characters = [c for c in characters if c['concept']['type'] == filter_type]
        
        return characters
    
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
            "skill_assignment",
            "character_export",
            "character_update"
        ]
    
    async def shutdown(self):
        """Cleanup resources"""
        self.logger.info("Shutting down CharacterCreationAgent...")
        
        # Clear model resources
        if hasattr(self, 'sd_pipeline'):
            del self.sd_pipeline
        if hasattr(self, 'lm_model'):
            del self.lm_model
        
        torch.cuda.empty_cache()
        self.logger.info("CharacterCreationAgent shutdown complete")
