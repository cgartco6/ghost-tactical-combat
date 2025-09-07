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

class WeaponCreationAgent:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        self.current_task = None
        self.weapon_db = {}
        self.logger = self.setup_logging()
        
        # Initialize models
        self.setup_models()
    
    def setup_logging(self):
        """Setup logging for the agent"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/weapon_agent.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('WeaponCreationAgent')
    
    def setup_models(self):
        """Initialize AI models for weapon creation"""
        try:
            self.logger.info("Loading weapon creation models...")
            
            # Load Stable Diffusion for weapon design
            self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None
            ).to(self.device)
            
            # Load language model for weapon description
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.lm_model = GPT2LMHeadModel.from_pretrained("gpt2")
            self.lm_model.to(self.device)
            
            self.model_loaded = True
            self.logger.info("Weapon creation models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            self.model_loaded = False
    
    async def initialize(self):
        """Initialize the agent"""
        self.logger.info("Initializing WeaponCreationAgent...")
        
        # Load existing weapons from database
        await self.load_weapon_database()
        
        self.logger.info("WeaponCreationAgent initialized successfully")
    
    async def load_weapon_database(self):
        """Load weapon database from file"""
        try:
            db_path = Path("data/weapons/weapon_db.json")
            if db_path.exists():
                with open(db_path, 'r') as f:
                    self.weapon_db = json.load(f)
                self.logger.info(f"Loaded {len(self.weapon_db)} weapons from database")
            else:
                self.logger.info("No existing weapon database found")
        except Exception as e:
            self.logger.error(f"Error loading weapon database: {str(e)}")
    
    async def save_weapon_database(self):
        """Save weapon database to file"""
        try:
            db_path = Path("data/weapons/weapon_db.json")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(db_path, 'w') as f:
                json.dump(self.weapon_db, f, indent=2)
            
            self.logger.info("Weapon database saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving weapon database: {str(e)}")
    
    async def design_weapon(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Design a new weapon with the given parameters"""
        self.current_task = "design_weapon"
        
        try:
            # Generate weapon concept
            concept = await self.generate_weapon_concept(params)
            
            # Generate visual assets
            assets = await self.generate_weapon_assets(concept)
            
            # Generate stats and mechanics
            stats = await self.generate_weapon_stats(concept)
            
            # Generate upgrade path
            upgrades = await self.generate_upgrade_path(concept)
            
            # Generate audio assets
            audio = await self.generate_weapon_audio(concept)
            
            # Compile weapon data
            weapon_id = self.generate_weapon_id(concept)
            weapon_data = {
                'id': weapon_id,
                'concept': concept,
                'assets': assets,
                'stats': stats,
                'upgrades': upgrades,
                'audio': audio,
                'created_at': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            # Save to database
            self.weapon_db[weapon_id] = weapon_data
            await self.save_weapon_database()
            
            # Export to game format
            await self.export_weapon(weapon_data)
            
            return {
                'success': True,
                'weapon_id': weapon_id,
                'weapon_data': weapon_data
            }
            
        except Exception as e:
            self.logger.error(f"Error creating weapon: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            self.current_task = None
    
    async def generate_weapon_concept(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate weapon concept based on parameters"""
        # Extract parameters
        weapon_type = params.get('type', 'assault_rifle')
        rarity = params.get('rarity', 'common')
        era = params.get('era', 'modern')
        faction = params.get('faction', 'nato')
        
        # Generate concept using AI
        prompt = self.create_weapon_prompt(weapon_type, rarity, era, faction)
        
        # Generate description using language model
        description = await self.generate_weapon_description(prompt)
        
        return {
            'type': weapon_type,
            'rarity': rarity,
            'era': era,
            'faction': faction,
            'prompt': prompt,
            'description': description,
            'name': self.generate_weapon_name(weapon_type, rarity, faction),
            'manufacturer': self.generate_manufacturer(faction)
        }
    
    def create_weapon_prompt(self, weapon_type: str, rarity: str, era: str, faction: str) -> str:
        """Create a prompt for weapon generation"""
        era_descriptions = {
            'ww2': 'World War II era',
            'cold_war': 'Cold War era',
            'modern': 'modern',
            'futuristic': 'futuristic'
        }
        
        rarity_descriptions = {
            'common': 'standard issue',
            'uncommon': 'tactical',
            'rare': 'special forces',
            'epic': 'elite',
            'legendary': 'prototype'
        }
        
        faction_descriptions = {
            'nato': 'NATO forces',
            'warsaw_pact': 'Warsaw Pact',
            'mercenary': 'mercenary group',
            'terrorist': 'insurgent forces'
        }
        
        return f"{rarity_descriptions[rarity]} {era_descriptions[era]} {weapon_type} for {faction_descriptions[faction]}, 4k, photorealistic, game asset, unreal engine 5"
    
    async def generate_weapon_description(self, prompt: str) -> str:
        """Generate weapon description using language model"""
        description_prompt = f"Describe this weapon in detail: {prompt}"
        
        inputs = self.tokenizer.encode(description_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.lm_model.generate(
                inputs, 
                max_length=100, 
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def generate_weapon_name(self, weapon_type: str, rarity: str, faction: str) -> str:
        """Generate a name for the weapon"""
        prefixes = {
            'common': ['Standard', 'Issue', 'Service'],
            'uncommon': ['Tactical', 'Combat', 'Field'],
            'rare': ['Special', 'Advanced', 'Elite'],
            'epic': ['Superior', 'Enhanced', 'Precision'],
            'legendary': ['Prototype', 'Experimental', 'Omega']
        }
        
        bases = {
            'assault_rifle': ['Assault Rifle', 'Battle Rifle', 'Combat Rifle'],
            'sniper_rifle': ['Sniper Rifle', 'Designated Marksman Rifle', 'Precision Rifle'],
            'smg': ['SMG', 'Submachine Gun', 'PDW'],
            'shotgun': ['Shotgun', 'Combat Shotgun', 'Tactical Shotgun'],
            'lmg': ['LMG', 'Squad Automatic Weapon', 'Light Machine Gun'],
            'pistol': ['Pistol', 'Handgun', 'Sidearm']
        }
        
        faction_suffixes = {
            'nato': ['NATO', 'Allied', 'Coalition'],
            'warsaw_pact': ['Warsaw', 'Eastern', 'Pact'],
            'mercenary': ['Merc', 'Contractor', 'Privateer'],
            'terrorist': ['Insurgent', 'Guerilla', 'Freedom Fighter']
        }
        
        prefix = random.choice(prefixes[rarity])
        base = random.choice(bases[weapon_type])
        suffix = random.choice(faction_suffixes[faction])
        
        return f"{prefix} {base} {suffix}"
    
    def generate_manufacturer(self, faction: str) -> str:
        """Generate a manufacturer for the weapon"""
        manufacturers = {
            'nato': ['Colt', 'Heckler & Koch', 'FN Herstal', 'SIG Sauer', 'Beretta'],
            'warsaw_pact': ['Kalashnikov Concern', 'Zbrojovka Brno', 'Norinco', 'Arsenal', 'Zastava'],
            'mercenary': ['Valkyrie Arms', 'Phoenix Defense', 'Aegis Technologies', 'Ronin Works'],
            'terrorist': ['Black Market Custom', 'Underground Arms', 'Freedom Works', 'Liberation Armory']
        }
        
        return random.choice(manufacturers.get(faction, ['Generic Arms']))
    
    async def generate_weapon_assets(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visual assets for the weapon"""
        if not self.model_loaded:
            raise Exception("Models not loaded")
        
        assets = {}
        weapon_type = concept['type']
        weapon_id = self.generate_weapon_id(concept)
        
        # Generate weapon texture
        texture = await self.generate_weapon_texture(concept)
        assets['texture'] = texture
        
        # Generate normal map
        normal_map = await self.generate_normal_map(concept, texture['diffuse'])
        assets['normal_map'] = normal_map
        
        # Generate model (placeholder - would use a 3D model generator)
        model = await self.generate_weapon_model(concept)
        assets['model'] = model
        
        # Generate icons and UI assets
        icons = await self.generate_weapon_icons(concept)
        assets['icons'] = icons
        
        return assets
    
    async def generate_weapon_texture(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Generate texture for the weapon"""
        prompt = f"{concept['prompt']}, texture, 4k, PBR, metallic, roughness"
        
        with torch.no_grad():
            image = self.sd_pipeline(
                prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                width=1024,
                height=1024
            ).images[0]
        
        # Save texture
        weapon_id = self.generate_weapon_id(concept)
        texture_path = f"assets/weapons/{weapon_id}/textures/diffuse.png"
        
        Path(texture_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(texture_path)
        
        # Generate additional texture maps
        metallic_path = texture_path.replace('diffuse', 'metallic')
        roughness_path = texture_path.replace('diffuse', 'roughness')
        ao_path = texture_path.replace('diffuse', 'ao')
        
        self.generate_metallic_map(concept, texture_path, metallic_path)
        self.generate_roughness_map(concept, texture_path, roughness_path)
        self.generate_ao_map(concept, texture_path, ao_path)
        
        return {
            'diffuse': texture_path,
            'metallic': metallic_path,
            'roughness': roughness_path,
            'ao': ao_path
        }
    
    def generate_metallic_map(self, concept: Dict[str, Any], texture_path: str, metallic_path: str):
        """Generate a metallic map for the weapon"""
        # In a real implementation, this would use proper material analysis
        # For now, we'll create a placeholder metallic map
        img = Image.open(texture_path)
        width, height = img.size
        
        metallic_img = Image.new('L', (width, height))
        pixels = []
        
        # Different metallic values based on weapon type
        metallic_values = {
            'assault_rifle': 0.7,
            'sniper_rifle': 0.8,
            'smg': 0.6,
            'shotgun': 0.5,
            'lmg': 0.7,
            'pistol': 0.8
        }
        
        base_metallic = metallic_values.get(concept['type'], 0.5)
        
        for _ in range(width * height):
            # Add some variation
            var = random.uniform(-0.2, 0.2)
            value = int(min(max(base_metallic + var, 0), 1) * 255)
            pixels.append(value)
        
        metallic_img.putdata(pixels)
        metallic_img.save(metallic_path)
    
    def generate_roughness_map(self, concept: Dict[str, Any], texture_path: str, roughness_path: str):
        """Generate a roughness map for the weapon"""
        img = Image.open(texture_path)
        width, height = img.size
        
        roughness_img = Image.new('L', (width, height))
        pixels = []
        
        # Different roughness values based on weapon type
        roughness_values = {
            'assault_rifle': 0.6,
            'sniper_rifle': 0.4,
            'smg': 0.5,
            'shotgun': 0.7,
            'lmg': 0.6,
            'pistol': 0.3
        }
        
        base_roughness = roughness_values.get(concept['type'], 0.5)
        
        for _ in range(width * height):
            # Add some variation
            var = random.uniform(-0.1, 0.1)
            value = int(min(max(base_roughness + var, 0), 1) * 255)
            pixels.append(value)
        
        roughness_img.putdata(pixels)
        roughness_img.save(roughness_path)
    
    def generate_ao_map(self, concept: Dict[str, Any], texture_path: str, ao_path: str):
        """Generate an ambient occlusion map for the weapon"""
        img = Image.open(texture_path)
        width, height = img.size
        
        ao_img = Image.new('L', (width, height))
        pixels = []
        
        for _ in range(width * height):
            # Random AO values with some pattern
            value = random.randint(200, 255)
            pixels.append(value)
        
        ao_img.putdata(pixels)
        ao_img.save(ao_path)
    
    async def generate_normal_map(self, concept: Dict[str, Any], texture_path: str) -> str:
        """Generate a normal map for the weapon"""
        # In a real implementation, this would use proper normal map generation
        # For now, we'll create a placeholder normal map
        weapon_id = self.generate_weapon_id(concept)
        normal_path = f"assets/weapons/{weapon_id}/textures/normal.png"
        
        img = Image.open(texture_path)
        width, height = img.size
        
        normal_img = Image.new('RGB', (width, height))
        pixels = []
        
        for y in range(height):
            for x in range(width):
                # Simple normal map (mostly blue with some red/green for variation)
                r = random.randint(120, 140)
                g = random.randint(120, 140)
                b = random.randint(240, 255)
                pixels.append((r, g, b))
        
        normal_img.putdata(pixels)
        normal_img.save(normal_path)
        
        return normal_path
    
    async def generate_weapon_model(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Generate 3D model data for the weapon"""
        weapon_id = self.generate_weapon_id(concept)
        model_path = f"assets/weapons/{weapon_id}/models"
        
        # Create directory structure
        Path(model_path).mkdir(parents=True, exist_ok=True)
        
        # Create placeholder model files
        model_files = {
            'mesh': f"{model_path}/weapon.fbx",
            'lod_meshes': [
                f"{model_path}/weapon_lod0.fbx",
                f"{model_path}/weapon_lod1.fbx",
                f"{model_path}/weapon_lod2.fbx"
            ],
            'skeleton': f"{model_path}/skeleton.fbx",
            'rig': f"{model_path}/rig.fbx",
            'materials': f"{model_path}/materials.json"
        }
        
        # Create placeholder material file
        materials = {
            'metal': {
                'texture': f"assets/weapons/{weapon_id}/textures/diffuse.png",
                'normal_map': f"assets/weapons/{weapon_id}/textures/normal.png",
                'metallic_map': f"assets/weapons/{weapon_id}/textures/metallic.png",
                'roughness_map': f"assets/weapons/{weapon_id}/textures/roughness.png",
                'ao_map': f"assets/weapons/{weapon_id}/textures/ao.png"
            },
            'plastic': {
                'texture': f"assets/weapons/{weapon_id}/textures/diffuse.png",
                'normal_map': f"assets/weapons/{weapon_id}/textures/normal.png",
                'metallic_map': f"assets/weapons/{weapon_id}/textures/metallic.png",
                'roughness_map': f"assets/weapons/{weapon_id}/textures/roughness.png",
                'ao_map': f"assets/weapons/{weapon_id}/textures/ao.png"
            }
        }
        
        with open(model_files['materials'], 'w') as f:
            json.dump(materials, f, indent=2)
        
        # Create animation references
        animations = {
            'idle': "animations/weapons/idle.fbx",
            'fire': "animations/weapons/fire.fbx",
            'reload': "animations/weapons/reload.fbx",
            'aim': "animations/weapons/aim.fbx",
            'sprint': "animations/weapons/sprint.fbx"
        }
        
        model_files['animations'] = animations
        
        return model_files
    
    async def generate_weapon_icons(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Generate icons and UI assets for the weapon"""
        weapon_id = self.generate_weapon_id(concept)
        icons_path = f"assets/weapons/{weapon_id}/ui"
        
        # Create directory structure
        Path(icons_path).mkdir(parents=True, exist_ok=True)
        
        # Generate icon prompt
        prompt = f"{concept['prompt']}, icon, white background, game UI asset"
        
        with torch.no_grad():
            icon = self.sd_pipeline(
                prompt,
                num_inference_steps=20,
                guidance_scale=7.0,
                width=256,
                height=256
            ).images[0]
        
        # Save icons
        icon_path = f"{icons_path}/icon.png"
        icon.save(icon_path)
        
        # Create different sizes
        sizes = {
            'small': (64, 64),
            'medium': (128, 128),
            'large': (256, 256)
        }
        
        icon_variants = {}
        for size_name, size in sizes.items():
            variant_path = f"{icons_path}/icon_{size_name}.png"
            resized_icon = icon.resize(size, Image.LANCZOS)
            resized_icon.save(variant_path)
            icon_variants[size_name] = variant_path
        
        return {
            'main': icon_path,
            'variants': icon_variants,
            'hud_element': f"{icons_path}/hud_element.png"
        }
    
    async def generate_weapon_stats(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Generate stats for the weapon"""
        weapon_type = concept['type']
        rarity = concept['rarity']
        
        # Base stats for different weapon types
        base_stats = {
            'assault_rifle': {
                'damage': 25,
                'fire_rate': 600,
                'accuracy': 70,
                'range': 300,
                'mobility': 70,
                'control': 60,
                'magazine_size': 30,
                'reload_time': 2.5
            },
            'sniper_rifle': {
                'damage': 100,
                'fire_rate': 40,
                'accuracy': 95,
                'range': 1000,
                'mobility': 40,
                'control': 50,
                'magazine_size': 5,
                'reload_time': 3.5
            },
            'smg': {
                'damage': 20,
                'fire_rate': 800,
                'accuracy': 60,
                'range': 150,
                'mobility': 85,
                'control': 40,
                'magazine_size': 25,
                'reload_time': 2.0
            },
            'shotgun': {
                'damage': 80,  # per pellet
                'fire_rate': 70,
                'accuracy': 50,  # spread-based
                'range': 50,
                'mobility': 60,
                'control': 30,
                'magazine_size': 8,
                'reload_time': 4.0
            },
            'lmg': {
                'damage': 30,
                'fire_rate': 700,
                'accuracy': 65,
                'range': 400,
                'mobility': 50,
                'control': 35,
                'magazine_size': 100,
                'reload_time': 5.0
            },
            'pistol': {
                'damage': 35,
                'fire_rate': 300,
                'accuracy': 75,
                'range': 100,
                'mobility': 90,
                'control': 70,
                'magazine_size': 12,
                'reload_time': 1.5
            }
        }
        
        # Rarity modifiers
        rarity_modifiers = {
            'common': 1.0,
            'uncommon': 1.1,
            'rare': 1.25,
            'epic': 1.5,
            'legendary': 2.0
        }
        
        # Get base stats for weapon type
        stats = base_stats.get(weapon_type, base_stats['assault_rifle']).copy()
        
        # Apply rarity modifier
        modifier = rarity_modifiers[rarity]
        for stat in ['damage', 'fire_rate', 'accuracy', 'range', 'mobility', 'control', 'magazine_size']:
            if stat != 'fire_rate':  # Fire rate is handled differently
                stats[stat] = int(stats[stat] * modifier)
        
        # Fire rate is inverse (higher is better)
        if 'fire_rate' in stats:
            stats['fire_rate'] = int(stats['fire_rate'] * (2 - modifier))
        
        # Reload time is inverse (lower is better)
        if 'reload_time' in stats:
            stats['reload_time'] = max(0.5, stats['reload_time'] / modifier)
        
        return stats
    
    async def generate_upgrade_path(self, concept: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate upgrade path for the weapon"""
        upgrades = []
        weapon_type = concept['type']
        rarity = concept['rarity']
        
        # Define possible upgrades based on weapon type
        upgrade_options = {
            'assault_rifle': [
                {'name': 'Extended Magazine', 'cost': 500, 'effect': {'magazine_size': 10}},
                {'name': 'Improved Barrel', 'cost': 800, 'effect': {'damage': 5, 'accuracy': 5}},
                {'name': 'Recoil Reduction', 'cost': 600, 'effect': {'control': 10}},
                {'name': 'Lightweight Stock', 'cost': 400, 'effect': {'mobility': 10}},
                {'name': 'Optical Sight', 'cost': 700, 'effect': {'accuracy': 15}}
            ],
            'sniper_rifle': [
                {'name': 'High-Power Scope', 'cost': 1000, 'effect': {'accuracy': 10, 'range': 100}},
                {'name': 'Bipod', 'cost': 600, 'effect': {'control': 15}},
                {'name': 'Suppressor', 'cost': 800, 'effect': {'stealth': True}},
                {'name': 'Extended Magazine', 'cost': 500, 'effect': {'magazine_size': 3}},
                {'name': 'Armor-Piercing Rounds', 'cost': 900, 'effect': {'damage': 20}}
            ],
            'smg': [
                {'name': 'Extended Magazine', 'cost': 400, 'effect': {'magazine_size': 10}},
                {'name': 'Rapid Fire', 'cost': 700, 'effect': {'fire_rate': 200}},
                {'name': 'Laser Sight', 'cost': 500, 'effect': {'accuracy': 10}},
                {'name': 'Lightweight Frame', 'cost': 600, 'effect': {'mobility': 10}},
                {'name': 'Suppressor', 'cost': 800, 'effect': {'stealth': True}}
            ],
            'shotgun': [
                {'name': 'Extended Tube', 'cost': 500, 'effect': {'magazine_size': 4}},
                {'name': 'Choke', 'cost': 600, 'effect': {'accuracy': 15}},
                {'name': 'Slug Rounds', 'cost': 800, 'effect': {'damage': 20, 'range': 50}},
                {'name': 'Quick Pump', 'cost': 700, 'effect': {'fire_rate': 20}},
                {'name': 'Dragon\'s Breath', 'cost': 900, 'effect': {'fire_damage': True}}
            ]
        }
        
        # Select upgrades based on rarity
        num_upgrades = {
            'common': 2,
            'uncommon': 3,
            'rare': 4,
            'epic': 5,
            'legendary': 6
        }
        
        options = upgrade_options.get(weapon_type, upgrade_options['assault_rifle'])
        selected_upgrades = random.sample(options, min(num_upgrades[rarity], len(options)))
        
        for upgrade in selected_upgrades:
            upgrades.append(upgrade)
        
        return upgrades
    
    async def generate_weapon_audio(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Generate audio assets for the weapon"""
        weapon_id = self.generate_weapon_id(concept)
        audio_path = f"assets/weapons/{weapon_id}/audio"
        
        # Create directory structure
        Path(audio_path).mkdir(parents=True, exist_ok=True)
        
        # Placeholder audio files (in a real implementation, these would be generated)
        audio_files = {
            'fire': f"{audio_path}/fire.wav",
            'reload': f"{audio_path}/reload.wav",
            'empty': f"{audio_path}/empty.wav",
            'equip': f"{audio_path}/equip.wav",
            'unequip': f"{audio_path}/unequip.wav"
        }
        
        # Create placeholder audio files
        for audio_file in audio_files.values():
            Path(audio_file).parent.mkdir(parents=True, exist_ok=True)
            # In a real implementation, this would generate actual audio
        
        return audio_files
    
    def generate_weapon_id(self, concept: Dict[str, Any]) -> str:
        """Generate a unique ID for the weapon"""
        timestamp = int(datetime.now().timestamp())
        type_abbr = concept['type'][:3].upper()
        rarity_abbr = concept['rarity'][:1].upper()
        random_suffix = np.random.randint(1000, 9999)
        
        return f"WPN_{type_abbr}_{rarity_abbr}_{timestamp}_{random_suffix}"
    
    async def export_weapon(self, weapon_data: Dict[str, Any]):
        """Export weapon data to game format"""
        try:
            weapon_id = weapon_data['id']
            export_path = f"exports/weapons/{weapon_id}"
            
            # Create export directory
            Path(export_path).mkdir(parents=True, exist_ok=True)
            
            # Export weapon data as JSON
            with open(f"{export_path}/weapon.json", 'w') as f:
                json.dump(weapon_data, f, indent=2)
            
            # Export for Unity
            await self.export_unity_weapon(weapon_data, export_path)
            
            # Export for Unreal Engine
            await self.export_unreal_weapon(weapon_data, export_path)
            
            self.logger.info(f"Exported weapon {weapon_id} to {export_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting weapon: {str(e)}")
    
    async def export_unity_weapon(self, weapon_data: Dict[str, Any], export_path: str):
        """Export weapon data for Unity game engine"""
        unity_data = {
            'name': weapon_data['id'],
            'prefabPath': f"Weapons/{weapon_data['id']}/Prefabs/Weapon.prefab",
            'stats': weapon_data['stats'],
            'upgrades': weapon_data['upgrades'],
            'model': weapon_data['assets']['model']['mesh'],
            'textures': {
                'diffuse': weapon_data['assets']['texture']['diffuse'],
                'normal': weapon_data['assets']['normal_map'],
                'metallic': weapon_data['assets']['texture']['metallic'],
                'roughness': weapon_data['assets']['texture']['roughness'],
                'ao': weapon_data['assets']['texture']['ao']
            },
            'audio': weapon_data['audio'],
            'icons': weapon_data['assets']['icons']
        }
        
        unity_path = f"{export_path}/unity"
        Path(unity_path).mkdir(parents=True, exist_ok=True)
        
        with open(f"{unity_path}/weapon_data.json", 'w') as f:
            json.dump(unity_data, f, indent=2)
    
    async def export_unreal_weapon(self, weapon_data: Dict[str, Any], export_path: str):
        """Export weapon data for Unreal Engine"""
        unreal_data = {
            'name': weapon_data['id'],
            'blueprintPath': f"/Game/Weapons/{weapon_data['id']}/BP_Weapon.BP_Weapon",
            'stats': weapon_data['stats'],
            'upgrades': weapon_data['upgrades'],
            'model': weapon_data['assets']['model']['mesh'],
            'materials': [
                {
                    'name': 'WeaponMaterial',
                    'textures': {
                        'baseColor': weapon_data['assets']['texture']['diffuse'],
                        'normal': weapon_data['assets']['normal_map'],
                        'metallic': weapon_data['assets']['texture']['metallic'],
                        'roughness': weapon_data['assets']['texture']['roughness'],
                        'ao': weapon_data['assets']['texture']['ao']
                    }
                }
            ],
            'audio': weapon_data['audio'],
            'ui': {
                'icons': weapon_data['assets']['icons']
            }
        }
        
        unreal_path = f"{export_path}/unreal"
        Path(unreal_path).mkdir(parents=True, exist_ok=True)
        
        with open(f"{unreal_path}/weapon_data.json", 'w') as f:
            json.dump(unreal_data, f, indent=2)
    
    async def update_weapon(self, weapon_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing weapon"""
        if weapon_id not in self.weapon_db:
            return {'success': False, 'error': 'Weapon not found'}
        
        try:
            # Update weapon data
            self.weapon_db[weapon_id].update(updates)
            self.weapon_db[weapon_id]['updated_at'] = datetime.now().isoformat()
            self.weapon_db[weapon_id]['version'] = str(float(self.weapon_db[weapon_id]['version']) + 0.1)
            
            # Save to database
            await self.save_weapon_database()
            
            # Re-export weapon
            await self.export_weapon(self.weapon_db[weapon_id])
            
            return {
                'success': True,
                'weapon_id': weapon_id,
                'updated_data': self.weapon_db[weapon_id]
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def get_weapon(self, weapon_id: str) -> Optional[Dict[str, Any]]:
        """Get weapon data by ID"""
        return self.weapon_db.get(weapon_id)
    
    async def list_weapons(self, filter_type: str = None) -> List[Dict[str, Any]]:
        """List all weapons, optionally filtered by type"""
        weapons = list(self.weapon_db.values())
        
        if filter_type:
            weapons = [w for w in weapons if w['concept']['type'] == filter_type]
        
        return weapons
    
    def get_status(self) -> str:
        """Get current agent status"""
        if not self.model_loaded:
            return "offline"
        return "busy" if self.current_task else "idle"
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities"""
        return [
            "weapon_concept_generation",
            "weapon_modeling",
            "texture_creation",
            "stat_balancing",
            "upgrade_design",
            "weapon_export"
        ]
    
    async def shutdown(self):
        """Cleanup resources"""
        self.logger.info("Shutting down WeaponCreationAgent...")
        
        # Clear model resources
        if hasattr(self, 'sd_pipeline'):
            del self.sd_pipeline
        if hasattr(self, 'lm_model'):
            del self.lm_model
        
        torch.cuda.empty_cache()
        self.logger.info("WeaponCreationAgent shutdown complete")
